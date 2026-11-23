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
"""Offline tests of the state of the objects of a session and of a DataFrame.

PySpark keeps the state of the readers, the writers, the builders and the other objects of a
session or of a DataFrame in their `__dict__`, the instance attributes that its `__init__` and
its methods set, which `vars()` gives in the order that they were set, and its classes are plain
classes, which a user subclasses and constructs of the arguments of their `__init__`. Each
expectation is what official PySpark 4.3.0's Spark Connect client does, against the mock Spark
Connect server: the goldens of `_object_state_goldens.py`, which
`capture_object_state_goldens.py` captures of PySpark 4.3.0, and these tests pass on PySpark
4.3.0 itself, of the same mock server.
"""

import copy
import gc
import weakref

import pytest
from _mock_connect_server import mock_connect_session, parse_message
from _object_state_cases import (
    CLASSES,
    OBJECTS,
    Objects,
    _class,
    check_vars,
    no_init,
    outcome,
    run,
    signature_of,
)
from _object_state_goldens import CLASSES as CLASS_GOLDENS
from _object_state_goldens import OBJECTS as OBJECT_GOLDENS

try:
    from pyspark import _pyspark  # noqa: F401

    _DROP_IN = True
except ImportError:
    # PySpark 4.3.0, of which the goldens are.
    _DROP_IN = False


# The objects and the classes whose state, in the drop-in, is not PySpark's for a test, by the
# change that makes them PySpark's, which the rows "The `__dict__` of the objects of a session"
# and "Subclasses" of dev/parity/known-differences.md say: each change removes its own, and a test
# of one of them that passes fails the run (strict).
_DIFFERS_BY_CHANGE = {
    "the Column, the WindowSpec and the TableArg": {
        "vars": [
            "Column",
            "Column of a DataFrame",
            "WindowSpec",
            "TableArg",
            "TableArg of partitionBy",
        ],
        "attributes": [
            "Column",
            "Column of a DataFrame",
            "WindowSpec",
            "TableArg",
            "TableArg of partitionBy",
        ],
        "copies": [
            "Column: copy",
            "Column: deepcopy",
            "Column: pickle",
            "Column: cloudpickle",
            "Column of a DataFrame: copy",
            "Column of a DataFrame: deepcopy",
            "Column of a DataFrame: pickle",
            "Column of a DataFrame: cloudpickle",
            "WindowSpec: copy",
            "WindowSpec: deepcopy",
            "WindowSpec: pickle",
            "WindowSpec: cloudpickle",
            "TableArg: copy",
            "TableArg: deepcopy",
            "TableArg: pickle",
            "TableArg: cloudpickle",
            "TableArg of partitionBy: copy",
            "TableArg of partitionBy: deepcopy",
            "TableArg of partitionBy: pickle",
            "TableArg of partitionBy: cloudpickle",
        ],
        "construct": [
            "pyspark.sql.connect.column.Column",
            "pyspark.sql.connect.window.WindowSpec",
            "pyspark.sql.connect.table_arg.TableArg",
        ],
        "uninit": [
            "pyspark.sql.connect.column.Column",
            "pyspark.sql.connect.window.WindowSpec",
            "pyspark.sql.connect.table_arg.TableArg",
        ],
        "foreign": [
            "pyspark.sql.connect.column.Column",
            "pyspark.sql.connect.window.WindowSpec",
            "pyspark.sql.connect.table_arg.TableArg",
        ],
    },
    "the DataFrame": {
        "vars": [
            # The writers of `df.write` and `df.writeTo`, PySpark's classes, whose `_df` is the
            # `df._plan` of their making: the drop-in's DataFrame gives a new plan object at each
            # read of its `_plan`, where PySpark's is one object.
            "DataFrameWriter",
            "DataFrameWriter of its setters",
            "DataFrameWriterV2",
            "DataFrameWriterV2 of its setters",
            # And the stream writer of `df.writeStream`, the `_child` of whose `_write_stream` is the
            # `df._plan` of its making.
            "DataStreamWriter",
            "DataStreamWriter of its setters",
            # And the merge writer of `df.mergeInto`, whose `_source_plan` is the `df._plan` of its
            # making.
            "MergeIntoWriter",
            "MergeIntoWriter of its clauses",
            "DataFrame",
            "DataFrame of schema and isStreaming",
            "DataFrame of select",
            "DataFrame of filter of a cached schema",
        ],
        "attributes": [
            "DataFrame",
            "DataFrame of schema and isStreaming",
            "DataFrame of select",
            "DataFrame of filter of a cached schema",
        ],
        "copies": [
            "DataFrame: copy",
            "DataFrame: deepcopy",
            "DataFrame: cloudpickle",
            "DataFrame of schema and isStreaming: copy",
            "DataFrame of schema and isStreaming: deepcopy",
            "DataFrame of schema and isStreaming: cloudpickle",
            "DataFrame of select: copy",
            "DataFrame of select: deepcopy",
            "DataFrame of select: cloudpickle",
            "DataFrame of filter of a cached schema: copy",
            "DataFrame of filter of a cached schema: deepcopy",
            "DataFrame of filter of a cached schema: cloudpickle",
        ],
        "construct": [
            "pyspark.sql.connect.dataframe.DataFrame",
        ],
        "uninit": [
            "pyspark.sql.connect.dataframe.DataFrame",
        ],
        "foreign": [
            "pyspark.sql.connect.dataframe.DataFrame",
            # The DataFrame of the plan of a stream reader of an object that is not a session,
            # which PySpark's DataFrame holds until a request reads its client: the drop-in's
            # DataFrame is of the native session, which its `_df` reads.
            "pyspark.sql.connect.streaming.readwriter.DataStreamReader",
        ],
        "specific": [
            "test_the_private_state_of_a_dataframe_is_not_a_column",
            # Of `writer._write._child is df._plan`, as the cases above.
            "test_the_write_operation_of_a_writer_is_what_its_save_builds",
            "test_the_message_of_a_stream_writer_is_the_message_of_its_operation",
        ],
    },
}

_DIFFERS = {}
for _change in _DIFFERS_BY_CHANGE.values():
    for _test, _keys in _change.items():
        _DIFFERS.setdefault(_test, set()).update(_keys)

_REASON = "the drop-in's differs: dev/parity/known-differences.md, Subclasses and __dict__ rows"


def _cases(test, keys):
    marked = []
    for key in keys:
        if _DROP_IN and key in _DIFFERS.get(test, ()):
            marked.append(pytest.param(key, marks=pytest.mark.xfail(strict=True, reason=_REASON)))
        else:
            marked.append(key)
    return marked


def _of_pyspark(test):
    # A test of one behavior, marked as the cases of `_cases` are.
    if _DROP_IN and test.__name__ in _DIFFERS.get("specific", ()):
        return pytest.mark.xfail(strict=True, reason=_REASON)(test)
    return test


@pytest.fixture(scope="module")
def _shared_mock_connect():
    pytest.importorskip("grpc")
    with mock_connect_session() as session:
        yield session


@pytest.fixture
def objects(_shared_mock_connect):
    # The objects of each test are of a session of their own, as the objects that a session keeps
    # once they are read (its streams and its catalog) are of the session.
    from pyspark.sql import SparkSession

    _, service = _shared_mock_connect
    service.reset()
    spark = SparkSession.builder.remote(service.remote).getOrCreate()
    yield Objects(spark, service)
    spark.release_session_on_close = False
    spark.stop()


def _stop_sessions(*objs):
    # The sessions that a test constructed, which are neither the default nor the active session
    # (and which PySpark 4.3.0 would not exit with: see `capture_object_state_goldens.py`).
    from pyspark.sql.connect.session import SparkSession

    for obj in objs:
        if isinstance(obj, SparkSession) and "_client" in vars(obj):
            obj.release_session_on_close = False
            obj.stop()


@pytest.mark.parametrize("name", _cases("vars", list(OBJECTS)))
def test_vars_hold_the_state_that_pysparks_objects_hold(objects, name):
    # As PySpark's: the keys of `vars()`, in the order that PySpark's `__init__` and methods set
    # them, and what each holds, the objects of the session and of the DataFrame themselves
    # (`vars(spark.read)["_client"] is spark`), as PySpark 4.3.0 gives them.
    obj = OBJECTS[name](objects)
    golden = OBJECT_GOLDENS[name]
    assert f"{type(obj).__module__}.{type(obj).__qualname__}" == golden["class"]
    check_vars(objects, obj, golden["vars"])


@pytest.mark.parametrize(
    "name", _cases("attributes", [n for n in OBJECTS if OBJECT_GOLDENS[n]["vars"]])
)
def test_the_entries_of_vars_are_the_attributes_of_the_object(objects, name):
    # As PySpark's plain objects, whose state is their instance attributes: each attribute of the
    # state is the entry of `vars()`, an assignment and a deletion of it change `vars()`, and an
    # entry that `vars()` is given is the attribute.
    obj = OBJECTS[name](objects)
    keys = [key for key, _ in OBJECT_GOLDENS[name]["vars"]]
    state = vars(obj)
    for key in keys:
        assert getattr(obj, key) is state[key], key
    marker = object()
    for key in keys:
        original = vars(obj)[key]
        setattr(obj, key, marker)
        assert vars(obj)[key] is marker and getattr(obj, key) is marker, key
        delattr(obj, key)
        assert key not in vars(obj), key
        vars(obj)[key] = original
        assert getattr(obj, key) is original, key
    assert list(vars(obj)) == keys


_COPIES = ("copy", "deepcopy", "pickle", "cloudpickle")


@pytest.mark.parametrize(
    "case", _cases("copies", [f"{name}: {how}" for name in OBJECTS for how in _COPIES])
)
def test_copies_and_pickles_give_what_pysparks_give(objects, case):
    # As PySpark's: a shallow copy shares the entries of the state, a deep copy and the pickles
    # copy them (a session is the active session of its ID), and each fails where PySpark's
    # fails, at the same object, with its error.
    name, how = case.rsplit(": ", 1)
    obj = OBJECTS[name](objects)
    assert outcome(obj, how) == OBJECT_GOLDENS[name][how]


def _constructed(objects, path):
    case = CLASSES[path]
    return _class(path), case


@pytest.mark.parametrize("path", _cases("construct", list(CLASSES)))
def test_the_classes_construct_and_subclass_of_pysparks_arguments(objects, path):
    # As PySpark's plain classes: the signature of the class is that of its `__init__` (of its
    # `__new__` for a DataFrame and a Column), the class constructs of the arguments of PySpark's
    # constructor, of the state of the object that it makes, and so does a subclass of it, whose
    # objects are of the subclass.
    cls, case = _constructed(objects, path)
    golden = CLASS_GOLDENS[path]
    assert signature_of(cls) == golden["signature"]
    made = []

    def construct(of):
        obj = of(*case["args"](objects))
        made.append(obj)
        return obj

    try:
        assert run(lambda: construct(cls)) == golden["construct"]
        if "construct_vars" in golden:
            assert list(vars(construct(cls))) == golden["construct_vars"]
        sub = type("Sub", (cls,), {})
        assert run(lambda: type(construct(sub)).__qualname__) == golden["subclass_construct"]
    finally:
        _stop_sessions(*made)


@pytest.mark.parametrize("path", _cases("uninit", list(CLASSES)))
def test_an_object_whose_init_never_ran_raises_what_pysparks_raises(objects, path):
    # As PySpark's: an object of a subclass whose `__init__` does not call that of the class has
    # no state, and its first call raises the AttributeError of the first attribute that PySpark's
    # method reads (the RecursionError of the `__getattr__` of a DataFrame and of a Column, which
    # look the missing attribute up as a column), rather than run on a session of no server.
    cls, case = _constructed(objects, path)
    # PySpark's WindowSpec takes its three arguments in `__new__`.
    args = case["args"](objects) if path.endswith(".WindowSpec") else ()
    obj = no_init(cls)(*args)
    assert vars(obj) == {}
    assert run(lambda: case["uninit"](obj)) == CLASS_GOLDENS[path]["uninit"]


@pytest.mark.parametrize(
    "path", _cases("foreign", [p for p in CLASSES if "foreign" in CLASSES[p]])
)
def test_an_object_that_is_not_a_session_is_kept_until_it_is_used(objects, path):
    # As PySpark's: a constructor keeps the object that it is given where it takes a session, a
    # client, a DataFrame or grouped data, as it is, and the first use of it raises the
    # AttributeError of the attribute that PySpark reads of it (`'int' object has no attribute
    # 'client'`), rather than a TypeError of the constructor; the checks of PySpark's constructors
    # (the assertions of GroupedData and WindowSpec, the NOT_EXPECTED_TYPE of Column, the
    # `session.client` of MergeIntoWriter) raise as PySpark's do.
    cls, case = _constructed(objects, path)
    golden = CLASS_GOLDENS[path]
    foreign_args, use = case["foreign"]
    assert run(lambda: cls(*foreign_args(objects))) == golden["foreign_construct"]
    if "foreign_use" in golden:
        assert run(lambda: use(cls(*foreign_args(objects)))) == golden["foreign_use"]


# The objects that `Objects` keeps itself, which a collection does not free while the test runs.
_KEPT = {
    "SparkSession",
    "SparkSession of streams and catalog",
    "StreamingQueryManager",
    "StreamingQueryListenerBus",
    "Catalog",
    "DataFrame",
    "DataFrame of schema and isStreaming",
    "MergeIntoWriter",
    "MergeIntoWriter of its clauses",
}


@pytest.mark.parametrize("name", _cases("cycle", [n for n in OBJECTS if n not in _KEPT]))
def test_a_cycle_through_the_state_is_collected(objects, name):
    # As PySpark's plain objects: a reference cycle through the `__dict__` of an object, of an
    # attribute that refers to the object, is freed by a collection.
    obj = OBJECTS[name](objects)
    obj.dictsub_self = obj
    freed = weakref.ref(obj)
    del obj
    gc.collect()
    assert freed() is None


@_of_pyspark
def test_the_private_state_of_a_dataframe_is_not_a_column(objects):
    # PySpark's DataFrame holds its schema, the serialized schema of `schema`, whether
    # `_repr_html_` ran and the information of its last execution as attributes, which
    # `__getattr__`, which looks a missing attribute up as a column, does not see.
    from pyspark.sql.types import StructType

    df = objects.spark.range(3)
    assert df._support_repr_html is False
    assert df._cached_schema is None and df._cached_schema_serialized is None
    assert df._execution_info is None
    schema = df.schema
    assert type(df._cached_schema) is StructType and df._cached_schema == schema
    assert type(df._cached_schema_serialized) is bytes
    # `schema` is a copy of the cached schema, as PySpark's pickle of it gives.
    assert schema is not df._cached_schema
    assert df.filter(df.id > 1)._cached_schema is df._cached_schema


@_of_pyspark
def test_the_options_of_a_reader_are_the_dict_of_its_state(objects):
    # PySpark's reader keeps its options in one dict, which `option` and `options` set, a shallow
    # copy shares, and an assignment replaces.
    reader = objects.spark.read
    options = reader._options
    reader._options["a"] = "1"
    assert reader.option("b", 2) is reader and reader._options is options
    assert options == {"a": "1", "b": "2"}
    copy.copy(reader).option("c", True)
    assert options == {"a": "1", "b": "2", "c": "true"}
    reader._options = {"k": "v"}
    assert reader.options(k2=None)._options == {"k": "v", "k2": None}


@_of_pyspark
def test_the_write_operation_of_a_writer_is_what_its_save_builds(objects):
    # PySpark's writer keeps what its setters set on its WriteOperation, whose command its save
    # builds: a mode that a user sets on it is the mode of the save, which PySpark's command
    # refuses before it sends anything.
    writer = objects.df.write.format("noop")
    assert writer._write.source == "noop" and writer._write._child is objects.df._plan
    writer._write.mode = "bogus"
    assert run(writer.save) == (
        "err",
        "PySparkValueError: [UNSUPPORTED_OPERATION] bogus is not supported.",
    )


@_of_pyspark
def test_the_message_of_a_stream_writer_is_the_message_of_its_operation(objects):
    # PySpark's stream writer keeps the WriteStreamOperationStart of its WriteStreamOperation, one
    # message, which its setters set and its start sends.
    writer = objects.sdf.writeStream.format("memory").option("a", 1)
    assert writer._write_proto is writer._write_stream.write_op
    assert writer._write_stream._child is objects.sdf._plan
    assert str(writer._write_proto) == 'format: "memory"\noptions {\n  key: "a"\n  value: "1"\n}\n'


def _sent_plan(objects, df):
    # The plan of the ExecutePlanRequest of a collect of `df`: the mock server answers what a
    # client may not take, and the request is what is checked.
    import pyspark.sql.connect.proto as pb2

    objects.service.reset()
    try:
        df.collect()
    except Exception:
        pass
    return pb2.ExecutePlanRequest.FromString(objects.service.execute_requests[-1]).plan


class _Proxy:
    # A proxy of an object, which gives each attribute of it, as a wrapper of a session or of a
    # DataFrame does.
    def __init__(self, target):
        self._target = target

    def __getattr__(self, name):
        return getattr(self._target, name)


def test_grouped_data_assert_their_arguments_as_pysparks(objects):
    # PySpark's GroupedData asserts the arguments of its __init__, of a bare AssertionError: a
    # DataFrame, a grouping of its names, a list of Columns, a Column to pivot and a list of the
    # values of a pivot, and a list of grouping sets. The arguments of a pivot and the grouping
    # sets are not read of another grouping, and the pivot values are the `F.lit` of each.
    from pyspark.sql import functions as F
    from pyspark.sql.connect.group import GroupedData

    df, col = objects.df, F.col("id")
    for args in (
        (1, "groupby", []),
        (df, "bogus", []),
        (df, "groupby", (col,)),
        (df, "groupby", ["id"]),
        (df, "pivot", [col]),
        (df, "pivot", [col], "id"),
        (df, "pivot", [col], col, (1, 2)),
        (df, "grouping_sets", [col], None, None, ([col],)),
    ):
        with pytest.raises(AssertionError) as raised:
            GroupedData(*args)
        assert str(raised.value) == ""
    grouped = GroupedData(df, "groupby", [col], "x", "y", "z")
    assert grouped._pivot_col is None and grouped._pivot_values is None
    assert grouped._grouping_sets is None
    pivoted = GroupedData(df, "pivot", [col], col, [1, "a", True, 1.5])
    assert [repr(value) for value in pivoted._pivot_values] == [
        "Column<'1'>",
        "Column<'a'>",
        "Column<'True'>",
        "Column<'1.5'>",
    ]


def test_a_pivot_shares_the_dataframe_and_the_grouping_columns_as_pysparks(objects):
    # PySpark's `pivot` makes grouped data of the base class, of the same DataFrame and the same
    # list of grouping columns, and of the pivot column `self._df[pivot_col]`, once it read the
    # type of the grouping, of which only that of `groupBy` pivots.
    from pyspark.errors import PySparkNotImplementedError
    from pyspark.sql.connect.group import GroupedData

    class Sub(GroupedData):
        pass

    df = objects.df
    grouped = Sub(df, "groupby", df.groupBy("id")._grouping_cols)
    pivot = grouped.pivot("id", [1])
    assert type(pivot) is GroupedData
    assert pivot._df is df and pivot._grouping_cols is grouped._grouping_cols
    assert pivot._group_type == "pivot" and repr(pivot._pivot_col) == "Column<'id'>"
    for group_type, operation in [
        ("rollup", "PIVOT after ROLLUP"),
        ("grouping_sets", "PIVOT after GROUPING_SETS"),
        ("pivot", "Repeated PIVOT operation"),
        ("other", "PIVOT after OTHER"),
    ]:
        grouped._group_type = group_type
        with pytest.raises(PySparkNotImplementedError) as raised:
            grouped.pivot("id")
        assert raised.value.getMessageParameters() == {"operation": operation}


def test_the_methods_of_grouped_data_read_their_state_as_pysparks(objects):
    # PySpark's GroupedData reads its state as its methods make their plans: an aggregation of
    # grouped data whose state was assigned is of that state, with the assertions of its plan, a
    # proxy of a DataFrame is grouped as the DataFrame, and `count` is a call of `agg`, of a
    # subclass too.
    from pyspark.sql import functions as F
    from pyspark.sql.connect.group import GroupedData

    df = objects.df
    grouped = df.groupBy("id")
    grouped._grouping_cols = [F.col("x"), F.col("y")]
    grouped._group_type = "cube"
    aggregate = _sent_plan(objects, grouped.count()).root.aggregate
    # GROUP_TYPE_CUBE.
    assert len(aggregate.grouping_expressions) == 2 and aggregate.group_type == 3
    for name, value in [
        ("_grouping_cols", ("x",)),
        ("_group_type", "bogus"),
        ("_pivot_col", F.col("x")),
    ]:
        original = getattr(grouped, name)
        setattr(grouped, name, value)
        with pytest.raises(AssertionError):
            grouped.count()
        setattr(grouped, name, original)
    del grouped._pivot_col
    with pytest.raises(AttributeError, match="'GroupedData' object has no attribute '_pivot_col'"):
        grouped.count()
    proxied = df.groupBy("id")
    proxied._df = _Proxy(df)
    assert _sent_plan(objects, proxied.count()).root.aggregate.input.HasField("range")
    assert repr(proxied) == (
        "GroupedData[grouping expressions: [id], value: [id: bigint], type: GroupBy]"
    )

    class Sub(GroupedData):
        def agg(self, *exprs):
            return [repr(expr) for expr in exprs]

    assert Sub(df, "groupby", [F.col("id")]).count() == ["Column<'count(1) AS count'>"]


def test_cogrouped_data_are_of_any_grouped_data_of_the_same_session_as_pysparks(objects):
    # PySpark's PandasCogroupedOps checks `gd1._df._check_same_session(gd2._df)` and keeps its
    # grouped data as they are, any objects of a `_df` and `_grouping_cols`, of which its methods
    # make their plans.
    from pyspark.errors import SessionNotSameException
    from pyspark.sql import functions as F

    df = objects.df
    grouped = df.groupBy("id")

    class Grouped:
        pass

    duck = Grouped()
    duck._df, duck._grouping_cols = df, [F.col("id"), F.col("id")]
    cogrouped = grouped.cogroup(duck)
    assert vars(cogrouped) == {"_gd1": grouped, "_gd2": duck}
    mapped = cogrouped.applyInPandas(lambda left, right: left, "id long")
    assert len(_sent_plan(objects, mapped).root.co_group_map.other_grouping_expressions) == 2
    with pytest.raises(SessionNotSameException):
        grouped.cogroup(df)
    with pytest.raises(AttributeError, match="'int' object has no attribute '_df'"):
        grouped.cogroup(1)


def test_the_null_and_statistics_functions_call_the_methods_of_their_df_as_pysparks(objects):
    # PySpark's DataFrameNaFunctions and DataFrameStatFunctions call the methods of their `df`, of
    # any object, as PySpark's call them, so a proxy of a DataFrame works; an omitted value of
    # `replace` is PySpark's `_NoValue`, which a DataFrame takes for an omitted value.
    from pyspark._globals import _NoValue
    from pyspark.sql.connect.dataframe import DataFrameNaFunctions, DataFrameStatFunctions

    calls = []

    class Recorder:
        def __getattr__(self, name):
            def method(*args, **kwargs):
                calls.append((name, args, kwargs))
                return name

            return method

    na, stat = DataFrameNaFunctions(Recorder()), DataFrameStatFunctions(Recorder())
    assert na.drop() == "dropna"
    na.drop(how=None, thresh=1, subset=["a"])
    na.fill(1)
    na.replace([1], [2])
    na.replace({1: 2})
    stat.corr("a", "b")
    stat.cov("a", "b")
    stat.crosstab("a", "b")
    stat.freqItems(["a"])
    stat.sampleBy("a", {1: 0.5})
    stat.approxQuantile("a", [0.5], 0.1)
    assert calls == [
        ("dropna", (), {"how": "any", "thresh": None, "subset": None}),
        ("dropna", (), {"how": None, "thresh": 1, "subset": ["a"]}),
        ("fillna", (), {"value": 1, "subset": None}),
        ("replace", ([1], [2], None), {}),
        ("replace", ({1: 2}, _NoValue, None), {}),
        ("corr", ("a", "b", None), {}),
        ("cov", ("a", "b"), {}),
        ("crosstab", ("a", "b"), {}),
        ("freqItems", (["a"], None), {}),
        ("sampleBy", ("a", {1: 0.5}, None), {}),
        ("approxQuantile", ("a", [0.5], 0.1), {}),
    ]
    df = objects.df
    assert type(df.na.replace({1: 2})) is type(df)
    assert type(DataFrameNaFunctions(_Proxy(df)).drop()) is type(df)
    assert type(DataFrameStatFunctions(_Proxy(df)).crosstab("id", "id")) is type(df)


@pytest.mark.parametrize("method", ["start", "toTable"])
@pytest.mark.parametrize("change", ["none", "instance", "class", "subclass", "replaced"])
def test_the_started_event_is_posted_to_the_post_to_all_of_the_bus(
    objects, monkeypatch, method, change
):
    # As PySpark's DataStreamWriter, which posts the started event of the query that it starts to
    # `self._session.streams._sqlb.post_to_all(start_event)`, as the attribute resolves then: the
    # method of the session's bus, which calls the listeners of its list, here one put on the list
    # itself; and else a patch of the method on the bus or on its class, the method of a subclass of
    # the bus, or that of a bus of another class, which gets the decoded event, once. The listeners
    # of the list get it if what is called posts it to them, as the original method or `super()`.
    import json

    from pyspark.sql.streaming.listener import StreamingQueryListener
    from streaming_listener_cases import start_response

    spark, service = objects.spark, objects.service
    manager = spark.streams
    original = manager._sqlb
    posted, received = [], []

    class Listener(StreamingQueryListener):
        def onQueryStarted(self, event):
            received.append(event)

        def onQueryProgress(self, event):
            pass

        def onQueryIdle(self, event):
            pass

        def onQueryTerminated(self, event):
            pass

    class Bus(type(original)):
        def post_to_all(self, event):
            posted.append(event)
            super().post_to_all(event)

    class Recorder:
        def post_to_all(self, event):
            posted.append(event)

    own = type(original).__dict__["post_to_all"]

    def post_to_all(self, event):
        posted.append(event)
        own(self, event)

    listener = Listener()
    bus = Bus(manager) if change == "subclass" else original
    bus._listener_bus.append(listener)
    try:
        if change == "instance":
            original.post_to_all = posted.append
        elif change == "class":
            monkeypatch.setattr(type(original), "post_to_all", post_to_all)
        elif change == "subclass":
            manager._sqlb = bus
        elif change == "replaced":
            manager._sqlb = Recorder()
        service.execute_payload = start_response(
            json.dumps(
                {
                    "id": "00000000-0000-4000-8000-000000000001",
                    "runId": "00000000-0000-4000-8000-000000000002",
                    "name": "query",
                    "timestamp": "started",
                    "jobTags": ["tag"],
                }
            )
        )
        writer = objects.sdf.writeStream
        query = getattr(writer, method)(*(["t"] if method == "toTable" else []))
        assert query.id == "q"
        called = [] if change == "none" else ["QueryStartedEvent"]
        assert [type(event).__name__ for event in posted] == called
        delivered = change in ("none", "class", "subclass")
        assert [event.timestamp for event in received] == (["started"] if delivered else [])
        if posted and received:
            assert received[0] is posted[0]
    finally:
        if "post_to_all" in vars(original):
            del original.post_to_all
        bus._listener_bus.remove(listener)
        manager._sqlb = original


class _ConfigResult:
    # A ConfigResult of a client of another class, as PySpark's `client.config` returns one.
    def __init__(self, pairs, warnings=()):
        self.pairs, self.warnings = pairs, list(warnings)


class _ConfigClient:
    # A client of another class than the client of the session, whose `config` PySpark's
    # RuntimeConf calls with the ConfigRequest.Operation of each of its methods.
    def __init__(self, *results):
        self.results, self.operations = list(results), []

    def config(self, operation):
        self.operations.append(operation)
        return self.results.pop(0)


def test_a_runtime_conf_of_another_client_calls_its_config(objects):
    # PySpark's RuntimeConf makes each request of its `_client`, `self._client.config(operation)`,
    # of a ConfigRequest.Operation of the method, and reads the pairs and the warnings of the
    # result, of a client of any class.
    import warnings

    from pyspark.errors import PySparkValueError
    from pyspark.sql.connect.conf import RuntimeConf

    client = _ConfigClient(
        _ConfigResult([("k", "v")]),
        _ConfigResult([], ["of the server"]),
        _ConfigResult([("k", "d")]),
        _ConfigResult([("a", "1"), ("b", "2")]),
        _ConfigResult([("k", "true")]),
        _ConfigResult([("k", "maybe")]),
        _ConfigResult([], []),
    )
    conf = RuntimeConf(client)
    assert vars(conf) == {"_client": client}
    assert conf.get("k") == "v"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        conf.set("k", True)
    assert [str(w.message) for w in caught] == ["of the server"]
    assert conf.get("k", "d") == "d"
    assert conf.getAll == {"a": "1", "b": "2"}
    assert conf.isModifiable("k") is True
    with pytest.raises(PySparkValueError) as raised:
        conf.isModifiable("k")
    assert raised.value.getCondition() == "VALUE_NOT_ALLOWED"
    conf.unset("k")
    get, set_, get_with_default, get_all, modifiable, _, unset = client.operations
    assert list(get.get.keys) == ["k"]
    assert [(p.key, p.value) for p in set_.set.pairs] == [("k", "true")]
    assert [(p.key, p.value) for p in get_with_default.get_with_default.pairs] == [("k", "d")]
    assert get_all.HasField("get_all") and list(modifiable.is_modifiable.keys) == ["k"]
    assert list(unset.unset.keys) == ["k"]
    # The client of another object is what its `_client` is, of an assignment too.
    conf._client = _ConfigClient(_ConfigResult([("x", "y")]))
    assert conf.get("x") == "y"


def test_a_registration_of_another_client_calls_its_register_data_source(objects):
    # PySpark's DataSourceRegistration registers of `self.sparkSession._client`, by its
    # `register_data_source`, of a session and a client of any class.
    from pyspark.sql.connect.datasource import DataSourceRegistration

    registered = []

    class Client:
        def register_data_source(self, data_source):
            registered.append(data_source)

    class Session:
        _client = Client()

    registration = DataSourceRegistration(Session())
    assert registration.register(int) is None and registered == [int]


def test_the_table_valued_functions_range_is_the_range_of_their_session(objects):
    # PySpark's `spark.tvf.range` is `self._sparkSession.range(start, end, step, numPartitions)`,
    # of its default step of 1, of a session of any class.
    from pyspark.sql.connect.tvf import TableValuedFunction

    class Session:
        def range(self, *args):
            return ("range", args)

    tvf = TableValuedFunction(Session())
    assert tvf.range(5) == ("range", (5, None, 1, None))
    assert tvf.range(1, 9, 2, 3) == ("range", (1, 9, 2, 3))
    assert objects.spark.tvf.range(3).columns == ["id"]


def test_a_catalog_makes_its_requests_of_the_session_of_its_state(objects):
    # PySpark's Catalog reads its `_sparkSession` at each request: of an assignment, the request
    # is of the session that it holds then, and of a proxy of a session, of the session of its
    # `client`.
    import pyspark.sql.connect.proto as pb2
    from pyspark.sql import SparkSession

    spark, service = objects.spark, objects.service
    other = SparkSession.builder.remote(service.remote).create()

    class Proxy:
        def __init__(self, target):
            self._target = target

        def __getattr__(self, name):
            return getattr(self._target, name)

    try:
        catalog = spark.catalog
        for holder, session in ((other, other), (Proxy(other), other), (spark, spark)):
            catalog._sparkSession = holder
            service.reset()
            try:
                catalog.currentDatabase()
            except Exception:
                # The mock server answers no rows; the request is what is checked.
                pass
            sent = pb2.ExecutePlanRequest.FromString(service.execute_requests[-1])
            assert sent.session_id == session.session_id
    finally:
        other.release_session_on_close = False
        other.stop()


def _config_sets(objects):
    # The pairs and the silent flag of each ConfigRequest.Set that the mock server received, of a
    # value of None for a value that is not set.
    import pyspark.sql.connect.proto as pb2

    sets = []
    for raw in objects.service.config_requests:
        operation = pb2.ConfigRequest.FromString(raw).operation
        if operation.HasField("set"):
            pairs = operation.set.pairs
            sets.append(
                ([(p.key, p.value if p.HasField("value") else None) for p in pairs], operation.set.silent)
            )
    return sets


def test_a_builder_creates_its_session_of_the_state_that_it_holds_then_as_pysparks(objects):
    # PySpark's Builder reads its `_options`, `_channel_builder` and `_hook_factories` when it
    # creates a session: an assignment of them is what `create` reads, its options but
    # `spark.remote` and `spark.master` are set on the session in one silent request, after the
    # `spark.sql.` options of the last init conf of the environment, `remote` and `appName` call
    # its `config`, and `getOrCreate` calls its `create` and its `_apply_options`, of a subclass
    # too, on the session that it makes or finds.
    import os

    from pyspark.errors import PySparkValueError
    from pyspark.sql.connect.session import SparkSession

    remote, made, calls = objects.service.remote, [], []

    class Sub(SparkSession.Builder):
        def config(self, key=None, value=None, *, map=None):
            calls.append(("config", key))
            return super().config(key, value, map=map)

        def create(self):
            calls.append(("create",))
            made.append(super().create())
            return made[-1]

        def _apply_options(self, session):
            calls.append(("_apply_options", session is made[-1]))
            return super()._apply_options(session)

    try:
        builder = SparkSession.Builder().remote(remote)
        builder._options = {"spark.remote": remote, "k": 7, "spark.master": "m", "f": False}
        objects.service.reset()
        made.append(builder.create())
        assert _config_sets(objects) == [([("k", "7"), ("f", "false")], True)]
        environ = {
            "PYSPARK_REMOTE_INIT_CONF_LEN": "2",
            "PYSPARK_REMOTE_INIT_CONF_0": '{"spark.sql.a": "1"}',
            "PYSPARK_REMOTE_INIT_CONF_1": '{"spark.sql.b": "2", "spark.c": "3", "spark.sql.k": "i"}',
        }
        os.environ.update(environ)
        try:
            objects.service.reset()
            made.append(SparkSession.Builder().remote(remote).config("spark.sql.k", "b").create())
        finally:
            for key in environ:
                del os.environ[key]
        assert _config_sets(objects) == [([("spark.sql.b", "2"), ("spark.sql.k", "b")], True)]
        objects.service.reset()
        session = Sub().remote(remote).appName("n").config("spark.sql.z", 1).getOrCreate()
        assert session is objects.spark
        assert calls == [
            ("config", "spark.remote"),
            ("config", "spark.app.name"),
            ("config", "spark.sql.z"),
            ("_apply_options", False),
        ]
        assert _config_sets(objects) == [([("spark.app.name", "n"), ("spark.sql.z", "1")], True)]
        for builder in (SparkSession.Builder(), SparkSession.Builder().remote(remote)):
            builder._channel_builder = remote
            if "spark.remote" in builder._options:
                with pytest.raises(PySparkValueError) as raised:
                    builder.create()
                assert raised.value.getCondition() == "SESSION_NEED_CONN_STR_OR_BUILDER"
            else:
                made.append(builder.create())
        builder = SparkSession.Builder()
        builder._options["spark.remote"] = None
        with pytest.raises(AssertionError):
            builder.create()
    finally:
        _stop_sessions(*made)


def test_the_options_of_a_builder_are_sent_as_pysparks_set_all_sends_them(objects):
    # PySpark's `_apply_options` sends the options in the KeyValues of one ConfigRequest.Set, as
    # protobuf takes them: a key of None is unset, the empty key, bytes of UTF-8 are their text,
    # and the first that protobuf refuses fails the request before it is sent, of its error.
    from pyspark.sql.connect.session import SparkSession

    remote, made = objects.service.remote, []
    try:
        for key, value, sent in [
            (None, "v", [("", "v")]),
            ("k", None, [("k", None)]),
            (b"bk", "v", [("bk", "v")]),
        ]:
            objects.service.reset()
            made.append(SparkSession.Builder().remote(remote).config(key, value).create())
            assert _config_sets(objects) == [(sent, True)]
        for key, value, error in [
            (1, "v", TypeError),
            ("k\udc80", "v", UnicodeEncodeError),
            ("k", "v\udc80", UnicodeEncodeError),
            (b"\xff", "v", UnicodeDecodeError),
        ]:
            objects.service.reset()
            with pytest.raises(error):
                made.append(SparkSession.Builder().remote(remote).config(key, value).create())
            assert _config_sets(objects) == []
    finally:
        _stop_sessions(*made)


def test_a_session_reads_its_client_and_its_session_id_of_its_dict_as_pysparks(objects):
    # PySpark's session holds its `_client`, its `_session_id` and its `release_session_on_close`
    # in its `__dict__`, which its methods read at the call: `client` is the `_client`,
    # `session_id` the `_session_id`, of which the session pickles, and `stop` releases the session
    # unless `release_session_on_close` is false.
    from pyspark.sql.connect.session import SparkSession

    spark, service = objects.spark, objects.service
    assert list(vars(spark)) == ["_client", "_session_id", "release_session_on_close"]
    assert spark.client is spark._client
    assert spark.session_id == spark._session_id == spark._client._session_id
    own = spark._session_id
    try:
        spark._session_id = "another id"
        assert spark.session_id == "another id"
        assert spark.__reduce__()[1] == ("another id",)
    finally:
        spark._session_id = own
    made = [SparkSession.builder.remote(service.remote).create() for _ in range(2)]
    made[0].release_session_on_close = False
    for session, released in zip(made, (0, 1)):
        service.reset()
        session.stop()
        assert session.is_stopped and service.release_session_calls == released


def test_the_catalog_the_streams_and_the_version_of_a_session_are_in_its_dict_as_pysparks(
    objects,
):
    # PySpark's `catalog` and `version` are `functools.cached_property`s, whose first access keeps
    # the value in the `__dict__` of the session, which the next ones read, and which an
    # assignment or a `del` changes; `streams` keeps its StreamingQueryManager as the `_sqm` of the
    # `__dict__`, unless the session has one.
    import functools

    import pyspark.sql.connect.proto as pb2
    from pyspark.sql.connect.session import SparkSession

    spark, service = objects.spark, objects.service
    for name in ("catalog", "version"):
        assert type(vars(SparkSession)[name]) is functools.cached_property
    catalog = spark.catalog
    assert vars(spark)["catalog"] is catalog and spark.catalog is catalog
    assert vars(catalog)["_sparkSession"] is spark
    service.reset()
    assert spark.version == "4.3.0" and spark.version == "4.3.0"
    assert vars(spark)["version"] == "4.3.0"
    requests = [pb2.AnalyzePlanRequest.FromString(r) for r in service.analyze_requests]
    assert [r.WhichOneof("analyze") for r in requests] == ["spark_version"]
    streams = spark.streams
    assert vars(spark)["_sqm"] is streams and spark.streams is streams
    assert vars(streams)["_session"] is spark
    try:
        spark.catalog, spark.version, spark._sqm = "a catalog", "a version", "a manager"
        assert spark.catalog == "a catalog" and spark.version == "a version"
        assert spark.streams == "a manager"
        del spark.catalog
        assert spark.catalog is not catalog and type(spark.catalog) is type(catalog)
    finally:
        spark.catalog, spark._sqm = catalog, streams


def test_a_session_makes_its_requests_of_its_client_at_the_call_as_pysparks(objects):
    # PySpark's session makes its requests of its `_client`, or of its `client`, as it is at the
    # call: of the client of another session that is assigned, the requests of `sql`, `range`,
    # `createDataFrame`, `interruptAll`, `version`, of its catalog and of its `conf` are of that
    # session, and `is_stopped` is of that client; of its own client given back, they are of the
    # session again.
    from pyspark.sql.connect.session import SparkSession

    spark, service = objects.spark, objects.service
    other = SparkSession.builder.remote(service.remote).create()
    own = spark._client

    def session_ids(call):
        service.reset()
        try:
            call()
        except Exception:
            # The mock server answers no rows; the requests are what is checked.
            pass
        return {parse_message(request)[1][0].decode() for _, request in service.recorded_order}

    calls = [
        lambda: spark.sql("select 1").collect(),
        lambda: spark.range(3).collect(),
        lambda: spark.createDataFrame([(1,)], "a int").collect(),
        lambda: spark.interruptAll(),
        lambda: spark.version,
        lambda: spark.catalog.currentDatabase(),
        lambda: spark.conf.get("spark.sql.shuffle.partitions"),
    ]
    try:
        spark._client = other.client
        assert spark.client is other.client
        for call in calls:
            assert session_ids(call) == {other.session_id}
        other.release_session_on_close = False
        other.stop()
        assert spark.is_stopped
        spark._client = own
        assert not spark.is_stopped
        del spark.version
        for call in calls:
            assert session_ids(call) == {spark.session_id}
    finally:
        spark._client = own
        if not other.is_stopped:
            other.release_session_on_close = False
            other.stop()


def test_a_session_whose_catalog_or_streams_was_read_is_freed_at_a_collection_as_pysparks(
    objects,
):
    # PySpark's session and its catalog, which holds the session as its `_sparkSession`, or its
    # StreamingQueryManager, which holds it as its `_session`, are a reference cycle: a dropped
    # session whose catalog or streams was read is freed at a collection, and one of neither is
    # freed when it is dropped, also once its `client` or its `version` was read.
    from pyspark.sql.connect.session import SparkSession

    remote = objects.service.remote
    for read, cycle in [
        (None, False),
        ("client", False),
        ("version", False),
        ("catalog", True),
        ("streams", True),
    ]:
        session = SparkSession.builder.remote(remote).create()
        if read is not None:
            getattr(session, read)
        freed = weakref.ref(session)
        del session
        assert (freed() is not None) == cycle, read
        gc.collect()
        assert freed() is None, read


def test_the_methods_of_a_session_whose_init_never_ran_raise_pysparks_errors(objects):
    # PySpark's methods of a session whose `__init__` never ran check their arguments first, as
    # they do, and then raise the `AttributeError` of the first attribute that they read: its
    # `_client` at their first request, and its `_session_id` of `session_id` and of a pickle.
    import pickle

    from pyspark.sql.connect.session import SparkSession

    def outcome(call):
        try:
            call()
        except Exception as error:
            condition = getattr(error, "getCondition", lambda: None)()
            return condition or f"{type(error).__name__}: {error}"
        return "ok"

    session = no_init(SparkSession)()
    client = "AttributeError: 'NoInit' object has no attribute '_client'"
    session_id = "AttributeError: 'NoInit' object has no attribute '_session_id'"
    cases = [
        (lambda: session.createDataFrame([(1,)], 5), "NOT_EXPECTED_TYPE"),
        (lambda: session.createDataFrame([]), "CANNOT_INFER_EMPTY_SCHEMA"),
        (lambda: session.createDataFrame([(1,)], "a int"), client),
        (lambda: session.createDataFrame([(1,)], ["a"]), client),
        (lambda: session.createDataFrame([(1,)]), client),
        (lambda: session.range(1).collect(), client),
        (lambda: session.sql("select 1", args=5), "INVALID_TYPE"),
        (lambda: session.sql("select 1"), client),
        (lambda: session.table(5), "NOT_EXPECTED_TYPE"),
        (lambda: session.emptyDataFrame(None), "CANNOT_INFER_EMPTY_SCHEMA"),
        (
            lambda: session.addArtifacts("x", pyfile=True, archive=True),
            "INVALID_MULTIPLE_ARGUMENT_CONDITIONS",
        ),
        (lambda: session.addArtifacts("/nonexistent.py", pyfile=True), client),
        (lambda: session.client, client),
        (lambda: session.session_id, session_id),
        (lambda: pickle.dumps(session), session_id),
        (lambda: session.is_stopped, client),
        (lambda: session.version, client),
        (lambda: session.conf, client),
        (lambda: session.catalog.currentDatabase(), client),
        (lambda: session.interruptAll(), client),
        (lambda: session.interruptTag("t"), client),
        (lambda: session.addTag("t"), client),
        (lambda: session.getTags(), client),
        (lambda: session.newSession(), client),
        (lambda: session.cloneSession(), client),
        (lambda: session.profile, client),
        (lambda: session.registerProgressHandler(lambda **kwargs: None), client),
        (lambda: session.stop(), client),
        (lambda: session.__exit__(None, None, None), client),
    ]
    assert [outcome(call) for call, _ in cases] == [expected for _, expected in cases]
    assert list(vars(session)) == ["catalog"]


# The reads of the readers, as PySpark's plans check their arguments when they are made and when
# they are sent, of PySpark 4.3.0: what the call raises, and else what the first request of its
# DataFrame, its schema, raises.
_BAD_STR = "a\udc80"


def _read_cases(spark):
    BAD = _BAD_STR
    df = spark.range(1).selectExpr("'x' as v")
    return {
        "format(int).load()": lambda: spark.read.format(1).load(),
        "format(bad).load()": lambda: spark.read.format(BAD).load(),
        "format(None).load()": lambda: spark.read.format(None).load(),
        "schema(int)": lambda: spark.read.schema(1),
        "schema(bad).load()": lambda: spark.read.schema(BAD).load(),
        "option(int key).load()": lambda: spark.read.option(1, "v").load(),
        "option(key, None).load()": lambda: spark.read.option("k", None).load(),
        "option(key, None).table()": lambda: spark.read.option("k", None).table("t"),
        "option(bad key).load()": lambda: spark.read.option(BAD, "v").load(),
        "option(bad value).table()": lambda: spark.read.option("k", BAD).table("t"),
        "option(bytes key).table()": lambda: spark.read.option(b"k", "v").table("t"),
        "option(bytes key).load()": lambda: spark.read.option(b"k", "v").load(),
        "option(unhashable)": lambda: spark.read.option([1], "v"),
        "_options[k]=5 load()": lambda: (lambda x: (x._options.__setitem__("k", 5), x.load())[1])(spark.read),
        "_options[k]=5 table()": lambda: (lambda x: (x._options.__setitem__("k", 5), x.table("t"))[1])(spark.read),
        "_schema=5 load()": lambda: (lambda x: (setattr(x, "_schema", 5), x.load())[1])(spark.read),
        "load(path list)": lambda: spark.read.format("json").load(["/a", "/b"]),
        "load(path tuple)": lambda: spark.read.format("json").load(("/a", "/b")),
        "load(list of int)": lambda: spark.read.format("json").load(["/a", 1]),
        "load(bad path)": lambda: spark.read.format("json").load(BAD),
        "load(int path)": lambda: spark.read.format("json").load(5),
        "load(path, schema=int)": lambda: spark.read.load("/a", schema=1),
        "load(path, k=bad)": lambda: spark.read.format("json").load("/a", k=BAD),
        "json(int)": lambda: spark.read.json(5),
        "json(df)": lambda: spark.read.json(df),
        "json(df, schema=ddl)": lambda: spark.read.json(df, schema="v STRING"),
        "csv(int)": lambda: spark.read.csv(5),
        "csv(tuple)": lambda: spark.read.csv(("/a",)),
        "xml(int)": lambda: spark.read.xml(5),
        "parquet(int)": lambda: spark.read.parquet(5),
        "text(int)": lambda: spark.read.text(5),
        "orc(None)": lambda: spark.read.orc(None),
        "jdbc(column without bounds)": lambda: spark.read.jdbc("jdbc:x", "t", column="c"),
        "jdbc(predicates)": lambda: spark.read.jdbc("jdbc:x", "t", predicates=["a > 1"]),
        "jdbc(predicates tuple)": lambda: spark.read.jdbc("jdbc:x", "t", predicates=("a > 1",)),
        "jdbc(predicates of int)": lambda: spark.read.jdbc("jdbc:x", "t", predicates=[1]),
        "jdbc(properties int key)": lambda: spark.read.jdbc("jdbc:x", "t", properties={1: "v"}),
        "jdbc(url None)": lambda: spark.read.jdbc(None, "t"),
        "table(int)": lambda: spark.read.table(5),
        "table(bad)": lambda: spark.read.table(BAD),
        "table(bytes)": lambda: spark.read.table(b"t"),
        "changes(schema)": lambda: spark.read.schema("a INT").changes("t"),
        "stream format(int).load()": lambda: spark.readStream.format(1).load(),
        "stream schema(int)": lambda: spark.readStream.schema(1),
        "stream option(k, None)": lambda: spark.readStream.option("k", None)._options,
        "stream option(k, True)": lambda: spark.readStream.option("k", True)._options,
        "stream load(int)": lambda: spark.readStream.format("rate").load(5),
        "stream load('  ')": lambda: spark.readStream.format("rate").load("  "),
        "stream load(bad)": lambda: spark.readStream.format("json").load(BAD),
        "stream name(int)": lambda: spark.readStream.name(5),
        "stream name('a b')": lambda: spark.readStream.name("a b"),
        "stream json(int)": lambda: spark.readStream.json(5),
        "stream table(int)": lambda: spark.readStream.table(5),
        "stream option(k, None).table()": lambda: spark.readStream.options(k=None).table("t"),
        "stream changes(schema)": lambda: spark.readStream.schema("a INT").changes("t"),
        "stream _source_name=5 load()": lambda: (lambda x: (setattr(x, "_source_name", 5), x.format("rate").load())[1])(spark.readStream),
    }


_READ_OUTCOMES = {
    'format(int).load()': ('call', 'AssertionError', ''),
    'format(bad).load()': ('request', 'UnicodeEncodeError', "'utf-8' codec can't encode character '\\udc80' in position 1: surrogates not allowed"),
    'format(None).load()': ('ok',),
    'schema(int)': ('call', 'PySparkTypeError', '[NOT_EXPECTED_TYPE] Argument `schema` should be str or struct type, got int.'),
    'schema(bad).load()': ('request', 'UnicodeEncodeError', "'utf-8' codec can't encode character '\\udc80' in position 1: surrogates not allowed"),
    'option(int key).load()': ('call', 'AssertionError', ''),
    'option(key, None).load()': ('ok',),
    'option(key, None).table()': ('request', 'TypeError', 'bad argument type for built-in operation'),
    'option(bad key).load()': ('request', 'UnicodeEncodeError', "'utf-8' codec can't encode character '\\udc80' in position 1: surrogates not allowed"),
    'option(bad value).table()': ('request', 'UnicodeEncodeError', "'utf-8' codec can't encode character '\\udc80' in position 1: surrogates not allowed"),
    'option(bytes key).table()': ('ok',),
    'option(bytes key).load()': ('call', 'AssertionError', ''),
    'option(unhashable)': ('call', 'TypeError', "unhashable type: 'list'"),
    '_options[k]=5 load()': ('call', 'AssertionError', ''),
    '_options[k]=5 table()': ('request', 'TypeError', 'bad argument type for built-in operation'),
    '_schema=5 load()': ('call', 'AssertionError', ''),
    'load(path list)': ('ok',),
    'load(path tuple)': ('call', 'AssertionError', ''),
    'load(list of int)': ('call', 'AssertionError', ''),
    'load(bad path)': ('request', 'UnicodeEncodeError', "'utf-8' codec can't encode character '\\udc80' in position 1: surrogates not allowed"),
    'load(int path)': ('call', 'AssertionError', ''),
    'load(path, schema=int)': ('call', 'PySparkTypeError', '[NOT_EXPECTED_TYPE] Argument `schema` should be str or struct type, got int.'),
    'load(path, k=bad)': ('request', 'UnicodeEncodeError', "'utf-8' codec can't encode character '\\udc80' in position 1: surrogates not allowed"),
    'json(int)': ('call', 'PySparkTypeError', '[NOT_EXPECTED_TYPE] Argument `path` should be str, list, or DataFrame, got int.'),
    'json(df)': ('ok',),
    'json(df, schema=ddl)': ('ok',),
    'csv(int)': ('call', 'AssertionError', ''),
    'csv(tuple)': ('call', 'AssertionError', ''),
    'xml(int)': ('call', 'PySparkTypeError', '[NOT_EXPECTED_TYPE] Argument `path` should be str, list, or DataFrame, got int.'),
    'parquet(int)': ('call', 'AssertionError', ''),
    'text(int)': ('call', 'AssertionError', ''),
    'orc(None)': ('ok',),
    'jdbc(column without bounds)': ('call', 'AssertionError', 'lowerBound can not be None when ``column`` is specified'),
    'jdbc(predicates)': ('ok',),
    'jdbc(predicates tuple)': ('call', 'AssertionError', ''),
    'jdbc(predicates of int)': ('call', 'AssertionError', ''),
    'jdbc(properties int key)': ('call', 'TypeError', 'keywords must be strings'),
    'jdbc(url None)': ('ok',),
    'table(int)': ('request', 'TypeError', 'bad argument type for built-in operation'),
    'table(bad)': ('request', 'UnicodeEncodeError', "'utf-8' codec can't encode character '\\udc80' in position 1: surrogates not allowed"),
    'table(bytes)': ('ok',),
    'changes(schema)': ('call', 'AnalysisException', '[_LEGACY_ERROR_TEMP_1189] User specified schema not supported with `changes`.'),
    'stream format(int).load()': ('call', 'AssertionError', ''),
    'stream schema(int)': ('call', 'PySparkTypeError', '[NOT_EXPECTED_TYPE] Argument `schema` should be str or struct type, got int.'),
    'stream option(k, None)': ('value', '{}'),
    'stream option(k, True)': ('value', "{'k': 'True'}"),
    'stream load(int)': ('call', 'PySparkValueError', "[VALUE_NOT_NON_EMPTY_STR] Value for `path` must be a non-empty string, got '5'."),
    "stream load('  ')": ('call', 'PySparkValueError', "[VALUE_NOT_NON_EMPTY_STR] Value for `path` must be a non-empty string, got '  '."),
    'stream load(bad)': ('request', 'UnicodeEncodeError', "'utf-8' codec can't encode character '\\udc80' in position 1: surrogates not allowed"),
    'stream name(int)': ('call', 'PySparkTypeError', '[NOT_EXPECTED_TYPE] Argument `source_name` should be str, got int.'),
    "stream name('a b')": ('call', 'PySparkValueError', "[INVALID_STREAMING_SOURCE_NAME] Invalid streaming source name 'a b'. Source names must contain only ASCII letters, digit"),
    'stream json(int)': ('call', 'PySparkTypeError', '[NOT_EXPECTED_TYPE] Argument `path` should be str, got int.'),
    'stream table(int)': ('request', 'TypeError', 'bad argument type for built-in operation'),
    'stream option(k, None).table()': ('ok',),
    'stream changes(schema)': ('call', 'AnalysisException', '[_LEGACY_ERROR_TEMP_1189] User specified schema not supported with `changes`.'),
    'stream _source_name=5 load()': ('call', 'AssertionError', ''),
}


def _read_outcome(call):
    try:
        df = call()
    except BaseException as error:  # noqa: BLE001
        return ("call", type(error).__name__, str(error)[:120])
    if type(df).__name__ != "DataFrame":
        return ("value", repr(df)[:80])
    try:
        df.schema
    except BaseException as error:  # noqa: BLE001
        return ("request", type(error).__name__, str(error)[:120])
    return ("ok",)


@pytest.mark.parametrize("case", list(_READ_OUTCOMES))
def test_the_reads_raise_what_pysparks_plans_raise(objects, case):
    # As PySpark's DataFrameReader and DataStreamReader, of whose source the drop-in's are adapted:
    # the assertions of the plans of their reads (`DataSource`, `Read`, `RelationChanges` and
    # `Parse`), made when the plan is made, and the errors of the arguments that the plan cannot
    # send, which each request of its DataFrame raises, as PySpark's plan raises them when it is
    # sent, in the same order.
    assert _read_outcome(_read_cases(objects.spark)[case]) == _READ_OUTCOMES[case]


# The writes of the writers, as PySpark's writers and their write operations make them, of PySpark
# 4.3.0: what the call raises, and else the write command that the server receives of it.
def _sent_write(service):
    import pyspark.sql.connect.proto as pb2

    if not service.execute_requests:
        return None
    request = pb2.ExecutePlanRequest.FromString(service.execute_requests[-1])
    command = request.plan.command
    kind = command.WhichOneof("command_type")
    if kind == "write_operation":
        w = command.write_operation
        return {
            "kind": kind,
            "source": w.source if w.HasField("source") else None,
            "mode": int(w.mode),
            "save": w.WhichOneof("save_type"),
            "path": w.path if w.HasField("path") else None,
            "table": [w.table.table_name, int(w.table.save_method)] if w.HasField("table") else None,
            "options": [list(item) for item in sorted(w.options.items())],
            "partitioning": list(w.partitioning_columns),
            "sort": list(w.sort_column_names),
            "clustering": list(w.clustering_columns),
            "bucket": [list(w.bucket_by.bucket_column_names), w.bucket_by.num_buckets] if w.HasField("bucket_by") else None,
        }
    if kind == "write_operation_v2":
        w = command.write_operation_v2
        return {
            "kind": kind,
            "table": w.table_name,
            "provider": w.provider if w.HasField("provider") else None,
            "mode": int(w.mode),
            "options": [list(item) for item in sorted(w.options.items())],
            "properties": [list(item) for item in sorted(w.table_properties.items())],
            "partitioning": len(w.partitioning_columns),
            "clustering": list(w.clustering_columns),
            "condition": w.HasField("overwrite_condition"),
        }
    return {"kind": kind}


def _write_cases(df):
    BAD = _BAD_STR
    w = lambda: df.write  # noqa: E731
    v2 = lambda: df.writeTo("t")  # noqa: E731
    return {
        "mode(append).save(p)": lambda: w().mode("append").save("p"),
        "mode(bogus).save(p)": lambda: w().mode("bogus").save("p"),
        "mode(b'append').save(p)": lambda: w().mode(b"append").save("p"),
        "mode(5).save(p)": lambda: w().mode(5).save("p"),
        "mode(None).save(p)": lambda: w().mode(None).save("p"),
        "format(5).save(p)": lambda: w().format(5).save("p"),
        "format(bad).save(p)": lambda: w().format(BAD).save("p"),
        "format(parquet).save()": lambda: w().format("parquet").save(),
        "save(5)": lambda: w().save(5),
        "save(bad)": lambda: w().save(BAD),
        "save(p, format=json, mode=overwrite, k=1)": lambda: w().save("p", format="json", mode="overwrite", k=1),
        "option(k, None).save(p)": lambda: w().option("k", None).save("p"),
        "option(k, True).save(p)": lambda: w().option("k", True).save("p"),
        "option(5, v).save(p)": lambda: w().option(5, "v").save("p"),
        "option(b'k', v).save(p)": lambda: w().option(b"k", "v").save("p"),
        "option(bad, v).save(p)": lambda: w().option(BAD, "v").save("p"),
        "option(k, bad).save(p)": lambda: w().option("k", BAD).save("p"),
        "option([1], v)": lambda: w().option([1], "v"),
        "options(k=bad).save(p)": lambda: w().options(k=BAD).save("p"),
        "partitionBy(a, b).save(p)": lambda: w().partitionBy("a", "b").save("p"),
        "partitionBy([a, b]).save(p)": lambda: w().partitionBy(["a", "b"]).save("p"),
        "partitionBy(5).save(p)": lambda: w().partitionBy(5).save("p"),
        "partitionBy(bad).save(p)": lambda: w().partitionBy(BAD).save("p"),
        "bucketBy(2, a).save(p)": lambda: w().bucketBy(2, "a").save("p"),
        "bucketBy(x, a)": lambda: w().bucketBy("x", "a"),
        "bucketBy(True, a).save(p)": lambda: w().bucketBy(True, "a").save("p"),
        "bucketBy(0, a).save(p)": lambda: w().bucketBy(0, "a").save("p"),
        "bucketBy(2**40, a).save(p)": lambda: w().bucketBy(2**40, "a").save("p"),
        "bucketBy(2, [a], b)": lambda: w().bucketBy(2, ["a"], "b"),
        "bucketBy(2, [])": lambda: w().bucketBy(2, []),
        "bucketBy(2, a, 5)": lambda: w().bucketBy(2, "a", 5),
        "bucketBy(2, 5)": lambda: w().bucketBy(2, 5),
        "sortBy(a, b).save(p)": lambda: w().sortBy("a", "b").save("p"),
        "sortBy((a,), b)": lambda: w().sortBy(("a",), "b"),
        "sortBy(5)": lambda: w().sortBy(5),
        "clusterBy(a).save(p)": lambda: w().clusterBy("a").save("p"),
        "clusterBy()": lambda: w().clusterBy(),
        "clusterBy(5).save(p)": lambda: w().clusterBy(5).save("p"),
        "saveAsTable(t)": lambda: w().saveAsTable("t"),
        "saveAsTable(5)": lambda: w().saveAsTable(5),
        "saveAsTable(t, format=parquet, mode=append, partitionBy=a)": lambda: w().saveAsTable("t", format="parquet", mode="append", partitionBy="a"),
        "insertInto(t)": lambda: w().insertInto("t"),
        "insertInto(t, overwrite=True)": lambda: w().insertInto("t", overwrite=True),
        "insertInto(t, overwrite=False)": lambda: w().insertInto("t", overwrite=False),
        "insertInto(5)": lambda: w().insertInto(5),
        "jdbc(url, t)": lambda: w().jdbc("jdbc:x", "t"),
        "jdbc(url, t, mode=overwrite, properties)": lambda: w().jdbc("jdbc:x", "t", mode="overwrite", properties={"user": "u"}),
        "jdbc(url, t, properties int key)": lambda: w().jdbc("jdbc:x", "t", properties={1: "v"}),
        "csv(p, header=True)": lambda: w().csv("p", header=True),
        "json(p, mode=bogus)": lambda: w().json("p", mode="bogus"),
        "parquet(p, partitionBy=a, compression=snappy)": lambda: w().parquet("p", partitionBy="a", compression="snappy"),
        "text(p, lineSep=bad)": lambda: w().text("p", lineSep=BAD),
        "orc(p)": lambda: w().orc("p"),
        "xml(p, rowTag=r)": lambda: w().xml("p", rowTag="r"),
        "v2 create()": lambda: v2().create(),
        "v2 using(delta).create()": lambda: v2().using("delta").create(),
        "v2 using(5).create()": lambda: v2().using(5).create(),
        "v2 replace()": lambda: v2().replace(),
        "v2 createOrReplace()": lambda: v2().createOrReplace(),
        "v2 append()": lambda: v2().append(),
        "v2 overwrite(col)": lambda: v2().overwrite(df.id > 0),
        "v2 overwrite(str)": lambda: v2().overwrite("id"),
        "v2 overwrite(5)": lambda: v2().overwrite(5),
        "v2 overwritePartitions()": lambda: v2().overwritePartitions(),
        "v2 option(k, None).append()": lambda: v2().option("k", None).append(),
        "v2 option(k, True).append()": lambda: v2().option("k", True).append(),
        "v2 option(5, v).append()": lambda: v2().option(5, "v").append(),
        "v2 option(k, bad).append()": lambda: v2().option("k", BAD).append(),
        "v2 options(k=1).append()": lambda: v2().options(k=1).append(),
        "v2 tableProperty(p, v).create()": lambda: v2().tableProperty("p", "v").create(),
        "v2 tableProperty(p, None).create()": lambda: v2().tableProperty("p", None).create(),
        "v2 tableProperty(5, v).create()": lambda: v2().tableProperty(5, "v").create(),
        "v2 tableProperty(p, bad).create()": lambda: v2().tableProperty("p", BAD).create(),
        "v2 partitionedBy(col).create()": lambda: v2().partitionedBy(df.id).create(),
        "v2 partitionedBy(str, str).create()": lambda: v2().partitionedBy("id", "x").create(),
        "v2 partitionedBy(5)": lambda: v2().partitionedBy(5),
        "v2 clusterBy(a, b).create()": lambda: v2().clusterBy("a", "b").create(),
        "v2 clusterBy(5).create()": lambda: v2().clusterBy(5).create(),
        "v2 writeTo(5).create()": lambda: df.writeTo(5).create(),
        "v2 writeTo(bad).create()": lambda: df.writeTo(BAD).create(),
    }


_WRITE_OUTCOMES = {'mode(append).save(p)': ['sent',
                          {'kind': 'write_operation',
                           'source': None,
                           'mode': 1,
                           'save': 'path',
                           'path': 'p',
                           'table': None,
                           'options': [],
                           'partitioning': [],
                           'sort': [],
                           'clustering': [],
                           'bucket': None}],
 'mode(bogus).save(p)': ['call',
                         'PySparkValueError',
                         '[UNSUPPORTED_OPERATION] bogus is not supported.'],
 "mode(b'append').save(p)": ['call',
                             'PySparkValueError',
                             "[UNSUPPORTED_OPERATION] b'append' is not supported."],
 'mode(5).save(p)': ['call', 'AttributeError', "'int' object has no attribute 'lower'"],
 'mode(None).save(p)': ['sent',
                        {'kind': 'write_operation',
                         'source': None,
                         'mode': 0,
                         'save': 'path',
                         'path': 'p',
                         'table': None,
                         'options': [],
                         'partitioning': [],
                         'sort': [],
                         'clustering': [],
                         'bucket': None}],
 'format(5).save(p)': ['call', 'TypeError', 'bad argument type for built-in operation'],
 'format(bad).save(p)': ['call',
                         'UnicodeEncodeError',
                         "'utf-8' codec can't encode character '\\udc80' in position 1: "
                         'surrogates not allowed'],
 'format(parquet).save()': ['sent',
                            {'kind': 'write_operation',
                             'source': 'parquet',
                             'mode': 0,
                             'save': None,
                             'path': None,
                             'table': None,
                             'options': [],
                             'partitioning': [],
                             'sort': [],
                             'clustering': [],
                             'bucket': None}],
 'save(5)': ['call', 'TypeError', 'bad argument type for built-in operation'],
 'save(bad)': ['call',
               'UnicodeEncodeError',
               "'utf-8' codec can't encode character '\\udc80' in position 1: surrogates not "
               'allowed'],
 'save(p, format=json, mode=overwrite, k=1)': ['sent',
                                               {'kind': 'write_operation',
                                                'source': 'json',
                                                'mode': 2,
                                                'save': 'path',
                                                'path': 'p',
                                                'table': None,
                                                'options': [['k', '1']],
                                                'partitioning': [],
                                                'sort': [],
                                                'clustering': [],
                                                'bucket': None}],
 'option(k, None).save(p)': ['sent',
                             {'kind': 'write_operation',
                              'source': None,
                              'mode': 0,
                              'save': 'path',
                              'path': 'p',
                              'table': None,
                              'options': [],
                              'partitioning': [],
                              'sort': [],
                              'clustering': [],
                              'bucket': None}],
 'option(k, True).save(p)': ['sent',
                             {'kind': 'write_operation',
                              'source': None,
                              'mode': 0,
                              'save': 'path',
                              'path': 'p',
                              'table': None,
                              'options': [['k', 'true']],
                              'partitioning': [],
                              'sort': [],
                              'clustering': [],
                              'bucket': None}],
 'option(5, v).save(p)': ['call', 'TypeError', 'bad argument type for built-in operation'],
 "option(b'k', v).save(p)": ['sent',
                             {'kind': 'write_operation',
                              'source': None,
                              'mode': 0,
                              'save': 'path',
                              'path': 'p',
                              'table': None,
                              'options': [['k', 'v']],
                              'partitioning': [],
                              'sort': [],
                              'clustering': [],
                              'bucket': None}],
 'option(bad, v).save(p)': ['call',
                            'UnicodeEncodeError',
                            "'utf-8' codec can't encode character '\\udc80' in position 1: "
                            'surrogates not allowed'],
 'option(k, bad).save(p)': ['call',
                            'UnicodeEncodeError',
                            "'utf-8' codec can't encode character '\\udc80' in position 1: "
                            'surrogates not allowed'],
 'option([1], v)': ['call', 'TypeError', "unhashable type: 'list'"],
 'options(k=bad).save(p)': ['call',
                            'UnicodeEncodeError',
                            "'utf-8' codec can't encode character '\\udc80' in position 1: "
                            'surrogates not allowed'],
 'partitionBy(a, b).save(p)': ['sent',
                               {'kind': 'write_operation',
                                'source': None,
                                'mode': 0,
                                'save': 'path',
                                'path': 'p',
                                'table': None,
                                'options': [],
                                'partitioning': ['a', 'b'],
                                'sort': [],
                                'clustering': [],
                                'bucket': None}],
 'partitionBy([a, b]).save(p)': ['sent',
                                 {'kind': 'write_operation',
                                  'source': None,
                                  'mode': 0,
                                  'save': 'path',
                                  'path': 'p',
                                  'table': None,
                                  'options': [],
                                  'partitioning': ['a', 'b'],
                                  'sort': [],
                                  'clustering': [],
                                  'bucket': None}],
 'partitionBy(5).save(p)': ['call', 'TypeError', 'bad argument type for built-in operation'],
 'partitionBy(bad).save(p)': ['call',
                              'UnicodeEncodeError',
                              "'utf-8' codec can't encode character '\\udc80' in position 1: "
                              'surrogates not allowed'],
 'bucketBy(2, a).save(p)': ['sent',
                            {'kind': 'write_operation',
                             'source': None,
                             'mode': 0,
                             'save': 'path',
                             'path': 'p',
                             'table': None,
                             'options': [],
                             'partitioning': [],
                             'sort': [],
                             'clustering': [],
                             'bucket': [['a'], 2]}],
 'bucketBy(x, a)': ['call',
                    'PySparkTypeError',
                    '[NOT_EXPECTED_TYPE] Argument `numBuckets` should be int, got str.'],
 'bucketBy(True, a).save(p)': ['call',
                               'TypeError',
                               'Field spark.connect.WriteOperation.BucketBy.num_buckets: '
                               'Expected an int, got a boolean.'],
 'bucketBy(0, a).save(p)': ['sent',
                            {'kind': 'write_operation',
                             'source': None,
                             'mode': 0,
                             'save': 'path',
                             'path': 'p',
                             'table': None,
                             'options': [],
                             'partitioning': [],
                             'sort': [],
                             'clustering': [],
                             'bucket': None}],
 'bucketBy(2**40, a).save(p)': ['call', 'ValueError', 'Value out of range: 1099511627776'],
 'bucketBy(2, [a], b)': ['call',
                         'PySparkValueError',
                         '[CANNOT_SET_TOGETHER] `col` of type list and `cols` should not be '
                         'set together.'],
 'bucketBy(2, [])': ['call', 'IndexError', 'list index out of range'],
 'bucketBy(2, a, 5)': ['call',
                       'PySparkTypeError',
                       '[NOT_EXPECTED_TYPE] Argument `cols` should be list[str], got int.'],
 'bucketBy(2, 5)': ['call',
                    'PySparkTypeError',
                    '[NOT_EXPECTED_TYPE] Argument `col` should be list[str], got int.'],
 'sortBy(a, b).save(p)': ['sent',
                          {'kind': 'write_operation',
                           'source': None,
                           'mode': 0,
                           'save': 'path',
                           'path': 'p',
                           'table': None,
                           'options': [],
                           'partitioning': [],
                           'sort': ['a', 'b'],
                           'clustering': [],
                           'bucket': None}],
 'sortBy((a,), b)': ['call',
                     'PySparkValueError',
                     '[CANNOT_SET_TOGETHER] `col` of type tuple and `cols` should not be set '
                     'together.'],
 'sortBy(5)': ['call',
               'PySparkTypeError',
               '[NOT_EXPECTED_TYPE] Argument `col` should be list[str], got int.'],
 'clusterBy(a).save(p)': ['sent',
                          {'kind': 'write_operation',
                           'source': None,
                           'mode': 0,
                           'save': 'path',
                           'path': 'p',
                           'table': None,
                           'options': [],
                           'partitioning': [],
                           'sort': [],
                           'clustering': ['a'],
                           'bucket': None}],
 'clusterBy()': ['call', 'AssertionError', 'clusterBy needs one or more clustering columns.'],
 'clusterBy(5).save(p)': ['call', 'TypeError', 'bad argument type for built-in operation'],
 'saveAsTable(t)': ['sent',
                    {'kind': 'write_operation',
                     'source': None,
                     'mode': 0,
                     'save': 'table',
                     'path': None,
                     'table': ['t', 1],
                     'options': [],
                     'partitioning': [],
                     'sort': [],
                     'clustering': [],
                     'bucket': None}],
 'saveAsTable(5)': ['call', 'TypeError', 'bad argument type for built-in operation'],
 'saveAsTable(t, format=parquet, mode=append, partitionBy=a)': ['sent',
                                                                {'kind': 'write_operation',
                                                                 'source': 'parquet',
                                                                 'mode': 1,
                                                                 'save': 'table',
                                                                 'path': None,
                                                                 'table': ['t', 1],
                                                                 'options': [],
                                                                 'partitioning': ['a'],
                                                                 'sort': [],
                                                                 'clustering': [],
                                                                 'bucket': None}],
 'insertInto(t)': ['sent',
                   {'kind': 'write_operation',
                    'source': None,
                    'mode': 0,
                    'save': 'table',
                    'path': None,
                    'table': ['t', 2],
                    'options': [],
                    'partitioning': [],
                    'sort': [],
                    'clustering': [],
                    'bucket': None}],
 'insertInto(t, overwrite=True)': ['sent',
                                   {'kind': 'write_operation',
                                    'source': None,
                                    'mode': 2,
                                    'save': 'table',
                                    'path': None,
                                    'table': ['t', 2],
                                    'options': [],
                                    'partitioning': [],
                                    'sort': [],
                                    'clustering': [],
                                    'bucket': None}],
 'insertInto(t, overwrite=False)': ['sent',
                                    {'kind': 'write_operation',
                                     'source': None,
                                     'mode': 1,
                                     'save': 'table',
                                     'path': None,
                                     'table': ['t', 2],
                                     'options': [],
                                     'partitioning': [],
                                     'sort': [],
                                     'clustering': [],
                                     'bucket': None}],
 'insertInto(5)': ['call', 'TypeError', 'bad argument type for built-in operation'],
 'jdbc(url, t)': ['sent',
                  {'kind': 'write_operation',
                   'source': 'jdbc',
                   'mode': 0,
                   'save': None,
                   'path': None,
                   'table': None,
                   'options': [['dbtable', 't'], ['url', 'jdbc:x']],
                   'partitioning': [],
                   'sort': [],
                   'clustering': [],
                   'bucket': None}],
 'jdbc(url, t, mode=overwrite, properties)': ['sent',
                                              {'kind': 'write_operation',
                                               'source': 'jdbc',
                                               'mode': 2,
                                               'save': None,
                                               'path': None,
                                               'table': None,
                                               'options': [['dbtable', 't'],
                                                           ['url', 'jdbc:x'],
                                                           ['user', 'u']],
                                               'partitioning': [],
                                               'sort': [],
                                               'clustering': [],
                                               'bucket': None}],
 'jdbc(url, t, properties int key)': ['call', 'TypeError', 'keywords must be strings'],
 'csv(p, header=True)': ['sent',
                         {'kind': 'write_operation',
                          'source': 'csv',
                          'mode': 0,
                          'save': 'path',
                          'path': 'p',
                          'table': None,
                          'options': [['header', 'true']],
                          'partitioning': [],
                          'sort': [],
                          'clustering': [],
                          'bucket': None}],
 'json(p, mode=bogus)': ['call',
                         'PySparkValueError',
                         '[UNSUPPORTED_OPERATION] bogus is not supported.'],
 'parquet(p, partitionBy=a, compression=snappy)': ['sent',
                                                   {'kind': 'write_operation',
                                                    'source': 'parquet',
                                                    'mode': 0,
                                                    'save': 'path',
                                                    'path': 'p',
                                                    'table': None,
                                                    'options': [['compression', 'snappy']],
                                                    'partitioning': ['a'],
                                                    'sort': [],
                                                    'clustering': [],
                                                    'bucket': None}],
 'text(p, lineSep=bad)': ['call',
                          'UnicodeEncodeError',
                          "'utf-8' codec can't encode character '\\udc80' in position 1: "
                          'surrogates not allowed'],
 'orc(p)': ['sent',
            {'kind': 'write_operation',
             'source': 'orc',
             'mode': 0,
             'save': 'path',
             'path': 'p',
             'table': None,
             'options': [],
             'partitioning': [],
             'sort': [],
             'clustering': [],
             'bucket': None}],
 'xml(p, rowTag=r)': ['sent',
                      {'kind': 'write_operation',
                       'source': 'xml',
                       'mode': 0,
                       'save': 'path',
                       'path': 'p',
                       'table': None,
                       'options': [['rowTag', 'r']],
                       'partitioning': [],
                       'sort': [],
                       'clustering': [],
                       'bucket': None}],
 'v2 create()': ['sent',
                 {'kind': 'write_operation_v2',
                  'table': 't',
                  'provider': None,
                  'mode': 1,
                  'options': [],
                  'properties': [],
                  'partitioning': 0,
                  'clustering': [],
                  'condition': False}],
 'v2 using(delta).create()': ['sent',
                              {'kind': 'write_operation_v2',
                               'table': 't',
                               'provider': 'delta',
                               'mode': 1,
                               'options': [],
                               'properties': [],
                               'partitioning': 0,
                               'clustering': [],
                               'condition': False}],
 'v2 using(5).create()': ['call', 'TypeError', 'bad argument type for built-in operation'],
 'v2 replace()': ['sent',
                  {'kind': 'write_operation_v2',
                   'table': 't',
                   'provider': None,
                   'mode': 5,
                   'options': [],
                   'properties': [],
                   'partitioning': 0,
                   'clustering': [],
                   'condition': False}],
 'v2 createOrReplace()': ['sent',
                          {'kind': 'write_operation_v2',
                           'table': 't',
                           'provider': None,
                           'mode': 6,
                           'options': [],
                           'properties': [],
                           'partitioning': 0,
                           'clustering': [],
                           'condition': False}],
 'v2 append()': ['sent',
                 {'kind': 'write_operation_v2',
                  'table': 't',
                  'provider': None,
                  'mode': 4,
                  'options': [],
                  'properties': [],
                  'partitioning': 0,
                  'clustering': [],
                  'condition': False}],
 'v2 overwrite(col)': ['sent',
                       {'kind': 'write_operation_v2',
                        'table': 't',
                        'provider': None,
                        'mode': 2,
                        'options': [],
                        'properties': [],
                        'partitioning': 0,
                        'clustering': [],
                        'condition': True}],
 'v2 overwrite(str)': ['sent',
                       {'kind': 'write_operation_v2',
                        'table': 't',
                        'provider': None,
                        'mode': 2,
                        'options': [],
                        'properties': [],
                        'partitioning': 0,
                        'clustering': [],
                        'condition': True}],
 'v2 overwrite(5)': ['call',
                     'PySparkTypeError',
                     '[NOT_EXPECTED_TYPE] Argument `col` should be Column or str, got int.'],
 'v2 overwritePartitions()': ['sent',
                              {'kind': 'write_operation_v2',
                               'table': 't',
                               'provider': None,
                               'mode': 3,
                               'options': [],
                               'properties': [],
                               'partitioning': 0,
                               'clustering': [],
                               'condition': False}],
 'v2 option(k, None).append()': ['sent',
                                 {'kind': 'write_operation_v2',
                                  'table': 't',
                                  'provider': None,
                                  'mode': 4,
                                  'options': [],
                                  'properties': [],
                                  'partitioning': 0,
                                  'clustering': [],
                                  'condition': False}],
 'v2 option(k, True).append()': ['sent',
                                 {'kind': 'write_operation_v2',
                                  'table': 't',
                                  'provider': None,
                                  'mode': 4,
                                  'options': [['k', 'true']],
                                  'properties': [],
                                  'partitioning': 0,
                                  'clustering': [],
                                  'condition': False}],
 'v2 option(5, v).append()': ['call', 'TypeError', 'bad argument type for built-in operation'],
 'v2 option(k, bad).append()': ['call',
                                'UnicodeEncodeError',
                                "'utf-8' codec can't encode character '\\udc80' in position 1: "
                                'surrogates not allowed'],
 'v2 options(k=1).append()': ['sent',
                              {'kind': 'write_operation_v2',
                               'table': 't',
                               'provider': None,
                               'mode': 4,
                               'options': [['k', '1']],
                               'properties': [],
                               'partitioning': 0,
                               'clustering': [],
                               'condition': False}],
 'v2 tableProperty(p, v).create()': ['sent',
                                     {'kind': 'write_operation_v2',
                                      'table': 't',
                                      'provider': None,
                                      'mode': 1,
                                      'options': [],
                                      'properties': [['p', 'v']],
                                      'partitioning': 0,
                                      'clustering': [],
                                      'condition': False}],
 'v2 tableProperty(p, None).create()': ['sent',
                                        {'kind': 'write_operation_v2',
                                         'table': 't',
                                         'provider': None,
                                         'mode': 1,
                                         'options': [],
                                         'properties': [],
                                         'partitioning': 0,
                                         'clustering': [],
                                         'condition': False}],
 'v2 tableProperty(5, v).create()': ['call',
                                     'TypeError',
                                     'bad argument type for built-in operation'],
 'v2 tableProperty(p, bad).create()': ['call',
                                       'UnicodeEncodeError',
                                       "'utf-8' codec can't encode character '\\udc80' in "
                                       'position 1: surrogates not allowed'],
 'v2 partitionedBy(col).create()': ['sent',
                                    {'kind': 'write_operation_v2',
                                     'table': 't',
                                     'provider': None,
                                     'mode': 1,
                                     'options': [],
                                     'properties': [],
                                     'partitioning': 1,
                                     'clustering': [],
                                     'condition': False}],
 'v2 partitionedBy(str, str).create()': ['sent',
                                         {'kind': 'write_operation_v2',
                                          'table': 't',
                                          'provider': None,
                                          'mode': 1,
                                          'options': [],
                                          'properties': [],
                                          'partitioning': 2,
                                          'clustering': [],
                                          'condition': False}],
 'v2 partitionedBy(5)': ['call',
                         'PySparkTypeError',
                         '[NOT_EXPECTED_TYPE] Argument `col` should be Column or str, got '
                         'int.'],
 'v2 clusterBy(a, b).create()': ['sent',
                                 {'kind': 'write_operation_v2',
                                  'table': 't',
                                  'provider': None,
                                  'mode': 1,
                                  'options': [],
                                  'properties': [],
                                  'partitioning': 0,
                                  'clustering': ['a', 'b'],
                                  'condition': False}],
 'v2 clusterBy(5).create()': ['call', 'TypeError', 'bad argument type for built-in operation'],
 'v2 writeTo(5).create()': ['call', 'TypeError', 'bad argument type for built-in operation'],
 'v2 writeTo(bad).create()': ['call',
                              'UnicodeEncodeError',
                              "'utf-8' codec can't encode character '\\udc80' in position 1: "
                              'surrogates not allowed']}


def _write_outcome(service, call):
    before = len(service.execute_requests)
    try:
        value = call()
    except BaseException as error:  # noqa: BLE001
        return ["call", type(error).__name__, str(error)[:120]]
    if len(service.execute_requests) == before:
        return ["value", type(value).__name__]
    return ["sent", _sent_write(service)]


@pytest.mark.parametrize("case", list(_WRITE_OUTCOMES))
def test_the_writes_send_what_pysparks_writers_send(objects, case):
    # As PySpark's DataFrameWriter and DataFrameWriterV2, of whose source the drop-in's are
    # adapted: the errors of their setters and of the commands of their writes, in the same order,
    # and the write commands that they send, of their modes, formats, options, partitioning,
    # bucketing, sorting and clustering, tables and paths.
    call = _write_cases(objects.spark.range(1))[case]
    assert _write_outcome(objects.service, call) == _WRITE_OUTCOMES[case]


@pytest.mark.parametrize("kind", ["write", "writeTo", "writeStream", "mergeInto"])
def test_a_writer_in_a_cycle_through_a_udf_is_collected(objects, kind):
    # As PySpark's writers, plain objects of their DataFrame's plan (the `_df` and the `_child` of
    # the `_write` of a writer, and the callback of its DataFrame; the `_child` of the
    # `_write_stream` of a stream writer; the `_source_plan` of a merge writer, and the callback of
    # its DataFrame): a cycle of the writer, its DataFrame and a function of a UDF of the plan,
    # which refers to the writer, is freed by a collection.
    from pyspark.sql.functions import udf

    def f(x):
        return x

    df = objects.spark.range(1).select(udf(f, "long")("id"))
    if kind == "write":
        writer = df.write
    elif kind == "writeTo":
        writer = df.writeTo("t")
    elif kind == "writeStream":
        writer = df.writeStream
    else:
        writer = df.mergeInto("t", df.id > 0)
    f.writer = writer
    freed = [weakref.ref(writer), weakref.ref(df), weakref.ref(f)]
    del df, writer, f
    gc.collect()
    assert [ref() for ref in freed] == [None, None, None]


# The starts of the stream writer, as PySpark's DataStreamWriter and its WriteStreamOperation make
# them, of PySpark 4.3.0: what the call raises, and else the start command that the server receives
# of it and the query that it gives, of the start result of the server, without a started event.
def _start_result():
    from _mock_connect_server import message_field

    # ExecutePlanResponse.write_stream_operation_start_result=8 of query_id=1 (id=1, run_id=2)
    # and name=2.
    return message_field(
        8, message_field(1, message_field(1, "q") + message_field(2, "r")) + message_field(2, "n")
    )


def _sent_stream_write(service):
    import sys

    import pyspark.sql.connect.proto as pb2

    request = pb2.ExecutePlanRequest.FromString(service.execute_requests[-1])
    command = request.plan.command
    kind = command.WhichOneof("command_type")
    if kind != "write_stream_operation_start":
        return {"kind": kind}
    w = command.write_stream_operation_start
    trigger = w.WhichOneof("trigger")
    sink = w.WhichOneof("sink_destination")

    def function(f):
        python = f.python_function
        return [
            f.WhichOneof("function"),
            bool(python.command),
            python.python_ver == "{}.{}".format(*sys.version_info[:2]),
        ]

    return {
        "kind": kind,
        "input": w.HasField("input"),
        "format": w.format,
        "options": [list(item) for item in sorted(w.options.items())],
        "partitioning": list(w.partitioning_column_names),
        "clustering": list(w.clustering_column_names),
        "trigger": [trigger, getattr(w, trigger)] if trigger else None,
        "output_mode": w.output_mode,
        "query_name": w.query_name,
        "sink": [sink, getattr(w, sink)] if sink else None,
        "foreach_writer": function(w.foreach_writer) if w.HasField("foreach_writer") else None,
        "foreach_batch": function(w.foreach_batch) if w.HasField("foreach_batch") else None,
    }


class _Rows:
    def process(self, row):
        pass


class _Opened:
    def open(self, partition_id, epoch_id):
        return True

    def process(self, row):
        pass

    def close(self, error):
        pass


class _NoProcess:
    def open(self, partition_id, epoch_id):
        return True


class _ProcessNotCallable:
    process = 5


def _stream_write_cases(sdf):
    import threading

    BAD = _BAD_STR
    lock = threading.Lock()

    def f(*args):
        return None

    def unpicklable(*args):
        return lock

    w = lambda: sdf.writeStream  # noqa: E731
    return {
        "start()": lambda: w().start(),
        "start(p)": lambda: w().start("p"),
        "start('')": lambda: w().start(""),
        "start(5)": lambda: w().start(5),
        "start(bad)": lambda: w().start(BAD),
        "start(p, format=json, outputMode=append, partitionBy=a, queryName=q, k=1)": lambda: w().start(
            "p", format="json", outputMode="append", partitionBy="a", queryName="q", k=1
        ),
        "start(k=None)": lambda: w().start(k=None),
        "start(k=True)": lambda: w().start(k=True),
        "start(k=bad)": lambda: w().start(k=BAD),
        "start(**{1: v})": lambda: w().start(**{1: "v"}),
        "start(queryName='')": lambda: w().start(queryName=""),
        "start(partitionBy=[a, b])": lambda: w().start(partitionBy=["a", "b"]),
        "toTable(t)": lambda: w().toTable("t"),
        "toTable('')": lambda: w().toTable(""),
        "toTable(5)": lambda: w().toTable(5),
        "toTable(bad)": lambda: w().toTable(BAD),
        "toTable(t, format=delta, outputMode=complete, partitionBy=[a, b], queryName=q, k=v)": lambda: w().toTable(
            "t", format="delta", outputMode="complete", partitionBy=["a", "b"], queryName="q", k="v"
        ),
        "outputMode(append).start()": lambda: w().outputMode("append").start(),
        "outputMode(b'append').start()": lambda: w().outputMode(b"append").start(),
        "outputMode(5)": lambda: w().outputMode(5),
        "outputMode(None)": lambda: w().outputMode(None),
        "outputMode(bad)": lambda: w().outputMode(BAD),
        "format(console).start()": lambda: w().format("console").start(),
        "format(5)": lambda: w().format(5),
        "format(None)": lambda: w().format(None),
        "format(bad)": lambda: w().format(BAD),
        "option(k, v).start()": lambda: w().option("k", "v").start(),
        "option(k, None).start()": lambda: w().option("k", None).start(),
        "option(k, True).start()": lambda: w().option("k", True).start(),
        "option(k, 1.5).start()": lambda: w().option("k", 1.5).start(),
        "option(5, v)": lambda: w().option(5, "v"),
        "option(b'k', v).start()": lambda: w().option(b"k", "v").start(),
        "option(bad, v)": lambda: w().option(BAD, "v"),
        "option(k, bad)": lambda: w().option("k", BAD),
        "option(k, b'v').start()": lambda: w().option("k", b"v").start(),
        "options(a=1, b=None).start()": lambda: w().options(a=1, b=None).start(),
        "options(**{1: v})": lambda: w().options(**{1: "v"}),
        "partitionBy(a, b).start()": lambda: w().partitionBy("a", "b").start(),
        "partitionBy([a, b]).start()": lambda: w().partitionBy(["a", "b"]).start(),
        "partitionBy((a,)).start()": lambda: w().partitionBy(("a",)).start(),
        "partitionBy(a).partitionBy(b).start()": lambda: w().partitionBy("a").partitionBy("b").start(),
        "partitionBy().start()": lambda: w().partitionBy().start(),
        "partitionBy(5)": lambda: w().partitionBy(5),
        "partitionBy(a, 5)": lambda: w().partitionBy("a", 5),
        "partitionBy(bad)": lambda: w().partitionBy(BAD),
        "partitionBy([a], b)": lambda: w().partitionBy(["a"], "b"),
        "clusterBy(a).start()": lambda: w().clusterBy("a").start(),
        "clusterBy([a, b]).start()": lambda: w().clusterBy(["a", "b"]).start(),
        "clusterBy(5)": lambda: w().clusterBy(5),
        "queryName(q).start()": lambda: w().queryName("q").start(),
        "queryName('')": lambda: w().queryName(""),
        "queryName('  ')": lambda: w().queryName("  "),
        "queryName(5)": lambda: w().queryName(5),
        "queryName(None)": lambda: w().queryName(None),
        "queryName(bad)": lambda: w().queryName(BAD),
        "trigger()": lambda: w().trigger(),
        "trigger(5)": lambda: w().trigger(5),
        "trigger(processingTime=1 second).start()": lambda: w().trigger(processingTime="1 second").start(),
        "trigger(processingTime=' 1 second ').start()": lambda: w().trigger(processingTime=" 1 second ").start(),
        "trigger(processingTime='')": lambda: w().trigger(processingTime=""),
        "trigger(processingTime=5)": lambda: w().trigger(processingTime=5),
        "trigger(processingTime=bad)": lambda: w().trigger(processingTime=BAD),
        "trigger(once=True).start()": lambda: w().trigger(once=True).start(),
        "trigger(once=False)": lambda: w().trigger(once=False),
        "trigger(once=1)": lambda: w().trigger(once=1),
        "trigger(availableNow=True).start()": lambda: w().trigger(availableNow=True).start(),
        "trigger(availableNow=False)": lambda: w().trigger(availableNow=False),
        "trigger(continuous=1 second).start()": lambda: w().trigger(continuous="1 second").start(),
        "trigger(continuous='')": lambda: w().trigger(continuous=""),
        "trigger(realTime=5 seconds).start()": lambda: w().trigger(realTime="5 seconds").start(),
        "trigger(realTime=5)": lambda: w().trigger(realTime=5),
        "trigger(processingTime=1 s, once=True)": lambda: w().trigger(processingTime="1 s", once=True),
        "trigger(once=True).trigger(availableNow=True).start()": lambda: w().trigger(once=True).trigger(availableNow=True).start(),
        "foreach(f).start()": lambda: w().foreach(f).start(),
        "foreach(process).start()": lambda: w().foreach(_Rows()).start(),
        "foreach(open, process, close).start()": lambda: w().foreach(_Opened()).start(),
        "foreach(no process)": lambda: w().foreach(_NoProcess()),
        "foreach(process not callable)": lambda: w().foreach(_ProcessNotCallable()),
        "foreach(5)": lambda: w().foreach(5),
        "foreach(unpicklable)": lambda: w().foreach(unpicklable),
        "foreachBatch(f).start()": lambda: w().foreachBatch(f).start(),
        "foreachBatch(5).start()": lambda: w().foreachBatch(5).start(),
        "foreachBatch(unpicklable)": lambda: w().foreachBatch(unpicklable),
        "foreachBatch(f).foreach(f).start()": lambda: w().foreachBatch(f).foreach(f).start(),
    }


_STREAM_WRITE_OUTCOMES = {'start()': ['sent',
             {'kind': 'write_stream_operation_start',
              'input': True,
              'format': '',
              'options': [],
              'partitioning': [],
              'clustering': [],
              'trigger': None,
              'output_mode': '',
              'query_name': '',
              'sink': None,
              'foreach_writer': None,
              'foreach_batch': None},
             ['StreamingQuery', 'q', 'r', 'n']],
 'start(p)': ['sent',
              {'kind': 'write_stream_operation_start',
               'input': True,
               'format': '',
               'options': [],
               'partitioning': [],
               'clustering': [],
               'trigger': None,
               'output_mode': '',
               'query_name': '',
               'sink': ['path', 'p'],
               'foreach_writer': None,
               'foreach_batch': None},
              ['StreamingQuery', 'q', 'r', 'n']],
 "start('')": ['sent',
               {'kind': 'write_stream_operation_start',
                'input': True,
                'format': '',
                'options': [],
                'partitioning': [],
                'clustering': [],
                'trigger': None,
                'output_mode': '',
                'query_name': '',
                'sink': None,
                'foreach_writer': None,
                'foreach_batch': None},
               ['StreamingQuery', 'q', 'r', 'n']],
 'start(5)': ['call', 'TypeError', 'bad argument type for built-in operation'],
 'start(bad)': ['call',
                'UnicodeEncodeError',
                "'utf-8' codec can't encode character '\\udc80' in position 1: surrogates not "
                'allowed'],
 'start(p, format=json, outputMode=append, partitionBy=a, queryName=q, k=1)': ['sent',
                                                                               {'kind': 'write_stream_operation_start',
                                                                                'input': True,
                                                                                'format': 'json',
                                                                                'options': [['k',
                                                                                             '1']],
                                                                                'partitioning': ['a'],
                                                                                'clustering': [],
                                                                                'trigger': None,
                                                                                'output_mode': 'append',
                                                                                'query_name': 'q',
                                                                                'sink': ['path',
                                                                                         'p'],
                                                                                'foreach_writer': None,
                                                                                'foreach_batch': None},
                                                                               ['StreamingQuery',
                                                                                'q',
                                                                                'r',
                                                                                'n']],
 'start(k=None)': ['sent',
                   {'kind': 'write_stream_operation_start',
                    'input': True,
                    'format': '',
                    'options': [],
                    'partitioning': [],
                    'clustering': [],
                    'trigger': None,
                    'output_mode': '',
                    'query_name': '',
                    'sink': None,
                    'foreach_writer': None,
                    'foreach_batch': None},
                   ['StreamingQuery', 'q', 'r', 'n']],
 'start(k=True)': ['sent',
                   {'kind': 'write_stream_operation_start',
                    'input': True,
                    'format': '',
                    'options': [['k', 'true']],
                    'partitioning': [],
                    'clustering': [],
                    'trigger': None,
                    'output_mode': '',
                    'query_name': '',
                    'sink': None,
                    'foreach_writer': None,
                    'foreach_batch': None},
                   ['StreamingQuery', 'q', 'r', 'n']],
 'start(k=bad)': ['call',
                  'UnicodeEncodeError',
                  "'utf-8' codec can't encode character '\\udc80' in position 1: surrogates not "
                  'allowed'],
 'start(**{1: v})': ['call', 'TypeError', 'keywords must be strings'],
 "start(queryName='')": ['call',
                         'PySparkValueError',
                         '[VALUE_NOT_NON_EMPTY_STR] Value for `queryName` must be a non-empty '
                         "string, got ''."],
 'start(partitionBy=[a, b])': ['sent',
                               {'kind': 'write_stream_operation_start',
                                'input': True,
                                'format': '',
                                'options': [],
                                'partitioning': ['a', 'b'],
                                'clustering': [],
                                'trigger': None,
                                'output_mode': '',
                                'query_name': '',
                                'sink': None,
                                'foreach_writer': None,
                                'foreach_batch': None},
                               ['StreamingQuery', 'q', 'r', 'n']],
 'toTable(t)': ['sent',
                {'kind': 'write_stream_operation_start',
                 'input': True,
                 'format': '',
                 'options': [],
                 'partitioning': [],
                 'clustering': [],
                 'trigger': None,
                 'output_mode': '',
                 'query_name': '',
                 'sink': ['table_name', 't'],
                 'foreach_writer': None,
                 'foreach_batch': None},
                ['StreamingQuery', 'q', 'r', 'n']],
 "toTable('')": ['sent',
                 {'kind': 'write_stream_operation_start',
                  'input': True,
                  'format': '',
                  'options': [],
                  'partitioning': [],
                  'clustering': [],
                  'trigger': None,
                  'output_mode': '',
                  'query_name': '',
                  'sink': None,
                  'foreach_writer': None,
                  'foreach_batch': None},
                 ['StreamingQuery', 'q', 'r', 'n']],
 'toTable(5)': ['call', 'TypeError', 'bad argument type for built-in operation'],
 'toTable(bad)': ['call',
                  'UnicodeEncodeError',
                  "'utf-8' codec can't encode character '\\udc80' in position 1: surrogates not "
                  'allowed'],
 'toTable(t, format=delta, outputMode=complete, partitionBy=[a, b], queryName=q, k=v)': ['sent',
                                                                                         {'kind': 'write_stream_operation_start',
                                                                                          'input': True,
                                                                                          'format': 'delta',
                                                                                          'options': [['k',
                                                                                                       'v']],
                                                                                          'partitioning': ['a',
                                                                                                           'b'],
                                                                                          'clustering': [],
                                                                                          'trigger': None,
                                                                                          'output_mode': 'complete',
                                                                                          'query_name': 'q',
                                                                                          'sink': ['table_name',
                                                                                                   't'],
                                                                                          'foreach_writer': None,
                                                                                          'foreach_batch': None},
                                                                                         ['StreamingQuery',
                                                                                          'q',
                                                                                          'r',
                                                                                          'n']],
 'outputMode(append).start()': ['sent',
                                {'kind': 'write_stream_operation_start',
                                 'input': True,
                                 'format': '',
                                 'options': [],
                                 'partitioning': [],
                                 'clustering': [],
                                 'trigger': None,
                                 'output_mode': 'append',
                                 'query_name': '',
                                 'sink': None,
                                 'foreach_writer': None,
                                 'foreach_batch': None},
                                ['StreamingQuery', 'q', 'r', 'n']],
 "outputMode(b'append').start()": ['sent',
                                   {'kind': 'write_stream_operation_start',
                                    'input': True,
                                    'format': '',
                                    'options': [],
                                    'partitioning': [],
                                    'clustering': [],
                                    'trigger': None,
                                    'output_mode': 'append',
                                    'query_name': '',
                                    'sink': None,
                                    'foreach_writer': None,
                                    'foreach_batch': None},
                                   ['StreamingQuery', 'q', 'r', 'n']],
 'outputMode(5)': ['call', 'TypeError', 'bad argument type for built-in operation'],
 'outputMode(None)': ['call', 'TypeError', 'bad argument type for built-in operation'],
 'outputMode(bad)': ['call',
                     'UnicodeEncodeError',
                     "'utf-8' codec can't encode character '\\udc80' in position 1: surrogates not "
                     'allowed'],
 'format(console).start()': ['sent',
                             {'kind': 'write_stream_operation_start',
                              'input': True,
                              'format': 'console',
                              'options': [],
                              'partitioning': [],
                              'clustering': [],
                              'trigger': None,
                              'output_mode': '',
                              'query_name': '',
                              'sink': None,
                              'foreach_writer': None,
                              'foreach_batch': None},
                             ['StreamingQuery', 'q', 'r', 'n']],
 'format(5)': ['call', 'TypeError', 'bad argument type for built-in operation'],
 'format(None)': ['call', 'TypeError', 'bad argument type for built-in operation'],
 'format(bad)': ['call',
                 'UnicodeEncodeError',
                 "'utf-8' codec can't encode character '\\udc80' in position 1: surrogates not "
                 'allowed'],
 'option(k, v).start()': ['sent',
                          {'kind': 'write_stream_operation_start',
                           'input': True,
                           'format': '',
                           'options': [['k', 'v']],
                           'partitioning': [],
                           'clustering': [],
                           'trigger': None,
                           'output_mode': '',
                           'query_name': '',
                           'sink': None,
                           'foreach_writer': None,
                           'foreach_batch': None},
                          ['StreamingQuery', 'q', 'r', 'n']],
 'option(k, None).start()': ['sent',
                             {'kind': 'write_stream_operation_start',
                              'input': True,
                              'format': '',
                              'options': [],
                              'partitioning': [],
                              'clustering': [],
                              'trigger': None,
                              'output_mode': '',
                              'query_name': '',
                              'sink': None,
                              'foreach_writer': None,
                              'foreach_batch': None},
                             ['StreamingQuery', 'q', 'r', 'n']],
 'option(k, True).start()': ['sent',
                             {'kind': 'write_stream_operation_start',
                              'input': True,
                              'format': '',
                              'options': [['k', 'true']],
                              'partitioning': [],
                              'clustering': [],
                              'trigger': None,
                              'output_mode': '',
                              'query_name': '',
                              'sink': None,
                              'foreach_writer': None,
                              'foreach_batch': None},
                             ['StreamingQuery', 'q', 'r', 'n']],
 'option(k, 1.5).start()': ['sent',
                            {'kind': 'write_stream_operation_start',
                             'input': True,
                             'format': '',
                             'options': [['k', '1.5']],
                             'partitioning': [],
                             'clustering': [],
                             'trigger': None,
                             'output_mode': '',
                             'query_name': '',
                             'sink': None,
                             'foreach_writer': None,
                             'foreach_batch': None},
                            ['StreamingQuery', 'q', 'r', 'n']],
 'option(5, v)': ['call', 'TypeError', 'bad argument type for built-in operation'],
 "option(b'k', v).start()": ['sent',
                             {'kind': 'write_stream_operation_start',
                              'input': True,
                              'format': '',
                              'options': [['k', 'v']],
                              'partitioning': [],
                              'clustering': [],
                              'trigger': None,
                              'output_mode': '',
                              'query_name': '',
                              'sink': None,
                              'foreach_writer': None,
                              'foreach_batch': None},
                             ['StreamingQuery', 'q', 'r', 'n']],
 'option(bad, v)': ['call',
                    'UnicodeEncodeError',
                    "'utf-8' codec can't encode character '\\udc80' in position 1: surrogates not "
                    'allowed'],
 'option(k, bad)': ['call',
                    'UnicodeEncodeError',
                    "'utf-8' codec can't encode character '\\udc80' in position 1: surrogates not "
                    'allowed'],
 "option(k, b'v').start()": ['sent',
                             {'kind': 'write_stream_operation_start',
                              'input': True,
                              'format': '',
                              'options': [['k', "b'v'"]],
                              'partitioning': [],
                              'clustering': [],
                              'trigger': None,
                              'output_mode': '',
                              'query_name': '',
                              'sink': None,
                              'foreach_writer': None,
                              'foreach_batch': None},
                             ['StreamingQuery', 'q', 'r', 'n']],
 'options(a=1, b=None).start()': ['sent',
                                  {'kind': 'write_stream_operation_start',
                                   'input': True,
                                   'format': '',
                                   'options': [['a', '1']],
                                   'partitioning': [],
                                   'clustering': [],
                                   'trigger': None,
                                   'output_mode': '',
                                   'query_name': '',
                                   'sink': None,
                                   'foreach_writer': None,
                                   'foreach_batch': None},
                                  ['StreamingQuery', 'q', 'r', 'n']],
 'options(**{1: v})': ['call', 'TypeError', 'keywords must be strings'],
 'partitionBy(a, b).start()': ['sent',
                               {'kind': 'write_stream_operation_start',
                                'input': True,
                                'format': '',
                                'options': [],
                                'partitioning': ['a', 'b'],
                                'clustering': [],
                                'trigger': None,
                                'output_mode': '',
                                'query_name': '',
                                'sink': None,
                                'foreach_writer': None,
                                'foreach_batch': None},
                               ['StreamingQuery', 'q', 'r', 'n']],
 'partitionBy([a, b]).start()': ['sent',
                                 {'kind': 'write_stream_operation_start',
                                  'input': True,
                                  'format': '',
                                  'options': [],
                                  'partitioning': ['a', 'b'],
                                  'clustering': [],
                                  'trigger': None,
                                  'output_mode': '',
                                  'query_name': '',
                                  'sink': None,
                                  'foreach_writer': None,
                                  'foreach_batch': None},
                                 ['StreamingQuery', 'q', 'r', 'n']],
 'partitionBy((a,)).start()': ['sent',
                               {'kind': 'write_stream_operation_start',
                                'input': True,
                                'format': '',
                                'options': [],
                                'partitioning': ['a'],
                                'clustering': [],
                                'trigger': None,
                                'output_mode': '',
                                'query_name': '',
                                'sink': None,
                                'foreach_writer': None,
                                'foreach_batch': None},
                               ['StreamingQuery', 'q', 'r', 'n']],
 'partitionBy(a).partitionBy(b).start()': ['sent',
                                           {'kind': 'write_stream_operation_start',
                                            'input': True,
                                            'format': '',
                                            'options': [],
                                            'partitioning': ['b'],
                                            'clustering': [],
                                            'trigger': None,
                                            'output_mode': '',
                                            'query_name': '',
                                            'sink': None,
                                            'foreach_writer': None,
                                            'foreach_batch': None},
                                           ['StreamingQuery', 'q', 'r', 'n']],
 'partitionBy().start()': ['sent',
                           {'kind': 'write_stream_operation_start',
                            'input': True,
                            'format': '',
                            'options': [],
                            'partitioning': [],
                            'clustering': [],
                            'trigger': None,
                            'output_mode': '',
                            'query_name': '',
                            'sink': None,
                            'foreach_writer': None,
                            'foreach_batch': None},
                           ['StreamingQuery', 'q', 'r', 'n']],
 'partitionBy(5)': ['call', 'TypeError', 'bad argument type for built-in operation'],
 'partitionBy(a, 5)': ['call', 'TypeError', 'bad argument type for built-in operation'],
 'partitionBy(bad)': ['call',
                      'UnicodeEncodeError',
                      "'utf-8' codec can't encode character '\\udc80' in position 1: surrogates "
                      'not allowed'],
 'partitionBy([a], b)': ['call', 'TypeError', 'bad argument type for built-in operation'],
 'clusterBy(a).start()': ['sent',
                          {'kind': 'write_stream_operation_start',
                           'input': True,
                           'format': '',
                           'options': [],
                           'partitioning': [],
                           'clustering': ['a'],
                           'trigger': None,
                           'output_mode': '',
                           'query_name': '',
                           'sink': None,
                           'foreach_writer': None,
                           'foreach_batch': None},
                          ['StreamingQuery', 'q', 'r', 'n']],
 'clusterBy([a, b]).start()': ['sent',
                               {'kind': 'write_stream_operation_start',
                                'input': True,
                                'format': '',
                                'options': [],
                                'partitioning': [],
                                'clustering': ['a', 'b'],
                                'trigger': None,
                                'output_mode': '',
                                'query_name': '',
                                'sink': None,
                                'foreach_writer': None,
                                'foreach_batch': None},
                               ['StreamingQuery', 'q', 'r', 'n']],
 'clusterBy(5)': ['call', 'TypeError', 'bad argument type for built-in operation'],
 'queryName(q).start()': ['sent',
                          {'kind': 'write_stream_operation_start',
                           'input': True,
                           'format': '',
                           'options': [],
                           'partitioning': [],
                           'clustering': [],
                           'trigger': None,
                           'output_mode': '',
                           'query_name': 'q',
                           'sink': None,
                           'foreach_writer': None,
                           'foreach_batch': None},
                          ['StreamingQuery', 'q', 'r', 'n']],
 "queryName('')": ['call',
                   'PySparkValueError',
                   '[VALUE_NOT_NON_EMPTY_STR] Value for `queryName` must be a non-empty string, '
                   "got ''."],
 "queryName('  ')": ['call',
                     'PySparkValueError',
                     '[VALUE_NOT_NON_EMPTY_STR] Value for `queryName` must be a non-empty string, '
                     "got '  '."],
 'queryName(5)': ['call',
                  'PySparkValueError',
                  '[VALUE_NOT_NON_EMPTY_STR] Value for `queryName` must be a non-empty string, got '
                  "'5'."],
 'queryName(None)': ['call',
                     'PySparkValueError',
                     '[VALUE_NOT_NON_EMPTY_STR] Value for `queryName` must be a non-empty string, '
                     "got 'None'."],
 'queryName(bad)': ['call',
                    'UnicodeEncodeError',
                    "'utf-8' codec can't encode character '\\udc80' in position 1: surrogates not "
                    'allowed'],
 'trigger()': ['call',
               'PySparkValueError',
               '[ONLY_ALLOW_SINGLE_TRIGGER] Only a single trigger is allowed.'],
 'trigger(5)': ['call',
                'TypeError',
                'DataStreamWriter.trigger() takes 1 positional argument but 2 were given'],
 'trigger(processingTime=1 second).start()': ['sent',
                                              {'kind': 'write_stream_operation_start',
                                               'input': True,
                                               'format': '',
                                               'options': [],
                                               'partitioning': [],
                                               'clustering': [],
                                               'trigger': ['processing_time_interval', '1 second'],
                                               'output_mode': '',
                                               'query_name': '',
                                               'sink': None,
                                               'foreach_writer': None,
                                               'foreach_batch': None},
                                              ['StreamingQuery', 'q', 'r', 'n']],
 "trigger(processingTime=' 1 second ').start()": ['sent',
                                                  {'kind': 'write_stream_operation_start',
                                                   'input': True,
                                                   'format': '',
                                                   'options': [],
                                                   'partitioning': [],
                                                   'clustering': [],
                                                   'trigger': ['processing_time_interval',
                                                               '1 second'],
                                                   'output_mode': '',
                                                   'query_name': '',
                                                   'sink': None,
                                                   'foreach_writer': None,
                                                   'foreach_batch': None},
                                                  ['StreamingQuery', 'q', 'r', 'n']],
 "trigger(processingTime='')": ['call',
                                'PySparkValueError',
                                '[VALUE_NOT_NON_EMPTY_STR] Value for `processingTime` must be a '
                                "non-empty string, got ''."],
 'trigger(processingTime=5)': ['call',
                               'PySparkValueError',
                               '[VALUE_NOT_NON_EMPTY_STR] Value for `processingTime` must be a '
                               "non-empty string, got '5'."],
 'trigger(processingTime=bad)': ['call',
                                 'UnicodeEncodeError',
                                 "'utf-8' codec can't encode character '\\udc80' in position 1: "
                                 'surrogates not allowed'],
 'trigger(once=True).start()': ['sent',
                                {'kind': 'write_stream_operation_start',
                                 'input': True,
                                 'format': '',
                                 'options': [],
                                 'partitioning': [],
                                 'clustering': [],
                                 'trigger': ['once', True],
                                 'output_mode': '',
                                 'query_name': '',
                                 'sink': None,
                                 'foreach_writer': None,
                                 'foreach_batch': None},
                                ['StreamingQuery', 'q', 'r', 'n']],
 'trigger(once=False)': ['call',
                         'PySparkValueError',
                         '[VALUE_NOT_ALLOWED] Value for `once` has to be amongst the following '
                         'values: [True].'],
 'trigger(once=1)': ['call',
                     'PySparkValueError',
                     '[VALUE_NOT_ALLOWED] Value for `once` has to be amongst the following values: '
                     '[True].'],
 'trigger(availableNow=True).start()': ['sent',
                                        {'kind': 'write_stream_operation_start',
                                         'input': True,
                                         'format': '',
                                         'options': [],
                                         'partitioning': [],
                                         'clustering': [],
                                         'trigger': ['available_now', True],
                                         'output_mode': '',
                                         'query_name': '',
                                         'sink': None,
                                         'foreach_writer': None,
                                         'foreach_batch': None},
                                        ['StreamingQuery', 'q', 'r', 'n']],
 'trigger(availableNow=False)': ['call',
                                 'PySparkValueError',
                                 '[VALUE_NOT_ALLOWED] Value for `availableNow` has to be amongst '
                                 'the following values: [True].'],
 'trigger(continuous=1 second).start()': ['sent',
                                          {'kind': 'write_stream_operation_start',
                                           'input': True,
                                           'format': '',
                                           'options': [],
                                           'partitioning': [],
                                           'clustering': [],
                                           'trigger': ['continuous_checkpoint_interval',
                                                       '1 second'],
                                           'output_mode': '',
                                           'query_name': '',
                                           'sink': None,
                                           'foreach_writer': None,
                                           'foreach_batch': None},
                                          ['StreamingQuery', 'q', 'r', 'n']],
 "trigger(continuous='')": ['call',
                            'PySparkValueError',
                            '[VALUE_NOT_NON_EMPTY_STR] Value for `continuous` must be a non-empty '
                            "string, got ''."],
 'trigger(realTime=5 seconds).start()': ['sent',
                                         {'kind': 'write_stream_operation_start',
                                          'input': True,
                                          'format': '',
                                          'options': [],
                                          'partitioning': [],
                                          'clustering': [],
                                          'trigger': ['real_time_batch_duration', '5 seconds'],
                                          'output_mode': '',
                                          'query_name': '',
                                          'sink': None,
                                          'foreach_writer': None,
                                          'foreach_batch': None},
                                         ['StreamingQuery', 'q', 'r', 'n']],
 'trigger(realTime=5)': ['call',
                         'PySparkValueError',
                         '[VALUE_NOT_NON_EMPTY_STR] Value for `realTime` must be a non-empty '
                         "string, got '5'."],
 'trigger(processingTime=1 s, once=True)': ['call',
                                            'PySparkValueError',
                                            '[ONLY_ALLOW_SINGLE_TRIGGER] Only a single trigger is '
                                            'allowed.'],
 'trigger(once=True).trigger(availableNow=True).start()': ['sent',
                                                           {'kind': 'write_stream_operation_start',
                                                            'input': True,
                                                            'format': '',
                                                            'options': [],
                                                            'partitioning': [],
                                                            'clustering': [],
                                                            'trigger': ['available_now', True],
                                                            'output_mode': '',
                                                            'query_name': '',
                                                            'sink': None,
                                                            'foreach_writer': None,
                                                            'foreach_batch': None},
                                                           ['StreamingQuery', 'q', 'r', 'n']],
 'foreach(f).start()': ['sent',
                        {'kind': 'write_stream_operation_start',
                         'input': True,
                         'format': '',
                         'options': [],
                         'partitioning': [],
                         'clustering': [],
                         'trigger': None,
                         'output_mode': '',
                         'query_name': '',
                         'sink': None,
                         'foreach_writer': ['python_function', True, True],
                         'foreach_batch': None},
                        ['StreamingQuery', 'q', 'r', 'n']],
 'foreach(process).start()': ['sent',
                              {'kind': 'write_stream_operation_start',
                               'input': True,
                               'format': '',
                               'options': [],
                               'partitioning': [],
                               'clustering': [],
                               'trigger': None,
                               'output_mode': '',
                               'query_name': '',
                               'sink': None,
                               'foreach_writer': ['python_function', True, True],
                               'foreach_batch': None},
                              ['StreamingQuery', 'q', 'r', 'n']],
 'foreach(open, process, close).start()': ['sent',
                                           {'kind': 'write_stream_operation_start',
                                            'input': True,
                                            'format': '',
                                            'options': [],
                                            'partitioning': [],
                                            'clustering': [],
                                            'trigger': None,
                                            'output_mode': '',
                                            'query_name': '',
                                            'sink': None,
                                            'foreach_writer': ['python_function', True, True],
                                            'foreach_batch': None},
                                           ['StreamingQuery', 'q', 'r', 'n']],
 'foreach(no process)': ['call',
                         'PySparkAttributeError',
                         '[ATTRIBUTE_NOT_CALLABLE] Attribute `process` in provided object `f` is '
                         'not callable.'],
 'foreach(process not callable)': ['call',
                                   'PySparkAttributeError',
                                   '[ATTRIBUTE_NOT_CALLABLE] Attribute `process` in provided '
                                   'object `f` is not callable.'],
 'foreach(5)': ['call',
                'PySparkAttributeError',
                '[ATTRIBUTE_NOT_CALLABLE] Attribute `process` in provided object `f` is not '
                'callable.'],
 'foreach(unpicklable)': ['call',
                          'PySparkPicklingError',
                          '[STREAMING_CONNECT_SERIALIZATION_ERROR] Cannot serialize the function '
                          '`foreach`. If you accessed the Spark session, or a'],
 'foreachBatch(f).start()': ['sent',
                             {'kind': 'write_stream_operation_start',
                              'input': True,
                              'format': '',
                              'options': [],
                              'partitioning': [],
                              'clustering': [],
                              'trigger': None,
                              'output_mode': '',
                              'query_name': '',
                              'sink': None,
                              'foreach_writer': None,
                              'foreach_batch': ['python_function', True, True]},
                             ['StreamingQuery', 'q', 'r', 'n']],
 'foreachBatch(5).start()': ['sent',
                             {'kind': 'write_stream_operation_start',
                              'input': True,
                              'format': '',
                              'options': [],
                              'partitioning': [],
                              'clustering': [],
                              'trigger': None,
                              'output_mode': '',
                              'query_name': '',
                              'sink': None,
                              'foreach_writer': None,
                              'foreach_batch': ['python_function', True, True]},
                             ['StreamingQuery', 'q', 'r', 'n']],
 'foreachBatch(unpicklable)': ['call',
                               'PySparkPicklingError',
                               '[STREAMING_CONNECT_SERIALIZATION_ERROR] Cannot serialize the '
                               'function `foreachBatch`. If you accessed the Spark session,'],
 'foreachBatch(f).foreach(f).start()': ['sent',
                                        {'kind': 'write_stream_operation_start',
                                         'input': True,
                                         'format': '',
                                         'options': [],
                                         'partitioning': [],
                                         'clustering': [],
                                         'trigger': None,
                                         'output_mode': '',
                                         'query_name': '',
                                         'sink': None,
                                         'foreach_writer': ['python_function', True, True],
                                         'foreach_batch': ['python_function', True, True]},
                                        ['StreamingQuery', 'q', 'r', 'n']]}


def _stream_write_outcome(service, call):
    import contextlib
    import io

    before = len(service.execute_requests)
    try:
        # CloudPickleSerializer prints the traceback of a function that does not pickle.
        with contextlib.redirect_stderr(io.StringIO()):
            value = call()
    except BaseException as error:  # noqa: BLE001
        return ["call", type(error).__name__, str(error)[:120]]
    if len(service.execute_requests) == before:
        return ["value", type(value).__name__]
    query = [type(value).__name__, value.id, value.runId, value.name]
    return ["sent", _sent_stream_write(service), query]


@pytest.mark.parametrize("case", list(_STREAM_WRITE_OUTCOMES))
def test_the_stream_writes_send_what_pysparks_writer_sends(objects, case):
    # As PySpark's DataStreamWriter, of whose source the drop-in's is adapted: the errors of its
    # setters and of the command of its start, in the same order, the start commands that it sends,
    # of its format, options, partitioning, clustering, trigger, output mode, query name, path or
    # table and the functions of foreach and foreachBatch, and the query of the start result.
    objects.service.execute_payload = _start_result()
    call = _stream_write_cases(objects.sdf)[case]
    assert _stream_write_outcome(objects.service, call) == _STREAM_WRITE_OUTCOMES[case]


# The merges of the merge writer, as PySpark's MergeIntoWriter and its clauses make them, of PySpark
# 4.3.0: what the call raises, and else the merge command that the server receives of it, of the
# kinds, the conditions and the assignments of its actions.
def _merge_expression(e):
    kind = e.WhichOneof("expr_type")
    if kind == "unresolved_function":
        f = e.unresolved_function
        return [kind, f.function_name, [_merge_expression(a) for a in f.arguments]]
    if kind == "unresolved_attribute":
        return [kind, e.unresolved_attribute.unparsed_identifier]
    if kind == "literal":
        lit = e.literal
        which = lit.WhichOneof("literal_type")
        value = getattr(lit, which) if which else None
        return [kind, which, value if isinstance(value, (int, float, str, bool)) else str(value)]
    if kind == "expression_string":
        return [kind, e.expression_string.expression]
    if kind == "merge_action":
        a = e.merge_action
        return [
            kind,
            int(a.action_type),
            _merge_expression(a.condition) if a.HasField("condition") else None,
            [[_merge_expression(x.key), _merge_expression(x.value)] for x in a.assignments],
        ]
    return [kind]


def _sent_merge(service):
    import pyspark.sql.connect.proto as pb2

    request = pb2.ExecutePlanRequest.FromString(service.execute_requests[-1])
    command = request.plan.command
    kind = command.WhichOneof("command_type")
    if kind != "merge_into_table_command":
        return {"kind": kind}
    m = command.merge_into_table_command
    return {
        "kind": kind,
        "table": m.target_table_name,
        "source": m.HasField("source_table_plan"),
        "condition": _merge_expression(m.merge_condition) if m.HasField("merge_condition") else None,
        "matched": [_merge_expression(a) for a in m.match_actions],
        "not_matched": [_merge_expression(a) for a in m.not_matched_actions],
        "not_matched_by_source": [_merge_expression(a) for a in m.not_matched_by_source_actions],
        "schema_evolution": m.with_schema_evolution,
    }


def _merge_cases(df):
    from pyspark.sql import functions as F

    BAD = _BAD_STR
    cond = F.col("t.id") == F.col("s.id")
    m = lambda: df.mergeInto("t", cond)  # noqa: E731
    return {
        "whenMatched().updateAll().merge()": lambda: m().whenMatched().updateAll().merge(),
        "whenMatched(c).update({v: x}).merge()": lambda: m().whenMatched(F.col("s.v") > 0).update({"v": F.col("s.v")}).merge(),
        "whenMatched().delete().merge()": lambda: m().whenMatched().delete().merge(),
        "whenNotMatched().insertAll().merge()": lambda: m().whenNotMatched().insertAll().merge(),
        "whenNotMatched(c).insert({v: x, id: y}).merge()": lambda: m().whenNotMatched(F.col("s.v") < 9).insert({"v": F.col("s.v"), "id": F.lit(1)}).merge(),
        "whenNotMatchedBySource().updateAll().merge()": lambda: m().whenNotMatchedBySource().updateAll().merge(),
        "whenNotMatchedBySource(c).update({v: 0}).merge()": lambda: m().whenNotMatchedBySource(F.col("t.v") > 1).update({"v": F.lit(0)}).merge(),
        "whenNotMatchedBySource().delete().merge()": lambda: m().whenNotMatchedBySource().delete().merge(),
        "every clause, twice, in order": lambda: m().whenMatched(F.col("s.v") > 0).delete().whenMatched().updateAll().whenNotMatched().insertAll().whenNotMatchedBySource().delete().whenNotMatched(F.lit(True)).insert({"v": F.lit(2)}).merge(),
        "withSchemaEvolution().merge()": lambda: m().withSchemaEvolution().whenMatched().delete().merge(),
        "merge() of no clause": lambda: m().merge(),
        "mergeInto(5, c).merge()": lambda: df.mergeInto(5, cond).merge(),
        "mergeInto(bad, c).merge()": lambda: df.mergeInto(BAD, cond).merge(),
        "mergeInto(b't', c).merge()": lambda: df.mergeInto(b"t", cond).merge(),
        "mergeInto(None, c).merge()": lambda: df.mergeInto(None, cond).merge(),
        "mergeInto(t, 'id = 1').merge()": lambda: df.mergeInto("t", "id = 1").merge(),
        "mergeInto(t, 5)": lambda: df.mergeInto("t", 5),
        "mergeInto(t, None).merge()": lambda: df.mergeInto("t", None).merge(),
        "whenMatched('v > 0').updateAll()": lambda: m().whenMatched("v > 0").updateAll(),
        "whenMatched(5)": lambda: m().whenMatched(5),
        "whenMatched(5).delete()": lambda: m().whenMatched(5).delete(),
        "whenNotMatched(None).insertAll().merge()": lambda: m().whenNotMatched(None).insertAll().merge(),
        "update({v: 'x'})": lambda: m().whenMatched().update({"v": "x"}),
        "update({v: 5})": lambda: m().whenMatched().update({"v": 5}),
        "update({5: x})": lambda: m().whenMatched().update({5: F.col("s.v")}),
        "update({bad: x})": lambda: m().whenMatched().update({BAD: F.col("s.v")}),
        "update({'a b': x}).merge()": lambda: m().whenMatched().update({"a b": F.col("s.v")}).merge(),
        "update({}).merge()": lambda: m().whenMatched().update({}).merge(),
        "update(None).merge()": lambda: m().whenMatched().update(None).merge(),
        "update([(v, x)])": lambda: m().whenMatched().update([("v", F.col("s.v"))]),
        "insert({v: lit(bad)})": lambda: m().whenNotMatched().insert({"v": F.lit(BAD)}),
        "insert(5)": lambda: m().whenNotMatched().insert(5),
        "insert({v: x}).insert again": lambda: m().whenNotMatched().insert({"v": F.col("s.v")}).whenNotMatched().insert({"id": F.lit(3)}).merge(),
        "a clause that raised is not added": lambda: (lambda w: (_merge_raises(lambda: w.whenMatched(5).delete()), w.whenMatched().delete().merge())[1])(m()),
        "the clause after a change of its condition": lambda: (lambda w, c: (w.whenMatched(c), w.merge()))(m(), F.col("s.v") > 0)[1],
        "the writer of a clause is the writer": lambda: (lambda w: [w.whenMatched().writer is w, w.whenNotMatched().writer is w, w.whenNotMatchedBySource().writer is w, w.whenMatched().delete() is w])(m()),
        "the condition of a clause is its argument": lambda: (lambda w, c: w.whenMatched(c)._condition is c)(m(), F.col("s.v") > 0),
    }


def _merge_raises(call):
    try:
        call()
    except Exception:  # noqa: BLE001
        return True
    return False


_MERGE_OUTCOMES = {'whenMatched().updateAll().merge()': ['sent',
                                       {'kind': 'merge_into_table_command',
                                        'table': 't',
                                        'source': True,
                                        'condition': ['unresolved_function',
                                                      '==',
                                                      [['unresolved_attribute', 't.id'],
                                                       ['unresolved_attribute', 's.id']]],
                                        'matched': [['merge_action', 5, None, []]],
                                        'not_matched': [],
                                        'not_matched_by_source': [],
                                        'schema_evolution': False},
                                       'NoneType'],
 'whenMatched(c).update({v: x}).merge()': ['sent',
                                           {'kind': 'merge_into_table_command',
                                            'table': 't',
                                            'source': True,
                                            'condition': ['unresolved_function',
                                                          '==',
                                                          [['unresolved_attribute', 't.id'],
                                                           ['unresolved_attribute', 's.id']]],
                                            'matched': [['merge_action',
                                                         4,
                                                         ['unresolved_function',
                                                          '>',
                                                          [['unresolved_attribute', 's.v'],
                                                           ['literal', 'integer', 0]]],
                                                         [[['expression_string', 'v'],
                                                           ['unresolved_attribute', 's.v']]]]],
                                            'not_matched': [],
                                            'not_matched_by_source': [],
                                            'schema_evolution': False},
                                           'NoneType'],
 'whenMatched().delete().merge()': ['sent',
                                    {'kind': 'merge_into_table_command',
                                     'table': 't',
                                     'source': True,
                                     'condition': ['unresolved_function',
                                                   '==',
                                                   [['unresolved_attribute', 't.id'],
                                                    ['unresolved_attribute', 's.id']]],
                                     'matched': [['merge_action', 1, None, []]],
                                     'not_matched': [],
                                     'not_matched_by_source': [],
                                     'schema_evolution': False},
                                    'NoneType'],
 'whenNotMatched().insertAll().merge()': ['sent',
                                          {'kind': 'merge_into_table_command',
                                           'table': 't',
                                           'source': True,
                                           'condition': ['unresolved_function',
                                                         '==',
                                                         [['unresolved_attribute', 't.id'],
                                                          ['unresolved_attribute', 's.id']]],
                                           'matched': [],
                                           'not_matched': [['merge_action', 3, None, []]],
                                           'not_matched_by_source': [],
                                           'schema_evolution': False},
                                          'NoneType'],
 'whenNotMatched(c).insert({v: x, id: y}).merge()': ['sent',
                                                     {'kind': 'merge_into_table_command',
                                                      'table': 't',
                                                      'source': True,
                                                      'condition': ['unresolved_function',
                                                                    '==',
                                                                    [['unresolved_attribute',
                                                                      't.id'],
                                                                     ['unresolved_attribute',
                                                                      's.id']]],
                                                      'matched': [],
                                                      'not_matched': [['merge_action',
                                                                       2,
                                                                       ['unresolved_function',
                                                                        '<',
                                                                        [['unresolved_attribute',
                                                                          's.v'],
                                                                         ['literal',
                                                                          'integer',
                                                                          9]]],
                                                                       [[['expression_string', 'v'],
                                                                         ['unresolved_attribute',
                                                                          's.v']],
                                                                        [['expression_string',
                                                                          'id'],
                                                                         ['literal',
                                                                          'integer',
                                                                          1]]]]],
                                                      'not_matched_by_source': [],
                                                      'schema_evolution': False},
                                                     'NoneType'],
 'whenNotMatchedBySource().updateAll().merge()': ['sent',
                                                  {'kind': 'merge_into_table_command',
                                                   'table': 't',
                                                   'source': True,
                                                   'condition': ['unresolved_function',
                                                                 '==',
                                                                 [['unresolved_attribute', 't.id'],
                                                                  ['unresolved_attribute',
                                                                   's.id']]],
                                                   'matched': [],
                                                   'not_matched': [],
                                                   'not_matched_by_source': [['merge_action',
                                                                              5,
                                                                              None,
                                                                              []]],
                                                   'schema_evolution': False},
                                                  'NoneType'],
 'whenNotMatchedBySource(c).update({v: 0}).merge()': ['sent',
                                                      {'kind': 'merge_into_table_command',
                                                       'table': 't',
                                                       'source': True,
                                                       'condition': ['unresolved_function',
                                                                     '==',
                                                                     [['unresolved_attribute',
                                                                       't.id'],
                                                                      ['unresolved_attribute',
                                                                       's.id']]],
                                                       'matched': [],
                                                       'not_matched': [],
                                                       'not_matched_by_source': [['merge_action',
                                                                                  4,
                                                                                  ['unresolved_function',
                                                                                   '>',
                                                                                   [['unresolved_attribute',
                                                                                     't.v'],
                                                                                    ['literal',
                                                                                     'integer',
                                                                                     1]]],
                                                                                  [[['expression_string',
                                                                                     'v'],
                                                                                    ['literal',
                                                                                     'integer',
                                                                                     0]]]]],
                                                       'schema_evolution': False},
                                                      'NoneType'],
 'whenNotMatchedBySource().delete().merge()': ['sent',
                                               {'kind': 'merge_into_table_command',
                                                'table': 't',
                                                'source': True,
                                                'condition': ['unresolved_function',
                                                              '==',
                                                              [['unresolved_attribute', 't.id'],
                                                               ['unresolved_attribute', 's.id']]],
                                                'matched': [],
                                                'not_matched': [],
                                                'not_matched_by_source': [['merge_action',
                                                                           1,
                                                                           None,
                                                                           []]],
                                                'schema_evolution': False},
                                               'NoneType'],
 'every clause, twice, in order': ['sent',
                                   {'kind': 'merge_into_table_command',
                                    'table': 't',
                                    'source': True,
                                    'condition': ['unresolved_function',
                                                  '==',
                                                  [['unresolved_attribute', 't.id'],
                                                   ['unresolved_attribute', 's.id']]],
                                    'matched': [['merge_action',
                                                 1,
                                                 ['unresolved_function',
                                                  '>',
                                                  [['unresolved_attribute', 's.v'],
                                                   ['literal', 'integer', 0]]],
                                                 []],
                                                ['merge_action', 5, None, []]],
                                    'not_matched': [['merge_action', 3, None, []],
                                                    ['merge_action',
                                                     2,
                                                     ['literal', 'boolean', True],
                                                     [[['expression_string', 'v'],
                                                       ['literal', 'integer', 2]]]]],
                                    'not_matched_by_source': [['merge_action', 1, None, []]],
                                    'schema_evolution': False},
                                   'NoneType'],
 'withSchemaEvolution().merge()': ['sent',
                                   {'kind': 'merge_into_table_command',
                                    'table': 't',
                                    'source': True,
                                    'condition': ['unresolved_function',
                                                  '==',
                                                  [['unresolved_attribute', 't.id'],
                                                   ['unresolved_attribute', 's.id']]],
                                    'matched': [['merge_action', 1, None, []]],
                                    'not_matched': [],
                                    'not_matched_by_source': [],
                                    'schema_evolution': True},
                                   'NoneType'],
 'merge() of no clause': ['sent',
                          {'kind': 'merge_into_table_command',
                           'table': 't',
                           'source': True,
                           'condition': ['unresolved_function',
                                         '==',
                                         [['unresolved_attribute', 't.id'],
                                          ['unresolved_attribute', 's.id']]],
                           'matched': [],
                           'not_matched': [],
                           'not_matched_by_source': [],
                           'schema_evolution': False},
                          'NoneType'],
 'mergeInto(5, c).merge()': ['call', 'TypeError', 'bad argument type for built-in operation'],
 'mergeInto(bad, c).merge()': ['call',
                               'UnicodeEncodeError',
                               "'utf-8' codec can't encode character '\\udc80' in position 1: "
                               'surrogates not allowed'],
 "mergeInto(b't', c).merge()": ['sent',
                                {'kind': 'merge_into_table_command',
                                 'table': 't',
                                 'source': True,
                                 'condition': ['unresolved_function',
                                               '==',
                                               [['unresolved_attribute', 't.id'],
                                                ['unresolved_attribute', 's.id']]],
                                 'matched': [],
                                 'not_matched': [],
                                 'not_matched_by_source': [],
                                 'schema_evolution': False},
                                'NoneType'],
 'mergeInto(None, c).merge()': ['sent',
                                {'kind': 'merge_into_table_command',
                                 'table': '',
                                 'source': True,
                                 'condition': ['unresolved_function',
                                               '==',
                                               [['unresolved_attribute', 't.id'],
                                                ['unresolved_attribute', 's.id']]],
                                 'matched': [],
                                 'not_matched': [],
                                 'not_matched_by_source': [],
                                 'schema_evolution': False},
                                'NoneType'],
 "mergeInto(t, 'id = 1').merge()": ['call',
                                    'AttributeError',
                                    "'str' object has no attribute 'to_plan'"],
 'mergeInto(t, 5)': ['value', 'MergeIntoWriter'],
 'mergeInto(t, None).merge()': ['call',
                                'AttributeError',
                                "'NoneType' object has no attribute 'to_plan'"],
 "whenMatched('v > 0').updateAll()": ['call',
                                      'AttributeError',
                                      "'str' object has no attribute 'to_plan'"],
 'whenMatched(5)': ['value', 'WhenMatched'],
 'whenMatched(5).delete()': ['call', 'AttributeError', "'int' object has no attribute 'to_plan'"],
 'whenNotMatched(None).insertAll().merge()': ['sent',
                                              {'kind': 'merge_into_table_command',
                                               'table': 't',
                                               'source': True,
                                               'condition': ['unresolved_function',
                                                             '==',
                                                             [['unresolved_attribute', 't.id'],
                                                              ['unresolved_attribute', 's.id']]],
                                               'matched': [],
                                               'not_matched': [['merge_action', 3, None, []]],
                                               'not_matched_by_source': [],
                                               'schema_evolution': False},
                                              'NoneType'],
 "update({v: 'x'})": ['call', 'AttributeError', "'str' object has no attribute 'to_plan'"],
 'update({v: 5})': ['call', 'AttributeError', "'int' object has no attribute 'to_plan'"],
 'update({5: x})': ['call', 'AssertionError', ''],
 'update({bad: x})': ['call',
                      'UnicodeEncodeError',
                      "'utf-8' codec can't encode character '\\udc80' in position 1: surrogates "
                      'not allowed'],
 "update({'a b': x}).merge()": ['sent',
                                {'kind': 'merge_into_table_command',
                                 'table': 't',
                                 'source': True,
                                 'condition': ['unresolved_function',
                                               '==',
                                               [['unresolved_attribute', 't.id'],
                                                ['unresolved_attribute', 's.id']]],
                                 'matched': [['merge_action',
                                              4,
                                              None,
                                              [[['expression_string', 'a b'],
                                                ['unresolved_attribute', 's.v']]]]],
                                 'not_matched': [],
                                 'not_matched_by_source': [],
                                 'schema_evolution': False},
                                'NoneType'],
 'update({}).merge()': ['sent',
                        {'kind': 'merge_into_table_command',
                         'table': 't',
                         'source': True,
                         'condition': ['unresolved_function',
                                       '==',
                                       [['unresolved_attribute', 't.id'],
                                        ['unresolved_attribute', 's.id']]],
                         'matched': [['merge_action', 4, None, []]],
                         'not_matched': [],
                         'not_matched_by_source': [],
                         'schema_evolution': False},
                        'NoneType'],
 'update(None).merge()': ['sent',
                          {'kind': 'merge_into_table_command',
                           'table': 't',
                           'source': True,
                           'condition': ['unresolved_function',
                                         '==',
                                         [['unresolved_attribute', 't.id'],
                                          ['unresolved_attribute', 's.id']]],
                           'matched': [['merge_action', 4, None, []]],
                           'not_matched': [],
                           'not_matched_by_source': [],
                           'schema_evolution': False},
                          'NoneType'],
 'update([(v, x)])': ['call', 'AttributeError', "'list' object has no attribute 'items'"],
 'insert({v: lit(bad)})': ['call',
                           'UnicodeEncodeError',
                           "'utf-8' codec can't encode character '\\udc80' in position 1: "
                           'surrogates not allowed'],
 'insert(5)': ['call', 'AttributeError', "'int' object has no attribute 'items'"],
 'insert({v: x}).insert again': ['sent',
                                 {'kind': 'merge_into_table_command',
                                  'table': 't',
                                  'source': True,
                                  'condition': ['unresolved_function',
                                                '==',
                                                [['unresolved_attribute', 't.id'],
                                                 ['unresolved_attribute', 's.id']]],
                                  'matched': [],
                                  'not_matched': [['merge_action',
                                                   2,
                                                   None,
                                                   [[['expression_string', 'v'],
                                                     ['unresolved_attribute', 's.v']]]],
                                                  ['merge_action',
                                                   2,
                                                   None,
                                                   [[['expression_string', 'id'],
                                                     ['literal', 'integer', 3]]]]],
                                  'not_matched_by_source': [],
                                  'schema_evolution': False},
                                 'NoneType'],
 'a clause that raised is not added': ['sent',
                                       {'kind': 'merge_into_table_command',
                                        'table': 't',
                                        'source': True,
                                        'condition': ['unresolved_function',
                                                      '==',
                                                      [['unresolved_attribute', 't.id'],
                                                       ['unresolved_attribute', 's.id']]],
                                        'matched': [['merge_action', 1, None, []]],
                                        'not_matched': [],
                                        'not_matched_by_source': [],
                                        'schema_evolution': False},
                                       'NoneType'],
 'the clause after a change of its condition': ['sent',
                                                {'kind': 'merge_into_table_command',
                                                 'table': 't',
                                                 'source': True,
                                                 'condition': ['unresolved_function',
                                                               '==',
                                                               [['unresolved_attribute', 't.id'],
                                                                ['unresolved_attribute', 's.id']]],
                                                 'matched': [],
                                                 'not_matched': [],
                                                 'not_matched_by_source': [],
                                                 'schema_evolution': False},
                                                'NoneType'],
 'the writer of a clause is the writer': ['value', 'list'],
 'the condition of a clause is its argument': ['value', 'bool']}


def _merge_outcome(service, call):
    before = len(service.execute_requests)
    try:
        value = call()
    except BaseException as error:  # noqa: BLE001
        return ["call", type(error).__name__, str(error)[:120]]
    if len(service.execute_requests) == before:
        return ["value", type(value).__name__]
    return ["sent", _sent_merge(service), type(value).__name__]


@pytest.mark.parametrize("case", list(_MERGE_OUTCOMES))
def test_the_merges_send_what_pysparks_merge_writer_sends(objects, case):
    # As PySpark's MergeIntoWriter and its clauses, of whose source the drop-in's are adapted: the
    # errors of the clauses, which build their actions when they are added, and of the merge, in
    # the same order, and the merge commands that they send, of the target table, the condition and
    # the actions of the clauses, in their order, and the schema evolution.
    call = _merge_cases(objects.df.alias("s"))[case]
    assert _merge_outcome(objects.service, call) == _MERGE_OUTCOMES[case]
