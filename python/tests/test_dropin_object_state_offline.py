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
    "the readers": {
        "vars": [
            "DataFrameReader",
            "DataFrameReader of its setters",
            "DataFrameReader of a struct schema",
            "DataStreamReader",
            "DataStreamReader of its setters",
        ],
        "attributes": [
            "DataFrameReader",
            "DataFrameReader of its setters",
            "DataFrameReader of a struct schema",
            "DataStreamReader",
            "DataStreamReader of its setters",
        ],
        "copies": [
            "DataFrameReader: copy",
            "DataFrameReader: deepcopy",
            "DataFrameReader: cloudpickle",
            "DataFrameReader of its setters: copy",
            "DataFrameReader of its setters: deepcopy",
            "DataFrameReader of its setters: cloudpickle",
            "DataFrameReader of a struct schema: copy",
            "DataFrameReader of a struct schema: deepcopy",
            "DataFrameReader of a struct schema: cloudpickle",
            "DataStreamReader: copy",
            "DataStreamReader: deepcopy",
            "DataStreamReader: cloudpickle",
            "DataStreamReader of its setters: copy",
            "DataStreamReader of its setters: deepcopy",
            "DataStreamReader of its setters: cloudpickle",
        ],
        "construct": [
            "pyspark.sql.connect.readwriter.DataFrameReader",
            "pyspark.sql.connect.streaming.readwriter.DataStreamReader",
        ],
        "uninit": [
            "pyspark.sql.connect.readwriter.DataFrameReader",
            "pyspark.sql.connect.streaming.readwriter.DataStreamReader",
        ],
        "foreign": [
            "pyspark.sql.connect.readwriter.DataFrameReader",
            "pyspark.sql.connect.streaming.readwriter.DataStreamReader",
        ],
        "specific": [
            "test_the_options_of_a_reader_are_the_dict_of_its_state",
        ],
    },
    "the writers": {
        "vars": [
            "DataFrameWriter",
            "DataFrameWriter of its setters",
            "DataFrameWriterV2",
            "DataFrameWriterV2 of its setters",
        ],
        "attributes": [
            "DataFrameWriter",
            "DataFrameWriter of its setters",
            "DataFrameWriterV2",
            "DataFrameWriterV2 of its setters",
        ],
        "copies": [
            "DataFrameWriter: copy",
            "DataFrameWriter: deepcopy",
            "DataFrameWriter: cloudpickle",
            "DataFrameWriter of its setters: copy",
            "DataFrameWriter of its setters: deepcopy",
            "DataFrameWriter of its setters: cloudpickle",
            "DataFrameWriterV2: copy",
            "DataFrameWriterV2: deepcopy",
            "DataFrameWriterV2: cloudpickle",
            "DataFrameWriterV2 of its setters: copy",
            "DataFrameWriterV2 of its setters: deepcopy",
            "DataFrameWriterV2 of its setters: cloudpickle",
        ],
        "construct": [
            "pyspark.sql.connect.readwriter.DataFrameWriter",
            "pyspark.sql.connect.readwriter.DataFrameWriterV2",
        ],
        "uninit": [
            "pyspark.sql.connect.readwriter.DataFrameWriter",
            "pyspark.sql.connect.readwriter.DataFrameWriterV2",
        ],
        "foreign": [
            "pyspark.sql.connect.readwriter.DataFrameWriter",
            "pyspark.sql.connect.readwriter.DataFrameWriterV2",
        ],
        "specific": [
            "test_the_write_operation_of_a_writer_is_what_its_save_builds",
        ],
    },
    "the stream writer": {
        "vars": [
            "DataStreamWriter",
            "DataStreamWriter of its setters",
        ],
        "attributes": [
            "DataStreamWriter",
            "DataStreamWriter of its setters",
        ],
        "copies": [
            "DataStreamWriter: copy",
            "DataStreamWriter: deepcopy",
            "DataStreamWriter: cloudpickle",
            "DataStreamWriter of its setters: copy",
            "DataStreamWriter of its setters: deepcopy",
            "DataStreamWriter of its setters: cloudpickle",
        ],
        "construct": [
            "pyspark.sql.connect.streaming.readwriter.DataStreamWriter",
        ],
        "uninit": [
            "pyspark.sql.connect.streaming.readwriter.DataStreamWriter",
        ],
        "foreign": [
            "pyspark.sql.connect.streaming.readwriter.DataStreamWriter",
        ],
        "specific": [
            "test_the_message_of_a_stream_writer_is_the_message_of_its_operation",
        ],
    },
    "the merge writer and its clauses": {
        "vars": [
            "MergeIntoWriter",
            "MergeIntoWriter of its clauses",
            "WhenMatched",
            "WhenNotMatched of a condition",
            "WhenNotMatchedBySource",
        ],
        "attributes": [
            "MergeIntoWriter",
            "MergeIntoWriter of its clauses",
            "WhenMatched",
            "WhenNotMatched of a condition",
            "WhenNotMatchedBySource",
        ],
        "copies": [
            "MergeIntoWriter: copy",
            "MergeIntoWriter of its clauses: copy",
            "WhenMatched: copy",
            "WhenNotMatched of a condition: copy",
            "WhenNotMatchedBySource: copy",
        ],
        "construct": [
            "pyspark.sql.connect.merge.MergeIntoWriter",
            "pyspark.sql.connect.merge.MergeIntoWriter.WhenMatched",
            "pyspark.sql.connect.merge.MergeIntoWriter.WhenNotMatched",
            "pyspark.sql.connect.merge.MergeIntoWriter.WhenNotMatchedBySource",
        ],
        "uninit": [
            "pyspark.sql.connect.merge.MergeIntoWriter",
            "pyspark.sql.connect.merge.MergeIntoWriter.WhenMatched",
            "pyspark.sql.connect.merge.MergeIntoWriter.WhenNotMatched",
            "pyspark.sql.connect.merge.MergeIntoWriter.WhenNotMatchedBySource",
        ],
        "foreign": [
            "pyspark.sql.connect.merge.MergeIntoWriter",
            "pyspark.sql.connect.merge.MergeIntoWriter.WhenMatched",
            "pyspark.sql.connect.merge.MergeIntoWriter.WhenNotMatched",
            "pyspark.sql.connect.merge.MergeIntoWriter.WhenNotMatchedBySource",
        ],
    },
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
        ],
        "specific": [
            "test_the_private_state_of_a_dataframe_is_not_a_column",
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
