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
from _mock_connect_server import mock_connect_session
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
    "the grouped data, the cogrouped data, na and stat": {
        "vars": [
            "GroupedData of groupBy",
            "GroupedData of cube",
            "GroupedData of rollup",
            "GroupedData of pivot",
            "GroupedData of groupingSets",
            "PandasCogroupedOps",
            "DataFrameNaFunctions",
            "DataFrameStatFunctions",
        ],
        "attributes": [
            "GroupedData of groupBy",
            "GroupedData of cube",
            "GroupedData of rollup",
            "GroupedData of pivot",
            "GroupedData of groupingSets",
            "PandasCogroupedOps",
            "DataFrameNaFunctions",
            "DataFrameStatFunctions",
        ],
        "copies": [
            "GroupedData of groupBy: copy",
            "GroupedData of groupBy: deepcopy",
            "GroupedData of groupBy: cloudpickle",
            "GroupedData of cube: copy",
            "GroupedData of cube: deepcopy",
            "GroupedData of cube: cloudpickle",
            "GroupedData of rollup: copy",
            "GroupedData of rollup: deepcopy",
            "GroupedData of rollup: cloudpickle",
            "GroupedData of pivot: copy",
            "GroupedData of pivot: deepcopy",
            "GroupedData of pivot: cloudpickle",
            "GroupedData of groupingSets: copy",
            "GroupedData of groupingSets: deepcopy",
            "GroupedData of groupingSets: cloudpickle",
            "PandasCogroupedOps: copy",
            "PandasCogroupedOps: deepcopy",
            "PandasCogroupedOps: cloudpickle",
            "DataFrameNaFunctions: copy",
            "DataFrameNaFunctions: deepcopy",
            "DataFrameNaFunctions: cloudpickle",
            "DataFrameStatFunctions: copy",
            "DataFrameStatFunctions: deepcopy",
            "DataFrameStatFunctions: cloudpickle",
        ],
        "construct": [
            "pyspark.sql.connect.group.GroupedData",
            "pyspark.sql.connect.group.PandasCogroupedOps",
            "pyspark.sql.connect.dataframe.DataFrameNaFunctions",
            "pyspark.sql.connect.dataframe.DataFrameStatFunctions",
        ],
        "uninit": [
            "pyspark.sql.connect.group.GroupedData",
            "pyspark.sql.connect.group.PandasCogroupedOps",
            "pyspark.sql.connect.dataframe.DataFrameNaFunctions",
            "pyspark.sql.connect.dataframe.DataFrameStatFunctions",
        ],
        "foreign": [
            "pyspark.sql.connect.group.GroupedData",
            "pyspark.sql.connect.group.PandasCogroupedOps",
            "pyspark.sql.connect.dataframe.DataFrameNaFunctions",
            "pyspark.sql.connect.dataframe.DataFrameStatFunctions",
        ],
    },
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
    "the session and its builder": {
        "vars": [
            "SparkSession",
            "SparkSession of streams and catalog",
            "SparkSession.Builder",
            "SparkSession.Builder of its setters",
        ],
        "attributes": [
            "SparkSession",
            "SparkSession of streams and catalog",
            "SparkSession.Builder",
            "SparkSession.Builder of its setters",
        ],
        "copies": [
            "SparkSession.Builder: copy",
            "SparkSession.Builder: deepcopy",
            "SparkSession.Builder: pickle",
            "SparkSession.Builder: cloudpickle",
            "SparkSession.Builder of its setters: copy",
            "SparkSession.Builder of its setters: deepcopy",
            "SparkSession.Builder of its setters: pickle",
            "SparkSession.Builder of its setters: cloudpickle",
        ],
        "construct": [
            "pyspark.sql.connect.session.SparkSession",
            "pyspark.sql.connect.session.SparkSession.Builder",
        ],
        "uninit": [
            "pyspark.sql.connect.session.SparkSession",
            "pyspark.sql.connect.session.SparkSession.Builder",
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
