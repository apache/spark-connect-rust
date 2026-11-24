# SPDX-License-Identifier: Apache-2.0
"""The classes of the objects of a session as official PySpark 4.3.0's, without a server: of the
modules of PySpark Connect, `pyspark.sql.connect.*`, which `repr`, `type(x).__module__` and pickle
show. The expected names are official PySpark's, of the same objects."""

import pickle

import pytest
from _mock_connect_server import message_field, mock_connect_server, mock_connect_session
from pyspark.sql import Window
from pyspark.sql import functions as F

# The module and the qualified name of the class of each object, in official PySpark.
EXPECTED = {
    "spark": "pyspark.sql.connect.session.SparkSession",
    "conf": "pyspark.sql.connect.conf.RuntimeConf",
    "catalog": "pyspark.sql.connect.catalog.Catalog",
    "read": "pyspark.sql.connect.readwriter.DataFrameReader",
    "readStream": "pyspark.sql.connect.streaming.readwriter.DataStreamReader",
    "streams": "pyspark.sql.connect.streaming.query.StreamingQueryManager",
    "udf": "pyspark.sql.connect.udf.UDFRegistration",
    "udtf": "pyspark.sql.connect.udtf.UDTFRegistration",
    "tvf": "pyspark.sql.connect.tvf.TableValuedFunction",
    "dataSource": "pyspark.sql.connect.datasource.DataSourceRegistration",
    "client": "pyspark.sql.connect.client.core.SparkConnectClient",
    "df": "pyspark.sql.connect.dataframe.DataFrame",
    "write": "pyspark.sql.connect.readwriter.DataFrameWriter",
    "writeTo": "pyspark.sql.connect.readwriter.DataFrameWriterV2",
    "writeStream": "pyspark.sql.connect.streaming.readwriter.DataStreamWriter",
    "mergeInto": "pyspark.sql.connect.merge.MergeIntoWriter",
    "whenMatched": "pyspark.sql.connect.merge.MergeIntoWriter.WhenMatched",
    "whenNotMatched": "pyspark.sql.connect.merge.MergeIntoWriter.WhenNotMatched",
    "whenNotMatchedBySource": "pyspark.sql.connect.merge.MergeIntoWriter.WhenNotMatchedBySource",
    "na": "pyspark.sql.connect.dataframe.DataFrameNaFunctions",
    "stat": "pyspark.sql.connect.dataframe.DataFrameStatFunctions",
    "groupBy": "pyspark.sql.connect.group.GroupedData",
    "cogroup": "pyspark.sql.connect.group.PandasCogroupedOps",
    "column": "pyspark.sql.connect.column.Column",
    "window": "pyspark.sql.connect.window.WindowSpec",
    "asTable": "pyspark.sql.connect.table_arg.TableArg",
}


def _objects(spark):
    df = spark.range(1)
    merge = df.mergeInto("t", F.lit(True))
    return {
        "spark": spark,
        "conf": spark.conf,
        "catalog": spark.catalog,
        "read": spark.read,
        "readStream": spark.readStream,
        "streams": spark.streams,
        "udf": spark.udf,
        "udtf": spark.udtf,
        "tvf": spark.tvf,
        "dataSource": spark.dataSource,
        "client": spark.client,
        "df": df,
        "write": df.write,
        "writeTo": df.writeTo("t"),
        "writeStream": df.writeStream,
        "mergeInto": merge,
        "whenMatched": merge.whenMatched(),
        "whenNotMatched": merge.whenNotMatched(),
        "whenNotMatchedBySource": merge.whenNotMatchedBySource(),
        "na": df.na,
        "stat": df.stat,
        "groupBy": df.groupBy(),
        "cogroup": df.groupBy().cogroup(df.groupBy()),
        "column": F.col("id"),
        "window": Window.partitionBy("id"),
        "asTable": df.asTable(),
    }


@pytest.fixture(scope="module")
def objects():
    with mock_connect_session() as (spark, _service):
        yield _objects(spark)


@pytest.mark.parametrize("name", list(EXPECTED))
def test_the_class_of_an_object_of_a_session_is_pysparks(objects, name):
    cls = type(objects[name])
    assert f"{cls.__module__}.{cls.__qualname__}" == EXPECTED[name]
    # Pickled by reference to that name, which names the same class.
    assert pickle.loads(pickle.dumps(cls)) is cls


# The code of a program that makes `obj`, of a session of the mock server of `remote` (`SESSION`)
# or of none, and the class of `obj`: each runs in an interpreter of its own, which imports nothing
# else, as a new program that makes such an object for a UDF or a `foreachBatch` function does.
SESSION = "spark = SparkSession.builder.remote(remote).getOrCreate()\ndf = spark.range(1)\n"
MERGE = SESSION + "merge = df.mergeInto('t', F.lit(True))\n"
FRESH_OBJECTS = {
    "spark": (SESSION + "obj = spark", EXPECTED["spark"]),
    "conf": (SESSION + "obj = spark.conf", EXPECTED["conf"]),
    "catalog": (SESSION + "obj = spark.catalog", EXPECTED["catalog"]),
    "read": (SESSION + "obj = spark.read", EXPECTED["read"]),
    "readStream": (SESSION + "obj = spark.readStream", EXPECTED["readStream"]),
    "streams": (SESSION + "obj = spark.streams", EXPECTED["streams"]),
    "tvf": (SESSION + "obj = spark.tvf", EXPECTED["tvf"]),
    "dataSource": (SESSION + "obj = spark.dataSource", EXPECTED["dataSource"]),
    "df": (SESSION + "obj = df", EXPECTED["df"]),
    "write": (SESSION + "obj = df.write", EXPECTED["write"]),
    "writeTo": (SESSION + "obj = df.writeTo('t')", EXPECTED["writeTo"]),
    "writeStream": (SESSION + "obj = df.writeStream", EXPECTED["writeStream"]),
    "mergeInto": (MERGE + "obj = merge", EXPECTED["mergeInto"]),
    "whenMatched": (MERGE + "obj = merge.whenMatched()", EXPECTED["whenMatched"]),
    "whenNotMatched": (MERGE + "obj = merge.whenNotMatched()", EXPECTED["whenNotMatched"]),
    "whenNotMatchedBySource": (
        MERGE + "obj = merge.whenNotMatchedBySource()",
        EXPECTED["whenNotMatchedBySource"],
    ),
    "na": (SESSION + "obj = df.na", EXPECTED["na"]),
    "stat": (SESSION + "obj = df.stat", EXPECTED["stat"]),
    "groupBy": (SESSION + "obj = df.groupBy('id')", EXPECTED["groupBy"]),
    "pivot": (SESSION + "obj = df.groupBy('id').pivot('id', [1])", EXPECTED["groupBy"]),
    "cogroup": (SESSION + "obj = df.groupBy('id').cogroup(df.groupBy('id'))", EXPECTED["cogroup"]),
    "column": (SESSION + "obj = df.id", EXPECTED["column"]),
    "asTable": (SESSION + "obj = df.asTable()", EXPECTED["asTable"]),
    "streamingQuery": (
        SESSION + "obj = spark.readStream.format('rate').load().writeStream.start()",
        "pyspark.sql.connect.streaming.query.StreamingQuery",
    ),
    "column of no session": ("obj = F.col('id')", EXPECTED["column"]),
    "window of no session": ("obj = Window.partitionBy('id')", EXPECTED["window"]),
    "observation of no session": (
        "obj = Observation('o')",
        "pyspark.sql.connect.observation.Observation",
    ),
}

# The interpreter of an object of FRESH_OBJECTS: it prints the class of `obj`, whether cloudpickle
# pickles that class by reference, and the classes that cloudpickle pickles by value, as dynamic
# classes, in the pickle of `obj`. A pickle of the locks and thread-locals of the client of a
# session fails, as of PySpark's objects (`spark.conf`, `spark.streams`, a MergeIntoWriter).
FRESH_INTERPRETER = """
import json
import sys

from pyspark.sql import Observation, SparkSession, Window
from pyspark.sql import functions as F

remote = sys.argv[1]
exec(sys.argv[2])
from pyspark import cloudpickle
from pyspark.cloudpickle import cloudpickle as internals

by_value = []
of_dynamic_class = internals._dynamic_class_reduce


def recorded(cls):
    by_value.append(f"{cls.__module__}.{cls.__qualname__}")
    return of_dynamic_class(cls)


internals._dynamic_class_reduce = recorded
cls = type(obj)
by_reference = internals._should_pickle_by_reference(cls)
try:
    cloudpickle.dumps(obj)
except TypeError:
    pass
print(json.dumps([f"{cls.__module__}.{cls.__qualname__}", by_reference, by_value]))
"""


@pytest.fixture(scope="module")
def mock_server():
    pytest.importorskip("grpc")
    with mock_connect_server() as service:
        yield service


def test_the_classes_of_the_objects_of_a_new_interpreter_pickle_by_reference_as_pysparks(
    mock_server, tmp_path
):
    # As PySpark's, whose classes are those of their modules, which an object of one imports: in
    # an interpreter that made only that object, cloudpickle pickles its class by reference, which
    # it does only of a class of a module of `sys.modules`, and no class of the pickle of the object
    # by value. The native classes name PySpark Connect's modules, which the session, its DataFrame
    # and its properties import as PySpark's do, so that a function of a UDF or of `foreachBatch`
    # that holds such an object pickles as PySpark's, which a Python worker of the server
    # unpickles, where the dynamic class of a native class has none of its members (`AttributeError:
    # type object 'GroupedData' has no attribute 'agg'`).
    import json
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor

    script = tmp_path / "fresh_object.py"
    script.write_text(FRESH_INTERPRETER)
    objects = dict(FRESH_OBJECTS)
    # And a Column of a plain pickle of this process, which needs no session to unpickle, as
    # PySpark's unpickle imports the module of its class.
    pickled = pickle.dumps(F.col("id") + 1).hex()
    objects["column unpickled of no session"] = (
        f"import pickle\nobj = pickle.loads(bytes.fromhex({pickled!r}))",
        EXPECTED["column"],
    )
    service = mock_server
    service.reset()
    # The result of the start of a query: WriteStreamOperationStartResult (8) of a query_id (1) of
    # id 1 and run_id 2, of no started event.
    service.execute_payload = message_field(
        8, message_field(1, message_field(1, "q") + message_field(2, "r"))
    )

    def run(code):
        result = subprocess.run(
            [sys.executable, str(script), service.remote, code],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout.strip().splitlines()[-1])

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = dict(zip(objects, pool.map(run, [code for code, _ in objects.values()])))
    finally:
        service.reset()
    assert results == {name: [cls, True, []] for name, (_, cls) in objects.items()}


def test_the_repr_of_a_session_names_its_connect_class(objects):
    assert repr(objects["spark"]).startswith("<pyspark.sql.connect.session.SparkSession object at ")


def test_the_data_source_registration_is_pyspark_connects(objects):
    # As PySpark's: spark.dataSource is of PySpark Connect's class, which its module names and
    # which takes the session, and not of pyspark.sql.datasource's, PySpark's classic class.
    from pyspark.sql import datasource as classic
    from pyspark.sql.connect import datasource

    registration = objects["dataSource"]
    assert type(registration) is datasource.DataSourceRegistration
    assert not isinstance(registration, classic.DataSourceRegistration)
    assert classic.DataSourceRegistration.__module__ == "pyspark.sql.datasource"
    made = datasource.DataSourceRegistration(sparkSession=objects["spark"])
    assert type(made) is datasource.DataSourceRegistration
    assert made.sparkSession is objects["spark"]


def test_the_classic_conf_tvf_merge_and_observation_classes_are_pysparks(objects):
    # As PySpark's pyspark.sql.conf, tvf, merge and observation: their classes are PySpark's
    # classic ones, of which the objects of a session are not instances, and Observation(...)
    # makes an Observation of PySpark Connect's class, which is not an instance of the classic one.
    import pyspark.sql.conf
    import pyspark.sql.connect.observation
    import pyspark.sql.merge
    import pyspark.sql.tvf
    from pyspark.sql import MergeIntoWriter, Observation

    for name, classic, expected in [
        ("conf", pyspark.sql.conf.RuntimeConfig, "pyspark.sql.conf.RuntimeConfig"),
        ("tvf", pyspark.sql.tvf.TableValuedFunction, "pyspark.sql.tvf.TableValuedFunction"),
        ("mergeInto", pyspark.sql.merge.MergeIntoWriter, "pyspark.sql.merge.MergeIntoWriter"),
        (
            "whenMatched",
            pyspark.sql.merge.MergeIntoWriter.WhenMatched,
            "pyspark.sql.merge.MergeIntoWriter.WhenMatched",
        ),
    ]:
        assert f"{classic.__module__}.{classic.__qualname__}" == expected
        assert not isinstance(objects[name], classic), name
        assert pickle.loads(pickle.dumps(classic)) is classic
    assert MergeIntoWriter is pyspark.sql.merge.MergeIntoWriter
    observation = Observation("o")
    assert type(observation) is pyspark.sql.connect.observation.Observation
    assert not isinstance(observation, Observation)
    assert Observation.__module__ == "pyspark.sql.observation"


# The classic class of each object of a session, of its module of PySpark's, which pyspark.sql or
# pyspark.sql.streaming names too, as PySpark's do.
_STREAMING = "pyspark.sql.streaming"
CLASSIC = {
    "catalog": ("pyspark.sql.catalog", "Catalog", "pyspark.sql"),
    "read": ("pyspark.sql.readwriter", "DataFrameReader", "pyspark.sql"),
    "write": ("pyspark.sql.readwriter", "DataFrameWriter", "pyspark.sql"),
    "writeTo": ("pyspark.sql.readwriter", "DataFrameWriterV2", "pyspark.sql"),
    "readStream": (f"{_STREAMING}.readwriter", "DataStreamReader", _STREAMING),
    "writeStream": (f"{_STREAMING}.readwriter", "DataStreamWriter", _STREAMING),
    "streams": (f"{_STREAMING}.query", "StreamingQueryManager", _STREAMING),
    "udf": ("pyspark.sql.udf", "UDFRegistration", "pyspark.sql"),
    "udtf": ("pyspark.sql.udtf", "UDTFRegistration", "pyspark.sql"),
    "groupBy": ("pyspark.sql.group", "GroupedData", "pyspark.sql"),
    "cogroup": ("pyspark.sql.pandas.group_ops", "PandasCogroupedOps", "pyspark.sql"),
}


@pytest.mark.parametrize("name", list(CLASSIC))
def test_the_classic_classes_of_the_objects_of_a_session_are_pysparks(objects, name):
    # As PySpark's: the classic modules are PySpark's, of Apache Spark as they are, whose classes
    # are of the JVM of Spark Classic, and the objects of a session are of PySpark Connect's
    # classes, which are not their subclasses.
    import importlib

    module, qualname, package = CLASSIC[name]
    classic = getattr(importlib.import_module(module), qualname)
    assert f"{classic.__module__}.{classic.__qualname__}" == f"{module}.{qualname}"
    assert getattr(importlib.import_module(package), qualname) is classic
    assert not isinstance(objects[name], classic)
    assert type(objects[name]).__module__.startswith("pyspark.sql.connect.")
    assert pickle.loads(pickle.dumps(classic)) is classic


def test_the_classic_modules_of_the_objects_of_a_session_are_pysparks():
    # Of PySpark's own names of these modules, which the Connect classes do not have: the option
    # helpers of its readers and writers, and the decorators and the mixin of its grouped data.
    from pyspark.sql import group, readwriter
    from pyspark.sql.pandas import group_ops
    from pyspark.sql.streaming import query
    from pyspark.sql.streaming import readwriter as streaming_readwriter

    assert [readwriter.to_str(v) for v in (True, False, None, 1, "a")] == [
        "true",
        "false",
        None,
        "1",
        "a",
    ]
    for cls in (readwriter.DataFrameReader, readwriter.DataFrameWriter):
        assert cls.__mro__ == (cls, readwriter.OptionUtils, object)
    assert streaming_readwriter.DataStreamReader.__mro__[1] is readwriter.OptionUtils
    assert group.GroupedData.__mro__ == (
        group.GroupedData,
        group_ops.PandasGroupedOpsMixin,
        object,
    )
    assert callable(group.dfapi) and callable(group.df_varargs_api)
    assert query.StreamingCheckpointManager.__module__ == "pyspark.sql.streaming.query"


def test_the_udfs_and_udtfs_are_of_pyspark_connects_classes():
    # As PySpark's on Spark Connect: udf, pandas_udf, udtf and arrow_udtf make the UDFs and the
    # UDTFs of PySpark Connect's classes, of pyspark.sql.connect.udf and pyspark.sql.connect.udtf,
    # which are not instances of the classic classes of pyspark.sql.udf and pyspark.sql.udtf,
    # which pyspark.sql.functions names, as PySpark's does.
    import pandas as pd
    import pyarrow as pa
    import pyspark.sql.udf as classic_udf
    import pyspark.sql.udtf as classic_udtf
    from pyspark.sql.connect import udf as connect_udf
    from pyspark.sql.connect import udtf as connect_udtf

    class Handler:
        def eval(self, a):
            yield (a,)

    class ArrowHandler:
        def eval(self, a: pa.RecordBatch):
            yield pa.table({"a": [1]})

    for table_function in (
        F.udtf(Handler, returnType="a int"),
        F.arrow_udtf(ArrowHandler, returnType="a int"),
    ):
        assert type(table_function) is connect_udtf.UserDefinedTableFunction
        assert not isinstance(table_function, classic_udtf.UserDefinedTableFunction)

    def plus_one(s: pd.Series) -> pd.Series:
        return s + 1

    for function in (F.udf(lambda x: x), F.pandas_udf(plus_one, "long")):
        assert type(function._unwrapped) is connect_udf.UserDefinedFunction
        assert not isinstance(function._unwrapped, classic_udf.UserDefinedFunction)

    assert F.UserDefinedFunction is classic_udf.UserDefinedFunction
    assert F.UserDefinedTableFunction is classic_udtf.UserDefinedTableFunction
    for cls, module in [
        (connect_udf.UserDefinedFunction, "pyspark.sql.connect.udf"),
        (connect_udtf.UserDefinedTableFunction, "pyspark.sql.connect.udtf"),
    ]:
        assert cls.__module__ == module
        assert pickle.loads(pickle.dumps(cls)) is cls


# The names of pyspark-client 4.3.0's pyspark.sql.utils, but those of the dunder methods.
SQL_UTILS_NAMES = (
    "AnalysisException Any Callable CapturedException Dict Enum ForeachBatchFunction FuncT "
    "IllegalArgumentException JVM_INT_MAX List NumpyHelper Optional ParseException "
    "PySparkImportError PySparkNotImplementedError PySparkRuntimeError PythonException "
    "QueryExecutionException Sequence SparkUpgradeException StreamingQueryException StringConcat "
    "TYPE_CHECKING TypeVar Union UnknownException _find_spark_home cast dispatch_col_method "
    "dispatch_df_method dispatch_table_arg_method dispatch_window_method enum_to_value "
    "escape_meta_characters functools get_active_spark_context get_lit_sql_str inspect is_remote "
    "is_remote_only is_timestamp_ntz_preferred os overload pyspark_column_op remote_only "
    "require_minimum_plotly_version require_test_compiled to_java_array to_scala_map to_str "
    "try_partitioning_remote_functions try_remote_avro_functions try_remote_functions "
    "try_remote_protobuf_functions try_remote_session_classmethod"
).split()


def test_sql_utils_is_pysparks_module(tmp_path, monkeypatch):
    # As pyspark-client's pyspark.sql.utils, of its names and its helpers: a remote-only client,
    # whose is_remote() is True, and whose require_test_compiled raises of a SPARK_HOME without
    # the test classes of Spark SQL.
    import pyspark.sql.utils as utils
    from pyspark.errors import PySparkRuntimeError

    assert sorted(n for n in vars(utils) if not n.startswith("__")) == sorted(SQL_UTILS_NAMES)
    assert utils.is_remote_only() is True and utils.is_remote() is True
    assert utils.JVM_INT_MAX == (1 << 31) - 1
    assert utils.escape_meta_characters("a\nb\tc") == "a\\nb\\tc"
    concat = utils.StringConcat()
    concat.append("a")
    concat.append("b")
    assert concat.toString() == "ab"
    monkeypatch.setenv("SPARK_HOME", str(tmp_path))
    assert utils._find_spark_home() == str(tmp_path)
    with pytest.raises(PySparkRuntimeError) as raised:
        utils.require_test_compiled()
    assert raised.value.getCondition() == "TEST_CLASS_NOT_COMPILED"
    # A Connect-only client has no SparkContext, of which PySpark's classic helper imports.
    with pytest.raises(ImportError, match="SparkContext"):
        utils.get_active_spark_context()


def test_the_decorators_of_sql_utils_run_spark_connects(objects):
    # As PySpark's decorators of a remote client: the functions run those of PySpark Connect's
    # modules of their names, and the methods those of the classes of Spark Connect, which an
    # object of another class does not have.
    from pyspark.errors import PySparkNotImplementedError
    from pyspark.sql.connect.column import Column
    from pyspark.sql.connect.dataframe import DataFrame
    from pyspark.sql.connect.table_arg import TableArg
    from pyspark.sql.connect.window import WindowSpec
    from pyspark.sql.utils import (
        dispatch_col_method,
        dispatch_df_method,
        dispatch_table_arg_method,
        dispatch_window_method,
        try_partitioning_remote_functions,
        try_remote_avro_functions,
        try_remote_functions,
        try_remote_protobuf_functions,
    )

    def named(name, decorator):
        def classic(*args, **kwargs):
            raise AssertionError("of Spark Classic")

        classic.__name__ = classic.__qualname__ = name
        return decorator(classic)

    assert type(named("lit", try_remote_functions)(5)) is Column
    assert type(named("years", try_partitioning_remote_functions)(F.col("d"))) is Column
    assert type(named("to_avro", try_remote_avro_functions)(F.col("d"))) is Column
    protobuf = named("to_protobuf", try_remote_protobuf_functions)
    assert type(protobuf(F.col("d"), "M", binaryDescriptorSet=b"")) is Column
    limit = named("limit", dispatch_df_method)
    assert type(limit(objects["df"], 1)) is DataFrame
    alias = named("alias", dispatch_col_method)
    assert type(alias(objects["column"], "x")) is Column
    for method, feature in ((limit, "DataFrame.limit"), (alias, "Column.alias")):
        with pytest.raises(PySparkNotImplementedError) as raised:
            method(object(), 1)
        assert raised.value.getMessageParameters() == {"feature": feature}
    assert type(named("partitionBy", dispatch_window_method)("id")) is WindowSpec
    table_arg = named("partitionBy", dispatch_table_arg_method)(objects["asTable"], "id")
    assert type(table_arg) is TableArg


def test_the_connect_package_has_pysparks_names_and_imports_without_its_checks(tmp_path):
    # As pyspark-client's pyspark.sql.connect, whose __init__ imports the check_dependencies of
    # pyspark.sql.connect.utils: the package has those names, and importing pyspark imports no
    # module of it, nor do the functions of UDFs and UDTFs until they are asked for. Its import
    # does not run the check, which requires pandas, PyArrow and the packages of gRPC, so it
    # imports without them, and of a main module without a file, which `python -c` runs and
    # which PySpark's check takes for its doctests.
    import subprocess
    import sys

    script = tmp_path / "connect_names.py"
    script.write_text(
        """
import importlib.abc, pickle, sys

blocked = sys.argv[1:]


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if any(name == b or name.startswith(b + ".") for b in blocked):
            raise ImportError("blocked: " + name)


sys.meta_path.insert(0, Block())


def connect_modules():
    return sorted(m for m in sys.modules if m.startswith("pyspark.sql.connect"))


import pyspark
from pyspark.sql import SparkSession, functions as F

assert connect_modules() == [], connect_modules()
assert {"udf", "udtf", "arrow_udtf"} <= set(dir(F))
assert connect_modules() == [], connect_modules()
import pyspark.sql.connect as connect

assert sorted(n for n in vars(connect) if not n.startswith("__")) == ["check_dependencies", "utils"]
assert connect_modules() == ["pyspark.sql.connect", "pyspark.sql.connect.utils"], connect_modules()
assert connect.__doc__ == "Spark Connect client"
assert connect.check_dependencies is connect.utils.check_dependencies
from pyspark.sql.functions import udf
import pyspark.sql.functions.builtin as builtin

assert udf is F.udf is builtin.udf and F.udtf is builtin.udtf and F.arrow_udtf is builtin.arrow_udtf
assert udf.__module__ == "pyspark.sql.functions.builtin" and pickle.loads(pickle.dumps(udf)) is udf
print("ok")
"""
    )
    packages = ["pandas", "pyarrow", "numpy", "grpc", "grpc_status", "google.rpc", "zstandard"]
    for blocked in ([], packages):
        result = subprocess.run(
            [sys.executable, str(script), *blocked], capture_output=True, text=True, timeout=300
        )
        assert result.returncode == 0 and result.stdout.strip() == "ok", result.stderr
    result = subprocess.run(
        [sys.executable, "-c", "import pyspark.sql.connect as c; print(sorted(vars(c))[-1])"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0 and result.stdout.strip() == "utils", result.stderr


# The names of pyspark-client 4.3.0's pyspark.sql.connect.client, whose __init__ is
# `from pyspark.sql.connect.client.core import *` and the getLogLevel of
# pyspark.sql.connect.logging, of no `__all__`: those of `dir()` but its dunders, which its
# `import *` gives too, before and after its submodules are imported, as its core imports them.
REFERENCE_CLIENT_NAMES = [
    "ChannelBuilder",
    "DefaultChannelBuilder",
    "RpcDeadlines",
    "SparkConnectClient",
    "artifact",
    "core",
    "getLogLevel",
    "reattach",
    "retries",
]
# Its modules that the drop-in does not have, as the Rust transport replaces them.
CLIENT_MODULES_OF_THE_TRANSPORT = {"artifact", "reattach"}
# The `__all__` of pyspark-client 4.3.0's pyspark.sql.connect.client.core, and the names that the
# module's source binds at its top level, but by its imports.
REFERENCE_CORE_ALL = [
    "ChannelBuilder",
    "DefaultChannelBuilder",
    "RpcDeadlines",
    "SparkConnectClient",
]
REFERENCE_CORE_DEFINED = [
    "AnalyzeResult",
    "ChannelBuilder",
    "ConfigResult",
    "DefaultChannelBuilder",
    "PYSPARK_ROOT",
    "PlanObservedMetrics",
    "RpcDeadlines",
    "SparkConnectClient",
    "__all__",
    "_import_zstandard_if_available",
    "_is_pyspark_source",
]

CLIENT_NAMES_SCRIPT = """
import ast, importlib.abc, importlib.util, json, sys

blocked = sys.argv[1:]


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if any(name == b or name.startswith(b + ".") for b in blocked):
            raise ImportError("blocked: " + name)


sys.meta_path.insert(0, Block())


def names(module):
    star = {}
    exec("from " + module.__name__ + " import *", star)
    return [
        sorted(n for n in dir(module) if not n.startswith("__")),
        sorted(n for n in star if not n.startswith("__")),
    ]


def defined(module):
    bound = set()
    for node in ast.parse(open(module.__file__).read()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            bound.update(t.id for t in targets if isinstance(t, ast.Name))
    return sorted(bound)


import pyspark.sql.connect.client as client

out = {"package": names(client), "has_all": hasattr(client, "__all__"), "doc": client.__doc__}
core = client.core
out["core"] = names(core)
out["core_all"] = core.__all__
out["core_defined"] = defined(core)
# The public names of core that PySpark's core does not have, and whether each is an import.
reference = set(json.loads(sys.stdin.read()))
out["core_extra"] = sorted(
    (n, getattr(getattr(core, n), "__module__", "") != core.__name__)
    for n in names(core)[0]
    if not n.startswith("_") and n not in reference
)
if not blocked and importlib.util.find_spec("grpc") is not None:
    import pyspark.sql.connect.client.retries

    out["package_after"] = names(client)
print(json.dumps(out))
"""

# The names of pyspark-client 4.3.0's pyspark.sql.connect.client.core, those of `dir()` but its
# dunders, of its definitions and of its imports.
REFERENCE_CORE_NAMES = (
    "AnalyzeResult Any ArtifactManager CallSite ChannelBuilder CommonInlineUserDefinedDataSource "
    "CommonInlineUserDefinedFunction CommonInlineUserDefinedTableFunction ConfigResult "
    "ConnectProfilerCollector DEFAULT_MAX_RETRY_EXCEPTION_ELAPSED_TIME DataType "
    "DefaultChannelBuilder DefaultPolicy Dict ExecutePlanResponseReattachableIterator "
    "ExecutionInfo FetchErrorDetailsResponse Iterable Iterator JavaUDF List LiteralExpression "
    "Mapping MetricValue NoReturn Observation ObservedMetrics Optional PYSPARK_ROOT PlanMetrics "
    "PlanObservedMetrics Progress ProgressHandler PySparkAssertionError "
    "PySparkNotImplementedError PySparkValueError PythonDataSource PythonEvalType PythonUDF "
    "PythonUDTF ResourceInformation RetryPolicy Retrying RpcDeadlines Set SparkConnectClient "
    "SparkConnectException SparkConnectGrpcException SpecialAccumulatorIds StorageLevel "
    "StructType TYPE_CHECKING Tuple Type Union _convert_arrow_table_to_pandas "
    "_import_zstandard_if_available _is_pyspark_source any_pb2 atexit cast concurrent "
    "convert_exception convert_observation_errors copy dataclass disable_gc error_details_pb2 "
    "fields from_arrow_schema from_proto get_python_ver google grpc grpc_lib is_remote_only "
    "logger logging os pa pb2 pd pickle platform proto_to_remote_cached_dataframe "
    "proto_to_storage_level pyspark rpc_status specialAccumulatorSer storage_level_to_proto sys "
    "text_format threading time traceback types urllib uuid weakref"
).split()
# The packages that PySpark's core imports, and the drop-in's does not need.
CLIENT_OPTIONAL_PACKAGES = [
    "pandas",
    "pyarrow",
    "numpy",
    "grpc",
    "grpc_status",
    "google",
    "zstandard",
]


@pytest.mark.parametrize("blocked", [[], CLIENT_OPTIONAL_PACKAGES], ids=["all", "blocked"])
def test_the_client_package_and_its_core_have_pysparks_names(blocked):
    # As pyspark-client's pyspark.sql.connect.client, in a new process: of no `__all__` and no
    # docstring, its public names, of `dir()` and of its `import *`, are the classes of the
    # `__all__` of its core, its getLogLevel and its modules, but `artifact` and `reattach`, which
    # the Rust transport replaces, and `retries` until it is imported, as it imports grpc. Its
    # core has PySpark's `__all__` and defines PySpark's names, besides private helpers of its
    # own; its other public names are of the modules that it imports. Both import without
    # pandas, PyArrow, NumPy, grpc, protobuf and zstandard, which PySpark's core imports.
    import importlib.util
    import json
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", CLIENT_NAMES_SCRIPT, *blocked],
        input=json.dumps(REFERENCE_CORE_NAMES),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    unimported = CLIENT_MODULES_OF_THE_TRANSPORT | {"retries"}
    expected = [n for n in REFERENCE_CLIENT_NAMES if n not in unimported]
    assert out["package"] == [expected, expected]
    assert out["has_all"] is False and out["doc"] is None
    assert out["core_all"] == REFERENCE_CORE_ALL
    assert out["core"][1] == REFERENCE_CORE_ALL
    assert set(REFERENCE_CORE_DEFINED) <= set(out["core_defined"])
    own = set(out["core_defined"]) - set(REFERENCE_CORE_DEFINED)
    assert own and all(n.startswith("_") for n in own), sorted(own)
    # Its public names that PySpark's core does not have: of the stdlib that its helpers import.
    assert out["core_extra"] == [
        ["Path", True],
        ["contextvars", True],
        ["functools", True],
        ["inspect", True],
        ["math", True],
        ["warnings", True],
    ]
    public = {n for n in out["core"][0] if not n.startswith("_")}
    assert public - {n for n, _ in out["core_extra"]} <= set(REFERENCE_CORE_NAMES)
    if "package_after" in out:
        after = [n for n in REFERENCE_CLIENT_NAMES if n not in CLIENT_MODULES_OF_THE_TRANSPORT]
        assert out["package_after"] == [after, after]
    else:
        assert blocked or importlib.util.find_spec("grpc") is None


# The modules and qualified names of the classes of pyspark-client 4.3.0's
# pyspark.sql.connect.client.core, of the first line of their source, which `inspect` finds there.
CORE_CLASSES = {
    "SparkConnectClient": "class SparkConnectClient",
    "ChannelBuilder": "class ChannelBuilder",
    "DefaultChannelBuilder": "class DefaultChannelBuilder(ChannelBuilder):",
    "RpcDeadlines": "@dataclass(frozen=True)",
    "PlanObservedMetrics": "class PlanObservedMetrics(ObservedMetrics):",
    "AnalyzeResult": "class AnalyzeResult:",
    "ConfigResult": "class ConfigResult:",
}


@pytest.mark.parametrize("name", list(CORE_CLASSES))
def test_the_classes_of_the_client_and_their_functions_are_of_its_core_as_pysparks(name):
    # As PySpark's: the class, and each function of it, is of pyspark.sql.connect.client.core,
    # which defines it, so that `inspect` finds its source there and a pickle of a function
    # names it there, as PySpark's does.
    import inspect

    from pyspark.sql.connect.client import core

    cls = getattr(core, name)
    assert (cls.__module__, cls.__qualname__) == ("pyspark.sql.connect.client.core", name)
    assert inspect.getmodule(cls) is core
    assert inspect.getsourcefile(cls) == core.__file__
    assert inspect.getsourcelines(cls)[0][0].startswith(CORE_CLASSES[name])
    functions = [
        member
        for member in vars(cls).values()
        if inspect.isfunction(member) and member.__qualname__.startswith(name + ".")
    ]
    assert functions
    for function in functions:
        assert function.__module__ == "pyspark.sql.connect.client.core", function
        assert inspect.getmodule(function) is core
        assert pickle.loads(pickle.dumps(function)) is function


# The pickles of protocol 4 of pyspark-client 4.3.0's objects of `_pickled_objects`: of the
# results of the client, its RPC deadlines and functions of its classes, all of
# pyspark.sql.connect.client.core.
REFERENCE_PICKLES = {
    "ConfigResult": (
        "8004956b000000000000008c1f7079737061726b2e73716c2e636f6e6e6563742e636c69656e742e636f7265"
        "948c0c436f6e666967526573756c749493942981947d94288c057061697273945d94288c0161948c01319486"
        "948c0162944e8694658c087761726e696e6773945d948c0177946175622e"
    ),
    "AnalyzeResult": (
        "80049514010000000000008c1f7079737061726b2e73716c2e636f6e6e6563742e636c69656e742e636f7265"
        "948c0d416e616c797a65526573756c749493942981947d94288c06736368656d61944e8c0e6578706c61696e"
        "5f737472696e67948c0a3d3d20506c616e203d3d948c0b747265655f737472696e67944e8c0869735f6c6f63"
        "616c94888c0c69735f73747265616d696e6794898c0b696e7075745f66696c6573945d948c016694618c0d73"
        "7061726b5f76657273696f6e948c05342e332e30948c06706172736564944e8c1169735f73616d655f73656d"
        "616e7469637394888c0d73656d616e7469635f68617368944b2a8c0d73746f726167655f6c6576656c944e8c"
        "0a64646c5f737472696e67948c056120494e549475622e"
    ),
    "PlanObservedMetrics": (
        "800495e1000000000000008c1f7079737061726b2e73716c2e636f6e6e6563742e636c69656e742e636f7265"
        "948c13506c616e4f627365727665644d6574726963739493942981947d94288c055f6e616d65948c016f948c"
        "085f6d657472696373945d948c17676f6f676c652e70726f746f6275662e6d657373616765948c195f496e74"
        "65726e616c436f6e7374727563744d6573736167659493948c20737061726b2e636f6e6e6563742e45787072"
        "657373696f6e2e4c69746572616c94859452947d948c0a73657269616c697a65649443023803947362618c05"
        "5f6b657973945d948c016e946175622e"
    ),
    "PlanObservedMetrics of no metrics": (
        "80049566000000000000008c1f7079737061726b2e73716c2e636f6e6e6563742e636c69656e742e636f7265"
        "948c13506c616e4f627365727665644d6574726963739493942981947d94288c055f6e616d65948c016f948c"
        "085f6d657472696373945d948c055f6b657973945d9475622e"
    ),
    "RpcDeadlines": (
        "8004951b010000000000008c1f7079737061726b2e73716c2e636f6e6e6563742e636c69656e742e636f7265"
        "948c0c527063446561646c696e65739493942981947d94288c19726561747461636861626c655f6578656375"
        "74655f706c616e944d58028c1072656174746163685f65786563757465944d58028c0c616e616c797a655f70"
        "6c616e94473ff80000000000008c0d6164645f617274696661637473944d100e8c06636f6e666967944e8c09"
        "696e74657272757074944d58028c0f72656c656173655f73657373696f6e944d58028c0f6172746966616374"
        "5f737461747573944d58028c0d636c6f6e655f73657373696f6e944d58028c0a6765745f737461747573944d"
        "58028c1366657463685f6572726f725f64657461696c73944d580275622e"
    ),
    "SparkConnectClient.config": (
        "80049541000000000000008c1f7079737061726b2e73716c2e636f6e6e6563742e636c69656e742e636f7265"
        "948c19537061726b436f6e6e656374436c69656e742e636f6e6669679493942e"
    ),
    "DefaultChannelBuilder.__init__": (
        "80049546000000000000008c1f7079737061726b2e73716c2e636f6e6e6563742e636c69656e742e636f7265"
        "948c1e44656661756c744368616e6e656c4275696c6465722e5f5f696e69745f5f9493942e"
    ),
    "RpcDeadlines.disabled": (
        "8004955a000000000000008c086275696c74696e73948c07676574617474729493948c1f7079737061726b2e"
        "73716c2e636f6e6e6563742e636c69656e742e636f7265948c0c527063446561646c696e65739493948c0864"
        "697361626c656494869452942e"
    ),
}


def _pickled_objects():
    import pyspark.sql.connect.proto as pb2
    from pyspark.sql.connect.client.core import (
        AnalyzeResult,
        ConfigResult,
        DefaultChannelBuilder,
        PlanObservedMetrics,
        RpcDeadlines,
        SparkConnectClient,
    )

    return {
        "ConfigResult": ConfigResult([("a", "1"), ("b", None)], ["w"]),
        "AnalyzeResult": AnalyzeResult(
            None, "== Plan ==", None, True, False, ["f"], "4.3.0", None, True, 42, None, "a INT"
        ),
        "PlanObservedMetrics": PlanObservedMetrics("o", [pb2.Expression.Literal(long=3)], ["n"]),
        "PlanObservedMetrics of no metrics": PlanObservedMetrics("o", [], []),
        "RpcDeadlines": RpcDeadlines(analyze_plan=1.5, config=None),
        "SparkConnectClient.config": SparkConnectClient.config,
        "DefaultChannelBuilder.__init__": DefaultChannelBuilder.__init__,
        "RpcDeadlines.disabled": RpcDeadlines.disabled,
    }


@pytest.mark.parametrize("name", list(REFERENCE_PICKLES))
def test_the_pickles_of_the_clients_results_are_pysparks(name):
    # As PySpark's: the pickles of the results of the client, of its RPC deadlines and of the
    # functions of its classes are those of pyspark-client, which name their classes in
    # pyspark.sql.connect.client.core, and load as they were.
    if name == "PlanObservedMetrics":
        # Its metrics are messages, which pickle of protobuf's _InternalConstructMessage.
        pytest.importorskip("google.protobuf")
    data = pickle.dumps(_pickled_objects()[name], protocol=4)
    assert data.hex() == REFERENCE_PICKLES[name]
    assert pickle.dumps(pickle.loads(data), protocol=4) == data


def test_the_classes_of_the_modules_of_the_client_are_theirs():
    # The classes of the extension that PySpark has not, the responses of an execution of the
    # client and the stub and the stream of the transport of the pandas-parity gate, are of
    # pyspark._pyspark, which has them, and not of PySpark's modules of the client, whose names
    # are PySpark's. Every class of those modules is so a name of its module, which a pickle of it
    # names.
    import gc

    import pyspark.sql.connect.client as client
    from pyspark import _pyspark
    from pyspark.sql.connect.client import core

    for name in ("ExecutePlanResponses", "RustConnectStub", "ResponseStream"):
        cls = getattr(_pyspark, name)
        assert (cls.__module__, cls.__qualname__) == ("pyspark._pyspark", name)
        assert pickle.loads(pickle.dumps(cls)) is cls
        assert not hasattr(client, name) and not hasattr(core, name)
    modules = {m.__name__: m for m in (client, core)}
    classes = [
        cls
        for cls in gc.get_objects()
        if isinstance(cls, type)
        and cls.__module__ in modules
        and "<locals>" not in cls.__qualname__
    ]
    assert {cls.__qualname__ for cls in classes} >= set(CORE_CLASSES)
    for cls in classes:
        assert getattr(core, cls.__qualname__) is cls and cls.__module__ == core.__name__, cls
        assert pickle.loads(pickle.dumps(cls)) is cls


def test_pysparks_own_helpers_of_the_core_module_are_there():
    # As PySpark's core, which has the directory of the pyspark package, whose files
    # _is_pyspark_source tells, and _import_zstandard_if_available, which gives zstandard where
    # it is installed, else None.
    import os

    import pyspark
    from pyspark.sql.connect.client import core

    assert core.PYSPARK_ROOT == os.path.dirname(pyspark.__file__)
    assert core._is_pyspark_source(core.__file__)
    assert not core._is_pyspark_source(os.__file__)
    zstandard = core._import_zstandard_if_available()
    try:
        import zstandard as installed
    except ImportError:
        installed = None
    assert zstandard is installed


def test_the_ml_connect_package_has_check_dependencies_as_pysparks():
    # As PySpark's pyspark.ml.connect, whose __init__ imports the check_dependencies of
    # pyspark.sql.connect.utils (and runs it, which the drop-in's does not; see the test above).
    import pyspark.ml.connect as ml_connect
    from pyspark.sql.connect import utils

    assert ml_connect.check_dependencies is utils.check_dependencies


def test_the_udf_functions_are_names_of_the_functions_modules_as_pysparks():
    # The functions of UDFs and UDTFs that pyspark.sql.functions reads when they are asked for are
    # names of it and of its builtin module, which dir() lists, and a name that neither has raises
    # the AttributeError of a module, as PySpark's.
    import pyspark.sql.functions.builtin as builtin

    for module in (F, builtin):
        assert {"udf", "udtf", "arrow_udtf"} <= set(dir(module))
        assert module.udf is F.udf and module.arrow_udtf is F.arrow_udtf
        with pytest.raises(AttributeError) as raised:
            module.no_such_function
        assert str(raised.value) == f"module {module.__name__!r} has no attribute 'no_such_function'"


def test_the_native_trigger_and_listener_bus_are_of_the_connect_modules(objects):
    # The drop-in's Trigger of a stream writer, of its module of PySpark Connect, where it pickles
    # by reference, and the listener bus of a session's streams, of PySpark Connect's module and
    # name of the class of its bus.
    from pyspark.sql.connect.streaming import readwriter

    trigger = readwriter.Trigger
    assert (trigger.__module__, trigger.__qualname__) == (
        "pyspark.sql.connect.streaming.readwriter",
        "Trigger",
    )
    assert pickle.loads(pickle.dumps(trigger)) is trigger
    bus = type(objects["streams"]._sqlb)
    assert (bus.__module__, bus.__qualname__) == (
        "pyspark.sql.connect.streaming.query",
        "StreamingQueryListenerBus",
    )


def test_the_connect_window_is_a_class_of_its_own_as_pysparks():
    # As PySpark Connect's Window, a subclass of pyspark.sql.window.Window, whose static methods
    # make the WindowSpecs of PySpark Connect.
    from pyspark.sql.connect import window

    assert window.Window is not Window
    assert window.Window.__mro__ == (window.Window, Window, object)
    assert (window.Window.__module__, window.Window.__qualname__) == (
        "pyspark.sql.connect.window",
        "Window",
    )
    assert pickle.loads(pickle.dumps(window.Window)) is window.Window
    spec = window.Window.partitionBy("id").orderBy("id")
    assert type(spec) is window.WindowSpec
    assert repr(spec) == repr(Window.partitionBy("id").orderBy("id"))
    assert window.Window.unboundedPreceding == Window.unboundedPreceding


# PySpark's classic classes whose Connect classes are native: their import paths of PySpark, the
# MRO of PySpark's class, and the object of a session of each.
CLASSIC_OF_NATIVE = {
    "column": ("pyspark.sql.column", "Column", "pyspark.sql.connect.column"),
    "na": ("pyspark.sql.dataframe", "DataFrameNaFunctions", "pyspark.sql.connect.dataframe"),
    "stat": ("pyspark.sql.dataframe", "DataFrameStatFunctions", "pyspark.sql.connect.dataframe"),
    "window": ("pyspark.sql.window", "WindowSpec", "pyspark.sql.connect.window"),
    "asTable": ("pyspark.sql.table_arg", "TableArg", "pyspark.sql.connect.table_arg"),
}
CLASSIC_MRO = {
    "column": ["pyspark.sql.column.Column", "pyspark.sql.tvf_argument.TableValuedFunctionArgument"],
    "na": ["pyspark.sql.dataframe.DataFrameNaFunctions"],
    "stat": ["pyspark.sql.dataframe.DataFrameStatFunctions"],
    "window": ["pyspark.sql.window.WindowSpec"],
    "asTable": [
        "pyspark.sql.table_arg.TableArg",
        "pyspark.sql.tvf_argument.TableValuedFunctionArgument",
    ],
}

# The first line of the docstring of each of PySpark's classic classes.
CLASSIC_DOC = {
    "column": "A column in a DataFrame.",
    "na": "Functionality for working with missing data in :class:`DataFrame`.",
    "stat": "Functionality for statistic functions with :class:`DataFrame`.",
    "window": "A window specification that defines the partitioning, ordering,",
    "asTable": "Represents a table argument in PySpark.",
}


def _qualified(cls):
    return f"{cls.__module__}.{cls.__qualname__}"


@pytest.mark.parametrize("name", list(CLASSIC_OF_NATIVE))
def test_the_classic_classes_of_native_connect_classes_are_pysparks(objects, name):
    # As PySpark's: the classic Column, DataFrameNaFunctions, DataFrameStatFunctions, WindowSpec
    # and TableArg are PySpark's classes, of its modules and MROs, which pyspark.sql names, and the
    # Connect class of each, which its objects are of, is another class, a subclass of it: the
    # drop-in's native one, its virtual subclass.
    import importlib

    import pyspark.sql

    module, qualname, connect = CLASSIC_OF_NATIVE[name]
    classic = getattr(importlib.import_module(module), qualname)
    native = getattr(importlib.import_module(connect), qualname)
    assert _qualified(classic) == f"{module}.{qualname}"
    assert [_qualified(c) for c in classic.__mro__] == CLASSIC_MRO[name] + ["builtins.object"]
    assert classic is not native and _qualified(native) == f"{connect}.{qualname}"
    if hasattr(pyspark.sql, qualname):
        assert getattr(pyspark.sql, qualname) is classic
    assert issubclass(native, classic)
    assert isinstance(objects[name], classic) and type(objects[name]) is native
    assert pickle.loads(pickle.dumps(classic)) is classic
    assert classic.__doc__.strip().splitlines()[0] == CLASSIC_DOC[name]
    assert "_of_expression" not in vars(classic) and "to_plan" not in vars(classic)


def test_the_classic_classes_construct_and_subclass_as_pysparks(objects):
    # As PySpark's on Spark Connect: the classic Column and WindowSpec build the classic
    # implementation, which pyspark-client does not have, and the others construct as plain
    # classes; each subclasses as PySpark's, in the caller's module, of PySpark's MRO, and the
    # objects of a session are not instances of a subclass.
    import abc

    from pyspark import cloudpickle
    from pyspark.sql.column import Column
    from pyspark.sql.dataframe import DataFrameNaFunctions, DataFrameStatFunctions
    from pyspark.sql.table_arg import TableArg
    from pyspark.sql.tvf_argument import TableValuedFunctionArgument
    from pyspark.sql.window import Window, WindowSpec

    for make in (lambda: Column(), lambda: Column("a"), lambda: WindowSpec(None)):
        with pytest.raises(ModuleNotFoundError, match="No module named 'pyspark.sql.classic'"):
            make()
    df = objects["df"]
    for cls in (DataFrameNaFunctions, DataFrameStatFunctions):
        made = cls(df)
        assert type(made) is cls and vars(made) == {"df": df}
        with pytest.raises(TypeError, match="missing 1 required positional argument: 'df'"):
            cls()
    for cls in (Window, TableArg, TableValuedFunctionArgument):
        assert type(cls()) is cls
    for cls, name in [
        (Column, "column"), (DataFrameNaFunctions, "na"), (DataFrameStatFunctions, "stat"),
        (WindowSpec, "window"), (TableArg, "asTable"), (Window, None),
        (TableValuedFunctionArgument, None),
    ]:
        sub = type("Sub", (cls,), {})
        assert sub.__module__ == __name__ and sub.__mro__[1:] == cls.__mro__
        # Of a metaclass of the standard library, which a by-value cloudpickle of the subclass
        # names, and which the Python workers of PySpark import to make it again.
        assert type(sub) is (type if cls is Window else abc.ABCMeta)
        assert b"_classic_bases" not in cloudpickle.dumps(sub)
        assert cloudpickle.loads(cloudpickle.dumps(sub)) is sub
        if name is not None:
            assert not isinstance(objects[name], sub)
            assert not issubclass(type(objects[name]), sub)
        assert pickle.loads(pickle.dumps(cls)) is cls


def test_pickled_columns_and_table_args_load_as_their_classes_as_pysparks(objects):
    # As PySpark Connect's Columns and table arguments: pickle and cloudpickle give back an object
    # of the same class, the native one, of the same expression, which is an instance of the
    # classic class, of which the native class is a virtual subclass.
    from pyspark import cloudpickle
    from pyspark.sql.column import Column
    from pyspark.sql.table_arg import TableArg

    client, table = objects["client"], objects["asTable"]
    for obj, classic in [
        (objects["column"], Column),
        ((F.col("id") + 1).alias("x"), Column),
        (table, TableArg),
        (table.partitionBy("id").orderBy("id"), TableArg),
        (table.withSinglePartition(), TableArg),
    ]:
        for dumps in (pickle.dumps, cloudpickle.dumps):
            restored = pickle.loads(dumps(obj))
            assert type(restored) is type(obj) and type(restored).__module__.startswith(
                "pyspark.sql.connect."
            )
            assert isinstance(restored, classic)
            assert restored.to_plan(client) == obj.to_plan(client)


def test_the_methods_of_the_classic_classes_dispatch_as_pysparks(objects):
    # As PySpark's dispatch_col_method, dispatch_df_method, dispatch_window_method and
    # dispatch_table_arg_method: an explicit call of a classic method runs the Connect class's of
    # an object of it, and raises NOT_IMPLEMENTED of another, where the methods of the classic
    # WindowSpec, which PySpark does not dispatch, return None.
    from pyspark.errors import PySparkNotImplementedError
    from pyspark.sql.column import Column
    from pyspark.sql.connect.window import WindowSpec as ConnectWindowSpec
    from pyspark.sql.dataframe import DataFrameNaFunctions, DataFrameStatFunctions
    from pyspark.sql.table_arg import TableArg
    from pyspark.sql.window import WindowSpec

    column, df = objects["column"], objects["df"]
    assert repr(Column.alias(column, "x")) == repr(column.alias("x"))
    assert repr(Column.__add__(column, 1)) == repr(column + 1)
    assert Column.__hash__ is None
    for call, feature in [
        (lambda: Column.alias(object(), "x"), "Column.alias"),
        (lambda: DataFrameNaFunctions.drop(objects["na"]), "DataFrame.drop"),
        (lambda: DataFrameStatFunctions.cov(objects["stat"], "id", "id"), "DataFrame.cov"),
    ]:
        with pytest.raises(PySparkNotImplementedError) as error:
            call()
        assert error.value.getMessageParameters() == {"feature": feature}
    assert type(DataFrameNaFunctions.drop(df)) is type(df)
    spec = Window.partitionBy("id")
    assert type(spec) is ConnectWindowSpec and repr(spec) == "WindowSpec(PartitionBy(id))"
    assert WindowSpec.partitionBy(spec, "id") is None
    assert WindowSpec.orderBy(spec, "id") is None
    assert WindowSpec.rowsBetween(spec, 0, 1) is None
    assert WindowSpec.rangeBetween(spec, 0, 1) is None
    assert type(TableArg.partitionBy(objects["asTable"], "id")) is type(objects["asTable"])


def test_the_connect_window_is_pyspark_connects(objects):
    # As PySpark Connect's Window, of PySpark Connect's code: a subclass of the classic Window,
    # whose static methods make WindowSpecs of its `_spec`, the specification of no partitioning,
    # ordering and frame, as the classic Window's dispatch to them.
    from pyspark.sql.connect import window

    # Of the names of PySpark Connect's class (and the __slotnames__ that copyreg caches on a
    # class once an object of it is pickled, as of PySpark's).
    assert sorted(set(vars(window.Window)) - {"__slotnames__"}) == [
        "__doc__", "__module__", "_spec", "orderBy", "partitionBy", "rangeBetween", "rowsBetween",
    ]
    assert repr(window.Window._spec) == "WindowSpec()"
    assert type(window.Window._spec) is window.WindowSpec
    for build in (
        lambda w: w.partitionBy("id"),
        lambda w: w.orderBy("id"),
        lambda w: w.rowsBetween(w.unboundedPreceding, w.currentRow),
        lambda w: w.rangeBetween(-1, 1),
    ):
        assert repr(build(Window)) == repr(build(window.Window))
    assert Window.partitionBy.__qualname__ == "Window.partitionBy"


def test_patches_of_the_classic_classes_reach_the_objects_as_pysparks(objects):
    # As PySpark's Connect objects, of which the classic class and TableValuedFunctionArgument are
    # bases: a name patched on them is found of the objects, unless the Connect class has the name
    # (alias), or the objects set it on themselves (_expr) and the patch is not a data descriptor;
    # a deletion of the patch takes it back, and a patch of the class nearer the object wins.
    from pyspark.sql.column import Column
    from pyspark.sql.connect.column import Column as ConnectColumn
    from pyspark.sql.tvf_argument import TableValuedFunctionArgument

    column, table = objects["column"], objects["asTable"]
    expr = column._expr
    alias = vars(Column)["alias"]
    try:
        Column.c14_new = lambda self: "classic"
        TableValuedFunctionArgument.c14_base = lambda self: "base"
        assert column.c14_new() == "classic" and ConnectColumn.c14_new(column) == "classic"
        assert column.c14_base() == "base" and table.c14_base() == "base"
        Column.c14_base = lambda self: "nearer"
        assert column.c14_base() == "nearer" and table.c14_base() == "base"
        del Column.c14_base
        assert column.c14_base() == "base"
        Column.alias = lambda self, *args: "patched"
        assert repr(column.alias("x")) == "Column<'id AS x'>"
        Column._expr = "plain"
        assert not isinstance(column._expr, str) and repr(column._expr) == repr(expr)
        Column._expr = property(lambda self: "property")
        assert column._expr == "property"
        del Column._expr
        assert repr(column._expr) == repr(expr)
        # A patch of the native class itself hides the patch of a classic class, as the Connect
        # class's own attribute does.
        ConnectColumn.c14_own = lambda self: "own"
        Column.c14_own = lambda self: "classic"
        assert column.c14_own() == "own"
    finally:
        for cls, name in [
            (Column, "c14_new"), (TableValuedFunctionArgument, "c14_base"), (Column, "c14_own"),
            (ConnectColumn, "c14_own"), (Column, "_expr"),
        ]:
            if name in vars(cls):
                delattr(cls, name)
        Column.alias = alias
    assert not hasattr(ConnectColumn, "c14_new") and "c14_base" not in vars(ConnectColumn)
    assert repr(column.alias("x")) == "Column<'id AS x'>"


def test_the_streaming_query_class_is_pysparks():
    # As PySpark's: a query is of PySpark Connect's class, and the StreamingQuery of
    # pyspark.sql.streaming is PySpark's classic class, which is not its base.
    import pyspark.sql.streaming.query as classic
    from pyspark.sql.connect.streaming.query import StreamingQuery
    from pyspark.sql.streaming import StreamingQuery as ClassicName

    assert (StreamingQuery.__module__, StreamingQuery.__qualname__) == (
        "pyspark.sql.connect.streaming.query",
        "StreamingQuery",
    )
    assert ClassicName is classic.StreamingQuery
    assert classic.StreamingQuery.__module__ == "pyspark.sql.streaming.query"
    assert not issubclass(StreamingQuery, ClassicName)
    assert pickle.loads(pickle.dumps(StreamingQuery)) is StreamingQuery


def test_the_repr_of_a_catalog_is_pysparks(objects):
    # Of no `__repr__` of its own, as PySpark Connect's Catalog.
    assert repr(objects["catalog"]).startswith("<pyspark.sql.connect.catalog.Catalog object at 0x")


def test_the_repr_of_a_table_argument_is_pysparks(objects):
    # Of no `__repr__` of its own, as PySpark's TableArg.
    assert repr(objects["asTable"]).startswith(
        "<pyspark.sql.connect.table_arg.TableArg object at 0x"
    )


def test_cpythons_messages_name_a_class_by_its_bare_name_as_pysparks():
    # The C name of a class, which these messages give, is its `__name__`, as of PySpark's
    # Python classes, not the dotted path of its module.
    from pyspark.sql.types import IntegerType

    with pytest.raises(TypeError) as raised:
        True ^ F.col("a")
    assert str(raised.value) == "unsupported operand type(s) for ^: 'bool' and 'Column'"
    with pytest.raises(AttributeError) as raised:
        IntegerType().foo
    assert str(raised.value) == "'IntegerType' object has no attribute 'foo'"
    # The module and the qualified name stay those of PySpark Connect, in the repr and the pickle.
    cls = type(F.col("a"))
    assert repr(cls) == "<class 'pyspark.sql.connect.column.Column'>"
    assert pickle.loads(pickle.dumps(cls)) is cls


def test_every_native_class_has_its_bare_name_as_its_c_name(objects):
    # `object.__new__(cls)` refuses a native class, in a message of its C name. The native
    # classes, whose `__new__` is a builtin of their own, are those of the module and those of
    # objects that it makes outside it, such as the expression of a Column, the active session of
    # the class and the exit hook of a session, which the garbage collector finds.
    import gc
    import inspect
    import types

    import pyspark._pyspark as native
    from pyspark.sql.connect.session import SparkSession

    made = [
        F.col("a")._expr,
        SparkSession._active_session,
        objects["spark"]._downgrade(),
        inspect.getattr_static(native.LogisticRegression, "maxIter"),
    ]
    module_classes = {cls for cls in vars(native).values() if isinstance(cls, type)}
    outside = {type(obj) for obj in made}
    assert not outside & module_classes and len(outside) == len(made)
    classes = [
        cls
        for cls in gc.get_objects()
        if isinstance(cls, type)
        and isinstance(vars(cls).get("__new__"), types.BuiltinFunctionType)
        and str(getattr(cls, "__module__", "")).startswith("pyspark.")
    ]
    assert outside <= set(classes) and len(module_classes & set(classes)) > 50
    checked = set()
    for cls in classes:
        if issubclass(cls, BaseException):
            continue
        with pytest.raises(TypeError) as error:
            object.__new__(cls)
        assert f"{cls.__module__}.{cls.__name__}" not in str(error.value), str(error.value)
        checked.add(cls.__name__)
    assert len(checked) > 50 and "_ExitHook" in checked


# The docstrings that the modules of PySpark Connect copy of those of PySpark's classic classes
# (`format.__doc__ = PySparkDataStreamReader.format.__doc__`), of PySpark's source at the tracked
# tag: by the module and the class of PySpark Connect and of the classic class, the names of the
# members of the class whose docstrings the module copies, "-" for the class itself.
COPIED_DOCSTRINGS = {
    (
        "pyspark.sql.connect.udf",
        "UDFRegistration",
        "pyspark.sql.udf",
        "UDFRegistration",
    ): "register registerJavaFunction registerJavaUDAF",
    (
        "pyspark.sql.connect.udtf",
        "UDTFRegistration",
        "pyspark.sql.udtf",
        "UDTFRegistration",
    ): "register",
    (
        "pyspark.sql.connect.streaming.query",
        "StreamingQuery",
        "pyspark.sql.streaming.query",
        "StreamingQuery",
    ): (
        "id runId name isActive awaitTermination status recentProgress lastProgress "
        "processAllAvailable stop explain exception"
    ),
    (
        "pyspark.sql.connect.streaming.query",
        "StreamingQueryManager",
        "pyspark.sql.streaming.query",
        "StreamingQueryManager",
    ): "active get awaitAnyTermination resetTerminated addListener removeListener",
    (
        "pyspark.sql.connect.streaming.readwriter",
        "DataStreamReader",
        "pyspark.sql.streaming.readwriter",
        "DataStreamReader",
    ): "format schema option options name load json orc parquet text csv xml table changes -",
    (
        "pyspark.sql.connect.streaming.readwriter",
        "DataStreamWriter",
        "pyspark.sql.streaming.readwriter",
        "DataStreamWriter",
    ): (
        "outputMode format option options partitionBy clusterBy queryName trigger foreach "
        "foreachBatch start toTable"
    ),
    (
        "pyspark.sql.connect.catalog",
        "Catalog",
        "pyspark.sql.catalog",
        "Catalog",
    ): (
        "currentCatalog setCurrentCatalog listCatalogs currentDatabase setCurrentDatabase "
        "listDatabases getDatabase databaseExists listTables getTable listFunctions "
        "functionExists getFunction listColumns tableExists createExternalTable createTable "
        "dropTempView dropGlobalTempView isCached cacheTable uncacheTable clearCache refreshTable "
        "recoverPartitions refreshByPath dropTable dropView createDatabase dropDatabase "
        "listPartitions listViews getTableProperties getCreateTableString truncateTable "
        "analyzeTable registerFunction -"
    ),
    (
        "pyspark.sql.connect.conf",
        "RuntimeConf",
        "pyspark.sql.conf",
        "RuntimeConfig",
    ): "__init__ set get getAll unset isModifiable -",
    (
        "pyspark.sql.connect.datasource",
        "DataSourceRegistration",
        "pyspark.sql.datasource",
        "DataSourceRegistration",
    ): "register",
    (
        "pyspark.sql.connect.group",
        "GroupedData",
        "pyspark.sql.group",
        "GroupedData",
    ): (
        "agg min max sum avg count pivot apply applyInPandas applyInPandasWithState "
        "transformWithStateInPandas transformWithState applyInArrow cogroup -"
    ),
    (
        "pyspark.sql.connect.group",
        "PandasCogroupedOps",
        "pyspark.sql.pandas.group_ops",
        "PandasCogroupedOps",
    ): "applyInPandas applyInArrow -",
    (
        "pyspark.sql.connect.merge",
        "MergeIntoWriter",
        "pyspark.sql.merge",
        "MergeIntoWriter",
    ): (
        "whenMatched whenNotMatched whenNotMatchedBySource withSchemaEvolution merge "
        "WhenMatched.updateAll WhenMatched.update WhenMatched.delete WhenMatched "
        "WhenNotMatched.insertAll WhenNotMatched.insert WhenNotMatched "
        "WhenNotMatchedBySource.updateAll WhenNotMatchedBySource.update "
        "WhenNotMatchedBySource.delete WhenNotMatchedBySource -"
    ),
    (
        "pyspark.sql.connect.observation",
        "Observation",
        "pyspark.sql.observation",
        "Observation",
    ): "__init__ _on get -",
    (
        "pyspark.sql.connect.readwriter",
        "DataFrameReader",
        "pyspark.sql.readwriter",
        "DataFrameReader",
    ): "format schema option options load table changes json parquet text csv xml orc jdbc -",
    (
        "pyspark.sql.connect.readwriter",
        "DataFrameWriter",
        "pyspark.sql.readwriter",
        "DataFrameWriter",
    ): (
        "mode format option options partitionBy bucketBy sortBy clusterBy save insertInto "
        "saveAsTable json parquet text csv xml orc jdbc"
    ),
    (
        "pyspark.sql.connect.readwriter",
        "DataFrameWriterV2",
        "pyspark.sql.readwriter",
        "DataFrameWriterV2",
    ): (
        "using option options tableProperty partitionedBy clusterBy create replace "
        "createOrReplace append overwrite overwritePartitions"
    ),
    (
        "pyspark.sql.connect.tvf",
        "TableValuedFunction",
        "pyspark.sql.tvf",
        "TableValuedFunction",
    ): (
        "- range explode explode_outer inline inline_outer json_tuple posexplode posexplode_outer "
        "stack collations sql_keywords variant_explode variant_explode_outer python_worker_logs"
    ),
}
# The `__init__` of a native class is the slot of CPython, whose docstring is CPython's.
NATIVE_INITS = {("RuntimeConf", "__init__"), ("Observation", "__init__")}


def _resolved(module, qualname):
    import importlib

    found = importlib.import_module(module)
    for part in qualname.split("."):
        found = getattr(found, part)
    return found


@pytest.mark.parametrize(
    "connect, cls, classic, classic_cls, member",
    [(*key, member) for key, members in COPIED_DOCSTRINGS.items() for member in members.split()],
)
def test_the_connect_objects_have_the_docstrings_that_pyspark_connect_copies(
    connect, cls, classic, classic_cls, member
):
    # As PySpark Connect's modules, which give these classes and their members the docstrings of
    # PySpark's classic ones: the Python classes assign them, and the native classes are built with
    # them (scripts/gen_connect_docstrings.py). The classic modules are Apache Spark's as they are.
    suffix = "" if member == "-" else "." + member
    ours = _resolved(connect, cls + suffix)
    theirs = _resolved(classic, classic_cls + suffix)
    if (cls, member) in NATIVE_INITS:
        assert ours.__doc__ == object.__init__.__doc__ != theirs.__doc__
    else:
        assert ours.__doc__ == theirs.__doc__


# The modules of PySpark that the drop-in does not have (see dev/parity/known-differences.md).
MODULES_OF_PYSPARK_ONLY = {
    "pyspark.sql.connect.client.artifact",
    "pyspark.sql.connect.client.reattach",
    "pyspark.sql.connect.streaming.worker.foreach_batch_worker",
    "pyspark.sql.connect.streaming.worker.listener_worker",
    "pyspark.sql.streaming.python_streaming_source_runner",
    "pyspark.sql.transpile",
}


def test_the_docstrings_of_the_modules_are_pysparks():
    # As official PySpark's, of tests/golden/runtime_docstrings.json: the docstrings of every module
    # of pyspark.sql and pyspark.errors, of their classes and functions and of the members of the
    # classes and of the classes of their bodies (`SparkSession.Builder`), which the native classes
    # are built with (of scripts/gen_connect_docstrings.py) and the Python ones have: their
    # `__doc__`, or the `inspect.getdoc` that `help()` shows of one of no docstring of its own that
    # inherits one, such as a member of PySpark Connect's Column, of the classic Column's. The
    # `__init__` of a native class is CPython's slot, of CPython's docstring; a name that the
    # drop-in does not have is not compared. The sweep imports every module, so it runs in a child
    # interpreter: this process imports none of them, as a module that keeps the state of its
    # import (the `sys.stdout` of the progress bar of pyspark.sql.connect.shell.progress) would
    # keep that of this test.
    import json
    import subprocess
    import sys
    from pathlib import Path

    tests = Path(__file__).resolve().parent
    golden_path = tests.parents[1] / "tests/golden/runtime_docstrings.json"
    assert json.loads(golden_path.read_text())["pyspark"] == "4.3.0"
    run = subprocess.run(
        [sys.executable, str(tests / "_runtime_docstrings.py"), str(golden_path)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout.strip().splitlines()[-1])
    assert set(result["unimported"]) <= MODULES_OF_PYSPARK_ONLY, result["unimported"]
    differ = result["differ"]
    assert result["compared"] > 4450 and not differ, (result["compared"], len(differ), differ[:20])


def test_the_spatial_types_convert_of_their_own_methods_as_pysparks():
    # As PySpark's GeographyType and GeometryType (types.py), which define their own
    # needConversion and fromInternal, of no docstring: the Geography or the Geometry of a dict of
    # its "srid" and "bytes", None of one without either, and the dict of toInternal back.
    from pyspark.sql.types import Geography, GeographyType, Geometry, GeometryType

    for cls, value in ((GeographyType, Geography), (GeometryType, Geometry)):
        for member in ("needConversion", "fromInternal", "toInternal"):
            assert member in vars(cls) and getattr(cls, member).__doc__ is None, (cls, member)
        dt = cls(4326)
        assert dt.needConversion() is True
        made = dt.fromInternal({"srid": 4326, "bytes": b"\x01"})
        assert type(made) is value and (made.srid, made.wkb) == (4326, b"\x01")
        assert dt.fromInternal({"srid": 4326}) is None and dt.fromInternal(None) is None
        assert dt.toInternal(made) == {"srid": 4326, "wkb": b"\x01"}
