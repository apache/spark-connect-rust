# SPDX-License-Identifier: Apache-2.0
"""The classes of the objects of a session as official PySpark 4.3.0's, without a server: of the
modules of PySpark Connect, `pyspark.sql.connect.*`, which `repr`, `type(x).__module__` and pickle
show. The expected names are official PySpark's, of the same objects."""

import pickle

import pytest
from _mock_connect_server import mock_connect_session
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
        objects["spark"]._copy_thread_tags(),
        objects["spark"]._copy_thread_user_context_extensions(),
        objects["streams"]._sqlb,
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
