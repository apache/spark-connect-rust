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
"""The classic `pyspark.sql.SparkSession` and the Connect classes as official PySpark 4.3.0's,
without a server of Spark: PySpark's classic class, whose builder makes a session of Spark
Connect's class, PySpark Connect's builder, the functions of `pyspark.sql.connect.functions`, and
the typing of the package."""

import inspect
import os
import pickle

import pyspark
import pytest
from _mock_connect_server import mock_connect_session
from pyspark.errors import PySparkRuntimeError, PySparkValueError
from pyspark.sql import SparkSession
from pyspark.sql.connect.session import SparkSession as ConnectSession


@pytest.fixture(scope="module")
def connect():
    previous = os.environ.pop("SPARK_REMOTE", None)
    try:
        with mock_connect_session() as (spark, service):
            yield spark, service
    finally:
        if previous is not None:
            os.environ["SPARK_REMOTE"] = previous


def test_the_session_class_of_pyspark_sql_is_the_classic_one(connect):
    spark, _ = connect
    # PySpark's classic class, of which a session of Spark Connect is not an instance.
    assert (SparkSession.__module__, SparkSession.__qualname__) == (
        "pyspark.sql.session",
        "SparkSession",
    )
    assert type(spark) is ConnectSession
    assert ConnectSession.__module__ == "pyspark.sql.connect.session"
    assert not isinstance(spark, SparkSession)
    assert SparkSession is not type(spark)
    # Its classmethods give the session of Spark Connect, as PySpark's.
    assert SparkSession.getActiveSession() is spark
    assert SparkSession.active() is spark
    # Its type and messages are those of a Python class, as PySpark's.
    with pytest.raises(AttributeError, match="type object 'SparkSession' has no attribute 'nope'"):
        SparkSession.nope
    # A subclass of it is a class, as of PySpark's.
    assert issubclass(type("MySession", (SparkSession,), {}), SparkSession)


def test_the_classic_builder_needs_a_url_also_with_an_active_session(connect, monkeypatch):
    spark, service = connect
    monkeypatch.delenv("SPARK_REMOTE", raising=False)
    for make in (
        lambda: SparkSession.builder.getOrCreate(),
        lambda: SparkSession.builder.appName("app").getOrCreate(),
    ):
        with pytest.raises(PySparkRuntimeError) as error:
            make()
        assert type(error.value) is PySparkRuntimeError
        assert error.value.getCondition() == "CONNECT_URL_NOT_SET"
    # With a URL, of the option or of SPARK_REMOTE, it gives the active session of the URL.
    assert SparkSession.builder.remote(service.remote).getOrCreate() is spark
    monkeypatch.setenv("SPARK_REMOTE", service.remote)
    assert SparkSession.builder.getOrCreate() is spark
    assert os.environ["SPARK_CONNECT_MODE_ENABLED"] == "1"


def test_the_connect_builder_is_pyspark_connects(connect, monkeypatch):
    spark, service = connect
    # It returns the active session without a URL, and needs one, or a channel builder, to
    # create one: SPARK_REMOTE is read by the classic builder only.
    assert ConnectSession.builder.getOrCreate() is spark
    monkeypatch.setenv("SPARK_REMOTE", service.remote)
    with pytest.raises(PySparkValueError) as error:
        ConnectSession.builder.create()
    assert error.value.getCondition() == "SESSION_NEED_CONN_STR_OR_BUILDER"
    # remote(location="sc://localhost"), as PySpark Connect's, and remote(url) of the classic one.
    assert list(inspect.signature(ConnectSession.builder.remote).parameters) == ["location"]
    assert list(inspect.signature(SparkSession.builder.remote).parameters) == ["url"]
    with pytest.raises(TypeError, match="unexpected keyword argument 'url'"):
        ConnectSession.builder.remote(url=service.remote)
    with pytest.raises(TypeError, match="multiple values for argument 'location'"):
        ConnectSession.builder.remote(service.remote, location=service.remote)
    assert type(ConnectSession.builder).__qualname__ == "SparkSession.Builder"
    assert type(ConnectSession.builder).__module__ == "pyspark.sql.connect.session"


def test_the_builder_of_the_connect_class_is_pysparks_classproperty(connect):
    # As PySpark Connect's `builder = classproperty(lambda cls: cls.Builder())`, of PySpark's
    # classproperty of pyspark.sql.session, of the docstring of the classic builder: a new builder
    # of each access, of the class or of a session.
    import pyspark.sql.session as classic

    spark, _ = connect
    descriptor = inspect.getattr_static(ConnectSession, "builder")
    assert type(descriptor) is classic.classproperty
    assert pickle.loads(pickle.dumps(type(descriptor))) is classic.classproperty
    assert type(descriptor.fget(ConnectSession)) is ConnectSession.Builder
    assert descriptor.fget.__module__ == "pyspark.sql.connect.session"
    assert descriptor.__doc__ == SparkSession.builder.__doc__
    with pytest.raises(AttributeError, match=r"^'classproperty' object has no attribute 'nope'$"):
        descriptor.nope
    assert ConnectSession.builder is not ConnectSession.builder
    assert type(spark.builder) is ConnectSession.Builder


def test_only_the_classic_builder_enables_hive_support():
    # As PySpark's: the classic builder sets spark.sql.catalogImplementation, which the session of
    # Spark Connect does not take, and PySpark Connect's builder raises NOT_IMPLEMENTED.
    from pyspark.errors import PySparkNotImplementedError

    builder = SparkSession.builder.enableHiveSupport()
    assert type(builder) is SparkSession.Builder
    assert builder._options == {"spark.sql.catalogImplementation": "hive"}
    with pytest.raises(PySparkNotImplementedError) as error:
        ConnectSession.builder.enableHiveSupport()
    assert error.value.getCondition() == "NOT_IMPLEMENTED"
    assert error.value.getMessageParameters() == {"feature": "enableHiveSupport"}
    assert str(error.value) == "[NOT_IMPLEMENTED] enableHiveSupport is not implemented."


def test_the_classic_class_has_pysparks_signatures_and_docs():
    signature = inspect.signature(SparkSession.sql)
    assert list(signature.parameters) == ["self", "sqlQuery", "args", "kwargs"]
    assert signature.parameters["sqlQuery"].annotation is str
    assert SparkSession.sql.__doc__.lstrip().startswith(
        "Returns a :class:`DataFrame` representing the result of the given query."
    )
    assert SparkSession.builder.__doc__.lstrip().startswith("Builder for :class:`SparkSession`.")
    assert str(inspect.signature(SparkSession.Builder.getOrCreate)) == "(self) -> 'SparkSession'"


def test_the_functions_of_connect_are_of_its_modules(connect):
    import pyspark.sql.connect.functions as CF
    import pyspark.sql.connect.functions.builtin as CB
    import pyspark.sql.connect.functions.partitioning as CP
    import pyspark.sql.functions as F

    spark, _ = connect
    # As PySpark's: the functions of pyspark.sql.connect.functions are of its modules, and are
    # not those of pyspark.sql.functions, which run as they do.
    for module, name, expected in [
        (CF, "col", "pyspark.sql.connect.functions.builtin"),
        (CB, "array_join", "pyspark.sql.connect.functions.builtin"),
        (CF, "udf", "pyspark.sql.connect.functions.builtin"),
        (CP, "years", "pyspark.sql.connect.functions.partitioning"),
    ]:
        function = getattr(module, name)
        assert function.__module__ == expected, name
        assert function is not getattr(F if module is not CP else F.partitioning, name)
        # Pickled by reference to that module, which a worker of the server imports.
        assert pickle.loads(pickle.dumps(function)) is function
    assert F.col.__module__ == "pyspark.sql.functions.builtin"
    assert CF.pandas_udf is F.pandas_udf
    assert inspect.signature(CF.array_join) == inspect.signature(F.array_join)
    assert CF.array_join.__doc__ == F.array_join.__doc__
    assert str(CF.col("a") + 1) == str(F.col("a") + 1)
    assert spark.range(1).select(CF.lit(1)).columns == spark.range(1).select(F.lit(1)).columns


def test_the_aliases_of_functions_are_their_functions_as_pysparks():
    import types

    import pyspark.sql.connect.functions as CF
    import pyspark.sql.connect.functions.builtin as CB
    import pyspark.sql.functions as F

    # As PySpark's pyspark.sql.functions and pyspark.sql.connect.functions, which bind
    # column = col, negate = negative, power = pow and random = rand: an alias is its function, of
    # the function's name, and a pickle names it by that name.
    for module in (F, F.builtin, CF, CB):
        for alias, name in [("column", "col"), ("negate", "negative"), ("power", "pow"),
                            ("random", "rand")]:
            function = getattr(module, alias)
            assert function is getattr(module, name), (module.__name__, alias)
            assert function.__name__ == name
            assert pickle.loads(pickle.dumps(function)) is function
    # So every function of the builtin module of Spark Connect pickles by reference to itself.
    for name in dir(CB):
        function = getattr(CB, name)
        if not name.startswith("_") and isinstance(function, types.FunctionType):
            assert pickle.loads(pickle.dumps(function)) is function, name


def test_a_cogroup_is_of_pyspark_connects_class(connect):
    spark, _ = connect
    df = spark.range(1)
    cogrouped = df.groupBy("id").cogroup(df.groupBy("id"))
    cls = type(cogrouped)
    assert (cls.__module__, cls.__qualname__) == (
        "pyspark.sql.connect.group",
        "PandasCogroupedOps",
    )
    from pyspark.sql.connect.group import PandasCogroupedOps

    assert cls is PandasCogroupedOps
    assert pickle.loads(pickle.dumps(cls)) is cls


def test_the_package_is_typed_as_pysparks():
    package = os.path.dirname(pyspark.__file__)
    # PEP 561, and the stubs of PySpark's type aliases.
    for name in ["py.typed", "_typing.pyi", os.path.join("sql", "_typing.pyi")]:
        assert os.path.isfile(os.path.join(package, name)), name
