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
"""Offline tests of the table arguments and the copies of the windows, grouped data and catalog.

Each expectation is what official PySpark 4.3.0's Spark Connect client does, against the mock
Spark Connect server.
"""

import copy
import pickle

import pytest
from _mock_connect_server import mock_connect_session
from pyspark.errors import IllegalArgumentException
from pyspark.sql.window import Window


@pytest.fixture(scope="module")
def _shared_mock_connect():
    pytest.importorskip("grpc")
    with mock_connect_session() as session:
        yield session


@pytest.fixture
def spark(_shared_mock_connect):
    from pyspark.sql import SparkSession

    _, service = _shared_mock_connect
    service.reset()
    session = SparkSession.builder.remote(service.remote).getOrCreate()
    yield session
    session.stop()


def _table(spark):
    return spark.createDataFrame([(1, "a")], ["id", "name"]).asTable()


PARTITIONED_TWICE = (
    "Cannot call partitionBy() after partitionBy() or withSinglePartition() has been called."
)
ORDER_WITHOUT_PARTITION = (
    "Please call partitionBy() or withSinglePartition() before orderBy()."
)


@pytest.mark.parametrize("columns", [("name",), (1,), (None,)])
def test_table_arg_partition_by_after_partition_by_checks_the_state_first(spark, columns):
    # PySpark checks the partitioning before it converts the columns.
    table = _table(spark).partitionBy("id")
    with pytest.raises(IllegalArgumentException) as excinfo:
        table.partitionBy(columns)
    assert str(excinfo.value) == PARTITIONED_TWICE


def test_table_arg_partition_by_after_single_partition_checks_the_state_first(spark):
    table = _table(spark).withSinglePartition()
    with pytest.raises(IllegalArgumentException) as excinfo:
        table.partitionBy(("name",))
    assert str(excinfo.value) == PARTITIONED_TWICE


def test_table_arg_with_single_partition_after_partition_by_checks_the_state_first(spark):
    table = _table(spark).partitionBy("id")
    with pytest.raises(IllegalArgumentException) as excinfo:
        table.withSinglePartition()
    assert str(excinfo.value) == (
        "Cannot call withSinglePartition() after partitionBy() or withSinglePartition() "
        "has been called."
    )


@pytest.mark.parametrize("columns", [(("name",),), (1,), (None,), ()])
def test_table_arg_order_by_without_partition_checks_the_state_first(spark, columns):
    with pytest.raises(IllegalArgumentException) as excinfo:
        _table(spark).orderBy(*columns)
    assert str(excinfo.value) == ORDER_WITHOUT_PARTITION


def test_table_arg_partitioned_and_ordered_still_builds(spark):
    table = _table(spark).partitionBy("id").orderBy("name")
    assert table is not None
    assert _table(spark).withSinglePartition().orderBy("name") is not None


def test_table_arg_to_plan_is_its_subquery_expression_as_pyspark(spark):
    # As PySpark Connect's TableArg.to_plan(session), the Expression of its subquery expression: a
    # table argument of the plan of the DataFrame, with its partitioning, ordering and single
    # partition.
    import pyspark.sql.connect.proto as pb2
    from pyspark.sql import functions as F

    df = spark.range(1)
    plain = df.asTable().to_plan(spark.client)
    assert isinstance(plain, pb2.Expression)
    assert plain.subquery_expression.subquery_type == (
        pb2.SubqueryExpression.SUBQUERY_TYPE_TABLE_ARG)
    assert not plain.subquery_expression.HasField("table_arg_options")
    options = (
        df.asTable().partitionBy("id").orderBy(F.col("id").desc()).to_plan(spark.client)
        .subquery_expression.table_arg_options
    )
    assert options.partition_spec[0].unresolved_attribute.unparsed_identifier == "id"
    assert options.order_spec[0].direction == pb2.Expression.SortOrder.SORT_DIRECTION_DESCENDING
    single = df.asTable().withSinglePartition().to_plan(spark.client)
    assert single.subquery_expression.table_arg_options.with_single_partition


def test_table_arg_has_the_subquery_expression_and_partitioning_of_pyspark(spark):
    # As PySpark Connect's TableArg: its _subquery_expr is the expression of its subquery, whose
    # to_plan is the table argument's and whose repr is that of a SubqueryExpression (of the plan
    # ID of its plan, see docs/compatibility.md), and _is_partitioned() says whether partitionBy
    # or withSinglePartition made it, which orderBy keeps.
    table = spark.range(1).asTable()
    partitioned = table.partitionBy("id")
    for arg in (table, partitioned, table.withSinglePartition().orderBy("id")):
        assert arg._subquery_expr.to_plan(spark.client) == arg.to_plan(spark.client)
    plan_id = table.to_plan(spark.client).subquery_expression.plan_id
    assert repr(table._subquery_expr) == f"SubqueryExpression(plan=<plan {plan_id}>, type=table_arg)"
    assert str(partitioned._subquery_expr) == (
        f"SubqueryExpression(plan=<plan {plan_id}>, type=table_arg, partition_spec=[id])"
    )
    assert [arg._is_partitioned() for arg in (
        table,
        partitioned,
        table.withSinglePartition(),
        partitioned.orderBy("id"),
    )] == [False, True, True, True]


def test_table_arg_copy_keeps_user_attributes(spark):
    table = _table(spark)
    table.foo = [1, 2]
    assert copy.copy(table).foo == [1, 2]
    assert copy.copy(table).foo is table.foo


def test_table_arg_deepcopy_keeps_user_attributes(spark):
    table = _table(spark)
    table.foo = [1, 2]
    deep = copy.deepcopy(table)
    assert deep.foo == [1, 2]
    assert deep.foo is not table.foo


def test_table_arg_pickles_as_pyspark(spark, _shared_mock_connect):
    # As PySpark's table argument, a plain object of the objects of its plan, its partitioning,
    # its ordering and its single partition, which pickles at every protocol, plainly and by
    # cloudpickle, of no session: the unpickled table argument is of the relation of the plan, of
    # its plan ID, as the calls that made it, and of its user-set attributes, with no request, as
    # PySpark's unpickle sends none, and a UDTF of it sends what one of the table argument sends.
    # Of the active session of the ID of the session of the pickled one, it raises
    # NO_ACTIVE_SESSION where none is, as on another thread. Of a plan that does not build, it
    # unpickles, and its requests raise the error of the plan, as PySpark's do.
    import threading

    import cloudpickle
    from pyspark.errors import PySparkRuntimeError
    from pyspark.sql import functions as F
    from pyspark.sql.connect.proto import Relation

    _, service = _shared_mock_connect

    @F.udtf(returnType="id: bigint")
    class Echo:
        def eval(self, row):
            yield (row["id"],)

    def references(table):
        return list(Relation.FromString(Echo(table)._relation_bytes()).with_relations.references)

    df = spark.range(3).withColumn("k", F.col("id") % 2)
    tables = [
        df.asTable(),
        df.asTable().partitionBy("k"),
        df.asTable().partitionBy(F.col("k"), "id").orderBy(F.col("id").desc(), "k"),
        df.asTable().withSinglePartition().orderBy(F.col("id").asc_nulls_last()),
    ]
    tables[0].note = [1, 2]
    requests = (len(service.analyze_requests), len(service.execute_requests))
    for table in tables:
        pickles = [pickle.dumps(table, protocol) for protocol in range(pickle.HIGHEST_PROTOCOL + 1)]
        for loaded in [pickle.loads(data) for data in pickles] + [
                cloudpickle.loads(cloudpickle.dumps(table))]:
            assert type(loaded) is type(table)
            assert loaded.to_plan(spark.client) == table.to_plan(spark.client)
            assert loaded._is_partitioned() == table._is_partitioned()
            assert vars(loaded) == vars(table)
            assert references(loaded) == references(table)
    assert (len(service.analyze_requests), len(service.execute_requests)) == requests
    bad = spark.range(1).alias(5).asTable()
    loaded = cloudpickle.loads(cloudpickle.dumps(bad))
    for table in (bad, loaded):
        with pytest.raises(TypeError, match="^bad argument type for built-in operation$"):
            Echo(table)._relation_bytes()
    errors = []

    def unpickle(data):
        try:
            cloudpickle.loads(data)
        except PySparkRuntimeError as error:
            errors.append(error.getCondition())

    thread = threading.Thread(target=unpickle, args=(cloudpickle.dumps(tables[1]),))
    thread.start()
    thread.join()
    assert errors == ["NO_ACTIVE_SESSION"]


def test_window_objects_and_members_are_those_of_pyspark():
    # As PySpark's Window, a plain class of static methods and of the thresholds of the frames,
    # and PySpark Connect's subclass of `_spec`: its objects, also of a subclass, have no state,
    # take attributes and weak references, pickle as plain objects and take no arguments, as
    # `object.__new__` makes them; and as PySpark Connect's WindowSpec, whose `_partitionSpec`,
    # `_orderSpec` and `_frame` are the expressions of its partitioning and its ordering, which
    # show as PySpark's, and its WindowFrame. `pyspark.sql.window.__all__` is PySpark's.
    import sys
    import weakref

    import pyspark.sql.window as window_module
    from pyspark.sql import functions as F
    from pyspark.sql.connect.window import Window as ConnectWindow
    from pyspark.sql.connect.window import WindowFrame

    assert window_module.__all__ == ["Window", "WindowSpec"]
    assert Window._PRECEDING_THRESHOLD == max(-sys.maxsize, -(1 << 63))
    assert Window._FOLLOWING_THRESHOLD == min(sys.maxsize, (1 << 63) - 1)
    assert repr(ConnectWindow._spec) == "WindowSpec()"
    for cls in (Window, ConnectWindow):
        window = cls()
        assert type(window) is cls and vars(window) == {}
        window.note = 1
        assert weakref.ref(window)() is window
        copied = pickle.loads(pickle.dumps(window))
        assert type(copied) is cls and copied.note == 1
        assert repr(window).startswith(f"<{cls.__module__}.Window object at 0x")
        with pytest.raises(TypeError, match=r"^Window\(\) takes no arguments$"):
            cls(5)

    class Sub(Window):
        def __init__(self, x):
            self.x = x

    assert Sub(5).x == 5
    assert repr(Sub.partitionBy("a")) == "WindowSpec(PartitionBy(a))"
    spec = Window.partitionBy(F.col("x") + 1, "y").orderBy(F.col("z").desc(), "q")
    spec = spec.rowsBetween(-1, 1)
    assert repr(spec._partitionSpec) == "[+(x, 1), y]"
    assert repr(spec._orderSpec) == "[z DESC NULLS LAST, q ASC NULLS FIRST]"
    assert type(spec._frame) is WindowFrame
    assert (spec._frame._isRowFrame, spec._frame._start, spec._frame._end) == (True, -1, 1)
    assert repr(spec._frame) == "WindowFrame(ROW_FRAME, -1, 1)"
    ranged = Window.rangeBetween(Window.unboundedPreceding, 5)._frame
    assert repr(ranged) == "WindowFrame(RANGE_FRAME, -9223372036854775808, 5)"
    assert Window.partitionBy("a")._frame is None
    assert Window.partitionBy("a")._orderSpec == []


def test_window_copy_and_pickle_keep_user_attributes():
    window = Window.partitionBy("id")
    window.foo = 5
    assert copy.copy(window).foo == 5
    assert copy.deepcopy(window).foo == 5
    assert pickle.loads(pickle.dumps(window)).foo == 5


def test_window_without_user_attributes_copies_and_pickles():
    window = Window.partitionBy("id").orderBy("name")
    assert vars(copy.copy(window)) == {}
    assert vars(pickle.loads(pickle.dumps(window))) == {}
    assert repr(copy.copy(window)) == repr(window)


def test_grouped_data_copy_keeps_user_attributes(spark):
    grouped = spark.range(2).groupBy("id")
    grouped.foo = 5
    assert copy.copy(grouped).foo == 5
    assert copy.deepcopy(grouped).foo == 5


def test_catalog_copy_keeps_user_attributes(spark):
    catalog = spark.catalog
    catalog.foo = 5
    assert copy.copy(catalog).foo == 5
    assert copy.deepcopy(catalog).foo == 5
