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
