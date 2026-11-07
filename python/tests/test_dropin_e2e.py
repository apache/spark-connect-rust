#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Drop-in behaviors that need a live Spark Connect server. The official tests run the
reference pyspark over our transport rather than these Rust-backed classes, so the expected
values here are those of the reference client; the tests pass with it as well.
"""

import os
from collections import namedtuple
from datetime import date, datetime

import pyarrow as pa
import pytest
from pyspark.errors import AnalysisException, PySparkValueError
from pyspark.sql import Row, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    DateType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


@pytest.fixture
def session():
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live-server tests")
    session = SparkSession.builder.remote(remote).getOrCreate()
    try:
        yield session
    finally:
        session.stop()


@pytest.mark.parametrize("prefetch", [False, True])
def test_to_local_iterator_raises_the_error_of_the_execution_as_pyspark(session, prefetch):
    # As official PySpark 4.3.0: the error of a query that fails while it is iterated is
    # raised by the iteration, which then ends, rather than ending it as if complete.
    df = session.sql("SELECT assert_true(id < 0) AS x FROM range(3)")
    rows = df.toLocalIterator(prefetch)
    with pytest.raises(Exception) as error:
        next(rows)
    assert error.value.getCondition() == "USER_RAISED_EXCEPTION"
    assert list(rows) == []


def test_udf_and_udtf_calls_take_named_arguments_as_pyspark(session):
    # As PySpark's: the keyword arguments of a UDF or UDTF call are its named arguments, and
    # a bare string is a column name.
    import pandas as pd

    @F.udf("int")
    def plus(a, b):
        return a + 10 * b

    @F.pandas_udf("long")
    def pandas_plus(a: pd.Series, b: pd.Series) -> pd.Series:
        return a + 10 * b

    df = session.range(1)
    for fn in (plus, pandas_plus):
        assert df.select(fn(b=F.lit(2), a=F.lit(1)).alias("r")).collect() == [Row(r=21)]
        assert df.select(fn(F.lit(1), b=F.lit(2)).alias("r")).collect() == [Row(r=21)]

    @F.udtf(returnType="a int, b int")
    class Two:
        def eval(self, a, b):
            yield a, b

    assert Two(b=F.lit(2), a=F.lit(1)).collect() == [Row(a=1, b=2)]
    assert Two(F.lit(1), b=F.lit(2)).collect() == [Row(a=1, b=2)]
    with pytest.raises(AnalysisException) as error:
        Two("x", F.lit(2)).collect()
    assert error.value.getCondition() == "UNRESOLVED_COLUMN.WITHOUT_SUGGESTION"


def test_conf_get_with_a_none_default(session):
    key = "spark.connect.rust.test.unset"
    assert session.conf.get(key, None) is None
    assert session.conf.get(key, "x") == "x"
    with pytest.raises(Exception, match="SQL_CONF_NOT_FOUND"):
        session.conf.get(key)


def test_user_id_of_the_connection_string_reaches_every_request_as_pyspark():
    # The server keys a session by the user ID in each request. As in PySpark, the
    # queries carry the user ID of the connection string, as the configs do, so a config
    # set with spark.conf.set applies to the queries and the other way around.
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live-server tests")
    remote += ";user_id=user-context-probe" if ";" in remote else "/;user_id=user-context-probe"
    spark = SparkSession.builder.remote(remote).create()
    try:
        spark.conf.set("spark.sql.shuffle.partitions", "7")
        assert spark.sql("SET spark.sql.shuffle.partitions").collect()[0][1] == "7"
        spark.sql("SET spark.sql.shuffle.partitions=9").collect()
        assert spark.conf.get("spark.sql.shuffle.partitions") == "9"
    finally:
        spark.stop()


def test_get_or_create_returns_the_session_with_the_options_as_pyspark(session):
    # As PySpark's builder: getOrCreate() returns the active session, and sets the options
    # of the builder on it; one that fails, of a static configuration, is ignored.
    key = "spark.sql.shuffle.partitions"
    previous = session.conf.get(key)
    try:
        got = (
            SparkSession.builder.remote(os.environ["SPARK_REMOTE"])
            .config(key, "5")
            .config("spark.sql.warehouse.dir", "/not/set")
            .getOrCreate()
        )
        assert got is session
        assert session.conf.get(key) == "5"
        assert session.conf.get("spark.sql.warehouse.dir") != "/not/set"
    finally:
        session.conf.set(key, previous)


def test_repr_shows_the_rows_with_eager_evaluation_as_pyspark(session):
    df = session.range(3)
    assert repr(df) == "DataFrame[id: bigint]"
    keys = ["spark.sql.repl.eagerEval.enabled", "spark.sql.repl.eagerEval.maxNumRows",
            "spark.sql.repl.eagerEval.truncate"]
    try:
        session.conf.set(keys[0], "true")
        session.conf.set(keys[1], "2")
        assert repr(df) == "+---+\n| id|\n+---+\n|  0|\n|  1|\n+---+\nonly showing top 2 rows"
        session.conf.set(keys[2], "3")
        assert repr(session.createDataFrame([("abcdef",)], "s string")) == (
            "+---+\n|  s|\n+---+\n|abc|\n+---+\n")
    finally:
        for key in keys:
            session.conf.unset(key)
    assert repr(df) == "DataFrame[id: bigint]"


def test_replace_and_unpivot_take_their_arguments_as_pyspark(session):
    # As PySpark's: replace sends each value as a literal of its type, and None as a null, over
    # a subset of a column name or a list or tuple of them. unpivot and melt take a column or a
    # column name, or a list or tuple of them.
    df = session.createDataFrame(
        [(1, "a", 1.5, True, "1"), (2, "b", None, False, "2")],
        "id int, k string, v double, b boolean, s string",
    )

    def rows(*changes):
        base = [dict(id=1, k="a", v=1.5, b=True, s="1"), dict(id=2, k="b", v=None, b=False, s="2")]
        return [Row(**{**row, **change}) for row, change in zip(base, changes)]

    assert df.replace("a", None, "k").collect() == rows({"k": None}, {})
    # A string replaces in the string columns, and a number in the numeric ones.
    assert df.replace("1", "one").collect() == rows({"s": "one"}, {})
    assert df.replace(1, 10, ("id",)).collect() == rows({"id": 10}, {})
    assert df.replace([1.5, 2], [9.5, 20]).collect() == rows({"v": 9.5}, {"id": 20})
    assert df.replace(True, False).collect() == rows({"b": False}, {})
    assert df.na.replace({"a": "x", "b": "y"}, subset="k").collect() == rows({"k": "x"}, {"k": "y"})

    wide = session.createDataFrame([(1, 11, 12)], "id int, x int, y int")
    expected = [Row(id=1, var="x", val=11), Row(id=1, var="y", val=12)]
    assert wide.unpivot("id", ["x", "y"], "var", "val").collect() == expected
    assert wide.unpivot(F.col("id"), ("x", F.col("y")), "var", "val").collect() == expected
    assert wide.melt(["id"], None, "var", "val").collect() == expected
    assert wide.melt("id", "x", "var", "val").collect() == expected[:1]


def test_instr_and_regexp_replace_apply_their_optional_arguments(session):
    # Column b holds "c", so the literal "b" and the column b find different positions.
    df = session.createDataFrame([("abcabc", "c", "a", "z")], ["s", "b", "p", "r"])
    out = df.select(
        F.instr("s", "b"),
        F.instr("s", "b", 3),
        F.instr("s", "b", occurrence=2),
        F.instr(F.col("s"), F.col("b"), F.lit(4)),
        F.regexp_replace("s", "a", "z", 2),
        F.regexp_replace("s", F.col("p"), F.col("r")),
        F.call_function("instr", "s", F.lit("b")),
    )
    assert out.columns == [
        "instr(s, b)", "instr(s, b, 3, 1)", "instr(s, b, 1, 2)", "instr(s, b, 4, 1)",
        "regexp_replace(s, a, z, 2)", "regexp_replace(s, p, r, 1)", "instr(s, b)",
    ]
    assert tuple(out.first()) == (2, 5, 5, 6, "abczbc", "zbczbc", 2)


@pytest.mark.parametrize("by_name", [True, False])
def test_repartition_by_id_takes_a_column_name(session, by_name):
    df = session.range(12).withColumn("p", (F.col("id") % 3).cast("int"))
    partition_id_col = "p" if by_name else F.col("p")
    rows = (
        df.repartitionById(3, partition_id_col)
        .select("p", F.spark_partition_id().alias("pid"))
        .collect()
    )
    assert {(r.p, r.pid) for r in rows} == {(0, 0), (1, 1), (2, 2)}


@pytest.mark.parametrize(
    "schema",
    [
        StructType().add("a", LongType()).add("b", LongType()),
        StructType([StructField("a", LongType())]).add("b", LongType()),
    ],
)
def test_map_in_arrow_and_pandas_with_a_struct_type_built_with_add(session, schema):
    df = session.range(3).selectExpr("id AS a", "id * 2 AS b")
    assert df.mapInArrow(lambda batches: batches, schema).columns == ["a", "b"]
    rows = df.mapInPandas(lambda frames: frames, schema).orderBy("a").collect()
    assert [tuple(r) for r in rows] == [(0, 0), (1, 2), (2, 4)]


def test_udtf_takes_a_ddl_or_struct_return_type_as_pyspark(session):
    from pyspark.errors.exceptions.connect import ParseException

    class Echo:
        def eval(self, value):
            yield (value,)
            yield (value + 1,)

    struct = StructType([StructField("value", LongType())])
    for return_type in ["value BIGINT", "value: bigint", struct]:
        for fn in (F.udtf(Echo, returnType=return_type), F.udtf(returnType=return_type)(Echo)):
            assert fn(F.lit(1)).collect() == [Row(value=1), Row(value=2)]

    # The server parses the DDL, so an invalid one fails when the query runs.
    df = F.udtf(Echo, returnType="invalid DDL !!!")(F.lit(1))
    with pytest.raises(ParseException) as error:
        df.collect()
    assert error.value.getCondition() == "PARSE_SYNTAX_ERROR"


def test_arrow_udtf_takes_and_returns_arrow_data_as_pyspark(session):
    # As PySpark's arrow_udtf(), a PyArrow-native UDTF: eval() takes a pyarrow array of the
    # argument, and returns pyarrow tables.
    import pyarrow.compute as pc

    @F.arrow_udtf(returnType="x long, kind string")
    class PlusOne:
        def eval(self, x):
            yield pa.table({"x": pc.add(x, 1), "kind": pa.array([type(x).__name__] * len(x))})

    assert PlusOne(F.lit(1)).collect() == [Row(x=2, kind="Int32Array")]


def test_udfs_are_arrow_optimized_or_of_their_type_hints_as_pyspark(session):
    # As PySpark's udf(): Arrow-optimized as the server enables by default, so Arrow
    # converts a result to the return type; and a function with pandas type hints is a
    # pandas UDF.
    import pandas as pd

    df = session.range(2)
    assert df.select(F.udf(lambda x: str(x), "int")("id").alias("r")).collect() == [
        Row(r=0), Row(r=1)
    ]
    assert df.select(F.udf(lambda x: 1.5, "int")("id").alias("r")).collect() == [
        Row(r=1), Row(r=1)
    ]

    def plus_one(s: pd.Series) -> pd.Series:
        return s + 1

    hinted = F.udf(plus_one, "long")
    assert hinted.evalType == 200
    assert df.select(hinted("id").alias("r")).collect() == [Row(r=1), Row(r=2)]


def test_map_functions_take_a_ddl_schema(session):
    df = session.createDataFrame([(1, 1.0), (1, 2.0), (2, 3.0)], ["k", "v"])
    schema = "k long, v double"
    grouped = df.groupBy("k")
    for result in (
        df.mapInPandas(lambda frames: frames, schema),
        df.mapInArrow(lambda batches: batches, schema),
        grouped.applyInPandas(lambda pdf: pdf, schema),
        grouped.applyInArrow(lambda table: table, schema),
        grouped.cogroup(grouped).applyInPandas(lambda left, right: left, schema),
        grouped.cogroup(grouped).applyInArrow(lambda left, right: left, schema),
    ):
        assert sorted(tuple(r) for r in result.collect()) == [(1, 1.0), (1, 2.0), (2, 3.0)]


def test_map_functions_run_in_the_stage_of_their_resource_profile(session):
    # As PySpark's, mapInPandas and mapInArrow send their resource profile, which the server
    # creates: a server that does not support it, as one in local mode, rejects it, and
    # otherwise the stage runs with the profile, of the id that the server gave it.
    from pyspark.errors import PySparkException
    from pyspark.resource import ResourceProfileBuilder, TaskResourceRequests

    for name in ("mapInPandas", "mapInArrow"):
        profile = ResourceProfileBuilder().require(TaskResourceRequests().cpus(1)).build
        try:
            mapped = getattr(session.range(3), name)(lambda it: it, "id long", profile=profile)
            rows = mapped.collect()
        except PySparkException as error:
            assert "ResourceProfiles are only supported" in str(error)
        else:
            assert sorted(r.id for r in rows) == [0, 1, 2]
            assert isinstance(profile._id, int)
    assert session.range(3).mapInPandas(lambda it: it, "id long", profile=None).count() == 3


def test_transform_with_state_in_pandas_takes_a_ddl_output_type(session, tmp_path):
    import pandas as pd
    from pyspark.sql.streaming.stateful_processor import StatefulProcessor

    class Counter(StatefulProcessor):
        def init(self, handle):
            self.count = handle.getValueState("count", "c long")

        def handleInputRows(self, key, rows, timerValues):
            total = sum(len(pdf) for pdf in rows)
            self.count.update((total,))
            yield pd.DataFrame({"k": [key[0]], "c": [total]})

        def close(self):
            pass

    confs = {
        "spark.sql.streaming.stateStore.providerClass":
            "org.apache.spark.sql.execution.streaming.state.RocksDBStateStoreProvider",
        # A state store per shuffle partition; the default 200 take half a minute.
        "spark.sql.shuffle.partitions": "2",
    }
    for key, value in confs.items():
        session.conf.set(key, value)
    try:
        source = str(tmp_path / "source")
        session.createDataFrame([(1,), (1,), (2,)], "k long").write.parquet(source)
        query = (
            session.readStream.schema("k long").parquet(source).groupBy("k")
            .transformWithStateInPandas(Counter(), "k long, c long", "Update", "None")
            .writeStream.format("memory").queryName("tws_ddl_output").outputMode("update")
            .option("checkpointLocation", str(tmp_path / "checkpoint"))
            .trigger(availableNow=True).start()
        )
        try:
            query.awaitTermination(120)
            assert query.exception() is None
            rows = session.table("tws_ddl_output").collect()
            assert sorted(tuple(r) for r in rows) == [(1, 2), (2, 1)]
        finally:
            query.stop()
    finally:
        for key in confs:
            session.conf.unset(key)


def test_transform_with_state_returns_the_rows_of_its_output_schema(session, tmp_path):
    # As PySpark's, transformWithState sends its output schema, a DDL string or a
    # StructType, which the server reads as the schema of the rows of the processor.
    from pyspark.sql.streaming.stateful_processor import StatefulProcessor

    class Counter(StatefulProcessor):
        def init(self, handle):
            pass

        def handleInputRows(self, key, rows, timerValues):
            yield Row(k=key[0], c=sum(1 for _ in rows))

        def close(self):
            pass

    confs = {
        "spark.sql.streaming.stateStore.providerClass":
            "org.apache.spark.sql.execution.streaming.state.RocksDBStateStoreProvider",
        # A state store per shuffle partition; the default 200 take half a minute.
        "spark.sql.shuffle.partitions": "2",
    }
    for key, value in confs.items():
        session.conf.set(key, value)
    try:
        source = str(tmp_path / "source")
        session.createDataFrame([(1,), (1,), (2,)], "k long").write.parquet(source)
        struct = StructType([StructField("k", LongType()), StructField("c", LongType())])
        for name, schema in [("ddl", "k long, c long"), ("struct", struct)]:
            counted = (
                session.readStream.schema("k long").parquet(source).groupBy("k")
                .transformWithState(Counter(), schema, "Update", "None")
            )
            assert counted.schema == struct
            query = (
                counted.writeStream.format("memory").queryName(f"tws_rows_{name}")
                .outputMode("update")
                .option("checkpointLocation", str(tmp_path / f"checkpoint_{name}"))
                .trigger(availableNow=True).start()
            )
            try:
                query.awaitTermination(120)
                assert query.exception() is None
                rows = session.table(f"tws_rows_{name}").collect()
                assert sorted(tuple(r) for r in rows) == [(1, 2), (2, 1)]
            finally:
                query.stop()
    finally:
        for key in confs:
            session.conf.unset(key)


def _arrow_table():
    return pa.table(
        {
            "a": pa.array([1, 2, None], pa.int64()),
            "b": pa.array(["x", None, "z"]),
            "ts": pa.array(
                [datetime(2020, 1, 1, 12), None, datetime(2021, 6, 1)], pa.timestamp("us")
            ),
            "tz": pa.array(
                [datetime(2020, 1, 1, 12), None, datetime(2021, 6, 1)],
                pa.timestamp("us", tz="America/Los_Angeles"),
            ),
            "d": pa.array([date(2020, 1, 1), None, date(2021, 1, 1)]),
            "f": pa.array([1.5, None, 2.5]),
            "l": pa.array([[1, 2], None, []], pa.list_(pa.int32())),
        }
    )


def _rows(df):
    # Timestamps as strings in the session time zone, so that the client's does not matter.
    columns = [
        F.col(f.name).cast("string") if isinstance(f.dataType, TimestampType) else F.col(f.name)
        for f in df.schema.fields
    ]
    return [tuple(r) for r in df.select(*columns).collect()]


ARROW_TABLE_ROWS = [
    (1, "x", "2020-01-01 12:00:00", "2020-01-01 12:00:00", date(2020, 1, 1), 1.5, [1, 2]),
    (2, None, None, None, None, None, None),
    (None, "z", "2021-06-01 00:00:00", "2021-06-01 00:00:00", date(2021, 1, 1), 2.5, []),
]


@pytest.mark.parametrize(
    "schema, expected",
    [
        (None, "struct<a:bigint,b:string,ts:timestamp,tz:timestamp,d:date,f:double,l:array<int>>"),
        (
            ["x", "y"],
            "struct<x:bigint,y:string,_3:timestamp,_4:timestamp,_5:date,_6:double,_7:array<int>>",
        ),
        (
            "a int, b string, ts timestamp, tz timestamp, d date, f double, l array<int>",
            "struct<a:int,b:string,ts:timestamp,tz:timestamp,d:date,f:double,l:array<int>>",
        ),
        (
            StructType(
                [
                    StructField("a", IntegerType()),
                    StructField("b", StringType()),
                    StructField("ts", TimestampType()),
                    StructField("tz", TimestampType()),
                    StructField("d", DateType()),
                    StructField("f", DoubleType()),
                    StructField("l", ArrayType(IntegerType())),
                ]
            ),
            "struct<a:int,b:string,ts:timestamp,tz:timestamp,d:date,f:double,l:array<int>>",
        ),
    ],
)
def test_create_dataframe_from_arrow_table(session, schema, expected):
    session.conf.set("spark.sql.session.timeZone", "UTC")
    if schema is None:
        df = session.createDataFrame(_arrow_table())
    else:
        df = session.createDataFrame(_arrow_table(), schema)
    assert df.schema.simpleString() == expected
    assert _rows(df) == ARROW_TABLE_ROWS


def test_add_artifacts_reach_udfs_as_pyspark(session, tmp_path):
    import uuid
    import zipfile

    tag = uuid.uuid4().hex[:8]
    module, note, bundle = f"mod_{tag}", f"note_{tag}.txt", f"bundle_{tag}.zip"
    (tmp_path / f"{module}.py").write_text("def f():\n    return 'from the module'\n")
    (tmp_path / note).write_text("file content")
    with zipfile.ZipFile(tmp_path / bundle, "w") as archive:
        archive.writestr("inner.txt", "inside")
    session.addArtifacts(str(tmp_path / f"{module}.py"), pyfile=True)
    session.addArtifacts(str(tmp_path / note), file=True)
    session.addArtifacts(str(tmp_path / bundle) + f"#alias_{tag}", archive=True)

    @F.udf("string")
    def read(_):
        import importlib
        import os

        imported = importlib.import_module(module)
        # The directory of the session's files, which holds the module. Not SparkFiles: a
        # worker started with SPARK_SKIP_CONNECT_COMPAT_TESTS, as in CI, does not set it up.
        root = os.path.dirname(imported.__file__)
        return "|".join([
            imported.f(),
            open(os.path.join(root, note)).read(),
            open(os.path.join(root, f"alias_{tag}", "inner.txt")).read(),
        ])

    assert session.range(1).select(read("id")).first()[0] == "from the module|file content|inside"


def test_foreach_foreach_partition_and_to_json_as_pyspark(session):
    # As PySpark Connect's: foreach calls f with each row, and foreachPartition with an
    # iterator of the rows of each partition, on the server; toJSON is a DataFrame of each row
    # as a JSON object string.
    import re

    from pyspark.errors import PythonException

    df = session.createDataFrame(
        [(1, "a", b"x"), (2, None, None)], "id int, name string, bin binary"
    )

    def each(row):
        raise ValueError(f"row:{row!r}")

    with pytest.raises(PythonException, match=re.escape("row:Row(id=1, name='a', bin=b'x')")):
        df.where("id = 1").foreach(each)

    def partition(rows):
        raise ValueError(f"rows:{list(rows)!r}")

    with pytest.raises(
        PythonException,
        match=re.escape("rows:[Row(id=1, name='a', bin=b'x'), Row(id=2, name=None, bin=None)]"),
    ):
        df.coalesce(1).foreachPartition(partition)
    assert df.foreach(lambda row: None) is None
    assert df.foreachPartition(lambda rows: None) is None

    json = df.toJSON()
    assert json.columns == ["value"]
    assert json.collect() == [Row(value='{"id":1,"name":"a","bin":"eA=="}'), Row(value='{"id":2}')]


def test_clone_session_copies_the_state_of_the_session_as_pyspark(session):
    # As PySpark's: the clone has the configurations and the temporary views of the session,
    # of an id of its own or of the one given, and changes to either do not show in the other.
    import uuid

    session.conf.set("spark.sql.shuffle.partitions", "7")
    session.range(2).createOrReplaceTempView("cloned_view")
    try:
        clone = session.cloneSession()
        assert clone.session_id != session.session_id
        assert clone.conf.get("spark.sql.shuffle.partitions") == "7"
        assert clone.table("cloned_view").count() == 2
        clone.conf.set("spark.sql.shuffle.partitions", "9")
        assert session.conf.get("spark.sql.shuffle.partitions") == "7"
        new_id = str(uuid.uuid4())
        assert session.cloneSession(new_id).session_id == new_id
    finally:
        session.conf.unset("spark.sql.shuffle.partitions")


def test_execution_info_is_that_of_each_dataframe_as_pyspark(session):
    a = session.range(3)
    assert a.executionInfo is None
    a.collect()
    assert a.executionInfo is not None
    # Another DataFrame, and one that an action runs derived from it, have none.
    b = session.range(5)
    b.count()
    b.first()
    assert b.executionInfo is None and a.select("id").executionInfo is None
    for run in (
        lambda df: df.toArrow(),
        lambda df: df.toPandas(),
        lambda df: df.createOrReplaceTempView("execution_info_view"),
        lambda df: df.write.mode("overwrite").format("noop").save(),
    ):
        df = session.range(2)
        run(df)
        assert df.executionInfo is not None
        assert df.executionInfo.operation_id != a.executionInfo.operation_id


def test_registered_udtfs_are_called_from_sql_as_pyspark(session):
    # As PySpark's spark.udtf.register: the server knows the UDTF by its name.
    @F.udtf(returnType="x int")
    class Count:
        def eval(self, n):
            for i in range(n):
                yield (i,)

    @F.arrow_udtf(returnType=StructType([StructField("y", LongType())]))
    class Doubled:
        def eval(self, x):
            import pyarrow.compute as pc

            yield pa.table({"y": pc.multiply(x, 2)})

    assert session.udtf.register("count_udtf", Count) is Count
    assert session.sql("SELECT * FROM count_udtf(2)").collect() == [Row(x=0), Row(x=1)]
    session.udtf.register("doubled_udtf", Doubled)
    assert session.sql("SELECT * FROM doubled_udtf(21L)").collect() == [Row(y=42)]


def test_catalog_and_stream_start_errors_carry_their_operation_id(session):
    for run in (
        lambda: session.catalog.setCurrentDatabase("no_such_database_60960"),
        lambda: session.readStream.format("rate").load().writeStream.format("no_such_sink_60960").start(),
    ):
        with pytest.raises(Exception) as error:
            run()
        assert isinstance(error.value.operation_id, str) and error.value.operation_id


def test_create_dataframe_from_empty_arrow_table(session):
    empty = pa.table({"a": pa.array([], pa.int64()), "b": pa.array([], pa.string())})
    df = session.createDataFrame(empty, "a long, b string")
    assert df.schema.simpleString() == "struct<a:bigint,b:string>"
    assert df.count() == 0
    # Without types, an empty table has no schema to infer, as an empty list does not.
    for args in ((empty,), (empty, ["x", "y"])):
        with pytest.raises(PySparkValueError) as error:
            session.createDataFrame(*args)
        assert error.value.getCondition() == "CANNOT_INFER_EMPTY_SCHEMA"


_Pair = namedtuple("_Pair", ["x", "y"])


def test_grouped_aggregations_name_and_take_columns_as_pyspark(session):
    # As PySpark's: count is named count, the numeric aggregations of no columns aggregate
    # every numeric column, and df.agg takes a dict.
    from pyspark.errors import PySparkTypeError

    df = session.createDataFrame([(1, "b", 2.0), (2, "b", 3.0)], "x int, k string, v double")
    grouped = df.groupBy("k")
    assert grouped.count().columns == ["k", "count"]
    assert grouped.count().orderBy("count").collect() == [Row(k="b", count=2)]
    assert grouped.sum().collect() == [Row(k="b", **{"sum(x)": 3, "sum(v)": 5.0})]
    assert grouped.min().columns == ["k", "min(x)", "min(v)"]
    assert df.groupBy().count().collect() == [Row(count=2)]
    assert df.agg({"x": "max"}).collect() == [Row(**{"max(x)": 2})]
    with pytest.raises(PySparkTypeError) as error:
        grouped.avg("k")
    assert error.value.getMessageParameters() == {"invalid_columns": "['k']"}


@pytest.mark.parametrize(
    "data, schema, columns, rows",
    [
        ([(1, 2)], None, ["_1", "_2"], [(1, 2)]),
        ([[1, 2]], ["a"], ["a", "_2"], [(1, 2)]),
        ([(1, 2), (3, 4)], ["a"], ["a", "_2"], [(1, 2), (3, 4)]),
        ([(1, 2)], [], ["_1", "_2"], [(1, 2)]),
        ([(1, 2)], ("a", "b"), ["a", "b"], [(1, 2)]),
        ([Row(x=1, y=2)], ["a", "b"], ["a", "b"], [(1, 2)]),
        ([{"y": 2, "x": 1}], ["a", "b"], ["a", "b"], [(1, 2)]),
        # A namedtuple is named by its fields, as a Row is.
        ([_Pair(1, 2), _Pair(3, 4)], None, ["x", "y"], [(1, 2), (3, 4)]),
        ([_Pair(1, 2)], ["a", "b"], ["a", "b"], [(1, 2)]),
    ],
)
def test_create_dataframe_rows_column_names(session, data, schema, columns, rows):
    df = session.createDataFrame(data, schema)
    assert df.columns == columns
    assert [tuple(r) for r in df.collect()] == rows
    assert df.isLocal() is (schema is None)


@pytest.mark.parametrize(
    "frame, schema, columns",
    [
        ({"a": [1, 2], "b": ["x", "y"]}, None, ["a", "b"]),
        ({"a": [1, 2], "b": ["x", "y"]}, ["c"], ["c", "_2"]),
        ({"a": [1, 2], "b": ["x", "y"]}, "a long, b string", ["a", "b"]),
        ({0: [1, 2], 1: ["x", "y"]}, None, ["0", "1"]),
    ],
)
def test_create_dataframe_pandas_column_names(session, frame, schema, columns):
    pd = pytest.importorskip("pandas")
    df = session.createDataFrame(pd.DataFrame(frame), schema)
    assert df.columns == columns
    assert [tuple(r) for r in df.collect()] == [(1, "x"), (2, "y")]
    assert not df.isLocal()


class _Point:
    def __init__(self, b, a):
        self.b = b
        self.a = a


@pytest.mark.parametrize(
    "data, schema, columns, rows",
    [
        # As pyspark, a dict row is read by name: missing keys are null, others ignored.
        ([{"b": 1, "a": "x"}], "b long, a string", ["b", "a"], [(1, "x")]),
        ([{"a": 1}], "a long, b string", ["a", "b"], [(1, None)]),
        ([{"a": 1, "c": 3}], "a long", ["a"], [(1,)]),
        # Without a schema, the fields are the rows' sorted keys, in the order first seen.
        ([{"a": 1}, {"b": 2}], None, ["a", "b"], [(1, None), (None, 2)]),
        ([{"b": 2, "a": 1}], None, ["a", "b"], [(1, 2)]),
        ([{"b": 2, "a": 1}], ["x", "y"], ["x", "y"], [(1, 2)]),
        ([{"a": 1}, {"a": 2, "b": 3}], ["x", "y"], ["x", "y"], [(1, None), (2, 3)]),
        # Other objects are read through their __dict__.
        ([_Point(2, "x")], None, ["a", "b"], [("x", 2)]),
        ([_Point(2, "x")], "a string, b long", ["a", "b"], [("x", 2)]),
    ],
)
def test_create_dataframe_dict_rows_by_name(session, data, schema, columns, rows):
    df = session.createDataFrame(data, schema)
    assert df.columns == columns
    assert [tuple(r) for r in df.collect()] == rows


@pytest.mark.parametrize(
    "data, simple_string, rows",
    [
        ([(None, 1), (2, 3)], "struct<_1:bigint,_2:bigint>", [(None, 1), (2, 3)]),
        ([{"a": None}, {"a": 1}], "struct<a:bigint>", [(None,), (1,)]),
    ],
)
def test_create_dataframe_infers_types_past_nulls(session, data, simple_string, rows):
    # A field takes the type of its first non-null value.
    df = session.createDataFrame(data)
    assert df.schema.simpleString() == simple_string
    assert [tuple(r) for r in df.collect()] == rows


@pytest.mark.parametrize("ddl", [False, True], ids=["StructType", "DDL"])
def test_reader_schema_keeps_field_metadata(session, tmp_path, ddl):
    (tmp_path / "t.csv").write_text("1,x\n2,y\n")
    path = str(tmp_path / "t.csv")
    if ddl:
        schema = "a long COMMENT 'the key', b string"
    else:
        schema = StructType(
            [StructField("a", LongType(), True, {"comment": "the key"}), StructField("b", StringType())]
        )
    for df in (session.read.schema(schema).csv(path), session.read.csv(path, schema=schema)):
        assert df.schema.fields[0].metadata == {"comment": "the key"}
        assert [tuple(r) for r in df.collect()] == [(1, "x"), (2, "y")]
    stream = session.readStream.schema(schema).csv(str(tmp_path))
    assert stream.isStreaming
    assert stream.schema.fields[0].metadata == {"comment": "the key"}


def test_subqueries_and_table_arguments_return_the_rows_of_pyspark(session):
    # As PySpark's DataFrame.scalar(), DataFrame.exists(), Column.isin(df) and asTable():
    # scalar, correlated and IN subqueries, and the rows of a DataFrame as a UDTF argument.
    from pyspark.errors import IllegalArgumentException

    t1 = session.createDataFrame([(1, 2), (3, 4)], "a int, b int").alias("t1")
    t2 = session.createDataFrame([(1, 5), (5, 6), (3, 7)], "a int, c int").alias("t2")
    largest = t2.select(F.max("c")).scalar()
    assert sorted(t1.select("a", largest.alias("m")).collect()) == [Row(a=1, m=7), Row(a=3, m=7)]
    correlated = t2.where(F.col("t2.a") == F.col("t1.a").outer())
    assert sorted(t1.select("a", correlated.select(F.max("c")).scalar().alias("m")).collect()) == [
        Row(a=1, m=5), Row(a=3, m=7)
    ]
    assert t1.where(~correlated.exists()).collect() == []
    above = t2.where(F.col("c") > t2.select(F.min("c")).scalar()).select("a")
    assert t1.where(F.col("a").isin(above)).collect() == [Row(a=3, b=4)]
    pairs = session.createDataFrame([(1, 2)], "x int, y int")
    assert t1.where(F.struct("a", "b").isin(pairs)).collect() == [Row(a=1, b=2)]

    @F.udtf(returnType="id bigint, n int")
    class Numbered:
        def __init__(self):
            self.n = 0

        def eval(self, row):
            self.n += 1
            yield row["id"], self.n

    table = session.range(4).asTable().withSinglePartition().orderBy(F.col("id").desc())
    assert sorted(Numbered(table).collect()) == [
        Row(id=0, n=4), Row(id=1, n=3), Row(id=2, n=2), Row(id=3, n=1)
    ]
    assert sorted(Numbered(row=session.range(2).asTable()).collect()) == [
        Row(id=0, n=1), Row(id=1, n=1)
    ]
    with pytest.raises(IllegalArgumentException, match="before orderBy"):
        session.range(1).asTable().orderBy("id")


def test_grouped_map_functions_take_iterators_by_their_type_hints(session):
    from typing import Iterator, Tuple

    import numpy as np
    import pandas as pd

    def scaled(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        for frame in batches:
            yield frame.assign(v=frame.v * 10)

    def shifted(key: Tuple[np.int64], batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        for frame in batches:
            yield frame.assign(v=frame.v + key[0])

    def batches(it: Iterator[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
        yield from it

    def keyed_batches(
        key: Tuple[pa.Scalar, ...], it: Iterator[pa.RecordBatch]
    ) -> Iterator[pa.RecordBatch]:
        yield from it

    df = session.createDataFrame([(1, 1.0), (1, 2.0), (2, 3.0)], ["k", "v"])
    schema = StructType([StructField("k", LongType()), StructField("v", DoubleType())])
    grouped = df.groupBy("k")
    for result, expected in (
        (grouped.applyInPandas(scaled, schema), [(1, 10.0), (1, 20.0), (2, 30.0)]),
        (grouped.applyInPandas(shifted, schema), [(1, 2.0), (1, 3.0), (2, 5.0)]),
        (grouped.applyInArrow(batches, schema), [(1, 1.0), (1, 2.0), (2, 3.0)]),
        (grouped.applyInArrow(keyed_batches, schema), [(1, 1.0), (1, 2.0), (2, 3.0)]),
    ):
        assert sorted(tuple(r) for r in result.collect()) == expected


def test_dataframe_columns_resolve_against_their_dataframe_as_pyspark(session):
    # As PySpark Connect's: df[name] and df.name refer to the columns of their DataFrame by its
    # plan ID, so a join selects the columns of each side, and an ambiguous self-join of a
    # derived DataFrame raises PySpark's error.
    left = session.createDataFrame([(1, "a"), (2, "b")], "id int, v string")
    right = session.createDataFrame([(1, "x"), (3, "y")], "id int, w string")
    joined = left.join(right, left["id"] == right["id"])
    assert joined.select(left["id"], right["w"]).collect() == [Row(id=1, w="x")]
    assert joined.select(left.v, right.w).collect() == [Row(v="a", w="x")]
    assert joined.select(right["*"]).collect() == [Row(id=1, w="x")]
    assert joined.select(left.colRegex("`v`")).collect() == [Row(v="a")]
    base = session.range(3)
    derived = base.withColumn("x", F.col("id") * 2)
    with pytest.raises(AnalysisException) as error:
        base.join(derived, base["id"] == derived["id"]).select(base["id"]).collect()
    assert error.value.getCondition() == "AMBIGUOUS_COLUMN_REFERENCE"


def test_collect_reads_a_result_across_stream_ends(session):
    # Takes several response streams when the server's senderMaxStreamSize is small.
    n = 300_000
    assert len(session.range(n).collect()) == n
    assert session.range(n).toArrow().num_rows == n


def test_time_values_round_trip_as_pyspark(session):
    # As PySpark's: TIME values are sent, and collected, as datetime.time, when the session
    # enables the TIME type.
    import datetime

    t = datetime.time(1, 2, 3, 456789)
    session.conf.set("spark.sql.timeType.enabled", "true")
    try:
        assert session.createDataFrame([(t,)]).collect() == [Row(_1=t)]
        assert session.createDataFrame([(t,)], "t time").collect() == [Row(t=t)]
        assert session.range(1).select(F.lit(t).alias("t")).collect() == [Row(t=t)]
        assert session.sql("SELECT TIME'12:34:56.789' AS t").collect() == [
            Row(t=datetime.time(12, 34, 56, 789000))
        ]
        # Without it, the server does not support the type, as for PySpark. It is set to
        # false, as the server enables it by default when SPARK_TESTING is set.
        session.conf.set("spark.sql.timeType.enabled", "false")
        with pytest.raises(AnalysisException) as error:
            session.createDataFrame([(t,)], "t time").collect()
        assert error.value.getCondition() == "UNSUPPORTED_TIME_TYPE"
    finally:
        session.conf.unset("spark.sql.timeType.enabled")


def test_timedelta_is_a_day_time_interval_as_pyspark(session):
    import datetime

    import pyarrow as pa

    values = [datetime.timedelta(days=1, seconds=2, microseconds=3),
              datetime.timedelta(seconds=-1, microseconds=5)]
    df = session.range(1).select(*[F.lit(v).alias(f"t{i}") for i, v in enumerate(values)])
    assert df.dtypes == [("t0", "interval day to second"), ("t1", "interval day to second")]
    assert tuple(df.first()) == tuple(values)
    df = session.createDataFrame([(v,) for v in values] + [(None,)], ["t"])
    assert df.dtypes == [("t", "interval day to second")]
    assert [r.t for r in df.collect()] == values + [None]
    # As PyArrow's conversion to duration[us]: an int, or a float truncated, is microseconds.
    df = session.createDataFrame([(5,), (1.5,)], "d interval day to second")
    assert [r.d for r in df.collect()] == [
        datetime.timedelta(microseconds=5), datetime.timedelta(microseconds=1)]
    with pytest.raises(pa.ArrowTypeError, match="Expected integer, got bool"):
        session.createDataFrame([(True,)], "d interval day to second")
    with pytest.raises(pa.ArrowTypeError, match="object of type <class 'str'> cannot be converted to int"):
        session.createDataFrame([("x",)], "d interval day to second")


def test_errors_carry_the_dataframe_query_contexts_of_pyspark(session):
    # As PySpark's: an error in an expression that col() or a Column method built reports its
    # DataFrameQueryContext, of the function and of its call site in this file.
    import inspect

    from pyspark.errors import ArithmeticException, QueryContextType

    def contexts(df):
        with pytest.raises((AnalysisException, ArithmeticException)) as error:
            df.collect()
        return [
            (c.contextType(), c.fragment(), c.callSite().rsplit("/", 1)[-1])
            for c in error.value.getQueryContext()
        ]

    here = f"{os.path.basename(__file__)}:{{}}"
    line = inspect.currentframe().f_lineno
    assert contexts(session.range(1).select("no_such_col")) == [
        (QueryContextType.DataFrame, "col", here.format(line + 1))
    ]
    assert contexts(session.range(1).select(F.col("nope") + 1)) == [
        (QueryContextType.DataFrame, "col", here.format(line + 4))
    ]
    session.conf.set("spark.sql.ansi.enabled", "true")
    assert contexts(session.range(1).select(F.lit(1) / F.lit(0))) == [
        (QueryContextType.DataFrame, "__truediv__", here.format(line + 8))
    ]
    session.conf.unset("spark.sql.ansi.enabled")
    # repartitionById makes a column of a name with col(), while a UDF call refers to a
    # column by its name without an origin.
    assert contexts(session.range(1).repartitionById(2, "nope")) == [
        (QueryContextType.DataFrame, "col", here.format(line + 14))
    ]
    udf = F.udf(lambda x: x, "long")
    assert contexts(session.range(1).select(udf("nope"), udf(x="nope2"))) == []
    # sampleBy takes a column of a name with col(), as PySpark's F._to_col.
    assert contexts(session.range(1).sampleBy("nope", {0: 1.0})) == [
        (QueryContextType.DataFrame, "col", here.format(line + 20))
    ]
    # unpivot takes its columns of names with col(), as PySpark's F._to_col.
    assert contexts(session.range(1).unpivot("nope", ["id"], "v", "w")) == [
        (QueryContextType.DataFrame, "col", here.format(line + 24))
    ]


def test_server_errors_carry_their_condition_and_details(session):
    from pyspark.errors import AnalysisException, ArithmeticException, ParseException
    from pyspark.errors.exceptions.connect import SparkConnectGrpcException, SQLQueryContext

    with pytest.raises(ParseException) as error:
        session.sql("SELEC 1").collect()
    e = error.value
    assert isinstance(e, SparkConnectGrpcException)
    assert e.getCondition() == "PARSE_SYNTAX_ERROR"
    assert e.getSqlState() == "42601"
    assert e.getMessageParameters() == {"error": "'SELEC'", "hint": ""}
    # The server's message, which names the condition once.
    assert str(e).count("[PARSE_SYNTAX_ERROR]") == 1
    assert e.getStackTrace().startswith("org.apache.spark.sql.catalyst.parser.ParseException")

    with pytest.raises(AnalysisException) as error:
        session.table("no_such_table_xyz").collect()
    e = error.value
    assert e.getCondition() == "TABLE_OR_VIEW_NOT_FOUND"
    assert e.getSqlState() == "42P01"
    assert list(e.getMessageParameters()) == ["relationName", "searchPath"]
    assert e.getMessageParameters()["relationName"] == "`no_such_table_xyz`"

    with pytest.raises(ArithmeticException) as error:
        session.sql("SELECT 1/0").collect()
    e = error.value
    assert e.getCondition() == "DIVIDE_BY_ZERO"
    [context] = e.getQueryContext()
    assert isinstance(context, SQLQueryContext)
    assert context.fragment() == "1/0"
    # spark.sql.connect.serverStacktrace.enabled is true by default.
    assert "\n\nJVM stacktrace:\norg.apache.spark.SparkArithmeticException" in str(e)


def test_apply_in_pandas_with_state_runs_as_pyspark(session, tmp_path):
    import pandas as pd

    def count(key, frames, state):
        total = sum(len(frame) for frame in frames)
        state.update((total,))
        yield pd.DataFrame({"k": [key[0]], "c": [total]})

    output = StructType([StructField("k", LongType()), StructField("c", LongType())])
    state = StructType([StructField("c", LongType())])
    df = session.createDataFrame([(1,), (1,), (2,)], "k long")
    # The server takes the function in a streaming query only, as with PySpark.
    with pytest.raises(Exception, match="applyInPandasWithState is unsupported in batch query"):
        df.groupBy("k").applyInPandasWithState(count, output, state, "append", "NoTimeout").collect()

    # A state store per shuffle partition; the default 200 take half a minute.
    session.conf.set("spark.sql.shuffle.partitions", "2")
    try:
        source = str(tmp_path / "source")
        df.write.parquet(source)
        query = (
            session.readStream.schema("k long").parquet(source).groupBy("k")
            .applyInPandasWithState(count, output, state, "update", "NoTimeout")
            .writeStream.format("memory").queryName("apply_in_pandas_with_state")
            .outputMode("update").option("checkpointLocation", str(tmp_path / "checkpoint"))
            .trigger(availableNow=True).start()
        )
        try:
            query.awaitTermination(120)
            assert query.exception() is None
            rows = session.table("apply_in_pandas_with_state").collect()
            assert sorted(tuple(r) for r in rows) == [(1, 2), (2, 1)]
        finally:
            query.stop()
    finally:
        session.conf.unset("spark.sql.shuffle.partitions")


def test_variant_spatial_and_user_defined_values_round_trip_as_pyspark(session):
    # As PySpark's: createDataFrame sends a VariantVal, a Geometry, a Geography and a value of
    # a user-defined type, and collects them as they are.
    from pyspark.sql.types import Geography, Geometry
    from pyspark.testing.objects import ExamplePoint

    wkb = bytes.fromhex("0101000000000000000000F03F0000000000000040")
    variant = session.sql("""SELECT parse_json('{"a": 1}') AS v""").first()[0]
    df = session.createDataFrame([
        (variant, Geometry.fromWKB(wkb, 4326), Geography.fromWKB(wkb, 4326), ExamplePoint(1.0, 2.0))
    ])
    assert df.schema.simpleString() == (
        "struct<_1:variant,_2:geometry(4326),_3:geography(4326),_4:udt>"
    )
    row = df.first()
    assert row[0].toJson() == '{"a":1}'
    assert row[1] == Geometry.fromWKB(wkb, 4326) and row[2] == Geography.fromWKB(wkb, 4326)
    assert row[3] == ExamplePoint(1.0, 2.0)
    # A collected GEOMETRY is a Geometry.
    geometry = session.sql(
        "SELECT ST_GeomFromWKB(X'0101000000000000000000F03F0000000000000040') AS g"
    ).first()[0]
    assert geometry == Geometry.fromWKB(wkb, 0)


def test_create_dataframe_types_pandas_columns_as_pyspark(session):
    from decimal import Decimal

    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")
    cases = [
        # A column without values takes the type of its dtype.
        ({"id": [1, 2], "s": [np.nan, np.nan]}, "struct<id:bigint,s:double>", [(1, None), (2, None)]),
        ({"id": [1, 2], "s": [None, None]}, "struct<id:bigint,s:void>", [(1, None), (2, None)]),
        ({"id": [1, 2], "t": pd.to_datetime([None, None])}, "struct<id:bigint,t:timestamp>",
         [(1, None), (2, None)]),
        # A column of Decimals takes the precision and scale of all its values.
        ({"d": [Decimal("1.5"), Decimal("2.25")]}, "struct<d:decimal(3,2)>",
         [(Decimal("1.50"),), (Decimal("2.25"),)]),
        ({"d": [Decimal("1E+2"), Decimal("0.001")]}, "struct<d:decimal(6,3)>",
         [(Decimal("100.000"),), (Decimal("0.001"),)]),
        ({"d": [Decimal("1.5"), None]}, "struct<d:decimal(2,1)>", [(Decimal("1.5"),), (None,)]),
    ]
    for data, schema, rows in cases:
        df = session.createDataFrame(pd.DataFrame(data))
        assert df.schema.simpleString() == schema, data
        assert [tuple(r) for r in df.collect()] == rows, data


def test_collected_rows_behave_as_pyspark_rows(session):
    import pickle

    from pyspark.sql import Row

    rows = session.range(1).collect()
    assert repr(rows) == "[Row(id=0)]"
    assert rows[0] == Row(id=0) and hash(rows[0]) == hash(Row(id=0))
    assert repr(pickle.loads(pickle.dumps(rows[0]))) == "Row(id=0)"
    row = session.range(1).selectExpr(
        "named_struct('b', 1, 'c', array(1, 2)) AS a",
        "date'2020-01-02' AS d",
        "1.50BD AS n",
        "CAST(NULL AS INT) AS z",
    ).first()
    assert repr(row) == (
        "Row(a=Row(b=1, c=[1, 2]), d=datetime.date(2020, 1, 2), n=Decimal('1.50'), z=None)"
    )
    assert row.asDict(recursive=True)["a"] == {"b": 1, "c": [1, 2]}
    # Columns of the same name, as of a join, keep both values through pickling.
    duplicate = session.range(1).selectExpr("1 AS a", "2 AS a").first()
    assert repr(duplicate) == "Row(a=1, a=2)" and duplicate.asDict() == {"a": 2}
    assert repr(pickle.loads(pickle.dumps(duplicate))) == "Row(a=1, a=2)"


def test_sample_sort_group_by_drop_and_select_expr_as_pyspark(session):
    # As PySpark's: sample takes a fraction and a seed positionally, sort and groupBy take the
    # 1-based ordinals of columns, drop takes Columns, and selectExpr a list of expressions.
    df = session.createDataFrame(
        [(1, "b", 2.0), (2, "a", 1.0), (3, "c", 3.0)], "x int, k string, v double"
    )
    assert [r.k for r in df.sort(2).collect()] == ["a", "b", "c"]
    assert [r.k for r in df.orderBy(-2).collect()] == ["c", "b", "a"]
    one_partition = df.coalesce(1).sortWithinPartitions(3, ascending=False)
    assert [r.v for r in one_partition.collect()] == [3.0, 2.0, 1.0]
    grouped = df.groupBy(2).agg(F.max("x").alias("m")).orderBy("k").collect()
    assert grouped == [Row(k="a", m=2), Row(k="b", m=1), Row(k="c", m=3)]
    assert df.drop(df.x, "k").columns == ["v"]
    assert df.drop(F.col("v")).columns == ["x", "k"]
    assert df.selectExpr(["x", "k AS name"]).columns == ["x", "name"]
    assert df.sample(0.5, 42).collect() == df.sample(fraction=0.5, seed=42).collect()


def test_column_attributes_select_struct_fields_and_slices_substrings(session):
    df = session.sql("SELECT named_struct('x', 1, 'y', named_struct('z', 'q')) AS s, 'abcdef' AS t")
    out = df.select(df.s.x, df.s.y.z, F.col("s").x.alias("x2"), df.t[2:3], F.col("t")[F.lit(1):F.lit(2)])
    assert out.columns == ["s.x", "s.y.z", "x2", "substr(t, 2, 3)", "substr(t, 1, 2)"]
    assert tuple(out.first()) == (1, "q", 1, "bcd", "ab")


def test_collected_rows_keep_their_values_as_pyspark(session):
    import pickle

    # A row holds the same values for as long as it lives, as a tuple does, so it equals
    # itself with a NaN or a VARIANT value, and keeps its hash.
    row = session.sql("SELECT double('NaN') x, 1 y").first()
    assert row == row and hash(row) == hash(row)
    assert row[0] is row[0] and row.x is row[0]
    variant = session.sql("""SELECT parse_json('{"a": 1}') v""").first()
    assert variant == variant and hash(variant) == hash(variant)
    nested = session.sql("SELECT named_struct('a', double('NaN')) s").first()
    assert nested == nested and nested.s is nested.s
    # Rows of VARIANT values pickle, as in PySpark.
    rows = pickle.loads(pickle.dumps(session.sql("SELECT parse_json('1') v, 2 n").collect()))
    assert (rows[0].v.toJson(), rows[0].n) == ("1", 2)


def test_getitem_indexes_filters_and_selects_as_pyspark(session):
    from pyspark.errors import PySparkTypeError

    df = session.createDataFrame([(1, "a"), (2, "b"), (3, "c")], ["id", "s"])
    assert df.select(df[0], df[-1], df[True]).columns == ["id", "s", "s"]
    assert df[df["id"] > 1].collect() == [Row(id=2, s="b"), Row(id=3, s="c")]
    assert df[["s"]].collect() == [Row(s="a"), Row(s="b"), Row(s="c")]
    assert df[("s", "id")].columns == ["s", "id"]
    assert df[[["id"]]].columns == ["id"]
    with pytest.raises(IndexError):
        df[2]
    with pytest.raises(PySparkTypeError) as error:
        df[1.5]
    assert error.value.getCondition() == "NOT_EXPECTED_TYPE"


def test_plans_name_the_udfs_of_map_functions_after_the_functions_as_pyspark(session, capfd):
    # As PySpark's: the server's plan names the UDF of a map function after the function.
    def my_arrow_func(iterator):
        yield from iterator

    def my_grouped_func(pdf):
        return pdf

    capfd.readouterr()
    session.range(1).mapInArrow(my_arrow_func, "id long").explain()
    session.range(1).groupBy("id").applyInPandas(my_grouped_func, "id long").explain()
    out = capfd.readouterr().out
    assert "MapInArrow my_arrow_func(" in out
    assert "FlatMapGroupsInPandas [id#" in out and "], my_grouped_func(" in out


def test_sql_runs_statements_eagerly(session):
    # sql() sends the statement at once: a command takes effect without an action, and the
    # DataFrame of its result is the local relation the server returned.
    session.sql("CREATE OR REPLACE TEMP VIEW eager_sql_v AS SELECT 1 AS x")
    assert [tuple(r) for r in session.table("eager_sql_v").collect()] == [(1,)]
    session.sql("SET spark.connect.rust.test.eager=on")
    assert session.conf.get("spark.connect.rust.test.eager") == "on"
    databases = session.sql("SHOW DATABASES")
    assert databases.isLocal()
    assert "default" in [r[0] for r in databases.collect()]
    # A query still runs at the action.
    query = session.sql("SELECT 1 AS x")
    assert not query.isLocal()
    assert [tuple(r) for r in query.collect()] == [(1,)]


def test_lists_and_numpy_arrays_are_arrays_as_pyspark(session):
    np = pytest.importorskip("numpy")

    df = session.range(1).select(
        F.lit([1, 2]).alias("l"), F.lit(np.array([1, 2])).alias("n"),
        F.lit(np.array([1.5], dtype=np.float32)).alias("f"), F.lit(np.array(["x"])).alias("s"),
        F.array_contains(F.lit([[1], [2]]), [1]).alias("c"), F.array(["id", "id"]).alias("a"),
        F.struct(["id"]).alias("t"), F.when(F.lit(True), [3]).alias("w"),
    )
    assert df.dtypes == [
        ("l", "array<int>"), ("n", "array<bigint>"), ("f", "array<float>"), ("s", "array<string>"),
        ("c", "boolean"), ("a", "array<bigint>"), ("t", "struct<id:bigint>"), ("w", "array<int>"),
    ]
    assert [tuple(r) for r in df.collect()] == [
        ([1, 2], [1, 2], [1.5], ["x"], True, [0, 0], Row(id=0), [3])]


def test_sql_fails_at_a_statement_that_does_not_parse(session):
    with pytest.raises(AnalysisException, match="PARSE_SYNTAX_ERROR"):
        session.sql("SELEC 1")


def test_sql_formats_the_query_with_keyword_arguments_as_pyspark(session):
    from decimal import Decimal

    df = session.range(2)
    other = session.createDataFrame([(0, "a"), (1, "b")], "id long, s string")
    cases = [
        # A value becomes its string, a string its SQL literal, and a column reference its
        # name, as `str.format` with PySpark's formatter does.
        (lambda: session.sql("SELECT {x} AS v", x=1), [(1,)]),
        (lambda: session.sql("SELECT {s} AS s", s="it's a \\ test"), [("it's a \\ test",)]),
        (lambda: session.sql("SELECT {c} AS c FROM {df}", c=F.col("id"), df=df), [(0,), (1,)]),
        (lambda: session.sql("SELECT {df.id} AS c FROM {df}", df=df), [(0,), (1,)]),
        (lambda: session.sql("SELECT {x:.2f} AS v", x=1.5), [(Decimal("1.50"),)]),
        # A DataFrame is a reference sent with the query, one for each DataFrame.
        (lambda: session.sql("SELECT count(*) AS n FROM {df}", df=session.range(3)), [(3,)]),
        (lambda: session.sql("SELECT * FROM {a} UNION ALL SELECT * FROM {a}", a=df),
         [(0,), (1,), (0,), (1,)]),
        (lambda: session.sql(
            "SELECT a.id, b.s FROM {a} a JOIN {b} b ON a.id = b.id ORDER BY a.id", a=df, b=other),
         [(0, "a"), (1, "b")]),
        # `args` still binds parameters, and braces are only formatted with keyword arguments.
        (lambda: session.sql("SELECT {x} + :p AS v", args={"p": 1}, x=2), [(3,)]),
        (lambda: session.sql("SELECT '{{}}' AS v"), [("{{}}",)]),
        (lambda: session.sql("SELECT '{{}}' AS v, {x} AS x", x=1), [("{}", 1)]),
    ]
    for run, rows in cases:
        assert [tuple(r) for r in run().collect()] == rows


def test_show_prints_the_table_of_the_server_as_pyspark(session):
    import contextlib
    import io

    df = session.sql("SELECT id, repeat('x', 25) AS s FROM range(2)")

    def shown(**kwargs):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):  # Python's print, as PySpark's show.
            df.show(**kwargs)
        return out.getvalue()

    assert shown() == (
        "+---+--------------------+\n"
        "| id|                   s|\n"
        "+---+--------------------+\n"
        "|  0|xxxxxxxxxxxxxxxxx...|\n"
        "|  1|xxxxxxxxxxxxxxxxx...|\n"
        "+---+--------------------+\n\n")
    assert shown(n=1, truncate=5) == (
        "+---+-----+\n| id|    s|\n+---+-----+\n|  0|xx...|\n+---+-----+\nonly showing top 1 row\n")
    assert shown(truncate=False).splitlines()[3] == "|0  |" + "x" * 25 + "|"
    assert shown(n=1, vertical=True) == (
        "-RECORD 0-------------------\n"
        " id  | 0                    \n"
        " s   | xxxxxxxxxxxxxxxxx... \n"
        "only showing top 1 row\n")


def test_sql_keyword_arguments_are_not_parameters_as_pyspark(session):
    with pytest.raises(AnalysisException) as error:
        session.sql("SELECT :p AS v", p=5).collect()
    assert error.value.getCondition() == "UNBOUND_SQL_PARAMETER"


def test_explain_prints_with_python_print_as_pyspark(session):
    # As PySpark's: explain, and StreamingQuery.explain, print the plan with Python's print, to
    # sys.stdout, which redirect_stdout captures, and return None.
    import contextlib
    import io

    df = session.range(1)
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        assert df.explain() is None
        assert df.explain(mode="formatted") is None
    assert printed.getvalue().startswith("== Physical Plan ==\n")
    assert "\n(1) Range [codegen id : 1]\n" in printed.getvalue()
    assert df._explain_string(True).startswith("== Parsed Logical Plan ==\n")

    query = (
        session.readStream.format("rate").load()
        .writeStream.format("memory").queryName("explain_prints").start()
    )
    try:
        query.processAllAvailable()
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            assert query.explain(True) is None
        assert printed.getvalue().startswith("== Parsed Logical Plan ==\n")
    finally:
        query.stop()


def test_column_alias_names_every_column_and_keeps_its_metadata(session):
    df = session.sql("SELECT map('k1', 1) AS m").select(F.explode("m").alias("k", "v"))
    assert df.columns == ["k", "v"]
    assert [tuple(r) for r in df.collect()] == [("k1", 1)]
    df = session.range(1).select(F.col("id").alias("x", metadata={"k": 1, "n": [1, 2], "s": "v"}))
    assert df.schema.fields[0].metadata == {"k": 1, "n": [1, 2], "s": "v"}


def test_to_pandas_converts_the_columns_as_pyspark(session):
    # As official PySpark 4.3.0's SparkConnectClient.to_pandas: each column by its Spark type,
    # in the session time zone, with the metrics of the execution in attrs.
    from decimal import Decimal

    import pandas as pd
    from pyspark.ml.functions import array_to_vector
    from pyspark.ml.linalg import DenseVector

    df = session.sql(
        "SELECT TIMESTAMP'2024-01-02 03:04:05' AS ts, map('k', 1) AS m, "
        "named_struct('a', 1, 'b', 'x') AS st, 1.5BD AS d, array(1.5D, 2.0D) AS av"
    ).withColumn("vec", array_to_vector("av")).drop("av")
    pdf = df.toPandas()
    assert str(pdf.dtypes["ts"]) == "datetime64[ns]"
    assert pdf["ts"][0] == pd.Timestamp("2024-01-02 03:04:05")
    assert pdf["m"][0] == {"k": 1}
    assert pdf["st"][0] == {"a": 1, "b": "x"}
    assert pdf["d"][0] == Decimal("1.5")
    assert pdf["vec"][0] == DenseVector([1.5, 2.0])
    assert sorted(pdf.attrs) == ["metrics"]
    assert type(pdf.attrs["metrics"][0]).__name__ == "PlanMetrics"
    struct = df.select("st")._to_pandas(pandasStructHandlingMode="row")["st"][0]
    assert type(struct).__name__ == "Row" and struct == Row(a=1, b="x")
    session.conf.set("spark.sql.session.timeZone", "Asia/Seoul")
    try:
        local = session.sql("SELECT TIMESTAMP'2024-01-02 03:04:05' AS ts").toPandas()
        assert local["ts"][0] == pd.Timestamp("2024-01-02 03:04:05")
    finally:
        session.conf.unset("spark.sql.session.timeZone")


def test_sample_by_samples_the_fractions_of_the_strata_as_pyspark(session):
    # As PySpark's sampleBy: each stratum by its fraction, of a column name or a Column.
    df = session.createDataFrame([(1, "a"), (2, "b"), (3, "a")], "id int, k string")
    fractions = {"a": 1.0, "b": 0.0}
    for sampled in (
        df.sampleBy("k", fractions, seed=1),
        df.sampleBy(F.col("k"), fractions, seed=1),
        df.stat.sampleBy("k", fractions, seed=1),
    ):
        assert sorted(r.id for r in sampled.collect()) == [1, 3]


def test_window_order_by_a_column_sorts_nulls_first_as_pyspark(session):
    from pyspark.sql import Window

    df = session.sql("SELECT * FROM VALUES (1, CAST(NULL AS INT)), (1, 2), (1, 1) AS t(g, o)")
    # A column that is not a sort order is ascending with nulls first, as SortOrder(expr).
    for window, expected in (
        (Window.partitionBy("g").orderBy("o"), [(None, 1), (1, 2), (2, 3)]),
        (Window.partitionBy("g").orderBy(F.col("o")), [(None, 1), (1, 2), (2, 3)]),
        (Window.orderBy("o").partitionBy("g").orderBy("o"), [(None, 1), (1, 2), (2, 3)]),
        (Window.partitionBy("g").orderBy(F.col("o").desc()), [(2, 1), (1, 2), (None, 3)]),
    ):
        rows = df.select("o", F.row_number().over(window).alias("rn")).orderBy("rn").collect()
        assert [(r.o, r.rn) for r in rows] == expected


def test_create_dataframe_takes_the_ddl_of_spark_as_pyspark(session):
    # As PySpark Connect's, which parses a DDL schema on the server: nested structs, arrays and
    # maps of them, backtick-quoted names, and NOT NULL and COMMENT clauses.
    schema = (
        "st struct<a:int,b:double>, arr array<struct<c:int>>, m map<string,struct<x:int>>, "
        "`a b` int NOT NULL COMMENT 'c'"
    )
    df = session.createDataFrame([((1, 2.0), [(3,)], {"k": (4,)}, 5)], schema)
    assert df.schema.simpleString() == (
        "struct<st:struct<a:int,b:double>,arr:array<struct<c:int>>,"
        "m:map<string,struct<x:int>>,a b:int>"
    )
    assert df.collect() == [
        Row(st=Row(a=1, b=2.0), arr=[Row(c=3)], m={"k": Row(x=4)}, **{"a b": 5})
    ]
    # Collations, which the server names, and the errors of the server's parser.
    collated = session.createDataFrame([("x",)], "s STRING COLLATE UTF8_LCASE")
    assert collated.schema.simpleString() == "struct<s:string collate UTF8_LCASE>"
    from pyspark.errors import ParseException

    for ddl, condition in [("a INT,", "PARSE_SYNTAX_ERROR"), ("a BOGUS", "UNSUPPORTED_DATATYPE")]:
        with pytest.raises(ParseException) as error:
            session.createDataFrame([], ddl)
        assert error.value.getCondition() == condition
    # As do the map functions, of their schemas.
    mapped = session.range(2).mapInPandas(lambda it: it, "id long NOT NULL COMMENT 'k'")
    assert [(f.name, f.nullable, f.metadata) for f in mapped.schema.fields] == [
        ("id", False, {"comment": "k"})]
    assert mapped.collect() == [Row(id=0), Row(id=1)]


def test_functions_send_their_optional_arguments_as_pyspark(session):
    # The values of official PySpark 4.3.0; each optional argument used to be dropped.
    df = session.sql(
        "SELECT '02/01/2024' AS d, 8.0D AS x, array(2, 1, 3) AS a, 'a,b,c' AS s, "
        "'a_c' AS p, 1700000000L AS t, 2.5D AS h")
    row = df.select(
        F.to_date("d", "dd/MM/yyyy").alias("date"),
        F.log(2.0, "x").alias("log2"),
        F.sort_array("a", asc=False).alias("descending"),
        F.split("s", ",", 2).alias("split"),
        F.substr("s", F.lit(1), F.lit(2)).alias("substr"),
        F.like("p", F.lit("a!_c"), F.lit("!")).alias("like"),
        F.unix_timestamp(F.timestamp_seconds("t")).alias("seconds"),
        F.months_between(F.lit("2024-03-31"), F.lit("2024-02-28"), roundOff=False).alias("months"),
        # An explicit None is no scale, as in PySpark, rather than a NULL one.
        F.bround("h", None).alias("bround"),
    ).first()
    assert row.date == date(2024, 1, 2)
    assert row.log2 == 3.0
    assert row.descending == [3, 2, 1]
    assert row.split == ["a", "b,c"]
    assert row.substr == "a,"
    assert row.like is True
    assert row.seconds == 1700000000
    assert row.months == 1.096774193548387
    assert row.bround == 2.0
    nulls = session.sql("SELECT * FROM VALUES (1, NULL), (1, 5) AS t(g, v)")
    assert nulls.groupBy("g").agg(F.first("v", ignorenulls=True)).first()[1] == 5


def test_maps_keep_their_entries_in_order_with_keys_of_their_type(session):
    row = session.range(1).selectExpr(
        "map(2, 'b', 1, 'a') AS ints",
        "map('b', 1, 'a', 2) AS strs",
        "map(date'2020-01-02', 1) AS dates",
        "map(named_struct('a', 1), 'x') AS structs",
    ).first()
    assert list(row.ints.items()) == [(2, "b"), (1, "a")]
    assert list(row.strs.items()) == [("b", 1), ("a", 2)]
    assert row.dates == {date(2020, 1, 2): 1}
    assert row.structs == {Row(a=1): "x"}
    # A list key is unhashable, as in PySpark. (PySpark raises at collect(); this client
    # turns a row's values into Python ones when they are read.)
    with pytest.raises(TypeError, match="unhashable"):
        session.range(1).selectExpr("map(array(1), 'x') AS m").first().m
    df = session.createDataFrame([({2: "b", 1: "a"},)])
    assert df.schema.simpleString() == "struct<_1:map<bigint,string>>"
    assert list(df.first()[0].items()) == [(2, "b"), (1, "a")]


def test_decimal_literals_are_of_the_types_of_pyspark(session):
    # The types of official PySpark 4.3.0: at least the precision 10 of DecimalType().
    from decimal import Decimal

    df = session.range(1).select(
        F.lit(Decimal("1.5")).alias("a"),
        F.lit(Decimal(0)).alias("b"),
        F.lit(Decimal("123456789012.5")).alias("c"),
        F.lit(Decimal("-0.001")).alias("d"),
        (F.lit(Decimal("1.5")) * F.lit(Decimal("2.25"))).alias("product"),
    )
    assert df.dtypes == [
        ("a", "decimal(10,1)"), ("b", "decimal(10,0)"), ("c", "decimal(13,1)"),
        ("d", "decimal(10,3)"), ("product", "decimal(21,3)"),
    ]
    assert tuple(df.first()) == (
        Decimal("1.5"), Decimal("0"), Decimal("123456789012.5"), Decimal("-0.001"),
        Decimal("3.375"),
    )


def test_ml_vector_functions_as_pyspark(session):
    from pyspark.ml.functions import array_to_vector, vector_to_array

    df = session.sql("SELECT array(1.5D, 2.0D) AS a").select(array_to_vector("a").alias("vec"))
    row = df.select(vector_to_array("vec").alias("d"), vector_to_array("vec", "float32").alias("f"))
    assert tuple(row.first()) == ([1.5, 2.0], [1.5, 2.0])


def test_udfs_return_user_defined_types_of_main_as_pyspark():
    # As PySpark's: a UDT that __main__ defines, of parameters, is a UDF return type, as
    # cloudpickle sends its class by value and its state to the worker. In a new
    # interpreter, whose __main__ defines the classes.
    import subprocess
    import sys

    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live-server tests")
    subprocess.run([sys.executable, "-c", """
import sys
from pyspark.sql import Row, SparkSession, functions as F
from pyspark.sql.types import ArrayType, DoubleType, UserDefinedType

class Point:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __eq__(self, other):
        return isinstance(other, Point) and (self.x, self.y) == (other.x, other.y)

class PointUDT(UserDefinedType):
    def __init__(self, scale=1.0):
        self.scale = scale

    @classmethod
    def sqlType(cls):
        return ArrayType(DoubleType(), False)

    @classmethod
    def module(cls):
        return "__main__"

    def serialize(self, obj):
        return [obj.x * self.scale, obj.y * self.scale]

    def deserialize(self, datum):
        return Point(datum[0], datum[1])

Point.__UDT__ = PointUDT()
spark = SparkSession.builder.remote(sys.argv[1]).getOrCreate()
df = spark.range(2).select(F.udf(lambda i: Point(i, i + 1), PointUDT(10.0))("id").alias("p"))
assert df.collect() == [Row(p=Point(0.0, 10.0)), Row(p=Point(10.0, 20.0))]
assert df.select(F.to_json(F.struct("p"))).collect()[1][0] == '{"p":[10.0,20.0]}'
""", remote], check=True)


def test_collected_values_of_user_defined_types_are_deserialized_as_pyspark(session):
    # As official PySpark 4.3.0: a collected value of a UDT is its deserialized object, and
    # dtypes names the UDT by its simpleString(), in every way that rows are collected.
    from pyspark.ml.functions import array_to_vector
    from pyspark.ml.linalg import DenseVector

    df = session.sql(
        "SELECT array(1.5D, 2.0D) AS a, 1 AS k UNION ALL SELECT array(3.0D), 2"
    ).select(F.when(F.col("k") == 1, array_to_vector("a")).alias("vec"), "k").orderBy("k")
    vector = DenseVector([1.5, 2.0])
    rows = [Row(vec=vector, k=1), Row(vec=None, k=2)]
    assert df.dtypes == [("vec", "vector"), ("k", "int")]
    assert df.collect() == rows and df.first() == rows[0] and df.head() == rows[0]
    assert df.take(1) == rows[:1] and df.head(1) == rows[:1] and df.tail(1) == rows[1:]
    assert list(df.toLocalIterator()) == rows and list(df.toLocalIterator(True)) == rows
    assert type(df.first().vec) is DenseVector
    nested = df.limit(1).select(
        F.array("vec", "vec").alias("arr"),
        F.struct("vec", F.lit(1).alias("n")).alias("st"),
        F.create_map(F.lit("x"), "vec").alias("m"),
    )
    assert nested.dtypes == [
        ("arr", "array<vector>"), ("st", "struct<vec:vector,n:int>"), ("m", "map<string,vector>")]
    assert nested.first() == Row(arr=[vector, vector], st=Row(vec=vector, n=1), m={"x": vector})


def test_create_dataframe_merges_the_types_of_every_row_as_pyspark(session):
    import array
    import decimal

    Point = namedtuple("Point", ["x", "y"])

    class Attributes:
        def __init__(self):
            self.b = 1
            self.a = "x"

    cases = [
        # A string absorbs another atomic type, whose values become strings, and a null
        # takes the type of another row.
        ([(1, None), ("a", 2.5), (True, None)], "struct<_1:string,_2:double>",
         [("1", None), ("a", 2.5), ("true", None)]),
        ([([1, "a"],)], "struct<_1:array<string>>", [(["1", "a"],)]),
        ([{"a": 1}, {"a": "x"}], "struct<a:string>", [("1",), ("x",)]),
        ([({"a": [1]},), ({"a": [None]},)], "struct<_1:map<string,array<bigint>>>",
         [({"a": [1]},), ({"a": [None]},)]),
        # A decimal is decimal(38,18), a tuple, a namedtuple or an object a struct, and an
        # array.array an array of its typecode.
        ([(decimal.Decimal("1.5"),)], "struct<_1:decimal(38,18)>",
         [(decimal.Decimal("1.500000000000000000"),)]),
        ([((1, 2),)], "struct<_1:struct<_1:bigint,_2:bigint>>", [(Row(_1=1, _2=2),)]),
        ([(Point(1, 2),)], "struct<_1:struct<x:bigint,y:bigint>>", [(Row(x=1, y=2),)]),
        ([(Attributes(),)], "struct<_1:struct<a:string,b:bigint>>", [(Row(a="x", b=1),)]),
        ([(array.array("i", [1, 2]),)], "struct<_1:array<int>>", [([1, 2],)]),
    ]
    for data, schema, rows in cases:
        df = session.createDataFrame(data)
        assert df.schema.simpleString() == schema, data
        assert [tuple(r) for r in df.collect()] == rows, data
    for text in ["1.25", "a\x00b", "\uffff"]:
        # A string stays text, and shares a string column with an integer.
        df = session.createDataFrame([Row(value=text), Row(value=5)])
        assert [tuple(r) for r in df.collect()] == [(text,), ("5",)]


def test_dataframes_of_a_stream_are_streaming_as_pyspark(session):
    # As PySpark's, which asks the server: a DataFrame with a streaming source is streaming.
    stream = session.readStream.format("rate").load()
    for df in (
        stream,
        stream.select("value"),
        stream.where("value > 1"),
        stream.union(stream),
        session.range(1).join(stream.select(F.col("value").alias("id")), "id"),
    ):
        assert df.isStreaming is True
    assert session.range(1).isStreaming is False


def test_stream_writer_starts_a_query_of_its_real_time_trigger_as_pyspark(session):
    # As PySpark's, the real-time trigger reaches the server, which runs real-time mode for
    # some sinks only, not the memory sink.
    from pyspark.errors import IllegalArgumentException

    writer = session.readStream.format("rate").load().writeStream.format("memory")
    with pytest.raises(IllegalArgumentException) as error:
        query = writer.queryName("real_time_trigger").trigger(realTime="5 seconds").start()
        try:
            query.awaitTermination(10)
        finally:
            query.stop()
    assert error.value.getCondition() == (
        "STREAMING_REAL_TIME_MODE.OPERATOR_OR_SINK_NOT_IN_ALLOWLIST")


def test_create_dataframe_raises_merge_errors_as_pyspark(session):
    from pyspark.errors import PySparkException

    cases = [
        ([(1,), (2.5,)], "CANNOT_MERGE_TYPE", {"data_type1": "LongType", "data_type2": "DoubleType"}),
        # A merge error within a value names its field.
        ([([1, 2.5],)], "CANNOT_INFER_TYPE_FOR_FIELD", {"field_name": "_1"}),
        ([({"a": 1, "b": 2.5},)], "CANNOT_INFER_TYPE_FOR_FIELD", {"field_name": "_1"}),
        # A type that stays null cannot be determined.
        ([({"k": None},)], "CANNOT_DETERMINE_TYPE", {}),
        ([([None],)], "CANNOT_DETERMINE_TYPE", {}),
    ]
    for data, condition, parameters in cases:
        with pytest.raises(PySparkException) as error:
            session.createDataFrame(data)
        assert error.value.getCondition() == condition, data
        assert error.value.getMessageParameters() == parameters, data


def test_create_dataframe_converts_numpy_scalars_as_pyspark(session):
    np = pytest.importorskip("numpy")
    df = session.createDataFrame(
        [(np.int64(1), np.float32(0.5), np.bool_(True), np.float32(1.5))],
        "a bigint, b double, c boolean, d bigint")
    assert [tuple(r) for r in df.collect()] == [(1, 0.5, True, 1)]


def test_create_dataframe_infers_by_the_session_configurations_as_pyspark(session):
    naive = datetime(2020, 1, 2, 3, 4, 5)
    cases = [
        ("spark.sql.pyspark.legacy.inferMapTypeFromFirstPair.enabled", "true",
         [({"a": 1, "b": 2.5},)], "struct<_1:map<string,bigint>>", [({"a": 1, "b": 2},)]),
        ("spark.sql.pyspark.legacy.inferArrayTypeFromFirstElement.enabled", "true",
         [([1, 2.5],)], "struct<_1:array<bigint>>", [([1, 2],)]),
        ("spark.sql.pyspark.inferNestedDictAsStruct.enabled", "true",
         [({"a": {"b": 1}, "c": None},)], "struct<_1:struct<a:struct<b:bigint>>>",
         [(Row(a=Row(b=1)),)]),
        # A naive datetime is TIMESTAMP_NTZ, but not in a list, whose elements PySpark
        # infers without `prefer_timestamp_ntz`.
        ("spark.sql.timestampType", "TIMESTAMP_NTZ", [(naive,)], "struct<_1:timestamp_ntz>",
         [(naive,)]),
        ("spark.sql.timestampType", "TIMESTAMP_NTZ", [([naive],)], "struct<_1:array<timestamp>>",
         None),
        # PySpark passes the elements of a list `prefer_timestamp_ntz` for
        # `infer_map_from_first_pair`, so that a dict in a list takes the map type of its
        # first pair.
        ("spark.sql.timestampType", "TIMESTAMP_NTZ", [([{"a": 1, "b": 2.5}],)],
         "struct<_1:array<map<string,bigint>>>", [([{"a": 1, "b": 2}],)]),
    ]
    for key, value, data, schema, rows in cases:
        session.conf.set(key, value)
        try:
            df = session.createDataFrame(data)
            assert df.schema.simpleString() == schema, (key, data)
            if rows is not None:
                assert [tuple(r) for r in df.collect()] == rows, (key, data)
        finally:
            session.conf.unset(key)


def test_na_and_stat_functions_take_their_arguments_as_pyspark(session):
    # As PySpark's: the subset of dropna, a column name or a list or tuple of them, the columns
    # of describe, names or a list of them, and the quantiles of approxQuantile of a list or
    # tuple of columns, a list of those of each.
    df = session.createDataFrame(
        [(1, "a", None), (None, "b", 2.0), (3, None, 3.0)], "x int, k string, v double"
    )
    kept = [Row(x=1, k="a", v=None), Row(x=3, k=None, v=3.0)]
    assert df.dropna(subset="x").collect() == kept
    assert df.na.drop(subset=("x",)).collect() == kept
    assert df.dropna(how="all", thresh=3).collect() == []
    assert df.describe(["x", "v"]).columns == ["summary", "x", "v"]
    assert df.describe(("x", "v")).columns == ["summary", "x", "v"]
    assert df.approxQuantile(["x", "v"], [0.0, 1.0], 0.0) == [[1.0, 3.0], [2.0, 3.0]]
    assert df.stat.approxQuantile("x", (0.0, 1), 0) == [1.0, 3.0]
    linear = session.createDataFrame([(1, 2.0), (2, 4.0), (3, 6.0)], "a int, b double")
    assert linear.corr("a", "b", None) == linear.stat.corr("a", "b", "pearson") == 1.0
    assert df.freqItems(["k"]).columns == ["k_freqItems"]


def test_functions_apply_their_optional_arguments_as_pyspark(session):
    df = session.sql(
        "SELECT 'key' AS k, 'msg' AS m, parse_json('{\"a\": 1}') AS v, "
        "array(3.0F, 4.0F) AS vec, 1.0F AS one")
    row = df.select(
        F.hex(F.hmac("k", "m", F.lit("SHA-1"))).alias("sha1"),
        F.to_json(F.variant_set("v", "$.b", F.lit(2), False)).alias("kept"),
        F.to_json(F.try_variant_set("v", "$.b", F.lit(2))).alias("set"),
        F.vector_norm("vec", F.col("one")).alias("l1"),
        F.vector_normalize("vec", F.col("one")).alias("normalized"),
    ).first()
    assert row.sha1 == "102900B72B7BF1031EEC76B4804B66052376896B"
    assert row.kept == '{"a":1}'
    assert row.set == '{"a":1,"b":2}'
    assert row.l1 == 7.0
    assert row.normalized == pytest.approx([3 / 7, 4 / 7])


def test_logistic_regression_fits_the_data_and_predicts_on_new_rows(session):
    # The native estimator targets Spark's JVM LogisticRegression. Compare the same
    # behavior with that estimator in official PySpark, rather than its deprecated
    # torch-based pyspark.ml.connect.classification implementation.
    try:
        from pyspark._pyspark import LogisticRegression
    except ImportError:
        from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.connect.functions import array_to_vector

    training = session.sql(
        "SELECT array(CAST(x AS DOUBLE)) AS xs, "
        "CAST(CASE WHEN x > 0 THEN 1 ELSE 0 END AS DOUBLE) AS target "
        "FROM VALUES (-4), (-3), (-2), (-1), (1), (2), (3), (4) AS t(x)"
    ).withColumn("input", array_to_vector("xs"))
    held_out = session.sql(
        "SELECT id, array(CAST(x AS DOUBLE)) AS xs "
        "FROM VALUES (0, -3.5), (1, -1.5), (2, 1.5), (3, 3.5) AS t(id, x)"
    ).withColumn("input", array_to_vector("xs"))
    estimator = LogisticRegression(
        featuresCol="input", labelCol="target", predictionCol="pred", maxIter=30
    )
    first = estimator.fit(training)
    second = estimator.fit(training.withColumn("target", F.lit(1.0) - F.col("target")))
    assert [r.pred for r in first.transform(held_out).orderBy("id").collect()] == [
        0.0, 0.0, 1.0, 1.0
    ]
    assert [r.pred for r in second.transform(held_out).orderBy("id").collect()] == [
        1.0, 1.0, 0.0, 0.0
    ]
    with pytest.raises(Exception):
        estimator.fit(session.range(3))
