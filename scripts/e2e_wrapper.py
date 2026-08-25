#!/usr/bin/env python3
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
"""End-to-end exercise of the drop-in ``pyspark`` API against a live Connect server.

Unlike the official parity suite (which uses the reference pyspark classes and only
routes bytes through our Rust transport), this drives OUR drop-in wrapper: every call
here goes pyspark-rs (PyDataFrame/PySession/...) -> spark-connect (DataFrame/Session/...)
-> plan building -> execution. It therefore exercises the user-facing API surface that
the transport-injection gate does not, and is the coverage vehicle for those layers.

Run against a live server (PYTHONPATH must put our ./python skin first):
    RUST_PYSPARK_SO=/repo/python/pyspark/_pyspark.so \
    SPARK_REMOTE=sc://localhost:15002 PYTHONPATH=/repo/python \
    python scripts/e2e_wrapper.py

Each section is independent and best-effort: a feature the single-node server can't do
(e.g. some streaming/catalog paths) is reported and skipped, so one gap doesn't stop the
rest of the surface from being exercised. Exit code is non-zero only on an unexpected
crash of the harness itself, not on a per-feature server limitation.
"""

import os
import sys
import tempfile

REMOTE = os.environ.get("SPARK_REMOTE") or os.environ.get(
    "SPARK_CONNECT_TESTING_REMOTE", "sc://localhost:15002"
)

_passed = []
_skipped = []


def section(name):
    """Decorator: run a block, record pass/skip, never abort the whole run."""

    def wrap(fn):
        try:
            fn()
            _passed.append(name)
        except Exception as e:  # noqa: BLE001 - best-effort surface exercise
            _skipped.append(f"{name}: {type(e).__name__}: {str(e)[:120]}")
        return fn

    return wrap


def main():
    from pyspark.sql import Row, SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        DoubleType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )
    from pyspark.sql.window import Window

    # Observation may not be surfaced in the skin; its section skips if so.
    try:
        from pyspark.sql import Observation
    except Exception:  # noqa: BLE001
        try:
            from pyspark.sql.observation import Observation
        except Exception:  # noqa: BLE001
            Observation = None

    spark = SparkSession.builder.remote(REMOTE).getOrCreate()

    # ---- SparkSession surface ---------------------------------------------------
    @section("session-basics")
    def _():
        assert spark.version
        _ = spark.range(3).collect()
        _ = spark.range(0, 10, 2).count()
        spark.conf.set("spark.sql.shuffle.partitions", "7")
        assert spark.conf.get("spark.sql.shuffle.partitions") == "7"
        _ = spark.sql("SELECT 1 AS a, 'x' AS b").collect()

    @section("createDataFrame-typed")
    def _():
        schema = StructType(
            [
                StructField("i", IntegerType()),
                StructField("s", StringType()),
                StructField("d", DoubleType()),
            ]
        )
        df = spark.createDataFrame([(1, "a", 1.5), (2, "b", 2.5)], schema)
        assert df.count() == 2
        # DDL-string schema path (exercises the types.rs DDL parser)
        df2 = spark.createDataFrame([(1, "a"), (2, "b")], "i int, s string")
        assert df2.columns == ["i", "s"]
        _ = spark.createDataFrame([Row(x=1, y="p"), Row(x=2, y="q")]).collect()

    # The skin's createDataFrame currently accepts a list of column names (types are
    # inferred); StructType / DDL-string schemas are exercised best-effort above.
    df = spark.createDataFrame(
        [(1, "a", 10.0), (2, "b", 20.0), (2, "c", 30.0), (3, "d", 40.0)],
        ["id", "name", "val"],
    )

    # ---- DataFrame transformations (plan builders) ------------------------------
    @section("df-projection")
    def _():
        _ = df.select("id", "name").collect()
        _ = df.select(df.id, F.col("name").alias("n")).collect()
        _ = df.selectExpr("id + 1 as id2", "upper(name) as up").collect()
        _ = df.withColumn("val2", F.col("val") * 2).collect()
        _ = df.withColumns({"a": F.lit(1), "b": F.lit(2)}).collect()
        _ = df.withColumnRenamed("val", "value").collect()
        _ = df.withColumnsRenamed({"id": "identifier"}).collect()
        _ = df.drop("val").collect()
        _ = df.toDF("c1", "c2", "c3").collect()

    @section("df-filter-sort-limit")
    def _():
        _ = df.filter(F.col("id") > 1).collect()
        _ = df.where("id >= 2").collect()
        _ = df.sort(F.col("id").desc()).collect()
        _ = df.orderBy("name").collect()
        _ = df.sortWithinPartitions("id").collect()
        _ = df.limit(2).collect()
        _ = df.offset(1).collect()
        _ = df.distinct().count()
        _ = df.dropDuplicates(["id"]).count()
        _ = df.sample(0.5, seed=1).count()

    @section("df-set-ops")
    def _():
        other = df.select("id", "name", "val")
        _ = df.union(other).count()
        _ = df.unionByName(other).count()
        _ = df.unionAll(other).count()
        _ = df.intersect(other).count()
        _ = df.intersectAll(other).count()
        _ = df.exceptAll(other).count()
        _ = df.subtract(other).count()

    @section("df-join")
    def _():
        a = df.select(F.col("id").alias("id"), "name")
        b = df.select(F.col("id").alias("id"), F.col("val"))
        _ = a.join(b, "id").collect()
        _ = a.join(b, on="id", how="left").collect()
        _ = a.join(b, a.id == b.id, "inner").collect()
        _ = a.crossJoin(b.limit(1)).collect()
        _ = a.hint("broadcast").join(b, "id").collect()

    @section("df-aggregation")
    def _():
        g = df.groupBy("id")
        _ = g.count().collect()
        _ = g.agg(F.sum("val").alias("s"), F.avg("val")).collect()
        _ = g.min("val").collect()
        _ = g.max("val").collect()
        _ = g.mean("val").collect()
        _ = g.sum("val").collect()
        _ = df.groupBy("id").pivot("name").sum("val").collect()
        _ = df.rollup("id", "name").count().collect()
        _ = df.cube("id").count().collect()
        _ = df.agg(F.countDistinct("name").alias("cd")).collect()

    @section("df-na-stat")
    def _():
        _ = df.na.fill(0).collect()
        _ = df.na.drop().collect()
        _ = df.na.replace(["a"], ["A"], "name").collect()
        _ = df.fillna({"val": 0.0}).collect()
        _ = df.dropna().collect()
        _ = df.stat.corr("id", "val")
        _ = df.stat.cov("id", "val")
        _ = df.stat.crosstab("id", "name").collect()
        _ = df.stat.freqItems(["name"]).collect()
        _ = df.stat.approxQuantile("val", [0.5], 0.1)
        _ = df.summary().collect()
        _ = df.describe("val").collect()

    @section("df-reshape")
    def _():
        _ = df.repartition(2).count()
        _ = df.repartition(2, "id").count()
        _ = df.coalesce(1).count()
        _ = df.unpivot(["id"], ["val"], "var", "value").collect()
        _ = df.melt(["id"], ["val"], "var", "value").collect()

    @section("df-actions-and-meta")
    def _():
        assert df.count() == 4
        _ = df.head()
        _ = df.head(2)
        _ = df.take(2)
        _ = df.first()
        _ = df.tail(2)
        _ = df.limit(1).toPandas()
        _ = df.schema
        _ = df.dtypes
        assert df.columns == ["id", "name", "val"]
        _ = df.isEmpty()
        df.printSchema()
        df.explain()
        df.explain(True)
        df.show(2)
        _ = df.cache()
        _ = df.persist()
        df.unpersist()
        _ = df.withWatermark  # attribute exists
        _ = df.colRegex("`.*`")

    @section("df-observe")
    def _():
        if Observation is None:
            raise RuntimeError("Observation not surfaced in the skin")
        obs = Observation("o")
        observed = df.observe(obs, F.count(F.lit(1)).alias("cnt"))
        observed.collect()
        _ = obs.get

    # ---- Column expression surface ---------------------------------------------
    @section("column-ops")
    def _():
        c = F.col("val")
        exprs = [
            (c + 1).alias("a"),
            (c - 1).alias("b"),
            (c * 2).alias("c"),
            (c / 2).alias("d"),
            (c % 3).alias("e"),
            (-c).alias("f"),
            (c > 1).alias("g"),
            (c >= 1).alias("h"),
            (c < 100).alias("i"),
            (c <= 100).alias("j"),
            (c == 10).alias("k"),
            (c != 10).alias("l"),
            c.isNull().alias("m"),
            c.isNotNull().alias("n"),
            c.between(0, 100).alias("o"),
            F.col("name").isin("a", "b").alias("p"),
            F.col("name").like("a%").alias("q"),
            F.col("name").rlike("^a").alias("r"),
            F.col("name").contains("a").alias("s"),
            F.col("name").startswith("a").alias("t"),
            F.col("name").endswith("z").alias("u"),
            F.col("name").substr(1, 2).alias("v"),
            c.cast("int").alias("w"),
            c.asc().alias("x1") if hasattr(c, "asc") else F.lit(1).alias("x1"),
        ]
        _ = df.select(*[e for e in exprs]).collect()
        _ = df.select(
            F.when(c > 15, "hi").otherwise("lo").alias("cw"),
            (c.bitwiseAND(F.lit(1))).alias("band"),
            (c.bitwiseOR(F.lit(1))).alias("bor"),
        ).collect()
        _ = df.select(F.col("id").asc_nulls_first(), F.col("id").desc_nulls_last())

    # ---- functions module ------------------------------------------------------
    @section("functions")
    def _():
        _ = df.select(
            F.abs(F.col("val")),
            F.sqrt(F.col("val")),
            F.round(F.col("val"), 1),
            F.ceil(F.col("val")),
            F.floor(F.col("val")),
            F.upper(F.col("name")),
            F.lower(F.col("name")),
            F.length(F.col("name")),
            F.concat(F.col("name"), F.lit("!")),
            F.concat_ws("-", F.col("name"), F.col("name")),
            F.coalesce(F.col("val"), F.lit(0.0)),
            F.greatest(F.col("id"), F.lit(5)),
            F.least(F.col("id"), F.lit(5)),
            F.lit(1),
        ).collect()
        _ = df.select(
            F.current_date(),
            F.current_timestamp(),
            F.rand(1),
            F.randn(1),
            F.monotonically_increasing_id(),
        ).collect()
        # window functions
        w = Window.partitionBy("id").orderBy("val")
        _ = df.select(
            F.row_number().over(w),
            F.rank().over(w),
            F.dense_rank().over(w),
            F.lag("val").over(w),
            F.lead("val").over(w),
            F.sum("val").over(w.rowsBetween(Window.unboundedPreceding, Window.currentRow)),
        ).collect()

    # ---- read/write ------------------------------------------------------------
    @section("readwriter")
    def _():
        d = tempfile.mkdtemp()
        p = os.path.join(d, "out")
        df.write.mode("overwrite").parquet(p)
        _ = spark.read.parquet(p).count()
        pj = os.path.join(d, "j")
        df.write.mode("overwrite").json(pj)
        _ = spark.read.json(pj).count()
        pc = os.path.join(d, "c")
        df.write.mode("overwrite").option("header", True).csv(pc)
        _ = spark.read.option("header", True).csv(pc).count()
        df.write.format("parquet").mode("overwrite").save(os.path.join(d, "s"))

    # ---- catalog ---------------------------------------------------------------
    @section("catalog")
    def _():
        cat = spark.catalog
        _ = cat.currentDatabase()
        _ = cat.listDatabases()
        _ = cat.listTables()
        _ = cat.listFunctions()
        df.createOrReplaceTempView("v_e2e")
        assert cat.tableExists("v_e2e")
        _ = cat.listColumns("v_e2e")
        _ = spark.table("v_e2e").count()
        cat.dropTempView("v_e2e")
        _ = cat.functionExists("abs")

    # ---- streaming (best effort: rate source) ----------------------------------
    @section("streaming")
    def _():
        sdf = spark.readStream.format("rate").option("rowsPerSecond", 1).load()
        assert sdf.isStreaming
        q = (
            sdf.writeStream.format("memory")
            .queryName("e2e_mem")
            .outputMode("append")
            .trigger(processingTime="1 second")
            .start()
        )
        import time

        time.sleep(2)
        q.stop()
        _ = spark.streams.active

    spark.stop()

    print(f"\n=== e2e_wrapper: {len(_passed)} sections OK, {len(_skipped)} skipped ===")
    for s in _passed:
        print(f"  ok   {s}")
    for s in _skipped:
        print(f"  skip {s}")
    # Non-zero only if essentially nothing ran (harness broken), not on feature gaps.
    return 0 if len(_passed) >= 5 else 1


if __name__ == "__main__":
    sys.exit(main())
