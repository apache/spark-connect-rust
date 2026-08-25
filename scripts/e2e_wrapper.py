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

Unlike the official parity suite (which uses the reference pyspark classes and only routes
bytes through our Rust transport), this drives OUR drop-in wrapper: every call goes
pyspark-rs (PyDataFrame/PySession/...) -> spark-connect (DataFrame/Session/...) -> plan
building -> execution. It is the coverage vehicle for the high-level API that the
transport-injection gate does not exercise.

Each operation is checked independently (via ``ck``) so one gap neither stops the rest nor
hides which methods are missing/divergent. Run against a live server:

    RUST_PYSPARK_SO=/repo/python/pyspark/_pyspark.so \
    SPARK_REMOTE=sc://localhost:15002 PYTHONPATH=/repo/python \
    python scripts/e2e_wrapper.py
"""

import os
import sys
import tempfile

REMOTE = os.environ.get("SPARK_REMOTE") or os.environ.get(
    "SPARK_CONNECT_TESTING_REMOTE", "sc://localhost:15002"
)

_ok = []
_fail = []


def ck(label, fn):
    """Run one operation independently; record ok/fail, never abort the run.

    This only proves the call does not raise. For correctness (the bug class this
    client has hit is *silently wrong values*, e.g. a nullary ``F.struct()``
    returning an empty struct and throwing nothing), use ``ckv`` below, which
    asserts the actual result.
    """
    try:
        fn()
        _ok.append(label)
    except Exception as e:  # noqa: BLE001 - best-effort surface exercise
        _fail.append(f"{label}: {type(e).__name__}: {str(e)[:110]}")


def ckv(label, fn, expected):
    """Like ``ck`` but assert ``fn() == expected`` - catches silently-wrong values."""
    try:
        got = fn()
        if got != expected:
            _fail.append(f"{label}: expected {expected!r}, got {got!r}")
        else:
            _ok.append(label)
    except Exception as e:  # noqa: BLE001
        _fail.append(f"{label}: {type(e).__name__}: {str(e)[:110]}")


def main():
    from pyspark.sql import Row, SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    spark = SparkSession.builder.remote(REMOTE).getOrCreate()
    df = spark.createDataFrame(
        [(1, "a", 10.0), (2, "b", 20.0), (2, "c", 30.0), (3, "d", 40.0)],
        ["id", "name", "val"],
    )
    df.createOrReplaceTempView("t_e2e") if hasattr(df, "createOrReplaceTempView") else None
    other = df.select("id", "name", "val")

    # ---- SparkSession -----------------------------------------------------------
    ck("session.version", lambda: spark.version)
    ck("session.range", lambda: spark.range(3).collect())
    ck("session.range3", lambda: spark.range(0, 10, 2).count())
    ck("session.sql", lambda: spark.sql("SELECT 1 AS a, 'x' AS b").collect())
    ck("session.conf.set", lambda: spark.conf.set("spark.sql.shuffle.partitions", "7"))
    ck("session.conf.get", lambda: spark.conf.get("spark.sql.shuffle.partitions"))
    ck("session.read", lambda: spark.read)
    ck("session.catalog.prop", lambda: spark.catalog.currentDatabase())
    ck("session.udf", lambda: spark.udf)
    ck(
        "session.createDataFrame.names",
        lambda: spark.createDataFrame([(1, "a")], ["i", "s"]).count(),
    )
    ck(
        "session.createDataFrame.ddl",
        lambda: spark.createDataFrame([(1, "a")], "i int, s string").count(),
    )
    ck("session.createDataFrame.rows", lambda: spark.createDataFrame([Row(x=1), Row(x=2)]).count())

    # ---- DataFrame projection / columns ----------------------------------------
    ck("df.select.str", lambda: df.select("id", "name").collect())
    ck("df.attr.col", lambda: df.select(df.id).collect())
    ck("df.getitem.col", lambda: df.select(df["id"]).collect())
    ck("df.select.expr", lambda: df.select(F.col("name").alias("n")).collect())
    ck("df.selectExpr", lambda: df.selectExpr("id + 1 as id2", "upper(name) as up").collect())
    ck("df.withColumn", lambda: df.withColumn("v2", F.col("val") * 2).collect())
    ck("df.withColumns", lambda: df.withColumns({"a": F.lit(1), "b": F.lit(2)}).collect())
    ck("df.withColumnRenamed", lambda: df.withColumnRenamed("val", "value").collect())
    ck("df.withColumnsRenamed", lambda: df.withColumnsRenamed({"id": "identifier"}).collect())
    ck("df.drop", lambda: df.drop("val").collect())
    ck("df.toDF", lambda: df.toDF("c1", "c2", "c3").collect())
    ck("df.colRegex", lambda: df.colRegex("`.*`").collect())

    # ---- filter / sort / limit --------------------------------------------------
    ck("df.filter.col", lambda: df.filter(F.col("id") > 1).collect())
    ck("df.where.col", lambda: df.where(F.col("id") > 1).collect())
    ck("df.where.str", lambda: df.where("id >= 2").collect())
    ck("df.filter.str", lambda: df.filter("id >= 2").collect())
    ck("df.sort", lambda: df.sort(F.col("id").desc()).collect())
    ck("df.orderBy", lambda: df.orderBy("name").collect())
    ck("df.sortWithinPartitions", lambda: df.sortWithinPartitions("id").collect())
    ck("df.limit", lambda: df.limit(2).collect())
    ck("df.offset", lambda: df.offset(1).collect())
    ck("df.distinct", lambda: df.distinct().count())
    ck("df.dropDuplicates", lambda: df.dropDuplicates(["id"]).count())
    ck("df.sample", lambda: df.sample(0.5, seed=1).count())

    # ---- set ops ----------------------------------------------------------------
    ck("df.union", lambda: df.union(other).count())
    ck("df.unionByName", lambda: df.unionByName(other).count())
    ck("df.unionAll", lambda: df.unionAll(other).count())
    ck("df.intersect", lambda: df.intersect(other).count())
    ck("df.intersectAll", lambda: df.intersectAll(other).count())
    ck("df.exceptAll", lambda: df.exceptAll(other).count())
    ck("df.subtract", lambda: df.subtract(other).count())

    # ---- join -------------------------------------------------------------------
    a = df.select(F.col("id").alias("id"), "name")
    b = df.select(F.col("id").alias("id"), "val")
    b2 = df.select(F.col("id").alias("id2"), "val")
    ck("df.join.on", lambda: a.join(b, "id").collect())
    ck("df.join.how", lambda: a.join(b, on="id", how="left").collect())
    ck("df.join.cond", lambda: a.join(b2, F.col("id") == F.col("id2"), "inner").collect())
    ck("df.crossJoin", lambda: a.crossJoin(b.limit(1)).collect())
    ck("df.hint", lambda: a.hint("broadcast").join(b, "id").collect())

    # ---- aggregation ------------------------------------------------------------
    ck("gb.count", lambda: df.groupBy("id").count().collect())
    ck("gb.agg", lambda: df.groupBy("id").agg(F.sum("val").alias("s"), F.avg("val")).collect())
    ck("gb.min", lambda: df.groupBy("id").min("val").collect())
    ck("gb.max", lambda: df.groupBy("id").max("val").collect())
    ck("gb.mean", lambda: df.groupBy("id").mean("val").collect())
    ck("gb.sum", lambda: df.groupBy("id").sum("val").collect())
    ck("gb.pivot", lambda: df.groupBy("id").pivot("name").sum("val").collect())
    ck("df.rollup", lambda: df.rollup("id", "name").count().collect())
    ck("df.cube", lambda: df.cube("id").count().collect())
    ck("df.agg", lambda: df.agg(F.countDistinct("name").alias("cd")).collect())

    # ---- na / stat --------------------------------------------------------------
    ck("na.fill", lambda: df.na.fill(0).collect())
    ck("na.drop", lambda: df.na.drop().collect())
    ck("na.replace", lambda: df.na.replace(["a"], ["A"], "name").collect())
    ck("df.fillna", lambda: df.fillna({"val": 0.0}).collect())
    ck("df.dropna", lambda: df.dropna().collect())
    ck("stat.corr", lambda: df.stat.corr("id", "val"))
    ck("stat.cov", lambda: df.stat.cov("id", "val"))
    ck("stat.crosstab", lambda: df.stat.crosstab("id", "name").collect())
    ck("stat.freqItems", lambda: df.stat.freqItems(["name"]).collect())
    ck("stat.approxQuantile", lambda: df.stat.approxQuantile("val", [0.5], 0.1))
    ck("df.summary", lambda: df.summary().collect())
    ck("df.describe", lambda: df.describe("val").collect())

    # ---- reshape / partition ----------------------------------------------------
    ck("df.repartition.n", lambda: df.repartition(2).count())
    ck("df.repartition.cols", lambda: df.repartition(2, "id").count())
    ck("df.coalesce", lambda: df.coalesce(1).count())
    ck("df.unpivot", lambda: df.unpivot(["id"], ["val"], "var", "value").collect())
    ck("df.melt", lambda: df.melt(["id"], ["val"], "var", "value").collect())

    # ---- actions / metadata -----------------------------------------------------
    ck("df.count", lambda: df.count())
    ck("df.head", lambda: df.head())
    ck("df.head.n", lambda: df.head(2))
    ck("df.take", lambda: df.take(2))
    ck("df.first", lambda: df.first())
    ck("df.tail", lambda: df.tail(2))
    ck("df.toPandas", lambda: df.limit(1).toPandas())
    ck("df.schema", lambda: df.schema)
    ck("df.dtypes", lambda: df.dtypes)
    ck("df.columns", lambda: df.columns)
    ck("df.isEmpty", lambda: df.isEmpty())
    ck("df.printSchema", lambda: df.printSchema())
    ck("df.explain", lambda: df.explain())
    ck("df.show", lambda: df.show(2))
    ck("df.cache", lambda: df.cache())
    ck("df.persist", lambda: df.persist())
    ck("df.unpersist", lambda: df.unpersist())

    # ---- Column expression surface ---------------------------------------------
    c = F.col("val")
    ck("col.arith", lambda: df.select((c + 1), (c - 1), (c * 2), (c / 2), (c % 3), (-c)).collect())
    ck("col.cmp", lambda: df.select(c > 1, c >= 1, c < 100, c <= 100, c == 10, c != 10).collect())
    ck("col.isNull", lambda: df.select(c.isNull(), c.isNotNull()).collect())
    ck("col.between", lambda: df.select(c.between(0, 100)).collect())
    ck("col.isin", lambda: df.select(F.col("name").isin("a", "b")).collect())
    ck("col.like", lambda: df.select(F.col("name").like("a%"), F.col("name").rlike("^a")).collect())
    ck(
        "col.strfns",
        lambda: df.select(
            F.col("name").contains("a"),
            F.col("name").startswith("a"),
            F.col("name").endswith("z"),
            F.col("name").substr(1, 2),
        ).collect(),
    )
    ck("col.cast", lambda: df.select(c.cast("int")).collect())
    ck("col.when", lambda: df.select(F.when(c > 15, "hi").otherwise("lo")).collect())
    ck(
        "col.bitwise",
        lambda: df.select(
            F.col("id").bitwiseAND(F.lit(1)), F.col("id").bitwiseOR(F.lit(1))
        ).collect(),
    )
    ck("col.getField", lambda: hasattr(F.col("x"), "getField"))
    ck("col.sortmods", lambda: df.sort(F.col("id").asc()).sort(F.col("id").desc()).collect())

    # ---- functions --------------------------------------------------------------
    ck(
        "fn.math",
        lambda: df.select(F.abs(c), F.sqrt(c), F.round(c, 1), F.ceil(c), F.floor(c)).collect(),
    )
    ck(
        "fn.string",
        lambda: df.select(
            F.upper(F.col("name")),
            F.lower(F.col("name")),
            F.length(F.col("name")),
            F.concat(F.col("name"), F.lit("!")),
            F.concat_ws("-", F.col("name"), F.col("name")),
        ).collect(),
    )
    ck(
        "fn.cond",
        lambda: df.select(
            F.coalesce(c, F.lit(0.0)),
            F.greatest(F.col("id"), F.lit(5)),
            F.least(F.col("id"), F.lit(5)),
        ).collect(),
    )
    ck(
        "fn.gen",
        lambda: df.select(
            F.current_date(), F.current_timestamp(), F.monotonically_increasing_id()
        ).collect(),
    )
    w = Window.partitionBy("id").orderBy("val")
    ck(
        "fn.window",
        lambda: df.select(
            F.row_number().over(w),
            F.rank().over(w),
            F.dense_rank().over(w),
            F.lag("val").over(w),
            F.lead("val").over(w),
        ).collect(),
    )

    # ---- read / write -----------------------------------------------------------
    d = tempfile.mkdtemp()
    ck("write.parquet", lambda: df.write.mode("overwrite").parquet(os.path.join(d, "p")))
    ck("read.parquet", lambda: spark.read.parquet(os.path.join(d, "p")).count())
    ck("write.json", lambda: df.write.mode("overwrite").json(os.path.join(d, "j")))
    ck("read.json", lambda: spark.read.json(os.path.join(d, "j")).count())
    ck(
        "write.csv",
        lambda: df.write.mode("overwrite").option("header", True).csv(os.path.join(d, "c")),
    )
    ck("read.csv", lambda: spark.read.option("header", True).csv(os.path.join(d, "c")).count())
    ck(
        "write.save",
        lambda: df.write.format("parquet").mode("overwrite").save(os.path.join(d, "s")),
    )

    # ---- catalog ----------------------------------------------------------------
    ck("catalog.currentDatabase", lambda: spark.catalog.currentDatabase())
    ck("catalog.listDatabases", lambda: spark.catalog.listDatabases())
    ck("catalog.listTables", lambda: spark.catalog.listTables())
    ck("catalog.tempView", lambda: df.createOrReplaceTempView("v_e2e2"))
    ck("catalog.tableExists", lambda: spark.catalog.tableExists("v_e2e2"))
    ck("catalog.table", lambda: spark.table("v_e2e2").count())
    ck("catalog.dropTempView", lambda: spark.catalog.dropTempView("v_e2e2"))

    # ---- functions: build every function once (covers functions.rs) -------------
    def _build_all_functions():
        ci, cs, cd = F.col("id"), F.col("name"), F.col("val")
        lit1 = F.lit(1)
        variants = [
            (ci,),
            (ci, cd),
            (ci, cd, ci),
            (ci, cs),
            (ci, "x"),
            (ci, 1),
            (cs,),
            ("x",),
            (lit1,),
            (),
            (ci, ci, ci, ci),
            (cs, cs),
        ]
        built = 0
        for name in dir(F):
            if name.startswith("_"):
                continue
            fn = getattr(F, name)
            if not callable(fn):
                continue
            for a in variants:
                try:
                    fn(*a)
                    built += 1
                    break
                except Exception:  # noqa: BLE001 - wrong arity/type; try the next shape
                    continue
        assert built > 400, f"only {built} functions built"

    ck("functions.build_all", _build_all_functions)

    # ---- window frame specs -----------------------------------------------------
    ck(
        "window.rowsBetween",
        lambda: df.select(
            F.sum("val").over(
                Window.partitionBy("id")
                .orderBy("val")
                .rowsBetween(Window.unboundedPreceding, Window.currentRow)
            )
        ).collect(),
    )
    ck(
        "window.rangeBetween",
        lambda: df.select(
            F.sum("val").over(
                Window.partitionBy("id")
                .orderBy("val")
                .rangeBetween(Window.unboundedPreceding, Window.currentRow)
            )
        ).collect(),
    )

    # ---- session extras ---------------------------------------------------------
    ck("session.sessionId", lambda: spark.sessionId)
    ck("session.newSession", lambda: spark.newSession())
    ck("session.cloneSession", lambda: spark.cloneSession())
    ck("session.emptyDataFrame", lambda: spark.emptyDataFrame().count())
    ck("session.range_full", lambda: spark.range(0, 10, 2, 2).count())
    ck("session.addTag", lambda: spark.addTag("e2e-tag"))
    ck("session.getTags", lambda: spark.getTags())
    ck("session.removeTag", lambda: spark.removeTag("e2e-tag"))
    ck("session.clearTags", lambda: spark.clearTags())
    ck("session.interruptAll", lambda: spark.interruptAll())
    ck("session.getActiveSession", lambda: SparkSession.getActiveSession())
    ck("session.profile", lambda: spark.profile)
    ck("session.dataSource", lambda: spark.dataSource)

    # ---- resource profile -------------------------------------------------------
    def _resource():
        from pyspark.resource import (
            ExecutorResourceRequests,
            ResourceProfileBuilder,
            TaskResourceRequests,
        )

        rpb = ResourceProfileBuilder()
        rpb.require(ExecutorResourceRequests().cores(2))
        rpb.require(TaskResourceRequests().cpus(1))
        _ = rpb.build

    ck("resource.profile", _resource)

    # ---- Column: remaining surface ----------------------------------------------
    cid = F.col("id")
    ck("col.asc_nulls", lambda: df.sort(cid.asc_nulls_first(), cid.asc_nulls_last()).collect())
    ck(
        "col.desc_nulls",
        lambda: df.sort(cid.desc(), cid.desc_nulls_first(), cid.desc_nulls_last()).collect(),
    )
    ck("col.bitwiseXOR", lambda: df.select(cid.bitwiseXOR(F.lit(1))).collect())
    ck("col.eqNullSafe", lambda: df.select(cid.eqNullSafe(F.lit(1))).collect())
    ck("col.ilike", lambda: df.select(F.col("name").ilike("A%")).collect())
    ck("col.isNaN", lambda: df.select(F.col("val").isNaN()).collect())
    ck("col.alias", lambda: df.select(cid.alias("renamed")).collect())

    def _struct_fields():
        s = df.select(F.struct(F.col("id"), F.col("name"), F.col("val")).alias("s"))
        # withField adds/replaces a field; dropFields removes one (leaving others).
        s.select(F.col("s").withField("z", F.lit(1))).collect()
        s.select(F.col("s").dropFields("id")).collect()

    ck("col.withField/dropFields", _struct_fields)

    # ---- Catalog: full surface --------------------------------------------------
    df.createOrReplaceTempView("cat_view")
    ck("catalog.currentCatalog", lambda: spark.catalog.currentCatalog())
    ck("catalog.listCatalogs", lambda: spark.catalog.listCatalogs())
    ck("catalog.databaseExists", lambda: spark.catalog.databaseExists("default"))
    ck("catalog.functionExists", lambda: spark.catalog.functionExists("abs"))
    ck("catalog.getDatabase", lambda: spark.catalog.getDatabase("default"))
    ck("catalog.listColumns", lambda: spark.catalog.listColumns("cat_view"))
    ck("catalog.listFunctions", lambda: spark.catalog.listFunctions())
    ck("catalog.getTable", lambda: spark.catalog.getTable("cat_view"))
    ck("catalog.cacheTable", lambda: spark.catalog.cacheTable("cat_view"))
    ck("catalog.uncacheTable", lambda: spark.catalog.uncacheTable("cat_view"))
    ck("catalog.setCurrentDatabase", lambda: spark.catalog.setCurrentDatabase("default"))

    # ---- DataFrame: remaining surface -------------------------------------------
    ck("df.alias", lambda: df.alias("d2").select("id").collect())
    ck("df.toArrow", lambda: df.limit(2).toArrow())
    ck("df.toLocalIterator", lambda: list(df.limit(2).toLocalIterator()))
    ck("df.executionInfo", lambda: (df.limit(1).collect(), df.executionInfo))

    # ---- DataFrameWriter: remaining formats -------------------------------------
    ck("write.orc", lambda: df.write.mode("overwrite").orc(os.path.join(d, "o")))
    ck("read.orc", lambda: spark.read.orc(os.path.join(d, "o")).count())
    ck(
        "write.text",
        lambda: df.select(F.col("name")).write.mode("overwrite").text(os.path.join(d, "txt")),
    )
    ck("read.text", lambda: spark.read.text(os.path.join(d, "txt")).count())
    ck("read.load", lambda: spark.read.format("parquet").load(os.path.join(d, "p")).count())
    ck(
        "write.saveAsTable",
        lambda: df.write.mode("overwrite").saveAsTable("e2e_saved_table"),
    )
    ck("read.table.saved", lambda: spark.table("e2e_saved_table").count())

    # ---- Window standalone builders ---------------------------------------------
    ck("window.orderBy", lambda: F.row_number().over(Window.orderBy("id")))
    ck(
        "window.rangeBetween.unbounded",
        lambda: df.select(
            F.sum("val").over(
                Window.orderBy("id").rangeBetween(
                    Window.unboundedPreceding, Window.unboundedFollowing
                )
            )
        ).collect(),
    )

    # ---- streaming (rate source, best effort) -----------------------------------
    def _streaming():
        import time

        sdf = spark.readStream.format("rate").option("rowsPerSecond", 1).load()
        assert sdf.isStreaming
        q = (
            sdf.writeStream.format("memory")
            .queryName("e2e_stream")
            .outputMode("append")
            .trigger(processingTime="1 second")
            .start()
        )
        time.sleep(2)
        _ = (q.id, q.name, q.isActive, q.status, q.recentProgress)
        q.stop()
        _ = spark.streams.active
        spark.streams.resetTerminated()

    ck("streaming.rate", _streaming)

    # ---- correctness assertions (catch silently-wrong values, not just exceptions) ---
    # These pin actual results for the operations most prone to returning wrong-but-
    # non-throwing data - especially the variadic functions whose bug was an empty
    # result with no exception.
    ckv("val.range.count", lambda: spark.range(5).count(), 5)
    ckv("val.range.rows", lambda: [r.id for r in spark.range(3).collect()], [0, 1, 2])
    ckv("val.df.count", lambda: df.count(), 4)
    ckv("val.filter.count", lambda: df.filter(F.col("id") > 1).count(), 3)
    ckv("val.distinct.count", lambda: df.select("id").distinct().count(), 3)
    ckv(
        "val.groupby.sum",
        lambda: {r.id: r.s for r in df.groupBy("id").agg(F.sum("val").alias("s")).collect()},
        {1: 10.0, 2: 50.0, 3: 40.0},
    )
    ckv("val.union.count", lambda: df.union(other).count(), 8)
    ckv("val.join.count", lambda: df.select("id").join(other.select("id"), "id").count(), 6)
    # The variadic-function bug class: these must return populated results.
    ckv(
        "val.struct",
        # dict(...) normalizes whether the nested struct comes back as a Row or a dict;
        # the point is it is *populated* (the nullary-F.struct bug returned {}).
        lambda: dict(
            spark.range(1)
            .select(F.struct(F.lit(1).alias("a"), F.lit("x").alias("b")).alias("s"))
            .collect()[0]["s"]
        ),
        {"a": 1, "b": "x"},
    )
    ckv(
        "val.array",
        lambda: spark.range(1)
        .select(F.array(F.lit(1), F.lit(2), F.lit(3)).alias("a"))
        .collect()[0]["a"],
        [1, 2, 3],
    )
    ckv(
        "val.coalesce",
        lambda: spark.range(1)
        .select(F.coalesce(F.lit(None), F.lit(7)).alias("c"))
        .collect()[0]["c"],
        7,
    )
    ckv(
        "val.concat",
        lambda: spark.range(1)
        .select(F.concat(F.lit("a"), F.lit("b")).alias("c"))
        .collect()[0]["c"],
        "ab",
    )
    ckv(
        "val.create_map",
        lambda: spark.range(1)
        .select(F.create_map(F.lit("k"), F.lit(9)).alias("m"))
        .collect()[0]["m"],
        {"k": 9},
    )
    ckv("val.sql.lit", lambda: spark.sql("SELECT 1 AS a").collect()[0]["a"], 1)

    ck("session.stop", lambda: spark.stop())

    total = len(_ok) + len(_fail)
    print(f"\n=== e2e_wrapper: {len(_ok)}/{total} operations OK, {len(_fail)} failed ===")
    for s in _fail:
        print(f"  FAIL {s}")
    # Non-zero only if almost nothing ran (harness broken), not on per-op gaps.
    return 0 if len(_ok) >= total * 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
