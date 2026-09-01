"""Focused e2e validation of the parity fixes against a real Spark Connect server.
Exercises keyword-argument calls and newly-wired params end-to-end (real execution),
checking results — run with our Rust client on PYTHONPATH."""

import os
import tempfile

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.remote(
    os.environ.get("SPARK_CONNECT_TESTING_REMOTE", "sc://localhost:15002")
).getOrCreate()

ok = 0
fail = 0


def check(desc, got, want):
    global ok, fail
    if got == want:
        ok += 1
        print(f"OK    {desc}  -> {got}")
    else:
        fail += 1
        print(f"FAIL  {desc}  got={got!r} want={want!r}")


def run(desc, fn):
    global ok, fail
    try:
        fn()
        ok += 1
        print(f"OK    {desc}")
    except Exception as e:
        fail += 1
        print(f"FAIL  {desc} -> {type(e).__name__}: {str(e)[:80]}")


df = spark.range(10).toDF("id")

# GAP 1: keyword-argument function calls now execute and produce correct plans/results
check(
    "F.round(lit(3.14159), scale=2)",
    spark.range(1).select(F.round(F.lit(3.14159), scale=2).alias("r")).collect()[0]["r"],
    3.14,
)
check(
    "F.split(lit('a,b,c'), pattern=',') length",
    len(spark.range(1).select(F.split(F.lit("a,b,c"), pattern=",").alias("s")).collect()[0]["s"]),
    3,
)
check(
    "F.first(id, ignorenulls=True) over range",
    spark.range(5).select(F.first("id", ignorenulls=True).alias("f")).collect()[0]["f"],
    0,
)
run(
    "F.from_json(col, schema='a INT') builds",
    lambda: (
        spark.range(1).select(F.from_json(F.lit('{"a":1}'), schema="a INT").alias("j")).collect()
    ),
)
check(
    "F.concat_ws(sep, *cols)",
    spark.range(1).select(F.concat_ws("-", F.lit("a"), F.lit("b")).alias("c")).collect()[0]["c"],
    "a-b",
)

# GAP 2: method keyword/params
check(
    "df.sample(withReplacement=False, fraction=1.0, seed=1) count",
    spark.range(100).sample(withReplacement=False, fraction=1.0, seed=1).count() > 0,
    True,
)
check(
    "df.sort(id, ascending=False) first",
    spark.range(5).sort("id", ascending=False).collect()[0]["id"],
    4,
)
check(
    "df.orderBy(id, ascending=[False]) first",
    spark.range(5).orderBy("id", ascending=[False]).collect()[0]["id"],
    4,
)
check(
    "df.selectExpr('id * 2 as d') first",
    spark.range(3).selectExpr("id * 2 as d").collect()[1]["d"],
    2,
)
check("df.drop('id') columns", spark.range(3).toDF("id").drop("id").columns, [])
check(
    "df.toDF('a','b') columns",
    spark.range(1).select(F.lit(1), F.lit(2)).toDF("a", "b").columns,
    ["a", "b"],
)
run(
    "df.unpivot(ids, values, variableColumnName=, valueColumnName=)",
    lambda: (
        spark.range(1)
        .select(F.lit(1).alias("a"), F.lit(2).alias("b"))
        .unpivot(["a"], ["b"], variableColumnName="var", valueColumnName="val")
        .collect()
    ),
)
check(
    "spark.createDataFrame(data, schema, verifySchema=True)",
    spark.createDataFrame([(1, "x")], "a int, b string", verifySchema=True).count(),
    1,
)
check("spark.sql(':p', p=...) via kwargs", spark.sql("SELECT :p AS v", p=7).collect()[0]["v"], 7)

# write/read round-trip exercising save(format=,mode=) + read.load(format=)
d = tempfile.mkdtemp()
p = os.path.join(d, "t")
run(
    "df.write.save(path, format='parquet', mode='overwrite')",
    lambda: spark.range(5).toDF("id").write.save(p, format="parquet", mode="overwrite"),
)
check(
    "spark.read.load(path, format='parquet') count", spark.read.load(p, format="parquet").count(), 5
)

print(f"\n==== e2e parity: {ok} OK, {fail} FAIL ====")
spark.stop()
raise SystemExit(1 if fail else 0)
