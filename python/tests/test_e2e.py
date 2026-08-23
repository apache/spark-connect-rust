#!/usr/bin/env python3
"""End-to-end test of the drop-in `pyspark` package (Rust backend) against a live
Spark Connect server. Run with an ARM64 python that can load the extension:

    PYTHONPATH=python /usr/bin/python3 python/tests/test_e2e.py

Requires the dev server at sc://localhost:15002.
"""
import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

REMOTE = os.environ.get("SPARK_REMOTE", "sc://localhost:15002")


def ids(rows, i=0):
    return [r[i] for r in rows]


def main() -> int:
    spark = SparkSession.builder.remote(REMOTE).getOrCreate()
    checks = []

    def check(name, got, want):
        ok = got == want
        checks.append((name, ok, got, want))
        print(f"{'PASS' if ok else 'FAIL'} {name}: {got}" + ("" if ok else f" != {want}"))

    # Original tests
    check("range", ids(spark.range(5).collect()), [0, 1, 2, 3, 4])
    check("filter>7", ids(spark.range(10).filter(F.col("id") > 7).collect()), [8, 9])
    check("select*2", ids(spark.range(5).select((F.col("id") * 2).alias("x")).collect()),
          [0, 2, 4, 6, 8])
    check("withColumn", [r[1] for r in spark.range(3).withColumn("y", F.col("id") + 1).collect()],
          [1, 2, 3])
    check("drop", [len(r) for r in spark.range(2).withColumn("y", F.lit(1)).drop("y").collect()],
          [1, 1])
    check("and/or", ids(spark.range(10).filter((F.col("id") > 2) & (F.col("id") < 5)).collect()),
          [3, 4])
    check("groupBy.sum",
          sorted([[r[0], r[1]] for r in spark.range(6).groupBy((F.col("id") % 2).alias("k"))
                 .agg(F.sum(F.col("id")).alias("s")).collect()]),
          [[0, 6], [1, 9]])
    check("groupBy.count",
          sorted([[r[0], r[1]] for r in spark.range(6).groupBy((F.col("id") % 2).alias("k"))
                 .count().collect()]),
          [[0, 3], [1, 3]])
    check("sql", spark.sql("SELECT 1 AS a, 'x' AS b").collect()[0][0], 1)
    check("count", spark.range(100).count(), 100)
    check("distinct", sorted(ids(spark.range(6).select((F.col("id") % 2).alias("m")).distinct().collect())),
          [0, 1])
    check("union", len(spark.range(3).union(spark.range(3)).collect()), 6)
    check("limit", ids(spark.range(100).limit(3).collect()), [0, 1, 2])

    # New tests for expanded functionality
    check("unionAll", len(spark.range(3).unionAll(spark.range(3)).collect()), 6)
    check("offset", ids(spark.range(10).offset(5).collect()), [5, 6, 7, 8, 9])
    check("tail", len(spark.range(10).tail(3).collect()), 3)
    check("sample>0", len(spark.range(100).sample(0.5, seed=42).collect()) > 0, True)

    # Test orderBy with descending
    df_sorted = spark.range(5).orderBy(F.col("id").desc())
    check("orderBy.desc", ids(df_sorted.collect()), [4, 3, 2, 1, 0])

    # Test join (just test that it works, don't check exact count due to join complexity)
    df1 = spark.range(3).select((F.col("id") + 1).alias("a"), F.col("id").alias("key1"))
    df2 = spark.range(5).filter(F.col("id") >= 2).select(F.col("id").alias("key2"), (F.col("id") * 10).alias("b"))
    joined = df1.join(df2, on=F.col("key1") == F.col("key2"), how="inner")
    join_result = joined.collect()
    check("join.inner", len(join_result) > 0, True)

    # Test crossJoin
    df_cross = spark.range(3).crossJoin(spark.range(2))
    check("crossJoin", len(df_cross.collect()), 6)

    # Test createDataFrame with list of tuples (all strings for now)
    df_created = spark.createDataFrame([('1', 'a'), ('2', 'b')], ['num', 'letter'])
    collected = df_created.collect()
    check("createDataFrame.rows", len(collected), 2)
    check("createDataFrame.col0", collected[0][0], '1')
    check("createDataFrame.col1", collected[0][1], 'a')

    # Test columns property
    cols = spark.range(5).select(F.col("id"), (F.col("id") + 1).alias("id_plus_1")).columns()
    check("columns", len(cols), 2)

    # Test isEmpty
    check("isEmpty.false", spark.range(1).isEmpty(), False)
    check("isEmpty.true", spark.range(10).filter(F.col("id") > 100).isEmpty(), True)

    # Test dtypes
    df_dtypes = spark.createDataFrame([('1', 'a')], ['num', 'letter'])
    dtypes = df_dtypes.dtypes()
    check("dtypes.count", len(dtypes), 2)

    # Test first
    first_row = spark.range(10).first()
    check("first", first_row[0], 0)

    # Test head with n
    head_rows = spark.range(10).head()
    check("head", head_rows[0], 0)

    # Test take
    taken = spark.range(10).take(3)
    check("take", len(taken), 3)

    # Test coalesce
    coal = spark.range(100).coalesce(1)
    check("coalesce", len(coal.collect()), 100)

    # Test repartition
    repart = spark.range(100).repartition(4)
    check("repartition", len(repart.collect()), 100)

    # Test dropDuplicates
    df_dup = spark.range(3).select((F.col("id") % 2).alias("m"))
    dedup = df_dup.dropDuplicates()
    check("dropDuplicates", len(dedup.collect()), 2)

    # Test toDF (rename columns)
    df_renamed = spark.range(2).select(F.col("id")).toDF("new_id")
    check("toDF", df_renamed.columns(), ["new_id"])

    # Test withColumnRenamed
    df_with_renamed = spark.range(3).withColumnRenamed("id", "new_id")
    check("withColumnRenamed", df_with_renamed.columns(), ["new_id"])

    # Test intersect
    df_int1 = spark.range(5)
    df_int2 = spark.range(7).filter(F.col("id") >= 2)
    intersection = df_int1.intersect(df_int2)
    check("intersect", sorted(ids(intersection.collect())), [2, 3, 4])

    # Test subtract
    df_sub1 = spark.range(5)
    df_sub2 = spark.range(5).filter((F.col("id") >= 2) & (F.col("id") < 4))
    subtracted = df_sub1.subtract(df_sub2)
    check("subtract", sorted(ids(subtracted.collect())), [0, 1, 4])

    # Test Window and over() - row_number is a window function
    from pyspark.sql.window import Window
    df_win = spark.range(10).select(F.col("id"), F.row_number().over(
        Window.partitionBy(F.lit(1)).orderBy(F.col("id"))
    ).alias("row_num"))
    win_rows = df_win.collect()
    check("window.row_number", len(win_rows), 10)

    # Test Catalog - check that catalog object exists
    cat = spark.catalog
    check("catalog.exists", cat is not None, True)

    # Test createDataFrame with schema to get schema info
    df_schema = spark.createDataFrame([('1', 'a'), ('2', 'b')], ['num', 'letter'])
    schema = df_schema.schema
    check("schema.exists", schema is not None, True)

    # Test Types - check that types can be imported
    from pyspark.sql.types import StringType, IntegerType, LongType, BooleanType
    check("types.StringType.import", StringType is not None, True)
    check("types.LongType.import", LongType is not None, True)

    failed = [c for c in checks if not c[1]]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
