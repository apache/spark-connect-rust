#!/usr/bin/env python3
"""Capture golden Spark Connect protobuf plans/expressions from the real PySpark.

For a curated set of DataFrame operations and Column expressions, this serializes
the *unresolved* proto that the reference PySpark Connect client produces, then
normalizes out run-to-run noise (plan_id counters and Python `origin`/call-site
metadata). The result is a stable golden file the Rust plan/expression/function
builders are tested against for byte-level parity.

Usage:
    python3 scripts/capture_golden.py --remote sc://localhost:15002 \
        --out tests/golden

Requires a live Spark Connect server (the unresolved plan proto is built
client-side, but the client object is needed for plan-id allocation).
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def normalize(msg) -> None:
    """Recursively clear non-deterministic fields in a protobuf message in place.

    - ``plan_id`` (common.plan_id): a session-scoped counter.
    - ``origin`` / ``source_info``: Python source location metadata that a
      non-Python client will not reproduce identically.
    """
    from google.protobuf.message import Message

    for field, value in list(msg.ListFields()):
        if field.name in ("plan_id", "origin", "source_info"):
            msg.ClearField(field.name)
            continue
        if field.type == field.TYPE_MESSAGE:
            if field.label == field.LABEL_REPEATED:
                if field.message_type.GetOptions().map_entry:
                    # map<...>: values may be messages
                    for k in value:
                        v = value[k]
                        if isinstance(v, Message):
                            normalize(v)
                else:
                    for item in value:
                        if isinstance(item, Message):
                            normalize(item)
            elif isinstance(value, Message):
                normalize(value)


def cap_plan(spark, name, df):
    plan = df._plan.to_proto(spark.client)
    normalize(plan)
    return dict(
        name=name,
        kind="plan",
        b64=base64.b64encode(plan.SerializeToString()).decode(),
        text=str(plan),
    )


def cap_expr(spark, name, col):
    expr = col._expr.to_plan(spark.client)
    normalize(expr)
    return dict(
        name=name,
        kind="expr",
        b64=base64.b64encode(expr.SerializeToString()).decode(),
        text=str(expr),
    )


def plan_cases(spark):
    r = spark.range(10)
    yield "range", spark.range(10)
    yield "range_start_end_step", spark.range(2, 20, 3)
    yield "sql_select", spark.sql("SELECT 1 AS a, 'x' AS b")
    yield "filter_gt", r.filter(F.col("id") > 3)
    yield "where_gt", r.where(F.col("id") > 3)
    yield "select_alias", r.select((F.col("id") * 2).alias("x"))
    yield "select_star", r.select("*")
    yield "select_expr", r.selectExpr("id * 2 as x", "id + 1 as y")
    yield "with_column", r.withColumn("y", F.col("id") + F.lit(1))
    yield "with_column_renamed", r.withColumnRenamed("id", "n")
    yield "with_columns_renamed", r.withColumnsRenamed({"id": "n"})
    yield "drop", r.withColumn("y", F.lit(1)).drop("y")
    yield "to_df", r.toDF("renamed")
    yield "limit", r.limit(5)
    yield "offset", r.offset(3)
    yield "tail_via_limit", r.limit(5).offset(2)
    yield "distinct", r.select((F.col("id") % 2).alias("m")).distinct()
    yield "drop_duplicates", r.dropDuplicates(["id"])
    yield "sort", r.sort(F.col("id").desc())
    yield "order_by_multi", r.orderBy(F.col("id").asc_nulls_last())
    yield (
        "group_agg",
        r.groupBy((F.col("id") % 2).alias("k")).agg(
            F.count("*").alias("c"), F.sum("id").alias("s")
        ),
    )
    yield "group_count", r.groupBy("id").count()
    yield "cube", r.cube("id").count()
    yield "rollup", r.rollup("id").count()
    yield "pivot", r.groupBy("id").pivot("id").count()
    yield "union", spark.range(3).union(spark.range(3))
    yield "union_by_name", spark.range(3).unionByName(spark.range(3))
    yield "intersect", spark.range(5).intersect(spark.range(3))
    yield "intersect_all", spark.range(5).intersectAll(spark.range(3))
    yield "subtract", spark.range(5).subtract(spark.range(3))
    yield "except_all", spark.range(5).exceptAll(spark.range(3))
    yield (
        "join_inner",
        spark.range(5)
        .alias("a")
        .join(spark.range(5).alias("b"), F.col("a.id") == F.col("b.id"), "inner"),
    )
    yield (
        "join_left",
        spark.range(5)
        .alias("a")
        .join(spark.range(5).alias("b"), F.col("a.id") == F.col("b.id"), "left"),
    )
    yield "cross_join", spark.range(5).crossJoin(spark.range(3))
    yield "sample", r.sample(0.5, seed=42)
    yield "repartition", r.repartition(4)
    yield "repartition_by", r.repartition(4, F.col("id"))
    yield "coalesce", r.coalesce(1)
    yield "hint", r.hint("broadcast")
    yield "na_drop", r.withColumn("y", F.lit(1)).na.drop()
    yield "na_fill", r.withColumn("y", F.lit(1)).na.fill(0)
    yield "replace", r.replace(0, 100)
    yield "describe", r.describe("id")
    yield "summary", r.summary("count", "min")
    yield "col_regex", r.select(r.colRegex("`id`"))
    yield "unpivot", r.withColumn("y", F.lit(1)).unpivot(["id"], ["y"], "var", "val")


def expr_cases(spark):
    c = F.col("id")
    yield "col", F.col("x")
    yield "lit_int", F.lit(5)
    yield "lit_str", F.lit("hello")
    yield "lit_double", F.lit(3.14)
    yield "lit_bool", F.lit(True)
    yield "lit_null", F.lit(None)
    yield "add", c + 1
    yield "sub", c - 1
    yield "mul", c * 2
    yield "truediv", c / 2
    yield "mod", c % 3
    yield "eq", c == 5
    yield "gt", c > 5
    yield "and", (c > 1) & (c < 9)
    yield "or", (c > 1) | (c < 9)
    yield "not", ~(c > 1)
    yield "alias", c.alias("y")
    yield "cast", c.cast("string")
    yield "isnull", c.isNull()
    yield "when", F.when(c > 1, "a").when(c > 2, "b").otherwise("c")
    yield "upper", F.upper(F.col("s"))
    yield "concat", F.concat(F.col("a"), F.col("b"))
    yield "coalesce", F.coalesce(F.col("a"), F.col("b"), F.lit(0))
    yield "sum_agg", F.sum("id")
    yield "count_star", F.count("*")
    yield "substr", F.col("s").substr(1, 3)
    yield "getitem", F.col("m")["k"]
    yield "getfield", F.col("st").getField("f")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default="sc://localhost:15002")
    ap.add_argument("--out", default="tests/golden")
    args = ap.parse_args()

    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    plans = [cap_plan(spark, n, df) for n, df in plan_cases(spark)]
    exprs = [cap_expr(spark, n, c) for n, c in expr_cases(spark)]

    (out / "plans.jsonl").write_text("\n".join(json.dumps(p) for p in plans) + "\n")
    (out / "exprs.jsonl").write_text("\n".join(json.dumps(e) for e in exprs) + "\n")
    print(f"wrote {len(plans)} plans -> {out / 'plans.jsonl'}")
    print(f"wrote {len(exprs)} exprs -> {out / 'exprs.jsonl'}")
    spark.stop()


if __name__ == "__main__":
    main()
