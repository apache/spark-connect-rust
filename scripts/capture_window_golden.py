#!/usr/bin/env python3
"""Capture golden Spark Connect window expression protos from PySpark.

For window expressions built with the reference PySpark Window API, this
serializes the unresolved proto the PySpark Connect client produces, then
normalizes out run-to-run noise (plan_id counters and Python origin metadata).

Usage:
    python3 scripts/capture_window_golden.py --remote sc://localhost:15002 \
        --out tests/golden

Requires a live Spark Connect server.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def normalize(msg) -> None:
    """Recursively clear non-deterministic fields in a protobuf message in place."""
    from google.protobuf.message import Message

    for field, value in list(msg.ListFields()):
        if field.name in ("plan_id", "origin", "source_info"):
            msg.ClearField(field.name)
            continue
        if field.type == field.TYPE_MESSAGE:
            if field.label == field.LABEL_REPEATED:
                if field.message_type.GetOptions().map_entry:
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


def cap_expr(spark, name, col):
    expr = col._expr.to_plan(spark.client)
    normalize(expr)
    return dict(
        name=name,
        kind="expr",
        b64=base64.b64encode(expr.SerializeToString()).decode(),
        text=str(expr),
    )


def window_expr_cases(spark):
    """Generate window expression test cases."""
    # Basic window functions
    yield "window_row_number_partition", F.row_number().over(Window.partitionBy(F.col("a")))
    yield "window_row_number_orderby", F.row_number().over(Window.orderBy(F.col("b")))
    yield (
        "window_row_number_full",
        F.row_number().over(Window.partitionBy(F.col("a")).orderBy(F.col("b"))),
    )

    # Rank functions
    yield "window_rank", F.rank().over(Window.orderBy(F.col("b")))
    yield (
        "window_dense_rank",
        F.dense_rank().over(Window.partitionBy(F.col("a")).orderBy(F.col("b"))),
    )

    # Aggregate functions with window
    yield "window_sum_basic", F.sum(F.col("x")).over(Window.partitionBy(F.col("a")))
    yield "window_count_basic", F.count(F.col("x")).over(Window.partitionBy(F.col("a")))
    yield "window_avg_basic", F.avg(F.col("x")).over(Window.orderBy(F.col("b")))

    # Window with frame bounds (ROWS)
    yield (
        "window_sum_rows_unbounded_to_current",
        F.sum(F.col("x")).over(
            Window.partitionBy(F.col("a"))
            .orderBy(F.col("b"))
            .rowsBetween(Window.unboundedPreceding, Window.currentRow)
        ),
    )

    yield (
        "window_sum_rows_unbounded_to_unbounded",
        F.sum(F.col("x")).over(
            Window.partitionBy(F.col("a"))
            .orderBy(F.col("b"))
            .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        ),
    )

    # Window with frame bounds (RANGE)
    yield (
        "window_sum_range_unbounded_to_current",
        F.sum(F.col("x")).over(
            Window.partitionBy(F.col("a"))
            .orderBy(F.col("b"))
            .rangeBetween(Window.unboundedPreceding, Window.currentRow)
        ),
    )

    # Complex ordering (descending)
    yield "window_rank_desc", F.rank().over(Window.orderBy(F.col("b").desc()))

    # Multiple partitions and order by
    yield (
        "window_row_number_multi_partition",
        F.row_number().over(Window.partitionBy(F.col("a"), F.col("b")).orderBy(F.col("c"))),
    )

    # Lag/Lead functions (window functions)
    yield "window_lag", F.lag(F.col("x")).over(Window.partitionBy(F.col("a")).orderBy(F.col("b")))

    yield "window_lead", F.lead(F.col("x")).over(Window.partitionBy(F.col("a")).orderBy(F.col("b")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default="sc://localhost:15002")
    ap.add_argument("--out", default="tests/golden")
    args = ap.parse_args()

    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    exprs = [cap_expr(spark, n, c) for n, c in window_expr_cases(spark)]

    (out / "window.jsonl").write_text("\n".join(json.dumps(e) for e in exprs) + "\n")
    print(f"wrote {len(exprs)} window exprs -> {out / 'window.jsonl'}")
    spark.stop()


if __name__ == "__main__":
    main()
