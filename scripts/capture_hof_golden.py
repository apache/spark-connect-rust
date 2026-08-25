#!/usr/bin/env python3
"""Capture golden Expression protos for higher-order functions (lambda-taking).

These are the functions the generic auto-capture skips because they need Python
lambdas: transform, filter, exists, forall, aggregate, zip_with, map_filter,
transform_keys/values, map_zip_with, array_sort(comparator). Their protos use
LambdaFunction + UnresolvedNamedLambdaVariable, which the Rust expression layer
must support. Output: tests/golden/functions_hof.jsonl.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from capture_golden import normalize  # type: ignore
from pyspark.sql import SparkSession
from pyspark.sql.connect import functions as F


def cases(F):
    yield "transform", F.transform(F.col("a"), lambda x: x + 1)
    yield "transform_idx", F.transform(F.col("a"), lambda x, i: x + i)
    yield "filter", F.filter(F.col("a"), lambda x: x > 0)
    yield "exists", F.exists(F.col("a"), lambda x: x > 0)
    yield "forall", F.forall(F.col("a"), lambda x: x > 0)
    yield "aggregate", F.aggregate(F.col("a"), F.lit(0), lambda acc, x: acc + x)
    yield (
        "aggregate_finish",
        F.aggregate(F.col("a"), F.lit(0), lambda acc, x: acc + x, lambda acc: acc * 2),
    )
    yield "zip_with", F.zip_with(F.col("a"), F.col("b"), lambda x, y: x + y)
    yield "transform_keys", F.transform_keys(F.col("m"), lambda k, v: k)
    yield "transform_values", F.transform_values(F.col("m"), lambda k, v: v + 1)
    yield "map_filter", F.map_filter(F.col("m"), lambda k, v: v > 0)
    yield "map_zip_with", F.map_zip_with(F.col("m1"), F.col("m2"), lambda k, a, b: a + b)


def main() -> None:
    import sys

    remote = sys.argv[1] if len(sys.argv) > 1 else "sc://localhost:15002"
    spark = SparkSession.builder.remote(remote).getOrCreate()
    recs = []
    for name, col in cases(F):
        expr = col._expr.to_plan(spark.client)
        normalize(expr)
        recs.append(
            dict(
                name=name,
                kind="expr",
                b64=base64.b64encode(expr.SerializeToString()).decode(),
                text=str(expr),
            )
        )
    out = Path("tests/golden/functions_hof.jsonl")
    out.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    print(f"wrote {len(recs)} HOF goldens -> {out}")
    spark.stop()


if __name__ == "__main__":
    main()
