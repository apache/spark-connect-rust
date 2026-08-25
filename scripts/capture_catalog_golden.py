#!/usr/bin/env python3
"""Capture golden Spark Connect protobuf catalog operations from the real PySpark.

For a curated set of Catalog operations, this serializes the proto that the reference
PySpark Connect client produces, then normalizes out run-to-run noise (plan_id counters
and Python `origin`/call-site metadata). The result is a stable golden file the Rust
catalog builders are tested against for byte-level parity.

Usage:
    python3 scripts/capture_catalog_golden.py --remote sc://localhost:15002 \
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


def cap_catalog_plan(spark, name, catalog_func):
    """Capture a catalog operation that returns a DataFrame."""
    try:
        df = catalog_func(spark.catalog)
        plan = df._plan.to_proto(spark.client)
        normalize(plan)
        return dict(
            name=name,
            kind="plan",
            b64=base64.b64encode(plan.SerializeToString()).decode(),
            text=str(plan),
        )
    except Exception as e:
        print(f"Warning: Could not capture {name}: {e}")
        return None


def cap_catalog_scalar(spark, name, catalog_func):
    """Capture a catalog operation that returns a scalar value (string or bool)."""
    try:
        # For scalar results, we can't directly get a plan like we do for DataFrames
        # Instead, we'll try to get the internal plan if available
        result = catalog_func(spark.catalog)
        # For now, just print a note - these might need special handling
        print(f"Note: {name} returns scalar value: {result}")
        return None
    except Exception as e:
        print(f"Warning: Could not capture {name}: {e}")
        return None


def catalog_cases(spark):
    """Yield (name, capture_func, kind) tuples for catalog operations."""
    # DataFrame-returning operations
    yield ("list_catalogs", lambda cat: cat.listCatalogs(), "plan")
    yield ("list_databases", lambda cat: cat.listDatabases(), "plan")
    yield ("list_tables", lambda cat: cat.listTables(), "plan")
    yield ("list_columns", lambda cat: cat.listColumns("spark_catalog"), "plan")
    yield ("list_functions", lambda cat: cat.listFunctions(), "plan")

    # Scalar-returning operations (strings, booleans)
    # These are harder to capture as plans, but we include them for reference
    yield ("current_database", lambda cat: cat.currentDatabase(), "scalar")
    yield ("current_catalog", lambda cat: cat.currentCatalog(), "scalar")
    yield ("table_exists", lambda cat: cat.tableExists("nonexistent"), "scalar")
    yield ("database_exists", lambda cat: cat.databaseExists("default"), "scalar")
    yield ("function_exists", lambda cat: cat.functionExists("sum"), "scalar")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default="sc://localhost:15002")
    ap.add_argument("--out", default="tests/golden")
    args = ap.parse_args()

    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    plans = []
    for name, func, kind in catalog_cases(spark):
        if kind == "plan":
            result = cap_catalog_plan(spark, name, func)
            if result:
                plans.append(result)
        elif kind == "scalar":
            result = cap_catalog_scalar(spark, name, func)
            if result:
                plans.append(result)

    (out / "catalog.jsonl").write_text("\n".join(json.dumps(p) for p in plans) + "\n")
    print(f"wrote {len(plans)} catalog operations -> {out / 'catalog.jsonl'}")
    spark.stop()


if __name__ == "__main__":
    main()
