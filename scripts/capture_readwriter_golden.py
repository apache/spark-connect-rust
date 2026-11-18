#!/usr/bin/env python3
"""Capture golden Spark Connect protobuf plans for read/write operations.

This script captures the unresolved protos for DataFrameReader and
DataFrameWriter operations from the reference PySpark Connect client: the
relation of each read, and the command of each write, as the request of the
write carries it, which a hook of the session takes and stops before it is
sent.

Usage:
    python3 scripts/capture_readwriter_golden.py --remote sc://localhost:15002 \
        --out tests/golden

Requires a live Spark Connect server.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from pyspark.sql.connect.session import SparkSession


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
            # A singular field is a message; a repeated one or a map is a container of them. The
            # value tells which, as protobuf 7 has no `FieldDescriptor.label`.
            if isinstance(value, Message):
                normalize(value)
            elif field.message_type.GetOptions().map_entry:
                # map<...>: values may be messages
                for k in value:
                    v = value[k]
                    if isinstance(v, Message):
                        normalize(v)
            else:
                for item in value:
                    if isinstance(item, Message):
                        normalize(item)


def cap_plan(spark, name, df):
    plan = df._plan.to_proto(spark.client)
    normalize(plan)
    return dict(
        name=name,
        kind="plan",
        b64=base64.b64encode(plan.SerializeToString()).decode(),
        text=str(plan),
    )


# The plans of the requests that the hook of the session stopped, in order.
_STOPPED_PLANS = []


class _Stopped(Exception):
    """Raised by the hook of the session, so that a write is not sent."""


class _WriteRecorder:
    """A hook of the session that keeps the plan of the request of a write and stops it."""

    def __init__(self, session) -> None:
        pass

    def on_execute_plan(self, request):
        _STOPPED_PLANS.append(request.plan)
        raise _Stopped()


def cap_write(name, write):
    """The command of the request that ``write``, a call of a DataFrameWriter, sends."""
    _STOPPED_PLANS.clear()
    try:
        write()
    except _Stopped:
        pass
    (plan,) = _STOPPED_PLANS
    normalize(plan)
    return dict(
        name=name,
        kind="command",
        b64=base64.b64encode(plan.SerializeToString()).decode(),
        text=str(plan),
    )


def write_cases(spark):
    """Generate write operation test cases, as the calls that send them."""
    df = spark.range(3)
    yield "write_jdbc", lambda: df.write.jdbc("jdbc:h2:mem:db", "people")
    # The mode and the properties, of which `url` and `dbtable` are set over the same keys.
    yield (
        "write_jdbc_with_mode_and_properties",
        lambda: df.write.jdbc(
            "jdbc:h2:mem:db",
            "people",
            mode="overwrite",
            properties={"user": "sa", "dbtable": "other", "url": "jdbc:other"},
        ),
    )


def read_write_cases(spark):
    """Generate read and write operation test cases."""
    # Create a simple test data path (would be used in real scenario)
    spark.range(10)

    # Read operations
    yield "read_json", spark.read.json("data/test.json")
    yield "read_parquet", spark.read.parquet("data/test.parquet")
    yield "read_csv", spark.read.csv("data/test.csv")
    yield "read_csv_with_option", (spark.read.option("header", "true").csv("data/test.csv"))
    yield "read_orc", spark.read.orc("data/test.orc")
    yield "read_text", spark.read.text("data/test.txt")
    yield "read_table", spark.read.table("my_table")

    # More read cases with options and schema
    yield (
        "read_parquet_with_option",
        (spark.read.option("mergeSchema", "true").parquet("data/test.parquet")),
    )
    yield "read_json_with_schema", (spark.read.schema("id INT, name STRING").json("data/test.json"))
    yield (
        "read_csv_with_multiple_options",
        (spark.read.option("header", "true").option("sep", ",").csv("data/test.csv")),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default="sc://localhost:15002")
    ap.add_argument("--out", default="tests/golden")
    args = ap.parse_args()

    spark = SparkSession.builder.remote(args.remote)._registerHook(_WriteRecorder).getOrCreate()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    plans = [cap_plan(spark, n, df) for n, df in read_write_cases(spark)]
    plans += [cap_write(n, write) for n, write in write_cases(spark)]

    (out / "readwriter.jsonl").write_text("\n".join(json.dumps(p) for p in plans) + "\n")
    print(f"wrote {len(plans)} read/write plans -> {out / 'readwriter.jsonl'}")
    spark.stop()


if __name__ == "__main__":
    main()
