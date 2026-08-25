#!/usr/bin/env python3
"""Capture golden Spark Connect ML protobuf from the real PySpark.

For a curated set of ML operations (estimators, transformers, models),
this serializes the *unresolved* proto that the reference PySpark Connect
client produces, then normalizes out run-to-run noise.

Usage:
    python3 scripts/capture_ml_golden.py --remote sc://localhost:15002 \
        --out tests/golden

Requires a live Spark Connect server (Spark 4.0+ with ML support).
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from pyspark.ml.connect.feature import StandardScaler
from pyspark.sql import SparkSession


def normalize(msg) -> None:
    """Recursively clear non-deterministic fields in a protobuf message.

    - ``plan_id``: a session-scoped counter.
    - ``uid``: model/operator instance UIDs can vary.
    - ``origin`` / ``source_info``: Python source location metadata.
    """
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


def cap_ml_relation(spark, name, df):
    """Capture an ML relation proto from a transformed DataFrame."""
    plan = df._plan.to_proto(spark.client)
    normalize(plan)
    return dict(
        name=name,
        kind="ml_relation",
        b64=base64.b64encode(plan.SerializeToString()).decode(),
        text=str(plan),
    )


def ml_relation_cases(spark):
    """Yield (name, DataFrame) tuples for ML operations to capture."""
    # Create a sample dataset
    data = spark.createDataFrame(
        [
            ([1.0, 2.0],),
            ([2.0, -1.0],),
            ([-3.0, -2.0],),
        ],
        schema=["features"],
    )

    # StandardScaler: fit and transform
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    scaler_model = scaler.fit(data)
    transformed = scaler_model.transform(data)

    yield "standard_scaler_transform", transformed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default="sc://localhost:15002")
    ap.add_argument("--out", default="tests/golden")
    args = ap.parse_args()

    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ml_relations = [cap_ml_relation(spark, n, df) for n, df in ml_relation_cases(spark)]

    (out / "ml.jsonl").write_text("\n".join(json.dumps(m) for m in ml_relations) + "\n")
    print(f"wrote {len(ml_relations)} ML relations -> {out / 'ml.jsonl'}")
    spark.stop()


if __name__ == "__main__":
    main()
