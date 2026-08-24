#!/usr/bin/env python3
"""Capture golden Spark Connect ML protobuf for additional ML operators.

For additional ML operations (new feature transformers, classifiers, evaluators),
this serializes the *unresolved* proto that the reference PySpark Connect
client produces, then normalizes out run-to-run noise.

Usage:
    python3 scripts/capture_ml_extra_golden.py --remote sc://localhost:15002 \
        --out tests/golden

Requires a live Spark Connect server (Spark 4.0+ with ML support).
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from pyspark.ml.connect.classification import LogisticRegression
from pyspark.ml.connect.feature import (
    MaxAbsScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.connect.pipeline import Pipeline
from pyspark.sql import SparkSession


def normalize(msg) -> None:
    """Recursively clear non-deterministic fields in a protobuf message."""
    from google.protobuf.message import Message

    for field, value in list(msg.ListFields()):
        if field.name in ("plan_id", "origin", "source_info", "uid"):
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

    # VectorAssembler: transform columns into vector
    assembler = VectorAssembler(inputCols=["features"], outputCol="vector")
    assembled = assembler.transform(data)
    yield "vector_assembler_transform", assembled

    # StringIndexer: fit and transform
    string_data = spark.createDataFrame(
        [
            ("a",),
            ("b",),
            ("a",),
        ],
        schema=["category"],
    )
    indexer = StringIndexer(inputCol="category", outputCol="category_index")
    indexer_model = indexer.fit(string_data)
    indexed = indexer_model.transform(string_data)
    yield "string_indexer_transform", indexed

    # MaxAbsScaler: fit and transform
    max_abs_scaler = MaxAbsScaler(inputCol="features", outputCol="scaled_features")
    max_abs_model = max_abs_scaler.fit(data)
    scaled = max_abs_model.transform(data)
    yield "max_abs_scaler_transform", scaled

    # LogisticRegression: fit and transform
    label_data = spark.createDataFrame(
        [
            ([1.0, 2.0], 1.0),
            ([2.0, -1.0], 0.0),
            ([-3.0, -2.0], 0.0),
        ],
        schema=["features", "label"],
    )
    lr = LogisticRegression(featuresCol="features", labelCol="label", predictionCol="prediction")
    lr_model = lr.fit(label_data)
    predictions = lr_model.transform(label_data)
    yield "logistic_regression_transform", predictions

    # Pipeline: chain multiple stages
    pipeline = Pipeline(
        stages=[
            VectorAssembler(inputCols=["features"], outputCol="vector"),
            MaxAbsScaler(inputCol="vector", outputCol="scaled"),
        ]
    )
    pipeline_model = pipeline.fit(data)
    pipelined = pipeline_model.transform(data)
    yield "pipeline_transform", pipelined


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default="sc://localhost:15002")
    ap.add_argument("--out", default="tests/golden")
    args = ap.parse_args()

    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ml_relations = [cap_ml_relation(spark, n, df) for n, df in ml_relation_cases(spark)]

    (out / "ml_extra.jsonl").write_text("\n".join(json.dumps(m) for m in ml_relations) + "\n")
    print(f"wrote {len(ml_relations)} additional ML relations -> {out / 'ml_extra.jsonl'}")
    spark.stop()


if __name__ == "__main__":
    main()
