#!/usr/bin/env python3
"""Capture golden Spark Connect ML protobuf from the real PySpark.

For a curated set of ML transforms of the native `pyspark.ml` classes, which the Rust ML API
mirrors (not the deprecated `pyspark.ml.connect` ones, whose stages run pandas UDFs), this
records what the reference PySpark Connect client sends: the plan of the transformed DataFrame,
whose root is the MlRelation of the transform, and, of a fitted model, the ML commands of its fit
with the results that the server returns, which a replay of the fit needs.

The datasets are projections of ranges, which a client builds without a server, so the input of
a transform is compared too. Normalized as scripts/ml_golden_common.py normalizes its captures:
plan_id and origin are cleared, and the ObjectRef ids and the uids become <idN> and <uidN>
placeholders, numbered per case in order of first appearance, so that the model of a fit is the
<id0> of its transform. The messages are base64 of SerializeToString(deterministic=True).

Usage:
    python3 scripts/capture_ml_golden.py --remote sc://localhost:15002 \
        --out tests/golden

Requires a live Spark Connect server (Spark 4.0+ with ML support), which fits the models.
"""

from __future__ import annotations

import argparse
import base64
import gc
import json
import sys
from pathlib import Path

# The sibling module is importable when this one runs isolated (`python -I`) too.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ml_golden_common import _install, _scope, _scrub, _sink  # noqa: E402
from pyspark.ml.feature import StandardScaler  # noqa: E402
from pyspark.ml.functions import array_to_vector  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402


def features():
    """The vector [id, 1.0] of a row of a range."""
    return array_to_vector(F.array(F.col("id").cast("double"), F.lit(1.0)))


def vectors(spark, rows=3):
    """A vector column `features` of [id, 1.0] for each id of a range of `rows`."""
    return spark.range(rows).select(features().alias("features"))


def cap_ml_case(spark, name, transform):
    """Capture a case: `transform()` fits the stage of the case, if any, and transforms.

    The ML commands that the client sends meanwhile are the fit's, with the results it gets.
    Returns the entry, and the transformed DataFrame, which keeps the fitted model alive.
    """
    with _scope():
        with _sink() as fit:
            transformed = transform()
        plan = transformed._plan.to_proto(spark.client)
        # Scrubbed in the scope of the fit, so that the model's ObjectRef has the fit's placeholder.
        _scrub(plan)
    entry = dict(
        name=name,
        kind="ml_relation",
        b64=base64.b64encode(plan.SerializeToString(deterministic=True)).decode(),
        text=str(plan),
    )
    if fit["commands"]:
        entry["fit"] = {"commands": fit["commands"], "results": fit["results"]}
    return entry, transformed


def write_cases(spark, cases, path):
    """Capture `cases`, (name, transform) pairs, into the JSONL file `path`.

    A model that is released sends the Delete of its server-side object. The models are kept,
    and the collector is off, until every case is captured, so that no Delete is sent while the
    commands of a fit are recorded.
    """
    _install()
    gc.disable()
    try:
        captured = [cap_ml_case(spark, name, transform) for name, transform in cases]
    finally:
        gc.enable()
    entries = [entry for entry, _ in captured]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    print(f"wrote {len(entries)} ML relations -> {path}")


def ml_relation_cases(spark):
    """Yield (name, transform) for the ML transforms to capture."""

    def standard_scaler():
        data = vectors(spark)
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
        return scaler.fit(data).transform(data)

    yield "standard_scaler_transform", standard_scaler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default="sc://localhost:15002")
    ap.add_argument("--out", default="tests/golden")
    args = ap.parse_args()

    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    write_cases(spark, ml_relation_cases(spark), Path(args.out) / "ml.jsonl")
    spark.stop()


if __name__ == "__main__":
    main()
