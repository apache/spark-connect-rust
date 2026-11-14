#!/usr/bin/env python3
"""Capture the fitted feature models of official PySpark 4.3.0, with their requests.

Run with the official pyspark-client on PYTHONPATH and a live Connect server:

    python scripts/capture_feature_models_golden.py --remote sc://localhost:15002

Each estimator case of tests/golden/ml_features.json is fitted on that fixture's dataset with its
values (as the feature parity tests configure it), and its model gives the members of
scripts/ml_feature_models_table.py. Every transformer, model and estimator is saved and loaded
back.
The class methods that build a model client-side are captured with their transforms. Values,
errors and requests are recorded by scripts/ml_golden_common.py.
"""

import argparse
import json
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ml_feature_models_table import (  # noqa: E402
    ATTRIBUTES,
    CLASS_METHODS,
    arguments,
    persist_arguments,
)
from ml_golden_common import Golden  # noqa: E402
from pyspark.ml import feature  # noqa: E402
from pyspark.ml.functions import array_to_vector  # noqa: E402
from pyspark.ml.linalg import Vectors  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FEATURES = json.loads((ROOT / "tests/golden/ml_features.json").read_text())

# The model members that the tables cover elsewhere, which members() would otherwise call.
SKIP = {
    # QuantileDiscretizer's model is a transformer, which has the keyword setter of its params.
    "Bucketizer": {"setParams": "Params setter (client side)"},
    "CountVectorizerModel": {"from_vocabulary": "class method, in class_methods"},
    "StringIndexerModel": {
        "from_labels": "class method, in class_methods",
        "from_arrays_of_labels": "class method, in class_methods",
    },
}


def decode(value):
    # The values of tests/golden/ml_features.json, as python/tests/_ml_features.py decodes them.
    if isinstance(value, dict):
        if "float" in value:
            return float(value["float"])
        if "vector" in value:
            return Vectors.dense(value["vector"])
        return {key: decode(v) for key, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def configured(cls, values):
    # The stage of the values: the constructor takes those of its signature, the setters the rest.
    values = decode(values)
    defaults = FEATURES["api"][cls.__name__]["defaults"]
    stage = cls(**{k: v for k, v in values.items() if k in defaults})
    for key, value in values.items():
        if key not in defaults:
            getattr(stage, "set" + key[0].upper() + key[1:])(value)
    return stage


def capture(golden, frame, pairs):
    for case in FEATURES["cases"]:
        key = case["id"]
        stage = configured(getattr(feature, case["name"]), case["values"])
        if not FEATURES["api"][case["name"]]["estimator"]:
            golden.persist(key, stage, case["name"], "stage")
            continue
        model = golden.fit(key, stage, frame, key.replace("-", "_"))
        cls_name = type(model).__name__
        golden.record_members(
            golden.models,
            key,
            model,
            cls_name,
            "model",
            attributes=ATTRIBUTES[cls_name],
            arguments=partial(arguments, pairs=pairs),
            frame=frame,
            skip=SKIP.get(cls_name),
        )
        golden.persist(
            key + ".model",
            model,
            cls_name,
            "model",
            attributes=ATTRIBUTES[cls_name],
            arguments=persist_arguments,
            frame=frame,
        )
        golden.persist(key + ".estimator", stage, case["name"], "estimator")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="sc://localhost:15002")
    parser.add_argument("--out", type=Path, default=ROOT / "tests/golden/feature_models.json")
    args = parser.parse_args()
    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    golden = Golden(spark)
    try:
        # The frame of the feature parity tests: the SQL, then each vector column from its array.
        spec = {
            "sql": FEATURES["sql"],
            "vectors": {"features": "xs", "rawFeatures": "xs"},
            "coalesce": 1,
        }
        frame = spark.sql(spec["sql"])
        for column, source in spec["vectors"].items():
            frame = frame.withColumn(column, array_to_vector(source))
        frame = golden.register("features", frame.coalesce(spec["coalesce"]), spec)
        pairs = golden.register(
            "pairs",
            frame.select("id", "features"),
            {"of": "features", "select": ["id", "features"]},
        )
        capture(golden, frame, pairs)
        golden.class_methods(CLASS_METHODS, lambda name: getattr(feature, name), frame)
        golden.dump(
            args.out,
            {
                "fits": golden.fits,
                "models": golden.models,
                "class_methods": golden.class_methods_out,
            },
        )
        golden.dump(
            args.out.with_name("feature_models_persistence.json"),
            {"persistence": golden.persistence},
        )
    finally:
        golden.close()
        spark.stop()


if __name__ == "__main__":
    main()
