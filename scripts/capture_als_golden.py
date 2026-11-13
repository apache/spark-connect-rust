#!/usr/bin/env python3
"""Capture ALS fixtures with official PySpark 4.3.0 and a Spark Connect server."""

import argparse
import base64
import inspect
import json
import math
from pathlib import Path

import pyspark
from pyspark.ml.connect.serialize import serialize_ml_params
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession

TRAIN = "SELECT * FROM VALUES (0,0,5.0D),(0,1,1.0D),(0,2,2.0D),(1,0,1.0D),(1,1,5.0D),(1,3,3.0D),(2,0,3.0D),(2,2,5.0D),(2,3,1.0D),(3,1,2.0D),(3,2,1.0D),(3,3,5.0D) AS t(u,i,r)"
HELD = (
    "SELECT * FROM VALUES (0,0,3),(1,1,2),(2,2,1),(3,3,0),(4,99,0),(5,0,99),(6,99,99) AS t(id,u,i)"
)
USERS = "SELECT * FROM VALUES (0),(0),(2),(99) AS t(u)"
ITEMS = "SELECT * FROM VALUES (1),(3),(3),(99) AS t(i)"


def clean(value):
    if hasattr(value, "asDict"):
        return {k: clean(v) for k, v in value.asDict().items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"float": str(value)}
    return value


def result(frame):
    return {
        "schema": frame.schema.jsonValue(),
        "rows": sorted(
            [clean(row) for row in frame.collect()],
            key=lambda v: next(v[k] for k in ("id", "u", "i") if k in v),
        ),
    }


def encoded(message):
    return base64.b64encode(message.SerializeToString(deterministic=True)).decode()


def configured(values):
    defaults = inspect.signature(ALS).parameters
    estimator = ALS(**{k: v for k, v in values.items() if k in defaults})
    for key, value in values.items():
        if key not in defaults:
            getattr(estimator, "set" + key[0].upper() + key[1:])(value)
    return estimator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="sc://localhost:15002")
    parser.add_argument("--out", type=Path, default=Path("tests/golden/als.json"))
    args = parser.parse_args()
    assert pyspark.__version__ == "4.3.0"
    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    base = dict(
        rank=2,
        maxIter=8,
        regParam=0.1,
        numUserBlocks=1,
        numItemBlocks=1,
        implicitPrefs=False,
        alpha=1.0,
        userCol="u",
        itemCol="i",
        seed=42,
        ratingCol="r",
        nonnegative=False,
        checkpointInterval=-1,
        intermediateStorageLevel="MEMORY_AND_DISK",
        finalStorageLevel="MEMORY_AND_DISK",
        coldStartStrategy="nan",
        blockSize=2,
        predictionCol="score",
    )
    output = {
        "spark_version": pyspark.__version__,
        "train": TRAIN,
        "held": HELD,
        "users": USERS,
        "items": ITEMS,
        "defaults": {k: v.default for k, v in inspect.signature(ALS).parameters.items()},
        "params": [],
        "cases": [],
        "fetches": [],
    }
    try:
        variants = [{}, base, {k: None for k in base}] + [
            dict(base, seed=s) for s in [2**31 - 1, 2**31, 2**63 - 1, -(2**63)]
        ]
        for values in variants:
            stage = configured(values)
            output["params"].append(
                {"values": values, "params_b64": encoded(serialize_ml_params(stage, spark.client))}
            )
        for name, values in [
            ("explicit", base),
            ("implicit-nonnegative", dict(base, implicitPrefs=True, nonnegative=True, alpha=2.0)),
        ]:
            model = configured(values).fit(spark.sql(TRAIN).coalesce(1))
            case = {"name": name, "values": values, "rank": model.rank}
            tables = {
                "predictions": model.transform(spark.sql(HELD)),
                "user_factors": model.userFactors,
                "item_factors": model.itemFactors,
                "all_users": model.recommendForAllUsers(2),
                "all_items": model.recommendForAllItems(2),
                "user_subset": model.recommendForUserSubset(spark.sql(USERS), 2),
                "item_subset": model.recommendForItemSubset(spark.sql(ITEMS), 2),
            }
            for key, frame in tables.items():
                case[key] = result(frame)
            if name == "explicit":
                for count in [2, 2**31, None, True, 2.0, "2"]:
                    frame = model.recommendForAllUsers(count)
                    fetch = frame._plan.plan(spark.client).ml_relation.fetch
                    output["fetches"].append(
                        {
                            "count": count,
                            "method": fetch.methods[-1].method,
                            "arg_b64": encoded(fetch.methods[-1].args[0].param),
                        }
                    )
            model.setColdStartStrategy("drop").setPredictionCol("filtered")
            case["drop_predictions"] = result(model.transform(spark.sql(HELD)))
            output["cases"].append(case)
        args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
        print(
            "Captured explicit/implicit ALS predictions, factors, recommendations and parameter literals."
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
