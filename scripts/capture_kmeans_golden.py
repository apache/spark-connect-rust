#!/usr/bin/env python3
"""Capture KMeans parameter wire goldens using official PySpark 4.3.0."""

import argparse
import base64
import json
from pathlib import Path

from pyspark.ml.clustering import KMeans
from pyspark.ml.connect.serialize import serialize_ml_params
from pyspark.sql import SparkSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="sc://localhost:15002")
    parser.add_argument("--out", type=Path, default=Path("tests/golden/kmeans.json"))
    args = parser.parse_args()
    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    try:
        cases = []
        for seed in [42, -42, 2**31 - 1, 2**31, -(2**31), -(2**31) - 1]:
            estimator = KMeans(
                featuresCol="input",
                predictionCol="cluster",
                k=3,
                initMode="random",
                initSteps=3,
                tol=1e-5,
                maxIter=50,
                seed=seed,
                distanceMeasure="euclidean",
                weightCol="weight",
                solver="block",
                maxBlockSizeInMB=2.0,
            )
            params = serialize_ml_params(estimator, spark.client)
            cases.append(
                {
                    "seed": seed,
                    "params_b64": base64.b64encode(
                        params.SerializeToString(deterministic=True)
                    ).decode(),
                }
            )
        args.out.write_text(
            json.dumps(
                {
                    "source": "apache/spark v4.3.0 pyspark.ml.clustering.KMeans",
                    "cases": cases,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
