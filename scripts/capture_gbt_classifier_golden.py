#!/usr/bin/env python3
"""Capture GBTClassifier wire parameters using official PySpark 4.3.0."""

import argparse
import base64
import json
from pathlib import Path

from pyspark.ml.classification import GBTClassifier
from pyspark.ml.connect.serialize import serialize_ml_params
from pyspark.sql import SparkSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="sc://localhost:15002")
    parser.add_argument("--out", type=Path, default=Path("tests/golden/gbt_classifier.json"))
    args = parser.parse_args()
    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    # These literal-only parameters need no RPC, including when closing the client.
    spark.release_session_on_close = False
    try:
        cases = []
        for seed in [42, -42, 2**31 - 1, 2**31, -(2**31), -(2**31) - 1]:
            estimator = (
                GBTClassifier(
                    featuresCol="input",
                    labelCol="target",
                    predictionCol="class",
                    maxDepth=3,
                    maxBins=16,
                    minInstancesPerNode=2,
                    minInfoGain=0.01,
                    maxMemoryInMB=64,
                    cacheNodeIds=True,
                    checkpointInterval=-1,
                    impurity="variance",
                    featureSubsetStrategy="all",
                    seed=seed,
                    subsamplingRate=0.75,
                    leafCol="leaves",
                    minWeightFractionPerNode=0.1,
                    weightCol="weight",
                    lossType="logistic",
                    maxIter=4,
                    stepSize=0.2,
                    validationTol=0.05,
                    validationIndicatorCol="validation",
                )
                .setProbabilityCol("probs")
                .setRawPredictionCol("margin")
                .setThresholds([0.25, 0.75])
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
                    "source": "apache/spark v4.3.0 pyspark.ml.classification.GBTClassifier",
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
