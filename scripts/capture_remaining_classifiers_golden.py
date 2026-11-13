#!/usr/bin/env python3
"""Capture classification parameter goldens using official PySpark 4.3.0.

OneVsRest's classifier is client-side orchestration state, not a literal parameter.
"""

import argparse
import base64
import json
from pathlib import Path

from pyspark.ml import classification
from pyspark.ml.connect.serialize import serialize_ml_params
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession

VALUES = {
    "FMClassifier": {
        "featuresCol": "input",
        "labelCol": "target",
        "predictionCol": "pred",
        "probabilityCol": "prob",
        "rawPredictionCol": "scores",
        "factorSize": 2,
        "fitIntercept": False,
        "fitLinear": False,
        "regParam": 0.1,
        "miniBatchFraction": 0.5,
        "initStd": 0.2,
        "maxIter": 50,
        "stepSize": 0.3,
        "tol": 1e-08,
        "solver": "gd",
        "thresholds": [1.0, 2.0],
        "seed": 8589934592,
    },
    "MultilayerPerceptronClassifier": {
        "featuresCol": "input",
        "labelCol": "target",
        "predictionCol": "pred",
        "maxIter": 50,
        "tol": 1e-08,
        "seed": 8589934592,
        "layers": [2, 3, 2],
        "blockSize": 4,
        "stepSize": 0.1,
        "solver": "gd",
        "initialWeights": {
            "dense": [
                0.0,
                0.01,
                0.02,
                0.03,
                0.04,
                0.05,
                0.06,
                0.07,
                0.08,
                0.09,
                0.1,
                0.11,
                0.12,
                0.13,
                0.14,
                0.15,
                0.16,
            ]
        },
        "probabilityCol": "prob",
        "rawPredictionCol": "scores",
        "thresholds": [1.0, 2.0],
    },
    "OneVsRest": {
        "featuresCol": "input",
        "labelCol": "target",
        "predictionCol": "pred",
        "rawPredictionCol": "scores",
        "weightCol": "weight",
        "parallelism": 3,
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="sc://localhost:15002")
    parser.add_argument("--out", type=Path, default=Path("tests/golden/remaining_classifiers.json"))
    args = parser.parse_args()
    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    cases = []
    try:
        for name, values in VALUES.items():
            variants = [{}, values]
            if name == "MultilayerPerceptronClassifier":
                variants.append(
                    dict(
                        values,
                        initialWeights={"size": 17, "indices": [0, 16], "values": [0.1, -0.2]},
                    )
                )
            for variant in variants:
                options = dict(variant)
                if "initialWeights" in options:
                    vector = options["initialWeights"]
                    options["initialWeights"] = (
                        Vectors.dense(vector["dense"])
                        if "dense" in vector
                        else Vectors.sparse(vector["size"], vector["indices"], vector["values"])
                    )
                thresholds = (
                    options.pop("thresholds", None)
                    if name == "MultilayerPerceptronClassifier"
                    else None
                )
                estimator = getattr(classification, name)(**options)
                if thresholds is not None:
                    estimator.setThresholds(thresholds)
                params = serialize_ml_params(estimator, spark.client)
                cases.append(
                    {
                        "class": name,
                        "values": variant,
                        "params_b64": base64.b64encode(
                            params.SerializeToString(deterministic=True)
                        ).decode(),
                    }
                )
        args.out.write_text(
            json.dumps(
                {"source": "apache/spark v4.3.0 pyspark.ml.classification", "cases": cases},
                indent=2,
            )
            + "\n"
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
