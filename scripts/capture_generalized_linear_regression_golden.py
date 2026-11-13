#!/usr/bin/env python3
"""Capture native GeneralizedLinearRegression params with official PySpark 4.3.0."""

import argparse
import base64
import json
from pathlib import Path

from pyspark.ml.connect.serialize import serialize_ml_params
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.sql import SparkSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="sc://localhost:15002")
    parser.add_argument(
        "--out", type=Path, default=Path("tests/golden/generalized_linear_regression.json")
    )
    args = parser.parse_args()
    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    try:
        cases = []
        for values in [
            {},
            {
                "labelCol": "target",
                "featuresCol": "input",
                "predictionCol": "pred",
                "family": "tweedie",
                "link": "log",
                "fitIntercept": False,
                "maxIter": 50,
                "tol": 1e-08,
                "regParam": 0.25,
                "weightCol": "weight",
                "solver": "irls",
                "linkPredictionCol": "eta",
                "variancePower": 1.5,
                "linkPower": 0.0,
                "offsetCol": "offset",
                "aggregationDepth": 3,
            },
        ]:
            estimator = GeneralizedLinearRegression(**values)
            params = serialize_ml_params(estimator, spark.client)
            cases.append(
                {
                    "values": values,
                    "params_b64": base64.b64encode(
                        params.SerializeToString(deterministic=True)
                    ).decode(),
                }
            )
        args.out.write_text(
            json.dumps(
                {
                    "source": "apache/spark v4.3.0 pyspark.ml.regression.GeneralizedLinearRegression",
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
