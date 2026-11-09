#!/usr/bin/env python3
"""Capture native LinearRegression parameter wire goldens from official PySpark.

Run with official Spark 4.3.0 on PYTHONPATH and a Connect server available:
    python scripts/capture_linear_regression_golden.py --remote sc://localhost:15002
"""

import argparse
import base64
import json
from pathlib import Path

from pyspark.ml.connect.serialize import serialize_ml_params
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="sc://localhost:15002")
    parser.add_argument("--out", type=Path, default=Path("tests/golden/linear_regression.json"))
    args = parser.parse_args()
    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    try:
        estimator = LinearRegression(
            featuresCol="input",
            labelCol="target",
            predictionCol="pred",
            maxIter=50,
            regParam=0.25,
            elasticNetParam=0.5,
            tol=1e-5,
            fitIntercept=False,
            standardization=False,
            solver="l-bfgs",
            weightCol="weight",
            aggregationDepth=3,
            loss="squaredError",
            epsilon=1.5,
            maxBlockSizeInMB=2.0,
        )
        params = serialize_ml_params(estimator, spark.client)
        args.out.write_text(
            json.dumps(
                {
                    "source": "apache/spark v4.3.0 pyspark.ml.regression.LinearRegression",
                    "params_b64": base64.b64encode(
                        params.SerializeToString(deterministic=True)
                    ).decode(),
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
