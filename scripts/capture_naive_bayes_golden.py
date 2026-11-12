#!/usr/bin/env python3
"""Capture NaiveBayes wire parameters using official PySpark 4.3.0."""

import argparse
import base64
import json
from pathlib import Path

from pyspark.ml.classification import NaiveBayes
from pyspark.ml.connect.serialize import serialize_ml_params
from pyspark.sql import SparkSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="sc://localhost:15002")
    parser.add_argument("--out", type=Path, default=Path("tests/golden/naive_bayes.json"))
    args = parser.parse_args()
    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    spark.release_session_on_close = False
    try:
        cases = []
        for model_type in ["multinomial", "bernoulli", "complement", "gaussian"]:
            estimator = NaiveBayes(
                featuresCol="input",
                labelCol="target",
                predictionCol="class",
                probabilityCol="probs",
                rawPredictionCol="scores",
                smoothing=0.25,
                modelType=model_type,
                thresholds=[0.25, 0.5, 1.0],
                weightCol="weight",
            )
            params = serialize_ml_params(estimator, spark.client)
            cases.append(
                {
                    "model_type": model_type,
                    "params_b64": base64.b64encode(
                        params.SerializeToString(deterministic=True)
                    ).decode(),
                }
            )
        args.out.write_text(
            json.dumps(
                {
                    "source": "apache/spark v4.3.0 pyspark.ml.classification.NaiveBayes",
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
