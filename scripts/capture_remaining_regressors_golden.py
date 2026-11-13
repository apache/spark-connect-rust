#!/usr/bin/env python3
"""Capture parameter and AFT prediction goldens using official PySpark 4.3.0."""

import argparse
import base64
import inspect
import json
from pathlib import Path

from pyspark.ml import regression
from pyspark.ml.connect.serialize import serialize_ml_params
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession

VALUES = {
    "AFTSurvivalRegression": {
        "featuresCol": "input",
        "labelCol": "duration",
        "predictionCol": "pred",
        "fitIntercept": False,
        "maxIter": 50,
        "tol": 1e-08,
        "censorCol": "observed",
        "quantileProbabilities": [0.2, 0.5, 0.8],
        "quantilesCol": "quantiles",
        "aggregationDepth": 3,
        "maxBlockSizeInMB": 1.0,
    },
    "IsotonicRegression": {
        "featuresCol": "input",
        "labelCol": "target",
        "predictionCol": "pred",
        "weightCol": "weight",
        "isotonic": False,
        "featureIndex": 1,
    },
    "FMRegressor": {
        "featuresCol": "input",
        "labelCol": "target",
        "predictionCol": "pred",
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
        "seed": 8589934592,
    },
}

AFT_ROWS = [
    (1.218, 1.0, [1.560, -0.605]),
    (2.949, 0.0, [0.346, 2.158]),
    (3.627, 0.0, [1.380, 0.231]),
    (0.273, 1.0, [0.520, 1.151]),
    (4.199, 0.0, [0.795, -0.226]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="sc://localhost:15002")
    parser.add_argument("--out", type=Path, default=Path("tests/golden/remaining_regressors.json"))
    args = parser.parse_args()
    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    cases, defaults, aft = [], {}, []
    try:
        for name, values in VALUES.items():
            cls = getattr(regression, name)
            defaults[name] = {n: p.default for n, p in inspect.signature(cls).parameters.items()}
            for variant in [{}, values]:
                estimator = cls(**variant)
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
        for fit_intercept in [False, True]:
            for censored in [False, True]:
                rows = [
                    (i, y, c if censored else 1.0, Vectors.dense(x))
                    for i, (y, c, x) in enumerate(AFT_ROWS)
                ]
                train = spark.createDataFrame(rows, ["id", "duration", "observed", "input"])
                options = dict(
                    featuresCol="input",
                    labelCol="duration",
                    predictionCol="pred",
                    censorCol="observed",
                    fitIntercept=fit_intercept,
                    maxIter=100,
                    tol=1e-8,
                    quantileProbabilities=[0.2, 0.5, 0.8],
                    quantilesCol="quantiles",
                    aggregationDepth=2,
                    maxBlockSizeInMB=1.0,
                )
                model = regression.AFTSurvivalRegression(**options).fit(train)
                results = model.transform(train).orderBy("id").select("pred", "quantiles").collect()
                aft.append(
                    {
                        "options": options,
                        "censored": censored,
                        "predictions": [r.pred for r in results],
                        "quantiles": [list(r.quantiles) for r in results],
                    }
                )
        args.out.write_text(
            json.dumps(
                {
                    "source": "apache/spark v4.3.0 pyspark.ml.regression",
                    "defaults": defaults,
                    "cases": cases,
                    "aft_rows": AFT_ROWS,
                    "aft": aft,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
