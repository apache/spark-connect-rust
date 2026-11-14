#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture standard pyspark.ml.stat results and protocol with official PySpark 4.3.0."""

import argparse
import base64
import inspect
import json
import math
from pathlib import Path

import pyspark
from pyspark.ml import stat
from pyspark.ml.functions import array_to_vector
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

METRICS = [
    "mean",
    "sum",
    "variance",
    "std",
    "count",
    "numNonZeros",
    "max",
    "min",
    "normL1",
    "normL2",
]
DATASETS = {
    "categorical": "SELECT * FROM VALUES (0D,array(0D,1D)),(1D,array(1D,1D)),(1D,array(2D,1D)),(0D,array(0D,2D)),(1D,array(1D,2D)),(0D,array(2D,2D)) AS t(label,features)",
    "numeric": "SELECT * FROM VALUES (0,array(1D,0D,-2D),0.5D,-1D),(0,array(2D,0D,3D),2D,0D),(1,array(-1D,0D,0D),0D,1D),(1,array(4D,0D,5D),1.5D,2D) AS t(group_id,features,weight,sample)",
}


def normalize(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"float": str(value)}
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    return value


def result(frame):
    rows = normalize(frame.toArrow().to_pylist())
    return {
        "columns": frame.columns,
        "rows": sorted(rows, key=lambda row: json.dumps(row, sort_keys=True)),
    }


def encoded(message):
    return base64.b64encode(message.SerializeToString(deterministic=True)).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="sc://localhost:15002")
    parser.add_argument("--out", type=Path, default=Path("tests/golden/statistics.json"))
    args = parser.parse_args()
    assert pyspark.__version__ == "4.3.0"
    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    fixture = {
        "spark_version": pyspark.__version__,
        "datasets": DATASETS,
        "metrics": METRICS,
        "signatures": {},
        "helpers": [],
        "summaries": [],
    }
    try:
        for cls, methods in {
            "ChiSquareTest": ["test"],
            "Correlation": ["corr"],
            "KolmogorovSmirnovTest": ["test"],
            "Summarizer": ["metrics"] + METRICS,
            "SummaryBuilder": ["summary"],
            "MultivariateGaussian": ["__init__"],
        }.items():
            for method in methods:
                fixture["signatures"][cls + "." + method] = [
                    {
                        "name": p.name,
                        "kind": p.kind.name,
                        **(
                            {"default": p.default}
                            if p.default is not inspect.Parameter.empty
                            else {}
                        ),
                    }
                    for p in inspect.signature(
                        getattr(getattr(stat, cls), method)
                    ).parameters.values()
                    if p.name != "self"
                ]
        frames = {
            name: spark.sql(sql).withColumn("features", array_to_vector("features"))
            for name, sql in DATASETS.items()
        }
        specs = [
            ("chi", "categorical", ["features", "label", flatten]) for flatten in [False, True]
        ]
        specs += [("corr", "numeric", ["features", method]) for method in ["pearson", "spearman"]]
        specs += [
            ("ks", "numeric", ["sample", "norm"] + params)
            for params in [[], [0.0, 1.0], [1.0, 2.0], ["1", "2"]]
        ]
        for operation, name, params in specs:
            function = {
                "chi": stat.ChiSquareTest.test,
                "corr": stat.Correlation.corr,
                "ks": stat.KolmogorovSmirnovTest.test,
            }[operation]
            frame = function(frames[name], *params)
            fetch = frame._plan.plan(spark.client).ml_relation.fetch
            fixture["helpers"].append(
                {
                    "op": operation,
                    "dataset": name,
                    "params": params,
                    "method": fetch.methods[0].method,
                    "args_b64": [encoded(arg.param) for arg in fetch.methods[0].args[1:]],
                    "result": result(frame),
                }
            )
        for weighted in [False, True]:
            weight = F.col("weight") if weighted else None
            for metric in [None] + METRICS:
                column = (
                    stat.Summarizer.metrics(*METRICS).summary(F.col("features"), weight)
                    if metric is None
                    else getattr(stat.Summarizer, metric)(F.col("features"), weight)
                )
                fixture["summaries"].append(
                    {
                        "metric": metric,
                        "weighted": weighted,
                        "expr_b64": encoded(column._expr.to_plan(spark.client)),
                        "result": result(frames["numeric"].select(column)),
                    }
                )
        fixture["grouped"] = result(
            frames["numeric"]
            .groupBy("group_id")
            .agg(
                stat.Summarizer.metrics("mean", "count")
                .summary(F.col("features"), F.col("weight"))
                .alias("summary")
            )
        )
        args.out.write_text(json.dumps(fixture, indent=2, allow_nan=False) + "\n")
        print(
            f"Captured {len(fixture['helpers'])} helpers and {len(fixture['summaries'])} summaries."
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
