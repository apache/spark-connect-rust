# SPDX-License-Identifier: Apache-2.0
"""Shared standard-ML statistics fixtures and result comparisons."""

import json
import math
import os
from pathlib import Path

import pytest

FIXTURE = json.loads(
    (Path(__file__).resolve().parents[2] / "tests/golden/statistics.json").read_text()
)
REFERENCE = os.environ.get("STATISTICS_REFERENCE") == "1"


def api():
    from pyspark.ml import stat

    if not REFERENCE:
        from pyspark import _pyspark

        for name in stat.__all__:
            assert getattr(stat, name).__module__ == "pyspark.ml.stat"
            if name == "MultivariateGaussian":
                # PySpark's plain holder, kept in Python as PySpark's.
                assert not hasattr(_pyspark, name)
                continue
            assert getattr(stat, name) is getattr(_pyspark, name)
    return stat


def dataset(spark, name):
    from pyspark.ml.functions import array_to_vector

    return spark.sql(FIXTURE["datasets"][name]).withColumn("features", array_to_vector("features"))


def helper(case, frame):
    stat = api()
    return {
        "chi": stat.ChiSquareTest.test,
        "corr": stat.Correlation.corr,
        "ks": stat.KolmogorovSmirnovTest.test,
    }[case["op"]](frame, *case["params"])


def summary(case):
    from pyspark.sql import functions as F

    summarizer = api().Summarizer
    weight = F.col("weight") if case["weighted"] else None
    if case["metric"] is None:
        return summarizer.metrics(*FIXTURE["metrics"]).summary(F.col("features"), weight)
    return getattr(summarizer, case["metric"])(F.col("features"), weight)


def normalize(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"float": str(value)}
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    return value


def assert_close(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_close(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert_close(a, e)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-10, abs=1e-10)
    else:
        assert actual == expected


def assert_frame(frame, expected):
    assert frame.columns == expected["columns"]
    rows = normalize(frame.toArrow().to_pylist())
    assert_close(sorted(rows, key=lambda row: json.dumps(row, sort_keys=True)), expected["rows"])
