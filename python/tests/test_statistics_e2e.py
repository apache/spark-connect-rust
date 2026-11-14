# SPDX-License-Identifier: Apache-2.0
"""Reference-backed tests for all standard ML statistics on a live Connect server."""

import gc
import os

import numpy as np
import pytest
from _statistics_cases import FIXTURE, REFERENCE, api, assert_frame, dataset, helper, summary
from pyspark.ml.linalg import DenseMatrix, DenseVector, Vectors
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


@pytest.fixture(scope="module")
def session():
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live statistics tests")
    spark = SparkSession.builder.remote(remote).getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture(scope="module")
def frames(session):
    return {name: dataset(session, name) for name in FIXTURE["datasets"]}


@pytest.mark.parametrize("case", FIXTURE["helpers"])
def test_tests_and_correlations_match_official(case, frames):
    result = helper(case, frames[case["dataset"]])
    assert_frame(result, case["result"])
    row = result.first()
    if case["op"] == "corr":
        assert isinstance(row[0], DenseMatrix)
        assert np.isnan(row[0].toArray()[0, 1])
    elif case["op"] == "chi" and not case["params"][-1]:
        assert isinstance(row.pValues, DenseVector)
        assert isinstance(row.statistics, DenseVector)


@pytest.mark.parametrize("case", FIXTURE["summaries"])
def test_all_ten_weighted_and_unweighted_summaries(case, frames):
    frame = frames["numeric"].select(summary(case))
    assert_frame(frame, case["result"])
    if case["metric"] not in {None, "count"}:
        assert isinstance(frame.first()[0], DenseVector)


def test_grouped_summary(frames):
    frame = (
        frames["numeric"]
        .groupBy("group_id")
        .agg(
            api()
            .Summarizer.metrics("mean", "count")
            .summary(F.col("features"), F.col("weight"))
            .alias("summary")
        )
    )
    assert_frame(frame, FIXTURE["grouped"])


@pytest.mark.parametrize("op", ["chi", "corr", "summary"])
def test_sparse_and_dense_vectors_agree(session, op):
    dense = session.createDataFrame(
        [
            (float(i % 2), Vectors.dense(v))
            for i, v in enumerate([[0, 1, 0], [1, 0, 2], [0, 3, 0], [2, 0, 1]])
        ],
        ["label", "features"],
    )
    sparse = session.createDataFrame(
        [
            (float(i % 2), Vectors.sparse(3, {j: float(x) for j, x in enumerate(v) if x}))
            for i, v in enumerate([[0, 1, 0], [1, 0, 2], [0, 3, 0], [2, 0, 1]])
        ],
        ["label", "features"],
    )
    stat = api()
    operation = {
        "chi": lambda f: stat.ChiSquareTest.test(f, "features", "label"),
        "corr": lambda f: stat.Correlation.corr(f, "features"),
        "summary": lambda f: f.select(
            stat.Summarizer.metrics(*FIXTURE["metrics"]).summary(F.col("features"))
        ),
    }[op]
    expected = operation(dense)
    assert_frame(
        operation(sparse), {"columns": expected.columns, "rows": expected.toArrow().to_pylist()}
    )


@pytest.mark.parametrize("metric", ["", "invalid", "numNonzeros"])
def test_invalid_metrics_are_rejected_by_spark(frames, metric):
    # Metric names are case sensitive, including the capital Z in numNonZeros.
    with pytest.raises(Exception):
        frames["numeric"].select(
            api().Summarizer.metrics(metric).summary(F.col("features"))
        ).collect()


@pytest.mark.parametrize(
    "kind", ["method", "distribution", "ks-arity", "missing", "array", "weight"]
)
def test_invalid_statistics_inputs(frames, kind):
    stat = api()
    frame = frames["numeric"]
    with pytest.raises(Exception):
        if kind == "method":
            stat.Correlation.corr(frame, "features", "invalid").collect()
        elif kind == "distribution":
            stat.KolmogorovSmirnovTest.test(frame, "sample", "invalid").collect()
        elif kind == "ks-arity":
            stat.KolmogorovSmirnovTest.test(frame, "sample", "norm", 0.0).collect()
        elif kind == "missing":
            stat.ChiSquareTest.test(frame, "missing", "group_id").collect()
        elif kind == "array":
            stat.Correlation.corr(
                frame.select(F.array(F.lit(1.0)).alias("features")), "features"
            ).collect()
        elif kind == "weight":
            frame.select(stat.Summarizer.mean(F.col("features"), F.lit(-1.0))).collect()


def test_lazy_statistics_retain_upstream_model(frames):
    if REFERENCE:
        from pyspark.ml.feature import StandardScaler
    else:
        from pyspark._pyspark import StandardScaler
    stat = api()
    model = StandardScaler(
        inputCol="features", outputCol="scaled", withStd=False, withMean=False
    ).fit(frames["categorical"])
    transformed = model.transform(frames["categorical"])
    results = [
        stat.ChiSquareTest.test(transformed, "scaled", "label"),
        stat.Correlation.corr(transformed, "scaled"),
        stat.KolmogorovSmirnovTest.test(transformed, "label", "norm"),
        transformed.select(stat.Summarizer.mean(F.col("scaled"))),
    ]
    if not REFERENCE:
        del transformed, model
        gc.collect()
    for result in results:
        assert result.count() == 1


def test_empty_metrics_returns_an_empty_struct(frames):
    frame = frames["numeric"].select(
        api().Summarizer.metrics().summary(F.col("features")).alias("s")
    )
    assert frame.toArrow().to_pylist() == [{"s": {}}]
