#!/usr/bin/env python3
"""Capture golden Spark Connect ML protobuf for additional ML operators.

For more ML transforms of the native `pyspark.ml` classes (feature transformers and estimators,
a classifier, and a pipeline), this records what the reference PySpark Connect client sends, as
scripts/capture_ml_golden.py records and normalizes it: the plan of the transformed DataFrame,
and the ML commands of the fit of its model, with their results.

Usage:
    python3 scripts/capture_ml_extra_golden.py --remote sc://localhost:15002 \
        --out tests/golden

Requires a live Spark Connect server (Spark 4.0+ with ML support), which fits the models.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The sibling script is importable when this one runs isolated (`python -I`) too.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from capture_ml_golden import features, vectors, write_cases  # noqa: E402
from pyspark.ml import Pipeline  # noqa: E402
from pyspark.ml.classification import LogisticRegression  # noqa: E402
from pyspark.ml.feature import MaxAbsScaler, StringIndexer, VectorAssembler  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402


def ml_relation_cases(spark):
    """Yield (name, transform) for the ML transforms to capture."""

    # VectorAssembler: a transformer, which needs no fit.
    def vector_assembler():
        assembler = VectorAssembler(inputCols=["features"], outputCol="vector")
        return assembler.transform(vectors(spark))

    yield "vector_assembler_transform", vector_assembler

    # StringIndexer: fit and transform a string column.
    def string_indexer():
        categories = spark.range(3).select(F.col("id").cast("string").alias("category"))
        indexer = StringIndexer(inputCol="category", outputCol="category_index")
        return indexer.fit(categories).transform(categories)

    yield "string_indexer_transform", string_indexer

    # MaxAbsScaler: fit and transform.
    def max_abs_scaler():
        data = vectors(spark)
        scaler = MaxAbsScaler(inputCol="features", outputCol="scaled_features")
        return scaler.fit(data).transform(data)

    yield "max_abs_scaler_transform", max_abs_scaler

    # LogisticRegression: fit and transform a dataset of labels, 0 and 1 by turns.
    def logistic_regression():
        label = (F.col("id") % 2).cast("double")
        labelled = spark.range(4).select(features().alias("features"), label.alias("label"))
        lr = LogisticRegression(
            featuresCol="features", labelCol="label", predictionCol="prediction"
        )
        return lr.fit(labelled).transform(labelled)

    yield "logistic_regression_transform", logistic_regression

    # Pipeline: chain a transformer and an estimator, whose fit takes the transformer's output.
    def pipeline():
        data = vectors(spark)
        stages = [
            VectorAssembler(inputCols=["features"], outputCol="vector"),
            MaxAbsScaler(inputCol="vector", outputCol="scaled"),
        ]
        return Pipeline(stages=stages).fit(data).transform(data)

    yield "pipeline_transform", pipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default="sc://localhost:15002")
    ap.add_argument("--out", default="tests/golden")
    args = ap.parse_args()

    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    write_cases(spark, ml_relation_cases(spark), Path(args.out) / "ml_extra.jsonl")
    spark.stop()


if __name__ == "__main__":
    main()
