# SPDX-License-Identifier: Apache-2.0
"""Official Spark ML evaluator fixtures shared by offline and live tests."""

import json
from pathlib import Path

FIXTURE = json.loads(
    (Path(__file__).resolve().parents[2] / "tests/golden/evaluators.json").read_text()
)


def dataset(spark, spec):
    from pyspark.ml.functions import array_to_vector

    frame = spark.sql(spec["sql"])
    for column in spec["vectors"]:
        frame = frame.withColumn(column, array_to_vector(column))
    # One partition, as scripts/capture_evaluators_golden.py captures it: a binned metric of
    # BinaryClassificationMetrics depends on the partitioning, which local[*] takes from the cores.
    return frame.coalesce(1)
