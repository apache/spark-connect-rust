# SPDX-License-Identifier: Apache-2.0
"""Capture, on official PySpark, the cases of the pyspark.ml API of NaiveBayes,
MultilayerPerceptronClassifier, FMClassifier and OneVsRest (python/tests/_misc_api_cases.py) for
tests/golden/misc_api_cases.json, which test_misc_api_offline.py and test_misc_api_e2e.py compare
the native classes with.

    SPARK_REMOTE=sc://localhost:15002 python scripts/capture_misc_api_cases_golden.py \\
        > tests/golden/misc_api_cases.json

Run it with official pyspark-client 4.3.0 against a Spark 4.3.0 Connect server on the same
machine: the saves of the cases go to a scratch directory of the local file system.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "tests"))

from _misc_api_cases import CLASSES, live_cases, params_api_cases  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402


def main():
    spark = SparkSession.builder.remote(os.environ["SPARK_REMOTE"]).getOrCreate()
    out = {}
    for name in CLASSES:
        base = tempfile.mkdtemp(prefix="misc-api-cases-")
        out[name] = {
            "params_api": params_api_cases(name),
            "live": live_cases(name, spark, base),
        }
    spark.stop()
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
