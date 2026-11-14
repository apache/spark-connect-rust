# SPDX-License-Identifier: Apache-2.0
"""Capture, on official PySpark, the cases of the pyspark.ml.evaluation API of the native evaluators
(python/tests/_evaluation_api_cases.py) for tests/golden/evaluation_api_cases.json, which
test_evaluation_api_offline.py and test_evaluation_api_e2e.py compare the native evaluators with.

    /tmp/scr-ref-py scripts/capture_evaluation_api_cases_golden.py --remote sc://localhost:15013 \\
        > tests/golden/evaluation_api_cases.json

Run it with official pyspark-client 4.3.0 against a Spark 4.3.0 Connect server on the same
machine: the saves of the cases go to a scratch directory of the local file system. The output
is byte-identical from run to run, which the capture is checked for.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "tests"))

from _evaluation_api_cases import CLASSES, live_cases, params_api_cases  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", default=os.environ.get("SPARK_REMOTE"))
    args = parser.parse_args()
    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    out = {}
    for name in CLASSES:
        base = tempfile.mkdtemp(prefix="evaluation-api-cases-")
        out[name] = {
            "params_api": params_api_cases(name),
            "live": live_cases(name, spark, base),
        }
    spark.stop()
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
