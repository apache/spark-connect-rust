#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Capture, on official PySpark, the cases of the native feature classes (transformers,
estimators and models; python/tests/_feature_api_cases.py) for tests/golden/feature_api_cases.json,
which test_feature_api_offline.py and test_feature_api_e2e.py compare the native classes with.

    python scripts/capture_feature_api_cases_golden.py --remote sc://localhost:15013 \\
        > tests/golden/feature_api_cases.json

Run it with the official pyspark-client 4.3.0 against a Spark 4.3.0 Connect server on the same
machine: the saves of the cases go to a scratch directory of the local file system.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "tests"))

from _feature_api_cases import (  # noqa: E402
    ESTIMATORS,
    MODELS,
    TRANSFORMERS,
    live_cases,
    offline_cases,
)
from pyspark.sql import SparkSession  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", required=True)
    args = parser.parse_args()
    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    out = {}
    for name in TRANSFORMERS + ESTIMATORS:
        base = tempfile.mkdtemp(prefix="feature-api-cases-")
        out[name] = {"offline": offline_cases(name), "live": live_cases(name, spark, base)}
    for name in MODELS:
        base = tempfile.mkdtemp(prefix="feature-api-cases-")
        out[name] = {"live": live_cases(name, spark, base)}
    spark.stop()
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
