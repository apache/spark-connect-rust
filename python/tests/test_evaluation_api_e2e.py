# SPDX-License-Identifier: Apache-2.0
"""The pyspark.ml.evaluation API of the native evaluators, against a live Connect server: the
evaluate cases (with and without a param map), isLargerBetter and the persistence of each evaluator
(`_evaluation_api_cases.live_cases`) against tests/golden/evaluation_api_cases.json, from official
PySpark (see scripts/capture_evaluation_api_cases_golden.py). Official PySpark runs them too, as the
reference (`_native_classes.native_class`).
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
from _evaluation_api_cases import CLASSES, live_cases
from pyspark.sql import SparkSession

CASES = json.loads(
    (Path(__file__).resolve().parents[2] / "tests/golden/evaluation_api_cases.json").read_text()
)


@pytest.fixture(scope="module")
def session():
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live evaluation API tests")
    spark = SparkSession.builder.remote(remote).getOrCreate()
    yield spark
    spark.stop()


@pytest.mark.parametrize("name", CLASSES)
def test_live_cases_are_pyspark_mls(name, session):
    # The saves go to a scratch directory of the local file system, of the server that the tests
    # run against on the same machine, as in CI.
    ours = live_cases(name, session, tempfile.mkdtemp(prefix="evaluation-api-cases-"))
    golden = CASES[name]["live"]
    assert sorted(ours) == sorted(golden)
    differ = {tag: (ours[tag], golden[tag]) for tag in golden if ours[tag] != golden[tag]}
    assert not differ, json.dumps(differ, indent=1)
