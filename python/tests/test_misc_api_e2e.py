# SPDX-License-Identifier: Apache-2.0
"""The pyspark.ml API of NaiveBayes, MultilayerPerceptronClassifier, FMClassifier and OneVsRest
against a live Connect server: the cases of their models, summaries, transforms and fits
(`_misc_api_cases.live_cases`) against tests/golden/misc_api_cases.json, from official PySpark
(see scripts/capture_misc_api_cases_golden.py). Official PySpark runs them too, as the reference.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
from _misc_api_cases import CLASSES, live_cases
from pyspark.sql import SparkSession

CASES = json.loads(
    (Path(__file__).resolve().parents[2] / "tests/golden/misc_api_cases.json").read_text()
)


@pytest.fixture(scope="module")
def session():
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live ML API tests")
    spark = SparkSession.builder.remote(remote).getOrCreate()
    yield spark
    spark.stop()


@pytest.mark.parametrize("name", list(CLASSES))
def test_live_cases_are_pyspark_mls(name, session):
    # The saves go to a scratch directory of the local file system, of the server that the tests
    # run against on the same machine, as in CI.
    ours = live_cases(name, session, tempfile.mkdtemp(prefix="misc-api-cases-"))
    golden = CASES[name]["live"]
    assert sorted(ours) == sorted(golden)
    differ = {tag: (ours[tag], golden[tag]) for tag in golden if ours[tag] != golden[tag]}
    assert not differ, json.dumps(differ, indent=1)
