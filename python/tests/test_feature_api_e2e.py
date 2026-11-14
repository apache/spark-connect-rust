"""The transform, persistence and output of the native feature transformers, against a live
Connect server: their cases (`_feature_api_cases.live_cases`) against
tests/golden/feature_api_cases.json, from official PySpark (see
scripts/capture_feature_api_cases_golden.py). Official PySpark runs them too, as the reference.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
from _feature_api_cases import TRANSFORMERS, live_cases
from pyspark.sql import SparkSession

# StopWordsRemover's defaults (caseSensitive, locale, outputCol and stopWords) are missing from
# ml_params.json, a base file, so its cases fail on ours until that base change lands. Official
# PySpark's offline cases need a session too, as its constructor loads the default stop words.
NAMES = [
    pytest.param(
        name,
        marks=pytest.mark.xfail(
            reason="StopWordsRemover defaults are missing from ml_params.json (a base change)",
            strict=False,
        )
        if name == "StopWordsRemover"
        else (),
    )
    for name in TRANSFORMERS
]

CASES = json.loads(
    (Path(__file__).resolve().parents[2] / "tests/golden/feature_api_cases.json").read_text()
)


@pytest.fixture(scope="module")
def session():
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live feature API tests")
    spark = SparkSession.builder.remote(remote).getOrCreate()
    yield spark
    spark.stop()


@pytest.mark.parametrize("name", NAMES)
def test_live_cases_are_pyspark_mls(name, session):
    # The saves go to a scratch directory of the local file system, of the server that the tests
    # run against on the same machine, as in CI.
    ours = live_cases(name, session, tempfile.mkdtemp(prefix="feature-api-cases-"))
    golden = CASES[name]["live"]
    assert sorted(ours) == sorted(golden)
    differ = {tag: (ours[tag], golden[tag]) for tag in golden if ours[tag] != golden[tag]}
    assert not differ, json.dumps(differ, indent=1)
