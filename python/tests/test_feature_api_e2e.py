"""The transforms, fits, models and persistence of the native feature classes, against a live
Connect server: their cases (`_feature_api_cases.live_cases`) against
tests/golden/feature_api_cases.json, from official PySpark (see
scripts/capture_feature_api_cases_golden.py). Official PySpark runs them too, as the reference.
"""

import ast
import json
import math
import os
import tempfile
from pathlib import Path

import pytest
from _feature_api_cases import ESTIMATORS, MODELS, TRANSFORMERS, live_cases
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
    for name in TRANSFORMERS + ESTIMATORS + MODELS
]

CASES = json.loads(
    (Path(__file__).resolve().parents[2] / "tests/golden/feature_api_cases.json").read_text()
)


def _parsed(value):
    # A row of a DataFrame is recorded as the repr of its tuple, and a DataFrame as the repr of its
    # dict of columns and rows.
    if isinstance(value, str) and value[:1] in "({" and value[-1:] in ")}":
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def _close(ours, golden):
    """Whether a recorded value matches the golden's: a float within the tolerance of the e2e ALS
    tests, since the server computes single-precision results (Word2Vec vectors and similarities,
    scaled values) with last bits that differ between the server of the capture and that of the
    test; anything else exactly."""
    ours, golden = _parsed(ours), _parsed(golden)
    if isinstance(ours, float) and isinstance(golden, float):
        if math.isnan(ours) or math.isnan(golden):
            return math.isnan(ours) and math.isnan(golden)
        return abs(ours - golden) <= 2e-5 + 2e-4 * abs(golden)
    if isinstance(ours, (list, tuple)) and isinstance(golden, (list, tuple)):
        return len(ours) == len(golden) and all(_close(a, b) for a, b in zip(ours, golden))
    if isinstance(ours, dict) and isinstance(golden, dict):
        return ours.keys() == golden.keys() and all(_close(ours[k], golden[k]) for k in golden)
    return ours == golden


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
    differ = {tag: (ours[tag], golden[tag]) for tag in golden if not _close(ours[tag], golden[tag])}
    assert not differ, json.dumps(differ, indent=1)
