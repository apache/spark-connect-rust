# SPDX-License-Identifier: Apache-2.0
"""The pyspark.ml API of the native frequent-pattern and recommendation classes that follow it in
full, against a live Connect server: the cases of their fits, models, transforms, recommendations
and persistence (`_fpm_recommendation_api_cases.live_cases`) against
tests/golden/fpm_recommendation_api_cases.json, from official PySpark (see
scripts/capture_fpm_recommendation_api_cases_golden.py). Official PySpark runs them too, as the
reference.
"""

import ast
import json
import math
import os
import re
import tempfile
from pathlib import Path

import pytest
from _fpm_recommendation_api_cases import CLASSES, live_cases
from pyspark.sql import SparkSession

CASES = json.loads(
    (Path(__file__).resolve().parents[2] / "tests/golden/fpm_recommendation_api_cases.json").read_text()
)


def _parsed(value):
    # A row of a DataFrame is recorded as the repr of its tuple.
    if isinstance(value, str) and value.startswith("(") and value.endswith(")"):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


# A decimal number in a recorded text, such as a rating in the repr of a nested Row.
_DECIMAL = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?")


def _close_float(ours, golden):
    if math.isnan(ours) or math.isnan(golden):
        return math.isnan(ours) and math.isnan(golden)
    return abs(ours - golden) <= 2e-5 + 2e-4 * abs(golden)


def _close_text(ours, golden):
    # A text that is not a literal, such as a row with a nested Row: the same text around its
    # decimal numbers, which are close.
    numbers = _DECIMAL.findall(ours), _DECIMAL.findall(golden)
    return _DECIMAL.sub("#", ours) == _DECIMAL.sub("#", golden) and all(
        _close_float(float(a), float(b)) for a, b in zip(*numbers)
    )


def _close(ours, golden):
    """Whether a recorded value matches the golden's: a float within the tolerance of the e2e ALS
    tests, since the server computes ALS factors and predictions in single precision, whose last
    bits differ between the server of the capture and that of the test; anything else exactly."""
    ours, golden = _parsed(ours), _parsed(golden)
    if isinstance(ours, float) and isinstance(golden, float):
        return _close_float(ours, golden)
    if isinstance(ours, str) and isinstance(golden, str):
        return _close_text(ours, golden)
    if isinstance(ours, (list, tuple)) and isinstance(golden, (list, tuple)):
        return len(ours) == len(golden) and all(_close(a, b) for a, b in zip(ours, golden))
    if isinstance(ours, dict) and isinstance(golden, dict):
        return ours.keys() == golden.keys() and all(_close(ours[k], golden[k]) for k in golden)
    return ours == golden


@pytest.fixture(scope="module")
def session():
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live fpm and recommendation API tests")
    spark = SparkSession.builder.remote(remote).getOrCreate()
    yield spark
    spark.stop()


@pytest.mark.parametrize("name", list(CLASSES))
def test_live_cases_are_pyspark_mls(name, session):
    # The saves go to a scratch directory of the local file system, of the server that the tests
    # run against on the same machine, as in CI.
    ours = live_cases(name, session, tempfile.mkdtemp(prefix="fpm-recommendation-api-cases-"))
    golden = CASES[name]["live"]
    assert sorted(ours) == sorted(golden)
    differ = {tag: (ours[tag], golden[tag]) for tag in golden if not _close(ours[tag], golden[tag])}
    assert not differ, json.dumps(differ, indent=1)
