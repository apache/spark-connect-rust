# SPDX-License-Identifier: Apache-2.0
"""The pyspark.ml API of the native frequent-pattern and recommendation classes that follow it in
full, without a server: the cases of their `Params` API against
tests/golden/fpm_recommendation_api_cases.json (from
scripts/capture_fpm_recommendation_api_cases_golden.py), which official PySpark captures.

Official PySpark runs these tests too, as the reference (`_native_classes`).
"""

import json
from pathlib import Path

import pytest
from _fpm_recommendation_api_cases import CLASSES, params_api_cases

CASES = json.loads(
    (Path(__file__).resolve().parents[2] / "tests/golden/fpm_recommendation_api_cases.json").read_text()
)


@pytest.mark.parametrize("name", list(CLASSES))
def test_params_api_cases_are_pyspark_mls(name):
    ours = params_api_cases(name)
    golden = CASES[name]["params_api"]
    assert sorted(ours) == sorted(golden)
    differ = {tag: (ours[tag], golden[tag]) for tag in golden if ours[tag] != golden[tag]}
    assert not differ, json.dumps(differ, indent=1)
