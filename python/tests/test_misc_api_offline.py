# SPDX-License-Identifier: Apache-2.0
"""The pyspark.ml API of NaiveBayes, MultilayerPerceptronClassifier, FMClassifier and OneVsRest,
without a server: the cases of PySpark's `Params` API (`_misc_api_cases.params_api_cases`)
against tests/golden/misc_api_cases.json (from scripts/capture_misc_api_cases_golden.py).

Official PySpark runs these tests too, as the reference of the golden (see `_native_classes`).
"""

import json
from pathlib import Path

import pytest
from _misc_api_cases import CLASSES, params_api_cases

CASES = json.loads(
    (Path(__file__).resolve().parents[2] / "tests/golden/misc_api_cases.json").read_text()
)


@pytest.mark.parametrize("name", list(CLASSES))
def test_params_api_cases_are_pyspark_mls(name):
    ours = params_api_cases(name)
    golden = CASES[name]["params_api"]
    assert sorted(ours) == sorted(golden)
    differ = {tag: (ours[tag], golden[tag]) for tag in golden if ours[tag] != golden[tag]}
    assert not differ, json.dumps(differ, indent=1)
