# SPDX-License-Identifier: Apache-2.0
"""The Params API of the native tree classifiers (DecisionTreeClassifier, RandomForestClassifier and
GBTClassifier), without a server: the cases of `_tree_api_cases.params_api_cases` against
tests/golden/tree_api_cases.json (from scripts/capture_tree_api_cases_golden.py).

Official PySpark runs these tests too, as the reference of the goldens (see `_native_classes`).
"""

import json
from pathlib import Path

import pytest
from _tree_api_cases import CLASSES, params_api_cases

ROOT = Path(__file__).resolve().parents[2]
CASES = json.loads((ROOT / "tests/golden/tree_api_cases.json").read_text())


@pytest.mark.parametrize("name", list(CLASSES))
def test_params_api_cases_are_pyspark_mls(name):
    ours = params_api_cases(name)
    golden = CASES[name]["params_api"]
    assert sorted(ours) == sorted(golden)
    differ = {tag: (ours[tag], golden[tag]) for tag in golden if ours[tag] != golden[tag]}
    assert not differ, json.dumps(differ, indent=1)
