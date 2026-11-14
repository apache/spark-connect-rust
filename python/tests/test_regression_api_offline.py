# SPDX-License-Identifier: Apache-2.0
"""The cases of the pyspark.ml API of the native regression classes that follow it in full,
without a server: the cases of PySpark's `Params` API
(`_regression_api_cases.params_api_cases`) against tests/golden/regression_api_cases.json (from
scripts/capture_regression_api_cases_golden.py).

Official PySpark runs these tests too, as the reference of the goldens (see `_native_classes`).
"""

import json

import pytest
from _regression_api_cases import CLASSES, load_golden, params_api_cases

CASES = load_golden()


@pytest.mark.parametrize("name", list(CLASSES))
def test_params_api_cases_are_pyspark_mls(name):
    ours = params_api_cases(name)
    golden = CASES[name]["params_api"]
    assert sorted(ours) == sorted(golden)
    differ = {tag: (ours[tag], golden[tag]) for tag in golden if ours[tag] != golden[tag]}
    assert not differ, json.dumps(differ, indent=1)
