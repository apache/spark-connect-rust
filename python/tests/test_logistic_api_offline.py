# SPDX-License-Identifier: Apache-2.0
"""The pyspark.ml API of LogisticRegression, without a server: the cases of its Params API and its
thresholds (`_logistic_api_cases.params_api_cases`) against tests/golden/logistic_api_cases.json
(from scripts/capture_logistic_api_cases_golden.py).

Official PySpark runs these tests too, as the reference of the golden (see `_native_classes`).
"""

import json
from pathlib import Path

from _logistic_api_cases import params_api_cases

CASES = json.loads(
    (Path(__file__).resolve().parents[2] / "tests/golden/logistic_api_cases.json").read_text()
)


def test_params_api_cases_are_pyspark_mls():
    ours = params_api_cases("LogisticRegression")
    golden = CASES["LogisticRegression"]["params_api"]
    assert sorted(ours) == sorted(golden)
    differ = {tag: (ours[tag], golden[tag]) for tag in golden if ours[tag] != golden[tag]}
    assert not differ, json.dumps(differ, indent=1)
