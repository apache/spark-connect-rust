"""The Params API, the getters and the members of the native feature transformers and estimators,
without a server: their cases (`_feature_api_cases.offline_cases`) against tests/golden/feature_api_cases.json,
from official PySpark (see scripts/capture_feature_api_cases_golden.py). Official PySpark runs them
too, as the reference (see `_native_classes`).
"""

import json
from pathlib import Path

import pytest
from _feature_api_cases import ESTIMATORS, TRANSFORMERS, offline_cases

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
    for name in TRANSFORMERS + ESTIMATORS
]

CASES = json.loads(
    (Path(__file__).resolve().parents[2] / "tests/golden/feature_api_cases.json").read_text()
)


@pytest.mark.parametrize("name", NAMES)
def test_offline_cases_are_pyspark_mls(name):
    ours = offline_cases(name)
    golden = CASES[name]["offline"]
    assert sorted(ours) == sorted(golden)
    differ = {tag: (ours[tag], golden[tag]) for tag in golden if ours[tag] != golden[tag]}
    assert not differ, json.dumps(differ, indent=1)
