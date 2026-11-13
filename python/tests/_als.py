# SPDX-License-Identifier: Apache-2.0
"""Shared ALS fixtures and numeric comparisons."""

import json
import math
from pathlib import Path

import pytest

GOLDEN = json.loads((Path(__file__).parents[2] / "tests/golden/als.json").read_text())


def clean(value):
    if hasattr(value, "asDict"):
        return {k: clean(v) for k, v in value.asDict().items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def assert_values(actual, expected):
    actual = clean(actual)
    if isinstance(expected, dict):
        if "float" in expected:
            assert math.isnan(actual)
        else:
            assert set(actual) == set(expected)
            for k, v in expected.items():
                assert_values(actual[k], v)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert_values(a, e)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=2e-4, abs=2e-5)
    else:
        assert actual == expected


def assert_frame(frame, expected):
    rows = sorted(
        [clean(row) for row in frame.collect()],
        key=lambda v: next(v[k] for k in ("id", "u", "i") if k in v),
    )
    assert_values(rows, expected["rows"])
    assert frame.schema.jsonValue() == expected["schema"]


def configured(cls, values, setters=False):
    stage = (
        cls() if setters else cls(**{k: v for k, v in values.items() if k in GOLDEN["defaults"]})
    )
    for key, value in values.items():
        if setters or key not in GOLDEN["defaults"]:
            assert getattr(stage, "set" + key[0].upper() + key[1:])(value) is stage
    return stage
