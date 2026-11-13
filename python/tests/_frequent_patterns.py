# SPDX-License-Identifier: Apache-2.0
"""Reference inputs and output normalization for frequent-pattern mining."""

import json
from pathlib import Path

import pytest

GOLDEN = json.loads((Path(__file__).parents[2] / "tests/golden/frequent_patterns.json").read_text())


def configured(cls, values, setters=False):
    if not setters:
        return cls(**values)
    instance = cls()
    for key, value in values.items():
        assert getattr(instance, "set" + key[0].upper() + key[1:])(value) is instance
    return instance


def normalized(frame, kind):
    result = []
    for row in frame.collect():
        row = list(row)
        if kind == "predictions":
            row[-1] = sorted(row[-1])
        elif kind == "itemsets":
            row[0] = sorted(row[0])
        elif kind == "rules":
            row[0], row[1] = sorted(row[0]), sorted(row[1])
        else:
            row[0] = [sorted(v) for v in row[0]]
        result.append(row)
    return sorted(result, key=lambda v: json.dumps(v))


def assert_values(actual, expected):
    if isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert_values(a, e)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-10, abs=1e-12)
    else:
        assert actual == expected
