# SPDX-License-Identifier: Apache-2.0
"""Shared inputs and normalization for the feature parity tests."""
import json
import math
from pathlib import Path

from pyspark.ml.linalg import Vector, Vectors

GOLDEN = json.loads((Path(__file__).parents[2] / "tests/golden/ml_features.json").read_text())


def decode(value):
    if isinstance(value, dict):
        if "float" in value:
            return float(value["float"])
        if "vector" in value:
            return Vectors.dense(value["vector"])
        return {key: decode(v) for key, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def normalized(value):
    if isinstance(value, Vector):
        return value.toArray().tolist()
    if isinstance(value, (list, tuple)):
        return [normalized(v) for v in value]
    return value


def assert_values(actual, expected):
    import pytest
    actual, expected = normalized(actual), decode(expected)
    if isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert_values(a, e)
    elif isinstance(expected, (float, int)):
        if isinstance(expected, float) and math.isnan(expected):
            assert math.isnan(actual)
        else:
            assert actual == pytest.approx(expected, rel=1e-5, abs=1e-7)
    else:
        assert actual == expected


def configured(cls, values, setters=False, set_params=False):
    # The stage of the values, given to its constructor, to its setters (setters), or to its
    # setParams (set_params); the values that the constructor does not take go to the setters.
    values = decode(values)
    defaults = GOLDEN["api"][cls.__name__]["defaults"]
    given = {k: v for k, v in values.items() if k in defaults}
    if setters:
        stage = cls()
    elif set_params:
        stage = cls()
        assert stage.setParams(**given) is stage
    else:
        stage = cls(**given)
    for key, value in values.items():
        if setters or key not in defaults:
            assert getattr(stage, "set" + key[0].upper() + key[1:])(value) is stage
    return stage


def wire_schema(value):
    # Connect identifies JVM UDTs by class without transmitting their backing SQL schema.
    # Compare the transmitted type, nullability and ML metadata; values above check decoding.
    if isinstance(value, list):
        return [wire_schema(v) for v in value]
    if isinstance(value, dict):
        return {k: wire_schema(v) for k, v in value.items() if not (value.get("type") == "udt" and k == "sqlType")}
    return value
