# SPDX-License-Identifier: Apache-2.0
"""The pyspark.ml API of the native ML classes that follow it in full, without a server.

- Their members against official PySpark's, of tests/golden/ml_api.json (from
  scripts/capture_ml_api_golden.py): the names of the members and Params, the kinds of the
  members, the parameters of each method and of a keyword-only constructor.
- The cases of PySpark's `Params` API (`_ml_api_cases.params_api_cases`) against
  tests/golden/ml_api_cases.json (from scripts/capture_ml_api_cases_golden.py).

Official PySpark runs these tests too, as the reference of the goldens (see `_native_classes`).
"""

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from _ml_api_cases import CLASSES, params_api_cases
from _native_classes import native_class

ROOT = Path(__file__).resolve().parents[2]
API = json.loads((ROOT / "tests/golden/ml_api.json").read_text())
CASES = json.loads((ROOT / "tests/golden/ml_api_cases.json").read_text())

# The attributes of the instances of PySpark's classes, of their constructors, by class.
INSTANCE_ATTRIBUTES = {
    "CrossValidatorModel": {"avgMetrics", "bestModel", "stdMetrics", "subModels"},
    "OneVsRestModel": {"models"},
    "PipelineModel": {"stages"},
    "TrainValidationSplitModel": {"bestModel", "subModels", "validationMetrics"},
}

# The classes that follow pyspark.ml in full, by the module of pyspark.ml that has them: one per
# line, so that the classes of each module are added where they sort.
FULL = [
    (f"pyspark.ml.connect.{module}", name, f"pyspark.ml.{module}")
    for module, name in [
        ("classification", "LinearSVC"),
        ("classification", "LinearSVCModel"),
        ("classification", "LinearSVCSummary"),
        ("classification", "LinearSVCTrainingSummary"),
        ("clustering", "KMeans"),
        ("clustering", "KMeansModel"),
        ("clustering", "KMeansSummary"),
    ]
]


# The kinds of the golden that a native class has as another: a `functools.cached_property` is a
# property, and a method of `functools.cache` a method (the cases test that they cache).
_SAME_KINDS = {"cached_property": "property", "_lru_cache_wrapper": "method"}


def _kind(cls, name):
    """The kind of the member `name` of `cls`, as the capture script names it (of `_SAME_KINDS`)."""
    raw = inspect.getattr_static(cls, name)
    if type(raw).__name__ in _SAME_KINDS:
        return _SAME_KINDS[type(raw).__name__]
    if isinstance(raw, (property, type(int.real))) or type(raw).__name__ == "getset_descriptor":
        return "property"
    if isinstance(raw, staticmethod):
        return "staticmethod"
    if isinstance(raw, classmethod) or type(raw).__name__ == "classmethod_descriptor":
        return "classmethod"
    if inspect.isfunction(raw) or type(raw).__name__ == "method_descriptor":
        return "method"
    return type(raw).__name__


def _parameters(func, bound):
    parameters = list(inspect.signature(func).parameters.values())
    if bound:
        parameters = parameters[1:]
    return [
        [p.name, p.kind.name, None if p.default is inspect.Parameter.empty else repr(p.default)]
        for p in parameters
    ]


@pytest.mark.parametrize("spec", FULL, ids=lambda spec: spec[1])
def test_members_are_pyspark_mls(spec):
    cls = native_class(*spec)
    golden = API[spec[1]]
    expected = set(golden["members"]) | {p["name"] for p in golden["params"]}
    # The attributes that PySpark sets of each instance (the uid of `Identifiable`, those of the
    # constructors of the models of the meta-algorithms), which the class of a native object has
    # as properties (the cases read them of the instances).
    actual = {name for name in dir(cls) if not name.startswith("_")} - {"uid"}
    actual -= INSTANCE_ATTRIBUTES.get(spec[1], set())
    assert actual == expected


@pytest.mark.parametrize("spec", FULL, ids=lambda spec: spec[1])
def test_member_kinds_and_parameters_are_pyspark_mls(spec):
    cls = native_class(*spec)
    for name, member in API[spec[1]]["members"].items():
        kind = _SAME_KINDS.get(member["kind"], member["kind"])
        assert _kind(cls, name) == kind, name
        if "parameters" in member:
            # A method of the class has `self` (a classmethod of the class is bound to it).
            parameters = _parameters(getattr(cls, name), kind == "method")
            assert parameters == member["parameters"], name


@pytest.mark.parametrize("spec", FULL, ids=lambda spec: spec[1])
def test_constructors_are_pyspark_mls(spec):
    cls = native_class(*spec)
    init = API[spec[1]]["init"]
    if not init or any(kind != "KEYWORD_ONLY" for _, kind, _ in init):
        # The constructor of a model or a summary, of its server-side object, is PySpark's own;
        # the native classes make them of a fit, a load or a model.
        pytest.skip("not a public constructor")
    assert _parameters(cls, False) == init


@pytest.mark.parametrize("spec", FULL, ids=lambda spec: spec[1])
def test_params_are_pyspark_mls(spec):
    # The Params of the class (`LinearSVC.maxIter`), which a native class has as PySpark's: of the
    # parent "undefined", of the name, doc and converter of PySpark's class, in its order. Read of
    # the class, as a params walk of an instance would read its cached members, of a session.
    from pyspark.ml.param import Param

    cls = native_class(*spec)
    params = [
        getattr(cls, name)
        for name in sorted(dir(cls))
        if isinstance(getattr(cls, name, None), Param)
    ]
    golden = API[spec[1]]["params"]
    assert [(p.parent, p.name, p.doc, p.typeConverter.__name__) for p in params] == [
        ("undefined", g["name"], g["doc"], g["converter"]) for g in golden
    ]


@pytest.mark.parametrize("name", list(CLASSES))
def test_params_api_cases_are_pyspark_mls(name):
    ours = params_api_cases(name)
    golden = CASES[name]["params_api"]
    assert sorted(ours) == sorted(golden)
    differ = {tag: (ours[tag], golden[tag]) for tag in golden if ours[tag] != golden[tag]}
    assert not differ, json.dumps(differ, indent=1)


def test_native_classes_are_of_pyspark_mls_bases_on_a_fresh_import():
    # The registration runs at the import of any module of pyspark.ml.connect, before any check of
    # the user's.
    code = """
import importlib
try:
    import pyspark._pyspark  # noqa: F401
    module = importlib.import_module("pyspark.ml.connect.classification")
except ImportError:
    module = importlib.import_module("pyspark.ml.classification")
from pyspark.ml.base import Estimator, Model
from pyspark.ml.param import Params
print(isinstance(module.LinearSVC(), Estimator), isinstance(module.LinearSVC(), Params),
      issubclass(module.LinearSVCModel, Model))
"""
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=dict(os.environ)
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["True", "True", "True"]
