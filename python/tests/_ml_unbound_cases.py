# SPDX-License-Identifier: Apache-2.0
"""The models of no server-side object, as PySpark's `JavaModel(java_model=None)` makes them, of
every model class of tests/golden/ml_api.json whose constructor is that one, over a live server.

`scripts/capture_ml_unbound_models_golden.py` records them of official PySpark for
tests/golden/ml_unbound_models.json, and test_ml_api_e2e.py compares the native classes with it.
Such a model has its params, of their defaults, and no object: its Params API works, and its
members of the server, a transform and a save raise PySpark's bare AssertionError, as does the
`params` walk of a model of `functools.cached_property` members (`trees`).
"""

import json
import tempfile
from pathlib import Path

from _native_classes import native_class

GOLDEN = Path(__file__).resolve().parents[2] / "tests/golden/ml_unbound_models.json"

_API = json.loads((GOLDEN.parent / "ml_api.json").read_text())

# The model classes whose constructor is PySpark's `JavaModel(java_model=None)`, by name.
MODELS = sorted(
    name
    for name, entry in _API.items()
    if entry.get("init") == [["java_model", "POSITIONAL_OR_KEYWORD", "None"]]
)


def model_class(name):
    """The native model class `name` (PySpark's, of the reference)."""
    module = _API[name]["module"].rsplit(".", 1)[1]
    return native_class(f"pyspark.ml.connect.{module}", name, f"pyspark.ml.{module}")


def _outcome(f):
    """The class of the error of `f()`, or "ok" with its value's repr (a writer as `writer`)."""
    try:
        value = f()
    except Exception as e:  # noqa: BLE001 - every error is an outcome
        return [type(e).__name__, str(e).split("\n")[0][:120]]
    return ["ok", "writer" if "Writer" in type(value).__name__ else repr(value)[:120]]


def outcomes(spark, name):
    """The outcomes of the API of a model of the class `name` of no server-side object. Its uid is
    recorded as `uid` of its value, and the default of `seed` as `hash` of the hash of the name of
    the class (PySpark's default, which differs per process)."""
    cls = model_class(name)
    model = cls()
    uid = model.uid

    def default(param):
        if not model.hasDefault(param):
            return None
        value = model.getOrDefault(param)
        if param == "seed" and value == hash(type(model).__name__):
            return "hash"
        return repr(value).replace(uid, "uid")

    params = sorted(a for a in dir(cls) if type(getattr(cls, a, None)).__name__ == "Param")
    path = tempfile.mkdtemp(prefix="ml-unbound-") + "/model"
    return {
        "uid": [uid.startswith(name + "_"), len(uid) - len(name)],
        "params": {p: [model.isSet(p), model.hasDefault(p), default(p)] for p in params},
        "params_walk": _outcome(lambda: len(model.params))[0],
        "explainParams": _outcome(lambda: len(model.explainParams()) > 0)[0],
        "extractParamMap": _outcome(lambda: len(model.extractParamMap())),
        "copy": _outcome(lambda: type(model.copy()).__name__),
        "repr": _outcome(lambda: repr(model)),
        "transform": _outcome(lambda: model.transform(spark.range(1)))[0],
        "write": _outcome(lambda: model.write())[0] if hasattr(cls, "write") else None,
        "save": _outcome(lambda: model.save(path))[0] if hasattr(cls, "save") else None,
    }
