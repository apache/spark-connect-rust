# SPDX-License-Identifier: Apache-2.0
"""Capture the API of official pyspark.ml 4.3.0 for tests/golden/ml_api.json.

Run with official pyspark-client 4.3.0, which needs no server for this:
    python scripts/capture_ml_api_golden.py > tests/golden/ml_api.json

For every public class of pyspark.ml.{classification, regression, clustering, feature, fpm,
recommendation, evaluation, stat, tuning} and Pipeline/PipelineModel, it records:
- `params`: each Param in PySpark's `params` order (by name), with its doc, the name of its
  TypeConverters method, and its default, if any, of a default instance, as `_setDefault` sets it.
  A default that PySpark derives from the instance is recorded as it derives it, so the golden is
  the same on every run: `{"uid_suffix": "__output"}`, the uid of the instance and the suffix, and
  `{"class_name_hash": true}`, the `hash` of the name of its class in the Python process;
- `members`: each public member that is not a Param, with its kind (property, method,
  staticmethod, classmethod) and, of a method, its signature.

It is the API that the native classes of pyspark.ml.connect and the Rust API of
spark_connect::ml follow: their Params, with the docs, converters and defaults, and their members.
"""

import importlib
import inspect
import json
import os

os.environ.setdefault("SPARK_CONNECT_MODE_ENABLED", "1")

from pyspark.ml.linalg import DenseMatrix, DenseVector, SparseMatrix, SparseVector  # noqa: E402
from pyspark.ml.param import Param, Params  # noqa: E402

MODULES = [
    "classification",
    "regression",
    "clustering",
    "feature",
    "fpm",
    "recommendation",
    "evaluation",
    "stat",
    "tuning",
]


def render(value):
    """A JSON form of a default, typed where JSON is not."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return {"int": value}
    if isinstance(value, float):
        return {"float": repr(value)}
    if isinstance(value, (list, tuple)):
        return {"list": [render(v) for v in value]}
    if isinstance(value, DenseVector):
        return {"dense_vector": value.values.tolist()}
    if isinstance(value, SparseVector):
        return {"sparse_vector": [value.size, value.indices.tolist(), value.values.tolist()]}
    if isinstance(value, (DenseMatrix, SparseMatrix)):
        return {"matrix": repr(value)}
    return {"repr": repr(value)}


def default_of(instance, name, value):
    """The default of a Param, in the form that PySpark derives it from its instance where it does:
    `outputCol` as the uid of the instance and "__output" (HasOutputCol), and `seed` as the hash of
    the name of its class, which differs between Python processes (HasSeed)."""
    if name == "seed" and value == hash(type(instance).__name__):
        return {"class_name_hash": True}
    if isinstance(value, str) and value.startswith(instance.uid):
        return {"uid_suffix": value[len(instance.uid) :]}
    return render(value)


def converter_name(param):
    converter = param.typeConverter
    return getattr(converter, "__name__", repr(converter))


def instance_of(cls):
    try:
        return cls()
    except Exception:  # noqa: BLE001 - a class that needs arguments has no default instance
        return None


def member_kind(cls, name):
    for klass in cls.__mro__:
        if name in klass.__dict__:
            raw = klass.__dict__[name]
            if isinstance(raw, property):
                return "property", None
            if isinstance(raw, staticmethod):
                return "staticmethod", str(inspect.signature(raw.__func__))
            if isinstance(raw, classmethod):
                return "classmethod", str(inspect.signature(raw.__func__))
            if inspect.isfunction(raw):
                try:
                    return "method", str(inspect.signature(raw))
                except (TypeError, ValueError):
                    return "method", None
            if isinstance(raw, Param):
                return "param", None
            return type(raw).__name__, None
    return "unknown", None


def capture(cls):
    instance = instance_of(cls)
    params = []
    # The Params of the class, by name, as `Params.params` orders them. Read off the class, as
    # `params` itself reads every attribute, a remote one of a model too (such as `trees`).
    for name in sorted(dir(cls)):
        value = getattr(cls, name, None)
        if not isinstance(value, Param):
            continue
        entry = {"name": name, "doc": value.doc, "converter": converter_name(value)}
        if isinstance(instance, Params):
            own = instance.__dict__.get(name, value)
            if own in instance._defaultParamMap:
                entry["default"] = default_of(instance, name, instance._defaultParamMap[own])
        params.append(entry)
    param_names = {p["name"] for p in params}
    members = {}
    for name in sorted(dir(cls)):
        if name.startswith("_") or name in param_names:
            continue
        kind, signature = member_kind(cls, name)
        members[name] = {"kind": kind, **({"signature": signature} if signature else {})}
    return {
        "module": cls.__module__,
        "default_instance": instance is not None,
        "params": params,
        "members": members,
    }


def main():
    out = {}
    for module_name in MODULES:
        module = importlib.import_module("pyspark.ml." + module_name)
        for name in getattr(module, "__all__", []):
            cls = getattr(module, name, None)
            if inspect.isclass(cls):
                out[name] = capture(cls)
    import pyspark.ml as ml

    for name in ["Pipeline", "PipelineModel"]:
        out[name] = capture(getattr(ml, name))
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
