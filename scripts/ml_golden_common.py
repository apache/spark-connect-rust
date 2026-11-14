# SPDX-License-Identifier: Apache-2.0
"""Shared helpers that capture the native ML golden fixtures from official PySpark 4.3.0.

An area script (scripts/capture_<area>_models_golden.py) holds its tables: the attributes and
arguments of each model class, the class methods, and the datasets and estimator variants. This
module runs them on the official pyspark-client and writes the JSON fixture:

    golden = Golden(spark)
    frame = golden.dataset("binary_train", rows)
    model = golden.fit("LogisticRegression.binomial", estimator, frame, uid="...")
    golden.members("models", key, model, "LogisticRegressionModel", "model",
                   attributes=ATTRIBUTES["LogisticRegressionModel"], arguments=arguments,
                   frame=frame)
    golden.dump(path, {"fits": golden.fits, "models": golden.models})

Tables, as the area scripts write them:

- ATTRIBUTES: class name -> names read as properties, or called with no arguments when the name
  ends with "()" (for example "summary()" for a method, "numClasses" for a property).
- arguments(class_name, frame) -> [(label, method, args)], the labelled calls with arguments.
  `frame` is the area's dataset (a DataFrame, rendered as its dataset name).
- CLASS_METHODS: [(class name, method, label, args, kwargs)], each followed by a transform
  of the dataset (see Golden.class_methods).

Every public member of an object is either covered by a table entry or skipped with a reason
(Params API, Param attribute, Params getter or setter, persistence, or the area's `skip`). An
uncovered member fails the capture.

Recording: the official client's `_execute_and_fetch_as_iterator` (the path of `execute_command`
and of DataFrame collection) is wrapped. Each call captures the ML command or relation it sends
(`commands`) and the MlCommandResult it gets back (`results`), both as base64 of
SerializeToString(deterministic=True). Normalization: ObjectRef ids and summary ids become
<idN> placeholders, uids become <uidN>, both numbered per entry in order of first appearance (so
an entry replays alone); plan_id
fields and origin (the client call site) are cleared; LocalRelation data bytes are cleared;
Write and Read paths become <path>.

Values: floats are JSON numbers (Python repr, shortest round-trip); NaN and infinities are the
strings "NaN", "Infinity" and "-Infinity". Vectors are {"type": "dense" | "sparse", "size",
"values"} or {"type": "sparse", "size", "indices", "values"}. Matrices are {"type", "numRows",
"numCols", "isTransposed", "values"} (column-major), plus "colPtrs" and "rowIndices" when sparse.
DataFrames are {"schema": simpleString, "rows": [...]}, the rows as lists in schema order and
sorted by their canonical JSON. Model and summary objects are {"type": class, "ref": placeholder}.
Errors are {"class", "condition", "message"}. repr and str of an object are captured as "repr()"
and "str()", with 0x addresses replaced by 0x<addr>.
"""

import base64
import gc
import json
import math
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from functools import partial

import numpy as np
from google.protobuf.message import Message
from pyspark.ml.base import Estimator
from pyspark.ml.functions import array_to_vector
from pyspark.ml.linalg import DenseMatrix, DenseVector, SparseMatrix, SparseVector
from pyspark.ml.param import Param, Params
from pyspark.ml.stat import MultivariateGaussian
from pyspark.mllib.linalg import DenseMatrix as MllibDenseMatrix
from pyspark.mllib.linalg import DenseVector as MllibDenseVector
from pyspark.mllib.linalg import SparseMatrix as MllibSparseMatrix
from pyspark.mllib.linalg import SparseVector as MllibSparseVector
from pyspark.sql import Row
from pyspark.sql.connect.client.core import SparkConnectClient
from pyspark.sql.connect.dataframe import DataFrame as ConnectDataFrame

SOURCE = "apache/spark v4.3.0 pyspark.ml (official pyspark-client 4.3.0, Spark 4.3.0 Connect)"

PARAMS_API = frozenset(n for n in dir(Params) if not n.startswith("_")) | {"uid", "params"}
PERSISTENCE = frozenset(["write", "save", "read", "load"])
FITTING = {"fitMultiple": "client-side loop over fit (one fit command per param map)"}

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_ADDR = re.compile(r"0x[0-9a-fA-F]+")
_RANDOM_UID = re.compile(r"[A-Za-z]+_[0-9a-f]{12}")
_TMP = re.compile(r"/tmp/ml-golden-[A-Za-z0-9_]+")

_SINK = [None]
# The placeholders of the entry being captured: numbered per kind, in order of first
# appearance within the entry, so that one entry can be replayed alone.
_SCOPE = [None]


def _placeholder(kind, raw):
    if _SCOPE[0] is None:
        raise RuntimeError(f"a placeholder outside an entry scope: {kind} {raw!r}")
    table = _SCOPE[0][kind]
    if raw not in table:
        table[raw] = f"<{kind}{len(table)}>"
    return table[raw]


@contextmanager
def _scope():
    previous = _SCOPE[0]
    _SCOPE[0] = {"id": {}, "uid": {}}
    try:
        yield
    finally:
        _SCOPE[0] = previous


def _ref_placeholder(raw):
    # A summary id is "<model id>.summary"; the model id gets the placeholder.
    if "." in raw:
        head, tail = raw.split(".", 1)
        return _placeholder("id", head) + "." + tail
    return _placeholder("id", raw)


def _scrub_text(text):
    """A string value: the save directory is <tmp>, and server-generated uids are placeholders."""
    text = _TMP.sub("<tmp>", text)
    return _RANDOM_UID.sub(lambda m: _placeholder("uid", m.group(0)), text)


def _scrub(msg):
    full = msg.DESCRIPTOR.full_name
    if full == "spark.connect.LocalRelation":
        msg.ClearField("data")
    for fd, value in msg.ListFields():
        if fd.name in ("plan_id", "origin"):
            # origin is the client's call site (a source path and line), not part of the plan.
            msg.ClearField(fd.name)
            continue
        if fd.message_type is not None:
            if fd.message_type.GetOptions().map_entry:
                items = [value[k] for k in value]
            elif isinstance(value, Message):
                items = [value]
            else:
                items = list(value)
            for item in items:
                _scrub(item)
            continue
        if fd.type != fd.TYPE_STRING:
            continue
        if not isinstance(value, str):
            # A repeated string field: each element may carry a server-generated uid.
            texts = [_scrub_text(v) for v in value]
            del value[:]
            value.extend(texts)
            continue
        if fd.name == "path":
            setattr(msg, fd.name, "<path>")
        elif full == "spark.connect.ObjectRef" and fd.name == "id":
            msg.id = _ref_placeholder(value)
        elif full == "spark.connect.MlCommandResult" and fd.name == "summary":
            msg.summary = _ref_placeholder(value)
        elif fd.name == "uid":
            setattr(msg, fd.name, _placeholder("uid", value))
        else:
            # A uid inside text (a tree's debug string) or a save path in a method argument.
            setattr(msg, fd.name, _scrub_text(value))


def _b64(msg):
    out = type(msg)()
    out.CopyFrom(msg)
    _scrub(out)
    return base64.b64encode(out.SerializeToString(deterministic=True)).decode()


def _ml_part(req):
    plan = req.plan
    if plan.HasField("command") and plan.command.HasField("ml_command"):
        return plan.command.ml_command
    if plan.HasField("root") and plan.root.HasField("ml_relation"):
        return plan.root.ml_relation
    return None


def _install():
    """Wrap the official client's execute path once per process."""
    if getattr(SparkConnectClient, "_golden_recorder", False):
        return
    original = SparkConnectClient._execute_and_fetch_as_iterator

    def recorded(self, req, *args, **kwargs):
        sink = _SINK[0]
        if sink is not None:
            part = _ml_part(req)
            if part is not None:
                sink["commands"].append(_b64(part))
        for item in original(self, req, *args, **kwargs):
            if sink is not None and isinstance(item, dict) and "ml_command_result" in item:
                sink["results"].append(_b64(item["ml_command_result"]))
            yield item

    SparkConnectClient._execute_and_fetch_as_iterator = recorded
    SparkConnectClient._golden_recorder = True


@contextmanager
def _sink():
    previous = _SINK[0]
    sink = {"commands": [], "results": []}
    _SINK[0] = sink
    try:
        yield sink
    finally:
        _SINK[0] = previous


def _num(x):
    x = float(x)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    return x


def _error(e):
    condition = None
    get = getattr(e, "getCondition", None)
    if callable(get):
        try:
            condition = get()
        except Exception:
            condition = None
    text = str(e).strip()
    message = text.splitlines()[0] if text else ""
    return {
        "class": type(e).__name__,
        "condition": condition,
        # Called inside the entry's scope, so the uid placeholders are per entry.
        "message": _scrub_text(_UUID.sub("<uuid>", message)),
    }


def _common_skip(obj, name, param_names):
    if name in PARAMS_API:
        return "Params API (client side)"
    if name in PERSISTENCE:
        return "persistence (captured by Golden.persist)"
    if name in FITTING:
        return FITTING[name]
    if isinstance(inspect_static(type(obj), name), Param):
        return "Param attribute (client side)"
    if name[:3] in ("get", "set") and len(name) > 3:
        lowered = name[3].lower() + name[4:]
        if lowered in param_names:
            return "Params getter or setter (client side)"
    return None


def inspect_static(cls, name):
    for klass in cls.__mro__:
        if name in vars(klass):
            return vars(klass)[name]
    return None


class Golden:
    """Captures one area's fixture. Create one per script run; call close() at the end."""

    def __init__(self, spark):
        _install()
        # A finalized model sends its ObjectRef Delete when it is collected. The collector is off
        # while the golden is captured, and call() collects before each entry, so every Delete
        # is recorded outside the entries.
        gc.disable()
        self.spark = spark
        self.datasets = {}
        self.fits = {}
        self.models = {}
        self.summaries = {}
        self.class_methods_out = {}
        self.persistence = {}
        self._frames = {}
        self._keep = []
        self._pinned = set()
        self._tmp = tempfile.mkdtemp(prefix="ml-golden-", dir="/tmp")

    def close(self):
        shutil.rmtree(self._tmp, ignore_errors=True)
        gc.enable()

    # Datasets and values.

    def dataset(self, name, columns, rows=None):
        """A one-partition DataFrame; columns is [(name, type)], rows are tuples in its order.

        Types: double, int, long, string, boolean, vector (VectorUDT, given as a dense list of
        floats), and array<double>, array<string>, array<int>. None is null for any type.
        The shortcut dataset(name, rows) takes (label, weight, features) rows.
        """
        if rows is None:
            columns, rows = DEFAULT_COLUMNS, columns
        for _, kind in columns:
            if kind not in SPARK_TYPES:
                raise ValueError(f"unsupported column type {kind}")
        names = [c for c, _ in columns]
        ddl = ", ".join(f"{c} {SPARK_TYPES[k]}" for c, k in columns)
        data = [tuple(_python_value(kind, v) for (_, kind), v in zip(columns, row)) for row in rows]
        df = self.spark.createDataFrame(data, ddl)
        for c, kind in columns:
            if kind == "vector":
                df = df.withColumn(c, array_to_vector(c))
        df = df.select(*names).coalesce(1)
        self.datasets[name] = {
            "columns": [[c, k] for c, k in columns],
            "rows": [[_stored(k, v) for (_, k), v in zip(columns, row)] for row in data],
        }
        self._frames[id(df)] = name
        self._keep.append(df)
        return df

    def register(self, name, frame, spec):
        """Register an existing frame under name; spec (JSON) says how to rebuild it.

        A spec such as {"sql": "...", "vectors": ["features"], "coalesce": 1} lets the Rust and
        Python tests rebuild exactly the frame.
        """
        self._frames[id(frame)] = name
        self._keep.append(frame)
        self.datasets[name] = spec
        return frame

    def arg_text(self, a):
        if isinstance(a, DenseVector):
            return f"Vectors.dense({_floats(a.toArray())})"
        if isinstance(a, SparseVector):
            indices = [int(i) for i in a.indices]
            return f"Vectors.sparse({a.size}, {indices}, {_floats(a.values)})"
        if isinstance(a, ConnectDataFrame):
            return self._frames.get(id(a), "<dataframe>")
        if isinstance(a, (list, tuple)):
            return "[" + ", ".join(self.arg_text(x) for x in a) + "]"
        return repr(a)

    def render(self, v):
        if v is None or isinstance(v, (bool, np.bool_)):
            return None if v is None else bool(v)
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            return _num(v)
        if isinstance(v, str):
            return self.text(v)
        if isinstance(v, (DenseVector, MllibDenseVector)):
            return {"type": "dense", "size": len(v), "values": [_num(x) for x in v.toArray()]}
        if isinstance(v, (SparseVector, MllibSparseVector)):
            return {
                "type": "sparse",
                "size": int(v.size),
                "indices": [int(i) for i in v.indices],
                "values": [_num(x) for x in v.values],
            }
        if isinstance(v, (DenseMatrix, MllibDenseMatrix)):
            return {
                "type": "dense",
                "numRows": int(v.numRows),
                "numCols": int(v.numCols),
                "isTransposed": bool(v.isTransposed),
                "values": [_num(x) for x in v.values],
            }
        if isinstance(v, (SparseMatrix, MllibSparseMatrix)):
            return {
                "type": "sparse",
                "numRows": int(v.numRows),
                "numCols": int(v.numCols),
                "isTransposed": bool(v.isTransposed),
                "colPtrs": [int(i) for i in v.colPtrs],
                "rowIndices": [int(i) for i in v.rowIndices],
                "values": [_num(x) for x in v.values],
            }
        if isinstance(v, MultivariateGaussian):
            return {
                "type": "MultivariateGaussian",
                "mean": self.render(v.mean),
                "cov": self.render(v.cov),
            }
        if isinstance(v, np.ndarray):
            return self.render(v.tolist())
        if isinstance(v, Row):
            return [self.render(x) for x in v]
        if isinstance(v, (list, tuple)):
            return [self.render(x) for x in v]
        if isinstance(v, dict):
            return {str(k): self.render(x) for k, x in sorted(v.items())}
        if isinstance(v, ConnectDataFrame):
            rows = [self.render(r) for r in v.collect()]
            rows.sort(key=lambda r: json.dumps(r, sort_keys=True))
            return {"schema": v.schema.simpleString(), "rows": rows}
        if hasattr(v, "_java_obj") and not isinstance(v, Estimator):
            raw = v._java_obj
            raw = raw.ref_id if hasattr(raw, "ref_id") else str(raw)
            return {"type": type(v).__name__, "ref": _ref_placeholder(raw)}
        if isinstance(v, Params):
            return {"type": type(v).__name__, "uid": self.uid_value(v.uid)}
        raise TypeError(f"cannot render {type(v).__name__}: {v!r}")

    def uid_value(self, uid):
        """A pinned uid is kept as it is; any other uid (client-built models) is a placeholder."""
        return uid if uid in self._pinned else _placeholder("uid", uid)

    def text(self, value):
        """Text (a str value, repr or str), with the addresses and random uids normalized.

        The server gives the trees of an ensemble random uids (dtr_<hex>, dtc_<hex>), which the
        debug strings of the trees carry; they become <uidN> placeholders.
        """
        text = _ADDR.sub("0x<addr>", value)
        return _RANDOM_UID.sub(lambda m: _placeholder("uid", m.group(0)), text)

    # Calls.

    def call(self, call, fn):
        """Run fn with the request recorder on; return (raw result, entry)."""
        gc.collect()
        with _scope(), _sink() as sink:
            try:
                # Rendering is inside the try: a DataFrame can fail lazily, at its collect.
                raw = fn()
                entry = {"call": call, "value": self.render(raw)}
            except Exception as e:  # noqa: BLE001 - every error is part of the golden
                entry = {"call": call, "error": _error(e)}
                raw = None
        entry["commands"] = sink["commands"]
        entry["results"] = sink["results"]
        return raw, entry

    def _params(self, obj):
        out = {}
        for p in sorted(obj.params, key=lambda p: p.name):
            if obj.isDefined(p):
                _, entry = self.call(
                    f"{type(obj).__name__}.getOrDefault({p.name!r})",
                    partial(obj.getOrDefault, p),
                )
                out[p.name] = entry["value"] if "value" in entry else {"error": entry["error"]}
        return out

    def fit(self, key, estimator, frame, uid):
        """Fit with a pinned uid; the fit entry records the request, the result and uid_same."""
        estimator._resetUid(uid)
        self._pinned.add(uid)
        name = self._frames[id(frame)]
        call = f"{type(estimator).__name__}.fit({name})"
        model, entry = self.call(call, lambda: estimator.fit(frame))
        if model is None:
            raise RuntimeError("{} failed: {}".format(call, entry["error"]))
        self.fits[key] = {
            "estimator": type(estimator).__name__,
            "dataset": name,
            "uid": estimator.uid,
            "params": self._params(estimator),
            "fit": entry,
        }
        self.fits[key]["uid_same"] = model.uid == estimator.uid
        with _scope():
            self.fits[key]["model_uid"] = self.uid_value(model.uid)
        return model

    def members(
        self,
        obj,
        cls_name,
        expr,
        attributes=(),
        arguments=None,
        frame=None,
        skip=None,
        partial=False,
    ):
        """Capture the members of obj; return {"class", "members", "skipped"}.

        Raises AssertionError on a member that is neither covered nor skipped, and on a table
        name that the object does not have. With partial=True (a persisted copy), the members
        that the tables do not list are skipped rather than failing.
        """
        skip = dict(skip or {})
        dirs = set(dir(obj))
        entries = {}
        covered = set()
        for name in attributes:
            base = name[:-2] if name.endswith("()") else name
            if base not in dirs:
                raise AssertionError(f"{cls_name} has no member {base}")
            covered.add(base)
            entries[name] = self._read(obj, base, name.endswith("()"), f"{expr}.{name}")[1]
        if arguments is not None:
            for label, method, args in arguments(cls_name, frame):
                if method not in dirs:
                    raise AssertionError(f"{cls_name} has no method {method}")
                covered.add(method)
                call = "{}.{}({})".format(expr, method, ", ".join(self.arg_text(a) for a in args))
                entries[label] = self._invoke(obj, method, args, call)[1]
        _, entries["repr()"] = self.call(f"repr({expr})", lambda: self.text(repr(obj)))
        _, entries["str()"] = self.call(f"str({expr})", lambda: self.text(str(obj)))
        param_names = {p.name for p in obj.params} if isinstance(obj, Params) else set()
        skipped, unknown = {}, []
        for name in sorted(dirs):
            if name.startswith("_") or name in covered:
                continue
            if name in skip:
                skipped.setdefault(skip[name], []).append(name)
                continue
            reason = _common_skip(obj, name, param_names)
            if reason is None and partial:
                skipped.setdefault("not captured on this copy (partial)", []).append(name)
            elif reason is None:
                unknown.append(name)
            else:
                skipped.setdefault(reason, []).append(name)
        if unknown:
            raise AssertionError(f"{cls_name}: members neither covered nor skipped: {unknown}")
        return {"class": cls_name, "members": entries, "skipped": skipped}

    def _read(self, obj, base, is_method, call):
        def read():
            attr = getattr(obj, base)
            if callable(attr) and not is_method:
                raise AssertionError(f"{call} is a method, list it as {base}()")
            return attr() if is_method else attr

        return self.call(call, read)

    def _invoke(self, obj, method, args, call):
        return self.call(call, lambda: getattr(obj, method)(*args))

    def record_members(self, store, key, obj, cls_name, expr, **kwargs):
        """members() stored under store[key] (store is one of this object's output dicts)."""
        store[key] = self.members(obj, cls_name, expr, **kwargs)
        return store[key]

    def class_methods(self, rows, resolve, frame):
        """Each (class, method, label, args, kwargs) builds a model, then transform(frame)."""
        for cls_name, method, label, args, kwargs in rows:
            cls = resolve(cls_name)
            parts = [self.arg_text(a) for a in args]
            parts += [f"{k}={repr(v)}" for k, v in kwargs.items()]
            call = "{}.{}({})".format(cls_name, method, ", ".join(parts))
            model, construct = self.call(call, partial(getattr(cls, method), *args, **kwargs))
            transform = None
            if model is not None:
                _, transform = self.call(
                    f"model.transform({self._frames[id(frame)]})",
                    partial(model.transform, frame),
                )
            self.class_methods_out[f"{cls_name}.{method}.{label}"] = {
                "construct": construct,
                "transform": transform,
            }

    def persist(self, key, obj, cls_name, expr, attributes=(), arguments=None, frame=None):
        """Save obj, load it back, and capture the loaded object's members."""
        path = os.path.join(self._tmp, re.sub(r"[^A-Za-z0-9_]", "_", key))
        uid = obj.uid
        params = self._params(obj)
        _, save = self.call(
            f"{expr}.write().overwrite().save(path)",
            lambda: obj.write().overwrite().save(path),
        )
        cls = type(obj)
        loaded, load = self.call(f"{cls_name}.load(path)", lambda: cls.load(path))
        with _scope():
            uid_text = self.uid_value(uid)
        record = {"save": save, "load": load, "uid": uid_text, "params_before": params}
        if loaded is not None:
            record["uid_same"] = loaded.uid == uid
            record["params_after"] = self._params(loaded)
            if attributes or arguments is not None:
                record["members"] = self.members(
                    loaded,
                    cls_name,
                    "loaded",
                    attributes=attributes,
                    arguments=arguments,
                    frame=frame,
                    partial=True,
                )
        self.persistence[key] = record
        return loaded

    def dump(self, path, sections):
        """Write {"source", "encoding", "datasets", <sections>} as JSON."""
        doc = {"source": SOURCE, "encoding": ENCODING, "datasets": self.datasets}
        doc.update(sections)
        with open(path, "w") as f:
            json.dump(doc, f, indent=1, sort_keys=True, allow_nan=False)
            f.write("\n")


SPARK_TYPES = {
    "double": "double",
    "int": "int",
    "long": "bigint",
    "string": "string",
    "boolean": "boolean",
    "vector": "array<double>",
    "array<double>": "array<double>",
    "array<string>": "array<string>",
    "array<int>": "array<int>",
}
DEFAULT_COLUMNS = [("label", "double"), ("weight", "double"), ("features", "vector")]


def _stored(kind, v):
    """A dataset value as the golden stores it: non-finite doubles are the strings of _num."""
    if v is None:
        return None
    if kind == "double":
        return _num(v)
    if kind in ("vector", "array<double>"):
        return [_num(x) for x in v]
    return v


def _python_value(kind, v):
    if v is None:
        return None
    if kind == "double":
        return float(v)
    if kind in ("int", "long"):
        return int(v)
    if kind == "string":
        return str(v)
    if kind == "boolean":
        return bool(v)
    if kind in ("vector", "array<double>"):
        return [float(x) for x in v]
    if kind == "array<string>":
        return [str(x) for x in v]
    return [int(x) for x in v]


def _floats(values):
    return "[" + ", ".join(repr(float(x)) for x in values) + "]"


ENCODING = (
    "Floats are JSON numbers (Python repr); NaN and infinities are strings. Vectors and matrices "
    "as in scripts/ml_golden_common.py. DataFrames: schema and rows sorted by canonical JSON. "
    "Model and summary objects: {type, ref} with ref a placeholder: <idN> for an object id "
    "(<idN>.summary for a summary), numbered in order of first appearance. Errors: {class, "
    "condition, message}. commands: base64 of the MlCommand or MlRelation sent, results: "
    "base64 of the MlCommandResult received, both SerializeToString(deterministic=True) after "
    "normalization (object ids and uids -> placeholders, plan_id, origin and LocalRelation data "
    "cleared, paths -> <path>)."
)
