# SPDX-License-Identifier: Apache-2.0
"""The docstrings of the modules of pyspark.sql and pyspark.errors at run time, as digests of their
exact text, of scripts/capture_runtime_docstrings_golden.py (of official PySpark) and of the test of
the drop-in's (test_dropin_class_modules_offline.py): of each module, of each public class and
function that the module defines, and of each public member (and ``__init__``) of such a class,
and of a public class of its body (``SparkSession.Builder``), as Python finds it on the class, by
"module", "module:Name", "module:Class.member" and "module:Class.Inner.member". The digest
is of the kind ``doc``, of ``__doc__`` (None of none), but of an object of no ``__doc__`` of its own
that inherits a docstring, of the kind ``getdoc``, of ``inspect.getdoc``, which ``help()`` shows
(``{"getdoc": digest}``): such as of a member of PySpark Connect's ``Column``, a subclass of the
classic ``Column``, of no docstring of its own, which finds that of the classic class (``KINDS``).

The sweep imports every module of them, some of which keep the state of the process at their
import, such as the ``sys.stdout`` that ``pyspark.sql.connect.shell.progress`` binds as the default
output of its progress bar: the test runs it in a child interpreter, of this file as a script
(``python _runtime_docstrings.py <golden>``), which prints the comparison as JSON."""

import hashlib
import importlib
import inspect
import json
import pkgutil
import sys
import types
import warnings

# The kinds of the digests of the golden, which its note gives.
KINDS = {
    "doc": "a digest, or null: of the __doc__ of the object",
    "getdoc": (
        '{"getdoc": digest}: of the inspect.getdoc of an object of no __doc__ of its own that '
        "inherits a docstring, which help() shows"
    ),
}


def digest(doc):
    """The digest of a docstring: None of none."""
    if doc is None:
        return None
    if not isinstance(doc, str):
        return "not a str: " + type(doc).__name__
    return hashlib.sha256(doc.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def entry(obj):
    """The digest of the docstring of `obj` of the golden, of its kind (``KINDS``): of its
    ``__doc__``, or ``{"getdoc": digest}`` of ``inspect.getdoc`` of one of no ``__doc__`` of its own
    that inherits a docstring."""
    doc = getattr(obj, "__doc__", None)
    if doc is None:
        found = inspect.getdoc(obj)
        if found is not None:
            return {"getdoc": digest(found)}
    return digest(doc)


def module_names():
    """The modules of pyspark.sql and pyspark.errors, but the test modules and those of messages."""
    found = {"pyspark.sql", "pyspark.errors"}
    for root in ("pyspark.sql", "pyspark.errors"):
        package = importlib.import_module(root)
        for info in pkgutil.walk_packages(package.__path__, root + ".", onerror=lambda _: None):
            found.add(info.name)
    return sorted(
        name
        for name in found
        if ".tests" not in name
        and not name.endswith("_pb2")
        and ".proto." not in name + "."
    )


def docstrings(module_name):
    """The digests of the docstrings of the module `module_name`, by their keys."""
    warnings.simplefilter("ignore")
    module = importlib.import_module(module_name)
    out = {module_name: digest(module.__doc__)}
    for attr, obj in list(vars(module).items()):
        if attr.startswith("_") or isinstance(obj, types.ModuleType):
            continue
        if not (inspect.isclass(obj) or callable(obj)):
            continue
        if getattr(obj, "__module__", None) != module_name:
            continue
        out[f"{module_name}:{attr}"] = entry(obj)
        if inspect.isclass(obj):
            out.update(_members(module_name, attr, obj))
    return out


def _members(module_name, prefix, cls):
    """The digests of the docstrings of the public members (and ``__init__``) of the class `cls`
    of the key `prefix`, as Python finds them on the class, and of those of the public classes of
    its body (``SparkSession.Builder``), by their keys."""
    out = {}
    for member in sorted(set(dir(cls))):
        if member.startswith("_") and member != "__init__":
            continue
        definer = next((c for c in cls.__mro__ if member in vars(c)), None)
        if member != "__init__" and (definer is None or definer.__module__ in ("builtins", "abc", "typing", "enum", "collections")):
            continue
        value = None
        try:
            value = getattr(cls, member)
            found = entry(value)
        except Exception as e:  # noqa: BLE001
            found = digest("error: " + type(e).__name__)
        out[f"{module_name}:{prefix}.{member}"] = found
        if (
            inspect.isclass(value)
            and not member.startswith("_")
            and value.__qualname__ == f"{cls.__qualname__}.{member}"
        ):
            out.update(_members(module_name, f"{prefix}.{member}", value))
    return out


def resolve(key):
    """The docstring of the object of `key`, of the drop-in, or a KeyError if it has none."""
    module_name, _, rest = key.partition(":")
    found = importlib.import_module(module_name)
    for part in rest.split(".") if rest else []:
        if not hasattr(found, part):
            raise KeyError(key)
        found = getattr(found, part)
    return found


def compare(golden):
    """The comparison of the drop-in's docstrings with those of `golden`, the golden of
    capture_runtime_docstrings_golden.py, each of its kind: the number compared, the keys that
    differ, and the modules of the golden that do not import. The `__init__` of a native class is
    CPython's slot, of CPython's docstring, and a name that the drop-in does not have is not
    compared."""
    compared, differ, unimported = 0, [], []
    for module, entries in golden["modules"].items():
        try:
            importlib.import_module(module)
        except ImportError:
            unimported.append(module)
            continue
        for rest, expected in entries.items():
            key = f"{module}:{rest}" if rest else module
            try:
                found = resolve(key)
            except KeyError:
                continue
            if rest.endswith(".__init__") and isinstance(found, types.WrapperDescriptorType):
                continue
            compared += 1
            if isinstance(expected, dict):
                doc, expected = inspect.getdoc(found), expected["getdoc"]
            else:
                doc = found.__doc__
            if digest(doc) != expected:
                differ.append(key)
    return {"compared": compared, "differ": differ, "unimported": unimported}


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    import pyspark.sql.connect.session  # noqa: F401  (as the modules of PySpark import it)

    with open(sys.argv[1]) as golden_file:
        result = compare(json.load(golden_file))
    print(json.dumps(result))
