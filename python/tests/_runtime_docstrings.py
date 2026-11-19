# SPDX-License-Identifier: Apache-2.0
"""The docstrings of the modules of pyspark.sql and pyspark.errors at run time, as digests of their
exact text, of scripts/capture_runtime_docstrings_golden.py (of official PySpark) and of the test of
the drop-in's (test_dropin_class_modules_offline.py): of each module, of each public class and
function that the module defines, and of each public member (and ``__init__``) of such a class,
as Python finds it on the class, by "module", "module:Name" and "module:Class.member". The digest
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

# The golden leaves out the docstrings that are not all PySpark's yet: those of the DataFrame class
# of both DataFrame modules, and of the modules of the session and its client
# (pyspark.sql.connect.session and pyspark.sql.connect.client), which are left out here as a whole.
EXCLUDED = (
    "pyspark.sql.connect.session",
    "pyspark.sql.connect.client",
)
# Of the two DataFrame modules, the golden leaves out the DataFrame class, and the docstrings that
# can only match PySpark's once those of the class do: the docstrings of both modules, and those
# that the classic `DataFrameNaFunctions` and `DataFrameStatFunctions` copy of the class
# (`drop.__doc__ = DataFrame.dropna.__doc__`). Their other docstrings are compared.
DATAFRAME_MODULES = ("pyspark.sql.dataframe", "pyspark.sql.connect.dataframe")
COPIES_OF_THE_DATAFRAME = {
    "DataFrameNaFunctions.drop",
    "DataFrameNaFunctions.fill",
    "DataFrameNaFunctions.replace",
    "DataFrameStatFunctions.approxQuantile",
    "DataFrameStatFunctions.corr",
    "DataFrameStatFunctions.cov",
    "DataFrameStatFunctions.crosstab",
    "DataFrameStatFunctions.freqItems",
    "DataFrameStatFunctions.sampleBy",
}


def of_the_dataframe(module_name, rest):
    """Whether the key of `rest` of the module `module_name` is of the DataFrame class, of a
    DataFrame module: the module, the class, its members, and the classic copies of them."""
    if module_name not in DATAFRAME_MODULES:
        return False
    if rest in ("", "DataFrame") or rest.startswith("DataFrame."):
        return True
    return module_name == "pyspark.sql.dataframe" and rest in COPIES_OF_THE_DATAFRAME

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
    """The modules of pyspark.sql and pyspark.errors, but the excluded and the test modules."""
    found = {"pyspark.sql", "pyspark.errors"}
    for root in ("pyspark.sql", "pyspark.errors"):
        package = importlib.import_module(root)
        for info in pkgutil.walk_packages(package.__path__, root + ".", onerror=lambda _: None):
            found.add(info.name)
    return sorted(
        name
        for name in found
        if not any(name == e or name.startswith(e + ".") for e in EXCLUDED)
        and ".tests" not in name
        and not name.endswith("_pb2")
        and ".proto." not in name + "."
    )


def docstrings(module_name):
    """The digests of the docstrings of the module `module_name`, by their keys, but those of the
    DataFrame class (``of_the_dataframe``)."""
    out = _docstrings(module_name)
    return {
        key: value
        for key, value in out.items()
        if not of_the_dataframe(module_name, key.partition(":")[2])
    }


def _docstrings(module_name):
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
        if not inspect.isclass(obj):
            continue
        for member in sorted(set(dir(obj))):
            if member.startswith("_") and member != "__init__":
                continue
            definer = next((c for c in obj.__mro__ if member in vars(c)), None)
            if member != "__init__" and (definer is None or definer.__module__ in ("builtins", "abc", "typing", "enum", "collections")):
                continue
            try:
                found = entry(getattr(obj, member))
            except Exception as e:  # noqa: BLE001
                found = digest("error: " + type(e).__name__)
            out[f"{module_name}:{attr}.{member}"] = found
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
