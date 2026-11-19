# SPDX-License-Identifier: Apache-2.0
"""pyspark.sql.functions as official PySpark 4.3.0's, without a server: PySpark's package of the
modules builtin and partitioning, whose functions run the functions of pyspark.sql.connect.functions
of their names when the session is remote (try_remote_functions of pyspark.sql.utils). Those are
the drop-in's, of the Rust core, which carry the docstrings and the annotations of PySpark's, as
PySpark Connect's do. The expected facts are pyspark-client 4.3.0's, as it shows them."""

import importlib.util
import inspect
import os
import pickle
import pkgutil
import subprocess
import sys
import types
import warnings
from pathlib import Path

import pyspark.sql
import pyspark.sql.connect.functions as CF
import pyspark.sql.connect.functions.builtin as CB
import pyspark.sql.connect.functions.partitioning as CP
import pyspark.sql.functions as F
import pyspark.sql.functions.builtin as B
import pyspark.sql.functions.partitioning as P
import pyspark.sql.utils
import pytest
from pyspark import cloudpickle
from pyspark.errors import PySparkTypeError

SQL = Path(pyspark.sql.__file__).parent
# The functions of PySpark's module builtin that it imports from pyspark.sql.pandas.functions.
PANDAS = ("pandas_udf", "arrow_udf")


def _functions(module):
    """The public functions of `module` that it defines, by name."""
    return {
        name: value
        for name, value in vars(module).items()
        if not name.startswith("_")
        and isinstance(value, types.FunctionType)
        and value.__module__ == module.__name__
    }


def test_the_functions_are_a_package_of_the_modules_builtin_and_partitioning():
    # The drop-in's pyspark.sql.functions was the module functions.py, of no __path__, and its
    # builtin and partitioning were ModuleType objects that it put in sys.modules, of no __file__
    # or __spec__, which importlib.reload and importlib.util.find_spec refused.
    assert Path(F.__file__) == SQL / "functions" / "__init__.py"
    assert [Path(p) for p in F.__path__] == [SQL / "functions"]
    assert F.__package__ == F.__spec__.name == "pyspark.sql.functions"
    assert F.__doc__ == "PySpark Functions"
    modules = [(m.name, m.ispkg) for m in pkgutil.iter_modules(pyspark.sql.__path__)]
    assert ("functions", True) in modules and ("functions_generated", False) not in modules
    assert sorted((m.name, m.ispkg) for m in pkgutil.iter_modules(F.__path__)) == [
        ("builtin", False),
        ("partitioning", False),
    ]
    for module, name, doc in [
        (B, "builtin", "A collections of builtin functions"),
        (P, "partitioning", "A collections of partitioning functions"),
    ]:
        assert getattr(F, name) is module
        assert Path(module.__file__) == SQL / "functions" / f"{name}.py"
        assert module.__spec__.name == module.__name__ == f"pyspark.sql.functions.{name}"
        assert importlib.util.find_spec(module.__name__).origin == module.__file__
        assert module.__doc__.strip() == doc
        assert not hasattr(module, "__all__")


def test_the_modules_reload_as_pysparks():
    # A reload makes new functions of the same names: pyspark-client's modules do, where the
    # drop-in's ModuleType objects of no __spec__ raised. In a process of its own, as the functions
    # of the other tests are those of the first import.
    code = (
        "import importlib\n"
        "import pyspark.sql.functions as F\n"
        "import pyspark.sql.functions.builtin as B\n"
        "import pyspark.sql.functions.partitioning as P\n"
        "old_col, old_bucket = B.col, P.bucket\n"
        "assert importlib.reload(P) is P and importlib.reload(B) is B\n"
        "assert B.col is not old_col and P.bucket is not old_bucket\n"
        "assert F.col is old_col\n"  # the package binds them of its star import
        "assert importlib.reload(F) is F and F.col is B.col and F.partitioning is P\n"
        "print(B.col('a'), P.bucket(2, 'a'))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=os.environ, timeout=300
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["Column<'a'>", "Column<'bucket(2,", "a)'>"]


def test_the_functions_are_pysparks_which_run_those_of_spark_connect_of_their_names():
    # pyspark-client: every function of __all__ is of pyspark.sql.functions.builtin (but pandas_udf
    # and arrow_udf), the try_remote_functions wrapper of PySpark's function, which it wraps
    # (__wrapped__), with its docstring and signature, and which runs the function of
    # pyspark.sql.connect.functions of its name. The drop-in's were its own functions of the Rust
    # core, of no __wrapped__, of its docstrings, unannotated.
    utils = Path(pyspark.sql.utils.__file__)
    for name in F.__all__:
        function = getattr(F, name)
        if isinstance(function, type):
            continue
        assert function is getattr(B, name), name
        if name in PANDAS:
            assert function.__module__ == "pyspark.sql.pandas.functions"
            continue
        assert function.__module__ == "pyspark.sql.functions.builtin", name
        assert (Path(function.__code__.co_filename), function.__code__.co_name) == (
            utils,
            "wrapped",
        ), name
        wrapped = function.__wrapped__
        assert wrapped.__module__ == "pyspark.sql.functions.builtin"
        assert function.__doc__ == wrapped.__doc__ and function.__doc__, name
        assert inspect.signature(function) == inspect.signature(wrapped)
        assert callable(getattr(CF, function.__name__)), name
    for name, function in _functions(P).items():
        assert function.__wrapped__.__module__ == P.__name__
        assert callable(getattr(CP, name)), name
    assert str(inspect.signature(F.col)).startswith("(col: str) -> ")
    assert F.col.__doc__.lstrip().startswith(
        "Returns a :class:`~pyspark.sql.Column` based on the given column name."
    )
    assert inspect.getsource(F.col).splitlines()[0] == "@_try_remote_functions"
    # The aliases of PySpark bind their functions.
    assert F.column is F.col and F.random is F.rand and F.power is F.pow


def test_the_udf_functions_have_pysparks_docstrings():
    # The drop-in's udf, udtf and arrow_udtf had docstrings of their own ("Create a user-defined
    # function (UDF).", of 793 characters), where pyspark-client's are PySpark's, as those of
    # pyspark.sql.connect.functions, which PySpark Connect copies.
    for function, first in [
        (F.udf, "Creates a user defined function (UDF)."),
        (F.udtf, "Creates a user defined table function (UDTF)."),
        (F.arrow_udtf, "Creates a PyArrow-native user defined table function (UDTF)."),
    ]:
        assert function.__doc__.lstrip().startswith(first)
        assert "Examples\n    --------" in function.__doc__
        assert getattr(CF, function.__name__).__doc__ == function.__doc__


def test_the_functions_pickle_by_reference_to_pysparks_modules():
    # As pyspark-client's: of pyspark.sql.functions.builtin and partitioning, which the Python
    # worker of the server imports.
    for function in (F.col, F.array_sort, F.udf, F.partitioning.bucket, F.years):
        payload = pickle.dumps(function)
        assert pickle.loads(payload) is function
        assert function.__module__.encode() in payload and b"connect" not in payload
        assert pickle.loads(cloudpickle.dumps(function)) is function


def test_a_call_runs_the_function_of_spark_connect_of_its_name_at_the_call(monkeypatch):
    # pyspark-client: try_remote_functions takes the function of pyspark.sql.connect.functions
    # of its name at each call, so a patch of it is what the functions of pyspark.sql.functions
    # run; a patch of a module binds the name of that module only. The drop-in's functions
    # were those of Spark Connect, of their own module, which no patch of the other changed.
    calls = []

    def patched(*args, **kwargs):
        calls.append((args, kwargs))
        return "patched"

    monkeypatch.setattr(CF, "upper", patched)
    assert F.upper("a") == B.upper(col="a") == CF.upper("b") == "patched"
    assert calls == [(("a",), {}), ((), {"col": "a"}), (("b",), {})]
    assert str(CB.upper("a")) == "Column<'upper(a)'>"
    monkeypatch.undo()

    # pyspark.sql.functions binds the functions of builtin: a patch of one is not of the other.
    monkeypatch.setattr(F, "lower", patched)
    assert F.lower("a") == "patched" and str(B.lower("a")) == "Column<'lower(a)'>"
    monkeypatch.undo()
    monkeypatch.setattr(B, "lower", patched)
    assert B.lower("a") == "patched" and str(F.lower("a")) == "Column<'lower(a)'>"
    monkeypatch.undo()
    # pyspark.sql.connect.functions binds those of its builtin, which the functions run.
    monkeypatch.setattr(CB, "lower", patched)
    assert str(F.lower("a")) == "Column<'lower(a)'>" and CB.lower("a") == "patched"
    monkeypatch.undo()

    # The partition transforms, of pyspark.sql.connect.functions.partitioning, also those
    # deprecated of builtin, which call them.
    monkeypatch.setattr(CP, "bucket", patched)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        assert F.partitioning.bucket(2, "a") == F.bucket(2, "a") == "patched"


def test_the_functions_of_spark_connect_call_those_of_their_module_at_the_call(monkeypatch):
    # pyspark-client: PySpark Connect's functions call the lit, column and _to_col of their module
    # (pyspark.sql.connect.functions.builtin), so a patch of one is what they call. The drop-in's
    # generated functions were of a module of their own, functions_generated, which the patch
    # did not change.
    calls = []
    lit = CB.lit

    def patched(value):
        calls.append(value)
        return lit(value)

    monkeypatch.setattr(CB, "lit", patched)
    assert str(F.array_contains("a", 1)) == "Column<'array_contains(a, 1)'>"
    assert calls == [1]
    assert CB.array_contains.__globals__ is vars(CB)
    assert CB.array_contains.__module__ == CB.__name__


def test_without_the_shared_namespace_the_functions_run_pysparks_classic_bodies(monkeypatch):
    # pyspark-client: with PYSPARK_NO_NAMESPACE_SHARE, try_remote_functions runs the function's
    # own body, of Spark Classic, which a client of Spark Connect only cannot run: col imports the
    # SparkContext of pyspark, sum and bucket the module pyspark.sql.classic. The drop-in's
    # functions ignored it.
    monkeypatch.setenv("PYSPARK_NO_NAMESPACE_SHARE", "1")
    with pytest.raises(ImportError, match="cannot import name 'SparkContext' from 'pyspark'"):
        F.col("a")
    for call in (lambda: F.sum("a"), lambda: F.partitioning.bucket(2, "a")):
        with pytest.raises(ModuleNotFoundError, match="No module named 'pyspark.sql.classic'"):
            call()
    monkeypatch.delenv("PYSPARK_NO_NAMESPACE_SHARE")
    assert str(F.col("a")) == "Column<'a'>"


def test_the_names_of_the_package_are_pysparks():
    # pyspark-client: the star import of builtin, a module of no __all__, binds all of its public
    # names in the package, also the modules, the typing names and the classes that it imports;
    # __all__ lists PySpark's 562 functions and classes, which a star import of the package binds.
    assert len(F.__all__) == 562
    names = set(dir(F))
    assert {"builtin", "partitioning", "inspect", "functools", "Optional"} <= names
    assert not {n for n in names if n.startswith("_") and not n.startswith("__")}
    assert F.UserDefinedFunction is pyspark.sql.udf.UserDefinedFunction
    assert F.PandasUDFType.__module__ == "pyspark.sql.pandas.functions"
    namespace = {}
    exec("from pyspark.sql.functions import *", namespace)
    assert sorted(set(namespace) - {"__builtins__"}) == sorted(F.__all__)


def test_the_functions_of_spark_connect_have_pysparks_docstrings_and_annotations():
    # pyspark-client: PySpark Connect's functions are of its builtin, but pandas_udf and arrow_udf,
    # which are PySpark's; they have the docstrings of PySpark's (col.__doc__ =
    # pysparkfuncs.col.__doc__), but date_part, datepart and uuid, of none, and are annotated as
    # they are. The drop-in's had their own docstrings, or none, and no annotations.
    others = {
        name: value.__module__
        for name, value in vars(CB).items()
        if not name.startswith("_")
        and isinstance(value, types.FunctionType)
        and value.__module__ != CB.__name__
    }
    assert others == {
        "pandas_udf": "pyspark.sql.pandas.functions",
        "arrow_udf": "pyspark.sql.pandas.functions",
        "cast": "typing",
        "overload": "typing",
    }
    connect = _functions(CB)
    assert len(connect) == 562
    for name, function in connect.items():
        classic = getattr(F, name)
        if name in ("date_part", "datepart", "uuid"):
            assert function.__doc__ is None, name
        else:
            assert function.__doc__ == classic.__doc__, name
        assert inspect.signature(function) == inspect.signature(classic), name
    assert CF.pandas_udf is F.pandas_udf and CF.arrow_udf is F.arrow_udf
    assert str(inspect.signature(CF.col)).startswith("(col: str) -> ")
    for name in ("udf", "udtf", "arrow_udtf"):
        assert getattr(CF, name).__module__ == CB.__name__


def test_the_partition_transforms_of_spark_connect_are_pyspark_connects():
    # PySpark Connect's module pyspark.sql.connect.functions.partitioning, vendored, whose
    # functions have the docstrings and the annotations of PySpark's.
    assert Path(CP.__file__) == SQL / "connect" / "functions" / "partitioning.py"
    assert CF.partitioning is CP
    for name, function in _functions(CP).items():
        classic = getattr(P, name)
        assert function.__doc__ == classic.__doc__, name
        assert function.__doc__.lstrip().startswith("Partition transform function: A transform")
        assert inspect.signature(function) == inspect.signature(classic), name
    assert sorted(_functions(CP)) == ["bucket", "days", "hours", "months", "years"]
    assert str(F.partitioning.bucket(2, "a")) == "Column<'bucket(2, a)'>"
    with pytest.raises(PySparkTypeError, match="NOT_EXPECTED_TYPE"):
        F.partitioning.bucket("x", "a")
    with pytest.warns(FutureWarning, match="Deprecated in 4.0.0, use partitioning.years instead"):
        assert str(F.years("a")) == "Column<'years(a)'>"
