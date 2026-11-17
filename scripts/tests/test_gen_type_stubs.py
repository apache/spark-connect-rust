#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The transform of gen_type_stubs.py, its re-exports, its extras and its --check.

Each test makes a PySpark source tree and a drop-in of a few modules; none needs a reference
PySpark, which the generated-drift workflow checks the real stubs against.
"""

import importlib.util
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "gen_type_stubs.py"
SPEC = importlib.util.spec_from_file_location("gen_type_stubs", SCRIPT)
gen = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gen
SPEC.loader.exec_module(gen)

UTILS = """
from typing import Any, Callable, TypeVar

FuncT = TypeVar("FuncT", bound=Callable[..., Any])


def try_remote_functions(f: FuncT) -> FuncT:
    return f


def not_identity(f: FuncT) -> Callable[..., Any]:
    return f
"""

MODULE = '''
"""PySpark's module."""

import sys
import warnings
from typing import TYPE_CHECKING, Optional, Union, overload

from pyspark.sql.column import Column
from pyspark.sql.utils import try_remote_functions
from pyspark.sql.utils import helper, missing_helper

if TYPE_CHECKING:
    from pyspark.sql.dataframe import DataFrame

__all__ = ["col", "Reader"]


def _private(x: int) -> int:
    return x


def _kept(x: int) -> int:
    return x


alias = _kept


@try_remote_functions
def col(name: str, flag: bool = True, *, sep: str = ",") -> Column:
    """Returns a Column."""
    warnings.warn("x")
    return Column(name)


@overload
def lit(value: int) -> Column: ...


@overload
def lit(value: str) -> Column: ...


def lit(value: Union[int, str]) -> Column:
    return Column(value)


class Reader:
    """A reader."""

    limit: int = 10

    def __init__(self, df: "DataFrame") -> None:
        self._df = df

    @property
    def df(self) -> "DataFrame":
        return self._df

    def _private_member(self) -> Optional[str]:
        return None

    load = __init__


Reader.__doc__ = "docs"

if sys.version_info >= (3, 11):

    def new_api() -> None:
        pass


if __name__ == "__main__":
    import doctest

    globs = {}
    doctest.testmod(globs=globs)
'''

DROPIN_UTILS = """
from pyspark._pyspark import helper
"""

EXPECTED = """
import sys
from typing import Optional, overload
from pyspark.sql.column import Column as Column
from pyspark.sql.utils import helper as helper
from pyspark.sql.dataframe import DataFrame

__all__ = ['col', 'Reader']

def _kept(x: int) -> int: ...

alias = _kept

def col(name: str, flag: bool=True, *, sep: str=",") -> Column: ...
@overload
def lit(value: int) -> Column: ...
@overload
def lit(value: str) -> Column: ...

class Reader:
    limit: int = ...
    def __init__(self, df: 'DataFrame') -> None: ...
    @property
    def df(self) -> 'DataFrame': ...
    def _private_member(self) -> Optional[str]: ...
    load = __init__

if sys.version_info >= (3, 11):
    def new_api() -> None: ...
"""


class Tree:
    """A PySpark source tree and a repository of the drop-in, in a temporary directory."""

    def __init__(self, test: unittest.TestCase):
        tmp = tempfile.TemporaryDirectory()
        test.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.src = self.root / "spark" / "python"
        self.repo = self.root / "repo"
        self.write(self.src, "pyspark/sql/functions/builtin.py", "")
        self.write(self.src, "pyspark/sql/utils.py", UTILS)
        self.write(self.repo, "scripts/check_vendored_upstream.sh", 'SPARK_TAG="v9.9.9"\n')
        self.write(
            self.repo,
            "pyproject.toml",
            "extend-exclude = [\n" + gen.RUFF_BLOCK_BEGIN + gen.RUFF_BLOCK_END + "]\n",
        )
        self.write(self.repo, "python/pyspark/sql/column.py", "class Column: ...\n")
        self.write(self.repo, "python/pyspark/sql/utils.py", DROPIN_UTILS)

    @staticmethod
    def write(base: Path, path: str, text: str) -> None:
        (base / path).parent.mkdir(parents=True, exist_ok=True)
        (base / path).write_text(textwrap.dedent(text).lstrip("\n"))

    def generate(self, modules, extras=None):
        with (
            mock.patch.object(gen, "STUB_MODULES", tuple(modules)),
            mock.patch.object(gen, "EXTRAS", extras or {}),
            mock.patch.object(
                gen, "IDENTITY_DECORATORS", frozenset({"pyspark.sql.utils.try_remote_functions"})
            ),
        ):
            return gen.generate(self.repo, str(self.src), log=lambda *a: None)


def body(stub: str) -> str:
    """The stub without its header."""
    return stub.split("instead of the module of the drop-in.\n", 1)[1]


class TransformTests(unittest.TestCase):
    def test_keeps_the_declarations_and_their_imports(self):
        tree = Tree(self)
        tree.write(tree.src, "pyspark/sql/reader.py", MODULE)
        outputs = tree.generate(["pyspark.sql.reader"])
        stub = outputs[Path("python/pyspark/sql/reader.pyi")]
        self.assertIn("from python/pyspark/sql/reader.py of Apache Spark v9.9.9", stub)
        # Bodies, docstrings, the doctests, the implementation of the overloads, the identity
        # decorator, the unreferenced private function and the unneeded imports are dropped.
        # Column is re-exported, as PySpark's module imports it at run time, and so is helper,
        # which the drop-in's pyspark.sql.utils has, unlike missing_helper. DataFrame, of the
        # TYPE_CHECKING block, is imported for the annotations only. Defaults keep their text.
        self.assertEqual(body(stub), EXPECTED)

    def test_an_unknown_decorator_is_an_error(self):
        tree = Tree(self)
        tree.write(
            tree.src,
            "pyspark/sql/reader.py",
            """
            from pyspark.sql.utils import not_identity

            @not_identity
            def f() -> None: ...
            """,
        )
        with self.assertRaisesRegex(gen.GeneratorError, "unknown decorator not_identity"):
            tree.generate(["pyspark.sql.reader"])

    def test_identity_decorators_are_checked_in_pyspark(self):
        tree = Tree(self)
        tree.write(tree.src, "pyspark/sql/reader.py", "")
        with (
            mock.patch.object(
                gen, "IDENTITY_DECORATORS", frozenset({"pyspark.sql.utils.not_identity"})
            ),
            self.assertRaisesRegex(gen.GeneratorError, "not annotated as returning what it takes"),
        ):
            gen.check_identity_decorators(tree.src)

    def test_extras_add_the_names_of_the_dropin(self):
        tree = Tree(self)
        tree.write(
            tree.src,
            "pyspark/sql/window.py",
            """
            from typing import TYPE_CHECKING

            if TYPE_CHECKING:
                from pyspark.sql.column import Column

            __all__ = ["Window"]

            class Window:
                def over(self, c: "Column") -> "Column": ...
            """,
        )
        extras = {
            "pyspark.sql.window": gen.Extra(
                "from pyspark.sql.column import Column as Column\nclass FrameBound: ...\n",
                ("FrameBound",),
            )
        }
        stub = body(
            tree.generate(["pyspark.sql.window"], extras)[Path("python/pyspark/sql/window.pyi")]
        )
        # The extra re-exports Column, which PySpark's module imports under TYPE_CHECKING only.
        self.assertNotIn("from pyspark.sql.column import Column\n", stub)
        self.assertIn("__all__ = ['Window', 'FrameBound']", stub)
        self.assertTrue(
            stub.endswith(
                "# Names of the drop-in's module that PySpark's does not have.\n"
                "from pyspark.sql.column import Column as Column\nclass FrameBound: ...\n"
            ),
            stub,
        )
        # An extra that PySpark's module now has is an error.
        clash = {"pyspark.sql.window": gen.Extra("class Window: ...\n")}
        with self.assertRaisesRegex(gen.GeneratorError, r"now has \['Window'\]"):
            tree.generate(["pyspark.sql.window"], clash)


class CheckTests(unittest.TestCase):
    def test_reports_drift_stale_stubs_and_the_ruff_block(self):
        tree = Tree(self)
        tree.write(tree.src, "pyspark/sql/reader.py", MODULE)
        outputs = tree.generate(["pyspark.sql.reader"])
        self.assertIn('    "python/pyspark/sql/reader.pyi",\n', outputs[gen.PYPROJECT])
        for path, text in outputs.items():
            (tree.repo / path).write_text(text)
        log = []
        self.assertTrue(gen.check(tree.repo, outputs, log=log.append))

        stub = tree.repo / "python/pyspark/sql/reader.pyi"
        stub.write_text(stub.read_text() + "x: int\n")
        log = []
        self.assertFalse(gen.check(tree.repo, outputs, log=log.append))
        self.assertIn("-x: int", log)
        stub.write_text(outputs[Path("python/pyspark/sql/reader.pyi")])

        stale = tree.repo / "python/pyspark/sql/gone.pyi"
        stale.write_text(outputs[Path("python/pyspark/sql/reader.pyi")])
        log = []
        self.assertFalse(gen.check(tree.repo, outputs, log=log.append))
        self.assertIn("STALE python/pyspark/sql/gone.pyi", " ".join(log))


if __name__ == "__main__":
    unittest.main()
