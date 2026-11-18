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

import ast
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
    def __init__(self, df: 'DataFrame') -> None:
        self._df = df
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
        self.write(self.repo, "python/pyspark/sql/dataframe.py", "class DataFrame: ...\n")
        self.write(self.repo, "python/pyspark/sql/utils.py", DROPIN_UTILS)

    @staticmethod
    def write(base: Path, path: str, text: str) -> None:
        (base / path).parent.mkdir(parents=True, exist_ok=True)
        (base / path).write_text(textwrap.dedent(text).lstrip("\n"))

    def generate(self, modules, extras=None, attribute_types=None, unresolved_imports=None):
        with (
            mock.patch.object(gen, "STUB_MODULES", tuple(modules)),
            mock.patch.object(gen, "EXTRAS", extras or {}),
            mock.patch.object(gen, "ATTRIBUTE_TYPES", attribute_types or {}),
            mock.patch.object(gen, "UNRESOLVED_IMPORTS", unresolved_imports or {}),
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
        # decorator, the unreferenced private function and the unneeded imports are dropped,
        # but the body of __init__, which assigns the instance attribute _df. Column is
        # re-exported, as PySpark's module imports it at run time, and so is helper, which the
        # drop-in's pyspark.sql.utils has, unlike missing_helper. DataFrame, of the
        # TYPE_CHECKING block, is imported for the annotations only. Defaults keep their text.
        self.assertEqual(body(stub), EXPECTED)

    def test_keeps_the_methods_that_assign_instance_attributes(self):
        tree = Tree(self)
        tree.write(
            tree.src,
            "pyspark/sql/schema.py",
            '''
            import os
            from typing import Dict, List, Optional

            from pyspark.sql.column import Column


            class Field:
                kind: str = "field"

                def __init__(self, name: str, metadata: Optional[Dict[str, str]] = None) -> None:
                    """A field."""
                    from pyspark.sql.utils import helper

                    self.name = name  # as PySpark writes it
                    self.metadata = metadata or {}
                    self.helper = helper

                def rename(self, name: str) -> "Field":
                    self.name, self.kind = name, os.sep
                    return self

                def describe(self) -> str:
                    return self.name


            class Schema:
                def __init__(self, fields: Optional[List[Field]] = None) -> None:
                    if not fields:
                        self.fields = []
                    else:
                        self.fields = fields

                @staticmethod
                def empty(column: Column) -> "Schema":
                    return Schema()
            ''',
        )
        attribute_types = {"pyspark.sql.schema": {"Schema.fields": "List[Field]"}}
        stub = body(
            tree.generate(["pyspark.sql.schema"], attribute_types=attribute_types)[
                Path("python/pyspark/sql/schema.pyi")
            ]
        )
        # A type checker infers each attribute from the body that its method keeps, as
        # PySpark's source writes it, but its docstring: Field.name and Field.metadata, of
        # __init__ and rename. The other methods are stubs. pyright does not narrow fields in a
        # stub, which ATTRIBUTE_TYPES declares. The import of os is kept for the body that reads
        # it, not the function-local import of helper, nor the import of Column.
        self.assertEqual(
            stub,
            """
import os
from typing import Dict, List, Optional
from pyspark.sql.column import Column as Column

class Field:
    kind: str = ...
    def __init__(self, name: str, metadata: Optional[Dict[str, str]]=None) -> None:
        from pyspark.sql.utils import helper

        self.name = name  # as PySpark writes it
        self.metadata = metadata or {}
        self.helper = helper
    def rename(self, name: str) -> 'Field':
        self.name, self.kind = name, os.sep
        return self
    def describe(self) -> str: ...

class Schema:
    fields: List[Field]
    def __init__(self, fields: Optional[List[Field]]=None) -> None:
        if not fields:
            self.fields = []
        else:
            self.fields = fields
    @staticmethod
    def empty(column: Column) -> 'Schema': ...
""",
        )

    def test_attribute_types_and_the_attributes_are_checked(self):
        tree = Tree(self)
        tree.write(
            tree.src,
            "pyspark/sql/schema.py",
            """
            from typing import Optional, overload

            class Writer:
                source: str

                def __init__(self) -> None:
                    self._mode = None

                @overload
                def mode(self, mode: str) -> "Writer": ...

                @overload
                def mode(self, mode: None) -> "Writer": ...

                def mode(self, mode: Optional[str]) -> "Writer":
                    self.current = mode
                    return self
            """,
        )

        def generate(attribute_types):
            return tree.generate(["pyspark.sql.schema"], attribute_types=attribute_types)

        # The stub drops the implementation of an overloaded method, and with it the only
        # assignment of the public attribute current: an error, not a missing attribute.
        with self.assertRaisesRegex(
            gen.GeneratorError, r"declares no instance attribute \['Writer.current'\]"
        ):
            generate({})
        stub = generate({"pyspark.sql.schema": {"Writer.current": "Optional[str]"}})
        self.assertIn(
            "class Writer:\n    current: Optional[str]\n",
            body(stub[Path("python/pyspark/sql/schema.pyi")]),
        )
        # An entry that PySpark's module does not assign, declares or has is an error.
        for key, error in (
            ("Writer.mode_name", "does not assign it"),
            ("Writer.source", "declares it now"),
            ("Reader.current", "of no class"),
        ):
            with self.subTest(key), self.assertRaisesRegex(gen.GeneratorError, error):
                generate({"pyspark.sql.schema": {key: "int", "Writer.current": "Optional[str]"}})

    def test_check_attributes_reports_a_stub_without_them(self):
        # The check of each stub that the generator renders, on the stub that it rendered before
        # it kept the bodies of the methods that assign instance attributes.
        source = ast.parse(
            textwrap.dedent(
                """
                class StructType:
                    def __init__(self, fields):
                        self.fields = fields
                        self._needConversion = []
                """
            )
        )
        stub = "class StructType:\n    def __init__(self, fields): ...\n"
        with self.assertRaisesRegex(
            gen.GeneratorError, r"declares no instance attribute \['StructType.fields'\]"
        ):
            gen.check_attributes("pyspark.sql.types", source, stub)
        gen.check_attributes("pyspark.sql.types", source, "class StructType:\n    fields: list\n")

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

    def test_keeps_the_private_names_of_the_dropins_module(self):
        tree = Tree(self)
        tree.write(
            tree.src,
            "pyspark/sql/reader.py",
            """
            from typing import Dict

            _registry: Dict[str, int] = {}

            def _helper(x: int) -> int:
                return x

            def _unused(x: int) -> int:
                return x

            def read(x: int) -> int:
                return x
            """,
        )
        # The drop-in's module has _helper and _registry at run time, of its own source (not
        # of its stub), so the stub declares them; _unused, which it does not have, is dropped.
        tree.write(
            tree.repo,
            "python/pyspark/sql/reader.py",
            "from pyspark._pyspark import _helper, read\n_registry = {}\n",
        )
        stub = body(tree.generate(["pyspark.sql.reader"])[Path("python/pyspark/sql/reader.pyi")])
        self.assertEqual(
            stub,
            "\nfrom typing import Dict\n\n_registry: Dict[str, int]\n\n"
            "def _helper(x: int) -> int: ...\ndef read(x: int) -> int: ...\n",
        )

    def test_keeps_the_type_ignores_and_the_imports_of_the_annotation_of_all(self):
        tree = Tree(self)
        tree.write(
            tree.src,
            "pyspark/sql/reader.py",
            """
            from typing import List

            __all__: List[str] = ["Reader"]


            class Reader(object):  # type: ignore[misc]
                @property  # type: ignore[override]
                def name(self) -> str:
                    return ""

                def __eq__(  # type: ignore[override]
                    self,
                    other: "Reader",
                ) -> "Reader":  # type: ignore[misc]
                    return self

                same = __eq__  # type: ignore[assignment]
            """,
        )
        stub = body(tree.generate(["pyspark.sql.reader"])[Path("python/pyspark/sql/reader.pyi")])
        # The import of List, which only the annotation of __all__ reads, stays. Each comment
        # stays on the line that ast.unparse prints of its statement, those of a multi-line
        # signature merged.
        self.assertEqual(
            stub,
            """
from typing import List

__all__: List[str] = ['Reader']

class Reader(object):  # type: ignore[misc]
    @property  # type: ignore[override]
    def name(self) -> str: ...
    def __eq__(self, other: 'Reader') -> 'Reader': ...  # type: ignore[misc, override]
    same = __eq__  # type: ignore[assignment]
""",
        )

    def test_the_imports_of_a_stub_resolve_in_the_dropin(self):
        tree = Tree(self)
        tree.write(
            tree.src,
            "pyspark/sql/reader.py",
            """
            from pyspark.sql.utils import helper, missing_helper

            def read(x: missing_helper) -> helper: ...
            """,
        )
        # The drop-in's pyspark.sql.utils has helper, not missing_helper.
        with self.assertRaisesRegex(
            gen.GeneratorError, r"does not have: \{'pyspark.sql.utils.missing_helper'"
        ):
            tree.generate(["pyspark.sql.reader"])
        unresolved = {"pyspark.sql.utils.missing_helper": "the reason"}
        tree.generate(["pyspark.sql.reader"], unresolved_imports=unresolved)
        unresolved["pyspark.sql.utils.helper"] = "it has it"
        with self.assertRaisesRegex(gen.GeneratorError, r"has now: \['pyspark.sql.utils.helper'"):
            tree.generate(["pyspark.sql.reader"], unresolved_imports=unresolved)

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
