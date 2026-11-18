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

    def generate(
        self,
        modules,
        extras=None,
        attribute_types=None,
        unresolved_imports=None,
        dropped=None,
        dropped_star_imports=None,
        ml_connect=None,
    ):
        """The outputs of the generator of the stubs of `modules`, and of the modules of
        pyspark.ml.connect `ml_connect` and their package, if given."""
        with (
            mock.patch.object(gen, "STUB_MODULES", tuple(modules)),
            mock.patch.object(gen, "EXTRAS", extras or {}),
            mock.patch.object(gen, "ATTRIBUTE_TYPES", attribute_types or {}),
            mock.patch.object(gen, "UNRESOLVED_IMPORTS", unresolved_imports or {}),
            mock.patch.object(gen, "DROPPED", dropped or {}),
            mock.patch.object(gen, "DROPPED_STAR_IMPORTS", dropped_star_imports or {}),
            mock.patch.object(
                gen, "ML_CONNECT_PACKAGE", None if ml_connect is None else "pyspark.ml.connect"
            ),
            mock.patch.object(gen, "ML_CONNECT_MODULES", tuple(ml_connect or ())),
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

    def test_reexports_and_declares_only_what_the_dropins_module_has(self):
        tree = Tree(self)
        tree.write(
            tree.src,
            "pyspark/sql/reader.py",
            """
            from pyspark.sql.column import Column
            from pyspark.sql.utils import helper

            class Converter:
                def convert(self, x: int) -> int: ...

            class Reader:
                def load(self, path: str) -> Column: ...

                def to_plan(self) -> None: ...
            """,
        )
        # The drop-in's module binds helper at run time, and calls globals() only in the body
        # of a function, so its source tells its names: it does not import Column.
        tree.write(
            tree.repo,
            "python/pyspark/sql/reader.py",
            "from pyspark.sql.utils import helper\n"
            "from pyspark._pyspark import Reader\n"
            "def f():\n    return globals()\n",
        )
        dropped = {"pyspark.sql.reader": {"Converter", "Reader.to_plan"}}
        stub = body(
            tree.generate(["pyspark.sql.reader"], dropped=dropped)[
                Path("python/pyspark/sql/reader.pyi")
            ]
        )
        # Column, which load() needs, is imported, not re-exported; helper is re-exported. The
        # converter and Reader.to_plan, PySpark's plumbing, are left out.
        self.assertEqual(
            stub,
            "\nfrom pyspark.sql.column import Column\n"
            "from pyspark.sql.utils import helper as helper\n\n"
            "class Reader:\n    def load(self, path: str) -> Column: ...\n",
        )
        with self.assertRaisesRegex(gen.GeneratorError, r"DROPPED names that .* lacks: \['Gone'\]"):
            tree.generate(["pyspark.sql.reader"], dropped={"pyspark.sql.reader": {"Gone"}})

    def test_a_factory_of_identity_decorators_is_checked_in_pyspark(self):
        tree = Tree(self)
        tree.write(
            tree.src,
            "pyspark/__init__.py",
            """
            from typing import Any, Callable, TypeVar, Union

            _F = TypeVar("_F", bound=Callable)

            def since(version: Union[str, float]) -> Callable[[_F], _F]: ...

            def until(version: str) -> Callable[..., Any]: ...
            """,
        )
        # since("2.4.0") returns a decorator that returns what it takes, as PySpark annotates it.
        with mock.patch.object(gen, "IDENTITY_DECORATORS", frozenset({"pyspark.since"})):
            gen.check_identity_decorators(tree.src)
        with (
            mock.patch.object(gen, "IDENTITY_DECORATORS", frozenset({"pyspark.until"})),
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

    def test_a_star_import_of_a_module_that_the_dropin_lacks(self):
        tree = Tree(self)
        tree.write(
            tree.src,
            "pyspark/sql/proto/__init__.py",
            """
            from pyspark.sql.proto.base_pb2_grpc import *
            from pyspark.sql.proto.base_pb2 import *
            """,
        )
        # The drop-in ships the stub of base_pb2, not the gRPC module, and its package installs
        # the messages in its globals().
        tree.write(tree.repo, "python/pyspark/sql/proto/base_pb2.pyi", "class Plan: ...\n")
        tree.write(tree.repo, "python/pyspark/sql/proto/__init__.py", "install(globals())\n")
        with self.assertRaisesRegex(
            gen.GeneratorError, r"does not have: \{'pyspark.sql.proto.base_pb2_grpc.\*'"
        ):
            tree.generate(["pyspark.sql.proto"])
        grpc = {"pyspark.sql.proto": {"pyspark.sql.proto.base_pb2_grpc": "no gRPC"}}
        stub = tree.generate(["pyspark.sql.proto"], dropped_star_imports=grpc)[
            Path("python/pyspark/sql/proto/__init__.pyi")
        ]
        self.assertEqual(body(stub), "\nfrom pyspark.sql.proto.base_pb2 import *\n")
        stale = {"pyspark.sql.proto": {"pyspark.sql.proto.gone": "gone"}}
        with self.assertRaisesRegex(gen.GeneratorError, r"does not star-import: \['pyspark"):
            tree.generate(["pyspark.sql.proto"], dropped_star_imports=stale)


# PySpark's module of the classes of the native classes of the drop-in's module below: summaries
# of two branches, a generic base and a base of the module that is not native.
PYSPARKS_CLASSIFICATION = """
from typing import Generic, TypeVar

T = TypeVar("T")


class _Base(Generic[T]): ...


class Classifier(_Base["Model"]): ...


class Summary: ...


class BinarySummary(Summary): ...


class TrainingSummary(Summary): ...


class BinaryTrainingSummary(BinarySummary, TrainingSummary): ...


class Model(_Base["Model"]):
    def summary(self) -> BinaryTrainingSummary: ...


class Reader: ...
"""

DROPIN_CLASSIFICATION = """
\"\"\"The drop-in's module of native classes.\"\"\"

from pyspark._pyspark import (
    BinarySummary,
    BinaryTrainingSummary,
    Model,
    Summary,
    TrainingSummary,
)
from pyspark.ml.classification import Classifier  # noqa: F401

__all__ = ["Model", "Summary"]

from pyspark.ml._native_classes import publish as _publish  # noqa: E402

_publish(globals(), "pyspark.ml.classification")

import pyspark.ml.classification as _pysparks  # noqa: E402

Reader = _pysparks.Reader
del _pysparks
"""

DROPIN_ML_CONNECT = """
\"\"\"The drop-in's package of the native ML classes.\"\"\"

from pyspark.ml.connect.classification import Model, Summary
from pyspark.ml.connect.feature import ArrayAssembler, MaxAbsScaler
from pyspark._pyspark import MLModel
from pyspark import _pyspark as _native
from pyspark.ml.connect import feature

for _name in _native._ml_feature_names:
    globals()[_name] = getattr(_native, _name)

feature.__all__ = ["Binarizer", "MaxAbsScaler"]

__all__ = ["Model", "Summary", "ArrayAssembler", "MaxAbsScaler", "MLModel", "feature"]

__all__ += [name for name in _native._ml_feature_names if name not in __all__]


def _publish_native_classes() -> None:
    pass


_publish_native_classes()
"""


class MLConnectTests(unittest.TestCase):
    def tree(self) -> Tree:
        tree = Tree(self)
        tree.write(tree.src, "pyspark/ml/classification.py", PYSPARKS_CLASSIFICATION)
        tree.write(
            tree.src,
            "pyspark/ml/feature.py",
            "class Binarizer: ...\n\n\nclass MaxAbsScaler(Binarizer): ...\n",
        )
        tree.write(
            tree.src,
            "pyspark/ml/connect/feature.py",
            """
            from pyspark.sql.dataframe import DataFrame


            class MaxAbsScaler:
                def fit(self, dataset: DataFrame) -> "MaxAbsScaler": ...


            class ArrayAssembler:
                def transform(self, dataset: DataFrame) -> DataFrame: ...
            """,
        )
        # The drop-in's modules: PySpark's of pyspark.ml, as the vendored ones, and its own of
        # pyspark.ml.connect.
        tree.write(tree.repo, "python/pyspark/ml/classification.py", PYSPARKS_CLASSIFICATION)
        tree.write(tree.repo, "python/pyspark/ml/feature.py", "class Binarizer: ...\n")
        tree.write(tree.repo, "python/pyspark/ml/_typing.pyi", "ParamMap = dict\n")
        tree.write(tree.repo, "python/pyspark/ml/connect/classification.py", DROPIN_CLASSIFICATION)
        tree.write(tree.repo, "python/pyspark/ml/connect/__init__.py", DROPIN_ML_CONNECT)
        tree.write(tree.repo, "python/pyspark/ml/connect/feature.py", "")
        return tree

    def generate(self, tree: Tree):
        return tree.generate(
            ["pyspark.ml.connect.feature"], ml_connect=["pyspark.ml.connect.classification"]
        )

    def test_the_native_classes_are_the_public_classes_of_pysparks(self):
        tree = self.tree()
        stub = self.generate(tree)[Path("python/pyspark/ml/connect/classification.pyi")]
        self.assertIn(
            "from python/pyspark/ml/connect/classification.py of the drop-in and the classes of\n"
            "# pyspark.ml.classification of Apache Spark v9.9.9:\n",
            stub,
        )
        # Each native class is of the public classes of its nearest native bases, which come
        # first, and of PySpark's class; the module re-exports what it imports of pyspark.ml and
        # the class that it binds of PySpark's module, and keeps its __all__.
        self.assertEqual(
            body(stub),
            "\nimport pyspark.ml.classification as _pysparks\n"
            "from pyspark.ml.classification import Classifier as Classifier\n"
            "from pyspark.ml.classification import Reader as Reader\n\n"
            "__all__ = ['Model', 'Summary']\n\n"
            "class Summary(_pysparks.Summary): ...\n"
            "class BinarySummary(Summary, _pysparks.BinarySummary): ...\n"
            "class TrainingSummary(Summary, _pysparks.TrainingSummary): ...\n"
            "class BinaryTrainingSummary(BinarySummary, TrainingSummary, "
            "_pysparks.BinaryTrainingSummary): ...\n"
            "class Model(_pysparks.Model): ...\n",
        )

    def test_the_package_reexports_and_the_module_of_the_features_has_the_stages(self):
        outputs = self.generate(self.tree())
        package = body(outputs[Path("python/pyspark/ml/connect/__init__.pyi")])
        # The names that it imports, the native stages that it sets in the module of the
        # features, its own native class, and its __all__ with the stages.
        self.assertIn("from pyspark.ml.connect.classification import Model as Model\n", package)
        self.assertIn("from pyspark.ml.connect.feature import Binarizer as Binarizer\n", package)
        self.assertEqual(package.count("import MaxAbsScaler as MaxAbsScaler\n"), 1)
        self.assertIn("from pyspark.ml.connect import feature as feature\n", package)
        self.assertIn(
            "__all__ = ['Model', 'Summary', 'ArrayAssembler', 'MaxAbsScaler', 'MLModel', "
            "'feature', 'Binarizer']\n",
            package,
        )
        self.assertIn("class MLModel:\n    def transform(\n", package)
        self.assertNotIn("_native", package)
        feature = body(outputs[Path("python/pyspark/ml/connect/feature.pyi")])
        # PySpark's MaxAbsScaler of the module is replaced, its ArrayAssembler kept, and the
        # __all__ is the drop-in's.
        self.assertIn("__all__ = ['Binarizer', 'MaxAbsScaler']\n", feature)
        self.assertIn("class ArrayAssembler:\n", feature)
        self.assertNotIn("def fit(", feature)
        self.assertTrue(
            feature.endswith(
                "import pyspark.ml.feature as _pysparks\n"
                "class Binarizer(_pysparks.Binarizer): ...\n"
                "class MaxAbsScaler(Binarizer, _pysparks.MaxAbsScaler): ...\n"
            ),
            feature,
        )

    def test_what_the_generator_cannot_stub_is_an_error(self):
        tree = self.tree()
        path = "python/pyspark/ml/connect/classification.py"
        # A statement of another kind.
        tree.write(tree.repo, path, DROPIN_CLASSIFICATION + "\nx = 1\n")
        with self.assertRaisesRegex(gen.GeneratorError, r"classification:\d+: unexpected Assign"):
            self.generate(tree)
        # A native class of no class of PySpark's module.
        tree.write(
            tree.repo,
            path,
            DROPIN_CLASSIFICATION.replace("    Model,\n", "    Model,\n    Gone,\n"),
        )
        with self.assertRaisesRegex(
            gen.GeneratorError, r"no class of the native classes \['Gone'\]"
        ):
            self.generate(tree)
        # A native class of the package that ML_CONNECT_NATIVE_CLASSES does not declare.
        tree.write(tree.repo, path, DROPIN_CLASSIFICATION)
        tree.write(
            tree.repo,
            "python/pyspark/ml/connect/__init__.py",
            DROPIN_ML_CONNECT.replace("import MLModel", "import MLModel, Other"),
        )
        with self.assertRaisesRegex(gen.GeneratorError, "the native class Other"):
            self.generate(tree)


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
