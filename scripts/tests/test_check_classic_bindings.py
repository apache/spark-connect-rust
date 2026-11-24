# SPDX-License-Identifier: Apache-2.0
"""The names that check_classic_bindings.py works out of PySpark's classes, and its check.

Each test parses a class or makes a PySpark source tree of a few modules; none needs a reference
PySpark, which the generated-drift workflow checks the real bindings against.
"""

import ast
import contextlib
import importlib.util
import io
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "check_classic_bindings.py"
SPEC = importlib.util.spec_from_file_location("check_classic_bindings", SCRIPT)
check = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check
SPEC.loader.exec_module(check)


def _class(source):
    return ast.parse(textwrap.dedent(source)).body[0]


TVF_ARGUMENT = """
class TableValuedFunctionArgument:
    pass
"""

CLASSIC_COLUMN = """
from pyspark.sql.tvf_argument import TableValuedFunctionArgument


class Column(TableValuedFunctionArgument):
    def alias(self, *alias):
        ...
"""

CONNECT_COLUMN = """
from typing import TYPE_CHECKING

from pyspark.sql.column import Column as ParentColumn

if TYPE_CHECKING:
    from pyspark.sql.connect.expressions import Expression


class Column(ParentColumn):
    def __init__(self, expr: "Expression") -> None:
        self._expr = expr

    def __eq__(self, other):
        ...

    def alias(self, *alias):
        ...

    name = alias
"""

BINDINGS = """
_BINDINGS: dict[str, _Binding] = {
    "pyspark.sql.column.Column": _Binding(
        "Column",
        frozenset({"__doc__", "__eq__", "__hash__", "__init__", "__module__", "alias", "name"}),
        frozenset({"_expr"}),
    ),
}

_PUBLIC_BASES = frozenset({"pyspark.sql.tvf_argument.TableValuedFunctionArgument"})
"""


class ClassNamesTest(unittest.TestCase):
    def test_the_names_of_the_dict_of_a_class(self):
        # Of every statement of the body that binds a name, of its blocks too, in order.
        cls = _class(
            """
            class C(B):
                '''The class.'''
                import os.path
                from x import y as z

                def f(self):
                    inner = 1

                async def g(self):
                    ...

                class Nested:
                    hidden = 1

                a = b = 1
                c, (d, *e) = 1, (2, 3)
                f.attribute = 1
                counter += 1
                annotated: int = 1
                only_annotated: int
                __qualname__ = "Other"
                for loop in ():
                    in_for = 1
                with open() as (w1, w2):
                    in_with = 1
                if True:
                    in_if = 1
                else:
                    in_else = 1
                while False:
                    in_while = 1
                try:
                    in_try = 1
                except Exception as error:
                    in_except = 1
                finally:
                    in_finally = 1
                gone = 1
                del gone
                pass
            """
        )
        self.assertEqual(
            check.class_names(cls, "C"),
            {
                "__doc__", "__module__", "os", "z", "f", "g", "Nested", "a", "b", "c", "d", "e",
                "counter", "annotated", "loop", "in_for", "w1", "w2", "in_with", "in_if",
                "in_else", "in_while", "in_try", "in_except", "in_finally",
            },
        )  # fmt: skip

    def test_a_class_of_eq_and_no_hash_has_the_hash_of_none(self):
        self.assertIn("__hash__", check.class_names(_class("class C:\n    __eq__ = f\n"), "C"))
        names = check.class_names(_class("class C:\n    def __eq__(s, o): ...\n"), "C")
        self.assertEqual(names, {"__doc__", "__module__", "__eq__", "__hash__"})
        self.assertNotIn("__hash__", check.class_names(_class("class C:\n    pass\n"), "C"))

    def test_a_statement_that_the_check_does_not_read_fails_it(self):
        for body in ("global x", "from x import *", "match x:\n        case 1:\n            y = 1"):
            with self.subTest(body=body), self.assertRaises(check.CheckError):
                check.class_names(_class(f"class C:\n    {body}\n"), "C")


class InitNamesTest(unittest.TestCase):
    def test_the_attributes_that_init_sets_on_its_object(self):
        cls = _class(
            """
            class C:
                def __init__(me, a, b):
                    me.plain = a
                    me.first, (me.second, *me.rest) = a
                    me.annotated: int = 1
                    me.counted += 1
                    me.other.attribute = 1
                    other.attribute = 1
                    setattr(me, "set", b)
                    setattr(other, "not_set", b)
                    if a:
                        me.in_if = 1

                    def nested(me):
                        me.of_nested = 1

                    callback = lambda: setattr(me, "of_lambda", 1)

                def method(self):
                    self.of_method = 1
            """
        )
        self.assertEqual(
            check.init_names(cls, "C"),
            {"plain", "first", "second", "rest", "annotated", "counted", "set", "in_if"},
        )

    def test_an_init_that_the_check_does_not_follow_fails_it(self):
        for source in (
            "class C(B):\n    pass\n",
            "class C(B):\n    def __init__(self):\n        super().__init__()\n",
            "class C(B):\n    def __init__(self):\n        B.__init__(self)\n",
            "class C:\n    def __init__(self, name):\n        setattr(self, name, 1)\n",
            "class C:\n    def __init__():\n        pass\n",
        ):
            with self.subTest(source=source), self.assertRaises(check.CheckError):
                check.init_names(_class(source), "C")


class ReadBindingsTest(unittest.TestCase):
    def test_the_bindings_of_the_repository(self):
        bindings, public = check.read_bindings((check.REPO_ROOT / check.BASES).read_text())
        self.assertEqual(
            {key: binding.native_name for key, binding in bindings.items()},
            {
                "pyspark.sql.column.Column": "Column",
                "pyspark.sql.dataframe.DataFrameNaFunctions": "DataFrameNaFunctions",
                "pyspark.sql.dataframe.DataFrameStatFunctions": "DataFrameStatFunctions",
                "pyspark.sql.window.WindowSpec": "WindowSpec",
                "pyspark.sql.table_arg.TableArg": "TableArg",
            },
        )
        self.assertEqual(bindings["pyspark.sql.column.Column"].instance_names, {"_expr"})
        self.assertIn("alias", bindings["pyspark.sql.column.Column"].connect_names)
        self.assertEqual(public, {"pyspark.sql.tvf_argument.TableValuedFunctionArgument"})

    def test_bindings_that_the_check_does_not_read_fail_it(self):
        for source in (
            "_PUBLIC_BASES = frozenset()\n",
            "_BINDINGS = {}\n",
            "_BINDINGS = dict()\n_PUBLIC_BASES = frozenset()\n",
            '_BINDINGS = {"k": ("C", frozenset(), frozenset())}\n_PUBLIC_BASES = frozenset()\n',
            '_BINDINGS = {"k": _Binding("C", set(), frozenset())}\n_PUBLIC_BASES = frozenset()\n',
            '_BINDINGS = {"k": _Binding("C", frozenset({1}), frozenset())}\n'
            "_PUBLIC_BASES = frozenset()\n",
        ):
            with self.subTest(source=source), self.assertRaises(check.CheckError):
                check.read_bindings(source)


class CheckTest(unittest.TestCase):
    def directory(self):
        made = tempfile.TemporaryDirectory()
        self.addCleanup(made.cleanup)
        return Path(made.name)

    def tree(self, modules):
        """A PySpark source tree of ``modules``, by their paths under pyspark/sql."""
        root = self.directory()
        for name, source in modules.items():
            path = root / "pyspark" / "sql" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(textwrap.dedent(source))
        return root

    def run_main(self, root, bindings):
        bases = self.directory() / "_classic_bases.py"
        bases.write_text(bindings)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = check.main(["--pyspark-src", str(root), "--bases", str(bases)])
        return status, out.getvalue()

    def modules(self, connect_column=CONNECT_COLUMN):
        return {
            "tvf_argument.py": TVF_ARGUMENT,
            "column.py": CLASSIC_COLUMN,
            "connect/column.py": connect_column,
        }

    def test_bindings_of_the_names_of_pysparks_classes_pass(self):
        status, out = self.run_main(self.tree(self.modules()), BINDINGS)
        self.assertEqual(status, 0, out)
        self.assertTrue(out.startswith("OK    "), out)

    def test_bindings_that_differ_fail_with_the_difference(self):
        bindings = BINDINGS.replace('"alias", "name"', '"alias", "stale"').replace(
            'frozenset({"_expr"})', "frozenset()"
        )
        bindings = bindings.replace("TableValuedFunctionArgument", "Other")
        status, out = self.run_main(self.tree(self.modules()), bindings)
        self.assertEqual(status, 1, out)
        self.assertTrue(out.startswith("DRIFT "), out)
        lines = out.splitlines()
        for line in (
            '_BINDINGS["pyspark.sql.column.Column"].connect_names differs from the names of '
            "pyspark.sql.connect.column.Column:",
            "  + name",
            "  - stale",
            "  + _expr",
            "  + pyspark.sql.tvf_argument.TableValuedFunctionArgument",
            "  - pyspark.sql.tvf_argument.Other",
        ):
            self.assertIn(line, lines)
        self.assertIn(
            "  PySpark's: frozenset({'__doc__', '__eq__', '__hash__', '__init__', '__module__', "
            "'alias', 'name'})",
            lines,
        )

    def test_a_connect_class_of_relative_imports_is_read(self):
        relative = CONNECT_COLUMN.replace(
            "from pyspark.sql.column import Column as ParentColumn",
            "from ..column import Column as ParentColumn",
        )
        status, out = self.run_main(self.tree(self.modules(relative)), BINDINGS)
        self.assertEqual(status, 0, out)

    def test_a_connect_class_of_other_bases_fails(self):
        for replaced in (
            "class Column(ParentColumn, Mixin):",
            "class Column(ParentColumn, metaclass=Meta):",
            "class Column(Expression):",
        ):
            connect = CONNECT_COLUMN.replace("class Column(ParentColumn):", replaced)
            connect = connect.replace("import TYPE_CHECKING", "import TYPE_CHECKING, Mixin, Meta")
            with self.subTest(replaced=replaced):
                status, out = self.run_main(self.tree(self.modules(connect)), BINDINGS)
                self.assertEqual(status, 1, out)
                self.assertTrue(out.startswith("ERROR "), out)

    def test_a_missing_class_or_module_fails(self):
        modules = self.modules()
        del modules["connect/column.py"]
        status, out = self.run_main(self.tree(modules), BINDINGS)
        self.assertEqual((status, out.startswith("ERROR ")), (1, True), out)
        modules = self.modules(CONNECT_COLUMN.replace("class Column(", "class Other("))
        status, out = self.run_main(self.tree(modules), BINDINGS)
        self.assertEqual((status, out.startswith("ERROR ")), (1, True), out)


if __name__ == "__main__":
    unittest.main()
