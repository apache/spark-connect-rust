#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check of the names of PySpark's Connect classes that python/pyspark/sql/_classic_bases.py keeps.

PySpark's Connect classes of a Column, of the ``na`` and ``stat`` of a DataFrame, of a window
specification and of a table argument subclass PySpark's classic classes, so a name patched on a
classic class reaches their objects through the MRO, unless the Connect class has the name itself,
or its ``__init__`` sets it on each object and the patch is not a data descriptor. The drop-in's
Connect classes are native, and the metaclass of its classic classes forwards such a patch to them
in the same cases, of the names that ``_BINDINGS`` of python/pyspark/sql/_classic_bases.py keeps
of PySpark's Connect classes. This script works those names out of PySpark's source, which it
parses and does not import, and fails with their difference when ``_BINDINGS`` differs.

Of each binding, keyed by a classic class ``pyspark.sql.<module>.<Class>``, PySpark's Connect
class is the ``<Class>`` of pyspark/sql/connect/<module>.py, and:

- its ``connect_names`` are the names of the class's own ``__dict__``: those that its body binds,
  the ``__module__`` and ``__doc__`` of every class, and the ``__hash__`` (None) that ``type`` gives
  a class whose body defines ``__eq__`` and no ``__hash__``;
- its ``instance_names`` are the attributes that the class's ``__init__`` sets on ``self``.

The forwarding follows the MRO of PySpark's Connect objects, so the script checks too that the
bases of each Connect class are its classic class alone, and that the bases of the classic classes
(of pyspark/sql/<module>.py), to the root, are the public bases of ``_PUBLIC_BASES``.

Usage, from anywhere:

    python scripts/check_classic_bindings.py --pyspark-src <spark>/python

``<spark>/python`` is the python/ directory of an Apache Spark checkout at the tracked tag
(SPARK_TAG of scripts/check_vendored_upstream.sh). CI runs it
(.github/workflows/generated-drift.yml).
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]
BASES = Path("python/pyspark/sql/_classic_bases.py")

# The statements of a class body that bind no name of it.
_UNBINDING = (ast.Expr, ast.Pass, ast.Return, ast.Raise, ast.Assert, ast.Break, ast.Continue)
# The try statements, of ``except*`` too (Python 3.11).
_TRY = (ast.Try,) + ((ast.TryStar,) if hasattr(ast, "TryStar") else ())
# The nodes of scopes of their own, whose assignments are not those of the function around them.
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


class CheckError(Exception):
    """A class of PySpark's source, or a binding, that the check cannot work out."""


class Binding(NamedTuple):
    """A binding of ``_BINDINGS``, as _classic_bases.py's ``_Binding``."""

    native_name: str
    connect_names: frozenset[str]
    instance_names: frozenset[str]


def _frozenset(node: ast.expr) -> frozenset[str]:
    """The value of ``frozenset({...})``, or ``frozenset()``, of literal strings."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "frozenset"
        and not node.keywords
        and len(node.args) <= 1
    ):
        values = frozenset(ast.literal_eval(node.args[0])) if node.args else frozenset()
        if all(isinstance(value, str) for value in values):
            return values
    raise CheckError(f"not a frozenset of strings: {ast.unparse(node)}")


def read_bindings(source: str) -> tuple[dict[str, Binding], frozenset[str]]:
    """The ``_BINDINGS`` and the ``_PUBLIC_BASES`` of ``source``, that of _classic_bases.py."""
    values: dict[str, ast.expr] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target, value = node.targets[0].id, node.value
        else:
            continue
        if target in ("_BINDINGS", "_PUBLIC_BASES") and value is not None:
            values[target] = value
    missing = sorted({"_BINDINGS", "_PUBLIC_BASES"} - set(values))
    if missing:
        raise CheckError(f"{BASES} assigns no {' and no '.join(missing)}")
    table = values["_BINDINGS"]
    if not isinstance(table, ast.Dict):
        raise CheckError(f"_BINDINGS of {BASES} is not a dict display")
    bindings: dict[str, Binding] = {}
    for key, value in zip(table.keys, table.values):
        if not (
            key is not None
            and isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "_Binding"
            and len(value.args) == 3
            and not value.keywords
        ):
            raise CheckError(f"not a _Binding(name, names, names) of a key: {ast.unparse(value)}")
        bindings[ast.literal_eval(key)] = Binding(
            ast.literal_eval(value.args[0]), _frozenset(value.args[1]), _frozenset(value.args[2])
        )
    return bindings, _frozenset(values["_PUBLIC_BASES"])


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _runtime_statements(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """The statements of a module that run on its import: of its blocks too, but not those of
    ``if TYPE_CHECKING:``."""
    for node in body:
        if isinstance(node, ast.If):
            if not _is_type_checking(node.test):
                yield from _runtime_statements(node.body)
            yield from _runtime_statements(node.orelse)
        elif isinstance(node, _TRY):
            for block in (node.body, *(h.body for h in node.handlers), node.orelse, node.finalbody):
                yield from _runtime_statements(block)
        else:
            yield node


class Module:
    """A module of PySpark's source, parsed: its classes, and the names that its imports bind."""

    def __init__(self, pyspark_src: Path, name: str) -> None:
        self.name = name
        path = pyspark_src.joinpath(*name.split("."))
        # The package of the module, which its relative imports are of.
        if (path / "__init__.py").is_file():
            path, package = path / "__init__.py", name
        else:
            path, package = path.with_suffix(".py"), name.rpartition(".")[0]
        if not path.is_file():
            raise CheckError(f"PySpark's source has no module {name} ({path})")
        self.classes: dict[str, ast.ClassDef] = {}
        self.imported: dict[str, str] = {}
        for node in _runtime_statements(ast.parse(path.read_text(), str(path)).body):
            if isinstance(node, ast.ClassDef):
                self.classes[node.name] = node
            elif isinstance(node, ast.ImportFrom):
                origin = node.module or ""
                if node.level:
                    parent = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
                    origin = f"{parent}.{origin}" if origin else parent
                for alias in node.names:
                    self.imported[alias.asname or alias.name] = f"{origin}.{alias.name}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        self.imported[alias.asname] = alias.name
                    else:
                        top = alias.name.split(".")[0]
                        self.imported[top] = top

    def cls(self, name: str) -> ast.ClassDef:
        if name not in self.classes:
            raise CheckError(f"PySpark's {self.name} defines no class {name}")
        return self.classes[name]

    def qualified(self, node: ast.expr) -> str:
        """The qualified name of the class of the expression of a base: of a class of the module,
        of a name that it imports, or of an attribute of a module that it imports."""
        if isinstance(node, ast.Name):
            if node.id in self.classes:
                return f"{self.name}.{node.id}"
            if node.id in self.imported:
                return self.imported[node.id]
            if node.id == "object":
                return "builtins.object"
        elif isinstance(node, ast.Attribute):
            return f"{self.qualified(node.value)}.{node.attr}"
        raise CheckError(f"a base of a class of PySpark's {self.name}: {ast.unparse(node)}")


def _bind_target(target: ast.expr, names: set[str]) -> None:
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            _bind_target(element, names)
    elif isinstance(target, ast.Starred):
        _bind_target(target.value, names)


def _bind_body(body: list[ast.stmt], names: set[str], where: str) -> None:
    """Adds the names that the statements of a class body bind, in order (a ``del`` takes one
    away), of the blocks among them too, of all their branches."""
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                _bind_target(target, names)
        elif isinstance(node, ast.AugAssign):
            _bind_target(node.target, names)
        elif isinstance(node, ast.AnnAssign):
            # An annotation of no value binds no name.
            if node.value is not None:
                _bind_target(node.target, names)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    raise CheckError(f"{where} imports *")
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.discard(target.id)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            _bind_target(node.target, names)
            _bind_body(node.body, names, where)
            _bind_body(node.orelse, names, where)
        elif isinstance(node, (ast.While, ast.If)):
            _bind_body(node.body, names, where)
            _bind_body(node.orelse, names, where)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    _bind_target(item.optional_vars, names)
            _bind_body(node.body, names, where)
        elif isinstance(node, _TRY):
            # The name of a handler's exception is deleted at the end of the handler.
            for block in (node.body, *(h.body for h in node.handlers), node.orelse, node.finalbody):
                _bind_body(block, names, where)
        elif not isinstance(node, _UNBINDING):
            raise CheckError(f"{where} has a statement that the check does not read: {node!r}")


def class_names(cls: ast.ClassDef, where: str) -> frozenset[str]:
    """The names of the own ``__dict__`` of the class of ``cls``: those that its body binds, the
    ``__module__`` and ``__doc__`` of every class, and the ``__hash__`` that ``type`` sets to None
    of a class whose body defines ``__eq__`` and no ``__hash__``. ``type`` takes ``__qualname__``
    and ``__classcell__`` out of the namespace. The names that the interpreter adds of its own,
    which differ by its version (the ``__annotations__`` of a body of annotations, the
    ``__static_attributes__`` of Python 3.13), are left out: they describe the class, and the
    metaclass forwards none of them."""
    names: set[str] = set()
    _bind_body(cls.body, names, where)
    if "__eq__" in names and "__hash__" not in names:
        names.add("__hash__")
    names |= {"__module__", "__doc__"}
    names -= {"__qualname__", "__classcell__"}
    return frozenset(names)


def _walk(nodes: list[ast.stmt]) -> Iterator[ast.AST]:
    """The nodes of ``nodes``, but not those of the functions, lambdas and classes in them."""
    pending: list[ast.AST] = [node for node in nodes if not isinstance(node, _SCOPES)]
    while pending:
        node = pending.pop()
        yield node
        pending.extend(c for c in ast.iter_child_nodes(node) if not isinstance(c, _SCOPES))


def _set_attributes(target: ast.expr, me: str, names: set[str]) -> None:
    if isinstance(target, ast.Attribute):
        if isinstance(target.value, ast.Name) and target.value.id == me:
            names.add(target.attr)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            _set_attributes(element, me, names)
    elif isinstance(target, ast.Starred):
        _set_attributes(target.value, me, names)


def init_names(cls: ast.ClassDef, where: str) -> frozenset[str]:
    """The attributes that the ``__init__`` of the class of ``cls`` sets on its object: of an
    assignment to ``self.<name>``, or of ``setattr(self, "<name>", ...)``. An ``__init__`` that
    calls that of a base, which the check does not follow, or a class of none, fails it."""
    inits = [
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__"
    ]
    if not inits:
        raise CheckError(f"{where} has no __init__ of its own, which the check does not follow")
    init = inits[-1]
    parameters = init.args.posonlyargs + init.args.args
    if not parameters:
        raise CheckError(f"the __init__ of {where} takes no self")
    me = parameters[0].arg
    names: set[str] = set()
    for node in _walk(init.body):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                _set_attributes(target, me, names)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            _set_attributes(node.target, me, names)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "setattr" and len(node.args) >= 2:
                obj, name = node.args[0], node.args[1]
                if isinstance(obj, ast.Name) and obj.id == me:
                    if not (isinstance(name, ast.Constant) and isinstance(name.value, str)):
                        raise CheckError(f"the __init__ of {where} sets an attribute of no name")
                    names.add(name.value)
            elif isinstance(func, ast.Attribute) and func.attr == "__init__":
                raise CheckError(
                    f"the __init__ of {where} calls an __init__ of a base, which the check does "
                    "not follow"
                )
    return frozenset(names)


class Checker:
    """The check of the bindings of _classic_bases.py against PySpark's source."""

    def __init__(self, pyspark_src: Path) -> None:
        self.pyspark_src = pyspark_src
        self.modules: dict[str, Module] = {}

    def module(self, name: str) -> Module:
        if name not in self.modules:
            self.modules[name] = Module(self.pyspark_src, name)
        return self.modules[name]

    def connect_class(self, key: str) -> tuple[str, ast.ClassDef]:
        """The qualified name and the definition of PySpark's Connect class of the classic class
        ``key``, whose bases are ``key`` alone."""
        module, _, name = key.rpartition(".")
        if not module.startswith("pyspark.sql."):
            raise CheckError(f"a key of _BINDINGS is not of a module of pyspark.sql: {key}")
        connect = self.module("pyspark.sql.connect." + module[len("pyspark.sql.") :])
        cls = connect.cls(name)
        bases = [connect.qualified(base) for base in cls.bases]
        if bases != [key] or cls.keywords:
            raise CheckError(
                f"PySpark's {connect.name}.{name} is not a subclass of {key} alone, of no "
                f"metaclass: of the bases {bases}"
            )
        return f"{connect.name}.{name}", cls

    def classic_bases(self, key: str) -> list[str]:
        """The bases of the classic class ``key``, to the root, but ``object``."""
        found: list[str] = []
        pending = [key]
        while pending:
            module, _, name = pending.pop(0).rpartition(".")
            classic = self.module(module)
            for base in classic.cls(name).bases:
                qualified = classic.qualified(base)
                if qualified != "builtins.object" and qualified not in found:
                    found.append(qualified)
                    pending.append(qualified)
        return found

    def problems(self, source: str) -> list[str]:
        """The differences of the bindings of ``source``, that of _classic_bases.py, from those
        of PySpark's source: none when they agree."""
        bindings, public = read_bindings(source)
        problems: list[str] = []
        bases: list[str] = []
        for key, binding in bindings.items():
            where, cls = self.connect_class(key)
            for field, kept, expected in (
                ("connect_names", binding.connect_names, class_names(cls, where)),
                ("instance_names", binding.instance_names, init_names(cls, where)),
            ):
                if kept != expected:
                    title = f'_BINDINGS["{key}"].{field} differs from the names of {where}:'
                    problems.append(_difference(title, expected, kept))
            bases.extend(base for base in self.classic_bases(key) if base not in bases)
        if set(bases) != public:
            title = "_PUBLIC_BASES differs from the bases of PySpark's classic classes:"
            problems.append(_difference(title, frozenset(bases), public))
        return problems


def _difference(title: str, expected: frozenset[str], kept: frozenset[str]) -> str:
    """The lines of a difference: the names of PySpark only (+), those kept only (-), and the
    frozenset of PySpark's names, to paste."""
    literal = ", ".join(repr(name) for name in sorted(expected))
    return "\n".join(
        [title]
        + [f"  + {name}" for name in sorted(expected - kept)]
        + [f"  - {name}" for name in sorted(kept - expected)]
        + [f"  PySpark's: frozenset({{{literal}}})"]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--pyspark-src",
        required=True,
        help="Path to the python/ directory of an Apache Spark checkout at the tracked tag, "
        "whose classes the bindings are of. It is parsed, not imported.",
    )
    parser.add_argument(
        "--bases",
        default=str(REPO_ROOT / BASES),
        help=f"The module of the bindings to check (default: {BASES} of this repository).",
    )
    args = parser.parse_args(argv)
    bases = Path(args.bases)
    label = BASES if bases.resolve() == (REPO_ROOT / BASES).resolve() else bases
    try:
        problems = Checker(Path(args.pyspark_src)).problems(bases.read_text())
    except CheckError as error:
        print(f"ERROR {label}: {error}")
        return 1
    if problems:
        print(
            f"DRIFT {label} differs from PySpark's classes (+ of PySpark only, - of {label} only):"
        )
        for problem in problems:
            print(problem)
        return 1
    print(f"OK    {label}: _BINDINGS and _PUBLIC_BASES match PySpark's classes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
