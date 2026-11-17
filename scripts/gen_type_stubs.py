#!/usr/bin/env python3
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Generator of the type stubs of the drop-in's Rust-backed and dynamic modules.

Type checkers such as mypy and pyright read the source of a module, they do not run it. The
modules of the drop-in whose API is native (``from pyspark._pyspark import DataFrameReader``) or
injected at import time (``pyspark.sql.functions`` copies the functions of
``functions_generated.py`` into ``globals()``) show them none of PySpark's API: ``F.desc`` does
not exist, and ``spark.read.json(...)`` is ``Any``. Each of these modules gets a ``.pyi`` stub
next to it, which a type checker reads instead of the module, and which Python never imports.

A stub is PySpark's module of the tracked tag without its bodies, an AST transform of its
source: its classes, functions, overloads, properties, annotations, type aliases, constants,
``__all__`` and the imports that they need. So a type checker sees the API that it sees of the
official ``pyspark-client``, where these modules are annotated Python. In detail:

  * A body is ``...``; docstrings and statements without declarations are dropped. Default
    values stay as PySpark writes them, which editors show.
  * A method that assigns an instance attribute that its class does not declare
    (``self.fields = fields`` in ``StructType.__init__``) keeps its body, without its docstring:
    mypy and pyright infer the attribute from it as they do from PySpark's source, which declares
    it nowhere else. pyright does not narrow in a stub, so ``ATTRIBUTE_TYPES`` declares the few
    attributes that PySpark narrows before it assigns them, with the type that both infer from
    PySpark's source. A public instance attribute that the stub would miss is an error.
  * ``if TYPE_CHECKING:`` keeps its body, and ``if __name__ == "__main__":`` (the doctests)
    nothing. A version check (``sys.version_info``) stays, for the type checker to decide. Under
    any other condition, such as ``is_remote_only()``, both branches are kept, as a type checker
    of PySpark's source sees both.
  * The decorators that type checkers apply are kept (``overload``, ``property``,
    ``classmethod``, ``functools.cache``, ``dataclass``, ...), and those of the module itself
    (``dfapi`` of ``pyspark.sql.group``). PySpark's call-site and dispatch decorators are typed
    ``(f: FuncT) -> FuncT``, which ``IDENTITY_DECORATORS`` lists and which the generator checks
    in PySpark's source: they change no type and are dropped. Any other decorator is an error.
  * The implementation of an overloaded function is dropped, as stubs declare only overloads.
  * A module-level function, class or constant whose name is private is kept only when the
    stub refers to it; a member of a class is always kept.
  * An import is kept when the stub refers to its name. A name that PySpark's module imports
    from ``pyspark`` at run time, under its own name, is an attribute of PySpark's module, which
    a type checker of PySpark's source lets a user import from it: the stub re-exports it
    (``from m import X as X``) when the drop-in's module ``m`` has it, as its source or its stub
    declares it. The imports under ``TYPE_CHECKING`` and those of other packages are not
    re-exported.

``EXTRAS`` adds the names that the drop-in's module has and PySpark's does not, such as the
native ``Trigger`` of ``pyspark.sql.streaming.readwriter``. python/tests/test_type_stubs_offline.py
checks that every public name of each module of the drop-in is in its stub, that mypy types every
public instance attribute of the stubbed classes, and runs mypy on a user's application.

``pyspark.sql.functions`` is a package in PySpark (``functions/__init__.py``, ``builtin.py``,
``partitioning.py``) and a module here (``functions.py``, which registers ``builtin`` and
``partitioning`` in ``sys.modules``). Its stubs are the directory ``pyspark/sql/functions/`` with
the three ``.pyi`` files and no ``__init__.py``: mypy and pyright take a package with an
``__init__.pyi`` over a module of the same name, and give its ``builtin.pyi`` to
``import pyspark.sql.functions.builtin``. Python imports ``functions.py``, as a module of a
directory goes before a namespace package of the same name (PEP 420), and no loader reads
``.pyi`` files.

Usage, from anywhere:

    python scripts/gen_type_stubs.py --pyspark-src <spark>/python            # write
    python scripts/gen_type_stubs.py --pyspark-src <spark>/python --check    # verify

``<spark>/python`` is the python/ directory of an Apache Spark checkout at the tracked tag
(SPARK_TAG of scripts/check_vendored_upstream.sh). The generator only parses it, so it needs no
dependency of PySpark. CI runs it with Python 3.11 (.github/workflows/generated-drift.yml); the
stubs that ``ast.unparse`` prints are the same with Python 3.9 to 3.13. The generator also writes
the block of pyproject.toml that keeps the stubs out of ruff.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import copy
import difflib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = Path("python")
VENDORED_CHECK = Path("scripts/check_vendored_upstream.sh")
# The stubs keep PySpark's style, which ruff would rewrite: pyproject.toml excludes them from
# ruff in a block between these lines, which the generator writes.
PYPROJECT = Path("pyproject.toml")
RUFF_BLOCK_BEGIN = "    # --- generated type stubs: BEGIN ---\n"
RUFF_BLOCK_END = "    # --- generated type stubs: END ---\n"

# The modules that get a stub, by PySpark's import path: each Rust-backed or dynamic module of
# the drop-in whose objects a user imports. The stub is at the path of PySpark's module, with
# .pyi for .py (pyspark/sql/functions/__init__.pyi for the package pyspark.sql.functions).
STUB_MODULES = (
    # The modules of the native classes, which re-export them from pyspark._pyspark.
    "pyspark.storagelevel",
    "pyspark.sql.catalog",
    "pyspark.sql.column",
    "pyspark.sql.conf",
    "pyspark.sql.dataframe",
    "pyspark.sql.group",
    "pyspark.sql.merge",
    "pyspark.sql.observation",
    "pyspark.sql.pandas.group_ops",
    "pyspark.sql.readwriter",
    "pyspark.sql.table_arg",
    "pyspark.sql.tvf",
    "pyspark.sql.types",
    "pyspark.sql.window",
    "pyspark.sql.streaming.query",
    "pyspark.sql.streaming.readwriter",
    # The functions: those that functions.py copies into its globals() from
    # functions_generated.py, and those of Avro and Protobuf, unannotated calls of the Rust core.
    "pyspark.sql.functions",
    "pyspark.sql.functions.builtin",
    "pyspark.sql.functions.partitioning",
    "pyspark.sql.avro.functions",
    "pyspark.sql.protobuf.functions",
    # PySpark's modules with their Rust-backed parts swapped in: the native PythonEvalType and
    # SkipRestOfInputTableException, and user-defined functions that annotate fewer parameters.
    "pyspark.util",
    "pyspark.sql.udf",
    "pyspark.sql.udtf",
    # The modules of PySpark Connect, which re-export the drop-in's classes and functions.
    "pyspark.sql.connect.catalog",
    "pyspark.sql.connect.column",
    "pyspark.sql.connect.conf",
    "pyspark.sql.connect.dataframe",
    "pyspark.sql.connect.functions",
    "pyspark.sql.connect.functions.builtin",
    "pyspark.sql.connect.functions.partitioning",
    "pyspark.sql.connect.group",
    "pyspark.sql.connect.merge",
    "pyspark.sql.connect.observation",
    "pyspark.sql.connect.readwriter",
    "pyspark.sql.connect.session",
    "pyspark.sql.connect.table_arg",
    "pyspark.sql.connect.tvf",
    "pyspark.sql.connect.udf",
    "pyspark.sql.connect.udtf",
    "pyspark.sql.connect.window",
    "pyspark.sql.connect.streaming.query",
    "pyspark.sql.connect.streaming.readwriter",
)

# The decorators that a type checker applies: kept as they are. A decorator of the module
# itself (a function that it defines) is kept too.
KEPT_DECORATORS = frozenset(
    {
        "overload",
        "typing.overload",
        "property",
        "staticmethod",
        "classmethod",
        "abstractmethod",
        "abc.abstractmethod",
        "final",
        "typing.final",
        "functools.cache",
        "functools.cached_property",
        "functools.lru_cache",
        "dataclasses.dataclass",
        "contextlib.contextmanager",
    }
)
# ``@<property>.setter`` and its kin.
_PROPERTY_ACCESSOR = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*\.(setter|getter|deleter)$")

# PySpark's decorators that return the function or class that they take, by the qualified
# name of their definition: the stub drops them, which changes no type. Each one is checked in
# PySpark's source (``check_identity_decorators``): its first parameter and its return are
# annotated with the same type variable, or ``Type[T]`` for a class decorator.
IDENTITY_DECORATORS = frozenset(
    {
        "pyspark.errors.utils._with_origin",
        "pyspark.errors.utils.with_origin_to_class",
        "pyspark.sql.utils.dispatch_col_method",
        "pyspark.sql.utils.dispatch_df_method",
        "pyspark.sql.utils.dispatch_table_arg_method",
        "pyspark.sql.utils.dispatch_window_method",
        "pyspark.sql.utils.try_partitioning_remote_functions",
        "pyspark.sql.utils.try_remote_avro_functions",
        "pyspark.sql.utils.try_remote_functions",
        "pyspark.sql.utils.try_remote_protobuf_functions",
    }
)


@dataclass(frozen=True)
class Extra:
    """Declarations of the names that the drop-in's module has and PySpark's does not: stub
    source appended to the stub as it is, and the names of it that ``__all__`` lists."""

    source: str
    all_names: tuple[str, ...] = ()


EXTRAS = {
    # The drop-in's RuntimeConfig is its native RuntimeConf, which the module also exports.
    "pyspark.sql.conf": Extra("RuntimeConf = RuntimeConfig\n"),
    # The drop-in's pyspark.sql.connect.functions.builtin has every public name of its
    # pyspark.sql.functions.builtin (see pyspark/sql/connect/functions/_copies.py), with these,
    # which PySpark's module of PySpark Connect does not import.
    "pyspark.sql.connect.functions.builtin": Extra(
        "from pyspark.sql.connect.udf import UserDefinedFunction as UserDefinedFunction\n"
        "from pyspark.sql.connect.udtf import UserDefinedTableFunction as "
        "UserDefinedTableFunction\n"
        "from pyspark.sql.pandas.functions import ArrowUDFType as ArrowUDFType\n"
        "from pyspark.sql.pandas.functions import PandasUDFType as PandasUDFType\n"
        "from pyspark.sql.types import ByteType as ByteType\n"
        "from pyspark.sql.types import NumericType as NumericType\n"
        "from pyspark.sql.udtf import OrderingColumn as OrderingColumn\n"
        "from pyspark.sql.udtf import PartitioningColumn as PartitioningColumn\n"
        "from pyspark.sql.udtf import SelectedColumn as SelectedColumn\n"
        "from pyspark.sql.udtf import SkipRestOfInputTableException as "
        "SkipRestOfInputTableException\n"
    ),
    # The drop-in's DataFrame pickles its plan with this parser of protobuf messages.
    "pyspark.sql.connect.dataframe": Extra(
        "from pyspark.sql.connect._column_reduce import parse_deep as parse_deep\n"
    ),
    # The drop-in defines udf, udtf and arrow_udtf of pyspark.sql.functions in these modules,
    # and their evaluation types.
    "pyspark.sql.udf": Extra(
        "from pyspark.serializers import CloudPickleSerializer as CloudPickleSerializer\n"
        "from pyspark.sql.functions.builtin import udf as udf\n"
    ),
    "pyspark.sql.udtf": Extra(
        "from pyspark.serializers import CloudPickleSerializer as CloudPickleSerializer\n"
        "from pyspark.sql.functions.builtin import arrow_udtf as arrow_udtf\n"
        "from pyspark.sql.functions.builtin import udtf as udtf\n"
        "SQL_TABLE_UDF: int\n"
        "SQL_ARROW_TABLE_UDF: int\n"
        "SQL_ARROW_UDTF: int\n"
    ),
    # The drop-in's module dispatches the events of the listeners itself.
    "pyspark.sql.streaming.query": Extra(
        "from pyspark.sql.streaming.listener import QueryIdleEvent as QueryIdleEvent\n"
        "from pyspark.sql.streaming.listener import QueryProgressEvent as QueryProgressEvent\n"
        "from pyspark.sql.streaming.listener import QueryStartedEvent as QueryStartedEvent\n"
        "from pyspark.sql.streaming.listener import QueryTerminatedEvent as QueryTerminatedEvent\n"
    ),
    # The native classes of the frame bounds and of the triggers.
    "pyspark.sql.window": Extra("class FrameBound: ...\n", ("FrameBound",)),
    "pyspark.sql.streaming.readwriter": Extra(
        "class Trigger:\n"
        "    @staticmethod\n"
        "    def processingTime(interval: str) -> Trigger: ...\n"
        "    @staticmethod\n"
        "    def once() -> Trigger: ...\n"
        "    @staticmethod\n"
        "    def availableNow() -> Trigger: ...\n"
        "    @staticmethod\n"
        "    def continuous(interval: str) -> Trigger: ...\n",
        ("Trigger",),
    ),
}

# The instance attributes that PySpark's methods narrow before they assign them, by module and
# qualified name: mypy follows the code flow of the method that the stub keeps, pyright does not
# in a stub (it gives ``StructType.fields`` the Optional of the parameter, and so reports a loop
# over it). Each is declared in its class with the type that both infer from PySpark's source.
# The generator checks that PySpark's class assigns each of them, and declares none of them.
ATTRIBUTE_TYPES = {
    "pyspark.sql.types": {
        "StructType.fields": "List[StructField]",
        "GeometryType.srid": "int",
        "GeographyType.srid": "int",
    },
}

# The line of the header of each stub that marks it as generated.
GENERATED = "# Generated by scripts/gen_type_stubs.py"
HEADER = """\
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Generated by scripts/gen_type_stubs.py from python/{source} of Apache Spark {tag}:
# do not edit. Type checkers read this stub instead of the module of the drop-in.
"""


class GeneratorError(Exception):
    """PySpark's source has a construct that the generator does not know how to stub."""


# ---------------------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------------------


def source_path(pyspark_src: Path, module: str) -> Path:
    """The source of PySpark's module `module` under `pyspark_src` (relative to it)."""
    base = Path(*module.split("."))
    if (pyspark_src / base / "__init__.py").is_file():
        return base / "__init__.py"
    if (pyspark_src / base.with_suffix(".py")).is_file():
        return base.with_suffix(".py")
    raise GeneratorError(f"{module}: no such module in {pyspark_src}")


def tracked_tag(root: Path) -> str:
    match = re.search(r'^SPARK_TAG="([^"]+)"$', (root / VENDORED_CHECK).read_text(), re.M)
    if match is None:
        raise GeneratorError(f"no SPARK_TAG in {VENDORED_CHECK}")
    return match.group(1)


# ---------------------------------------------------------------------------------------
# The names that statements read and bind
# ---------------------------------------------------------------------------------------


def referenced_names(nodes) -> set[str]:
    """The names that `nodes` read: each Name (the first name of a dotted name too), those of
    the string annotations, the forward references in subscripts (``Optional["Row"]``) and the
    ``bound`` of a type variable, and those that ``__all__`` lists. The body that the stub keeps
    of a method is code, whose subscripts are no annotations and whose own names are not read
    from the module."""
    names: set[str] = set()

    def visit(node: ast.AST, annotation: bool, code: bool, into: set[str]) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not _is_ellipsis(
            node.body
        ):
            for signature in (node.decorator_list, node.args, node.returns):
                for child in signature if isinstance(signature, list) else [signature]:
                    if child is not None:
                        visit(child, child is node.returns, code, into)
            body: set[str] = set()
            for stmt in node.body:
                visit(stmt, False, True, body)
            into |= body - _local_names(node)
            return
        if isinstance(node, ast.Name):
            into.add(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and annotation:
            try:
                parsed = ast.parse(node.value.strip(), mode="eval")
            except SyntaxError:
                return
            visit(parsed, True, code, into)
        elif _all_target(node) and isinstance(node.value, (ast.List, ast.Tuple)):
            into.update(e.value for e in node.value.elts if isinstance(e, ast.Constant))
            return
        for name, value in ast.iter_fields(node):
            inner = (
                annotation
                or name in ("annotation", "returns")
                or (not code and isinstance(node, ast.Subscript) and name == "slice")
                or (not code and isinstance(node, ast.keyword) and node.arg == "bound")
            )
            for child in value if isinstance(value, list) else [value]:
                if isinstance(child, ast.AST):
                    visit(child, inner, code, into)

    for node in nodes:
        visit(node, False, False, names)
    return names


def _is_ellipsis(body: list[ast.stmt]) -> bool:
    """Whether `body` is the ``...`` of a stub."""
    return (
        len(body) == 1
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and body[0].value.value is Ellipsis
    )


def _is_docstring(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _local_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """The names that `function` binds: its parameters, and what its body assigns, imports or
    defines, at any depth (in its nested functions too), but what it declares global."""
    args = function.args
    names = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
    names |= {a.arg for a in (args.vararg, args.kwarg) if a is not None}
    outer: set[str] = set()
    for node in ast.walk(ast.Module(body=function.body, type_ignores=[])):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names if a.name != "*"}
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            outer |= set(node.names)
    return names - outer


# ---------------------------------------------------------------------------------------
# Instance attributes
# ---------------------------------------------------------------------------------------


def _methods(body: list[ast.stmt]):
    """The functions of the body of a class, under any condition."""
    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield stmt
        elif isinstance(stmt, ast.If):
            yield from _methods(stmt.body + stmt.orelse)
        elif isinstance(stmt, ast.Try):
            handlers = [s for h in stmt.handlers for s in h.body]
            yield from _methods(stmt.body + handlers + stmt.orelse + stmt.finalbody)


def _classes(body: list[ast.stmt], prefix: str = ""):
    """(qualified name, class) of each class of `body` and of its classes, under any condition."""
    for stmt in body:
        if isinstance(stmt, ast.ClassDef):
            yield prefix + stmt.name, stmt
            yield from _classes(stmt.body, f"{prefix}{stmt.name}.")
        elif isinstance(stmt, ast.If):
            yield from _classes(stmt.body + stmt.orelse, prefix)
        elif isinstance(stmt, ast.Try):
            handlers = [s for h in stmt.handlers for s in h.body]
            yield from _classes(stmt.body + handlers + stmt.orelse + stmt.finalbody, prefix)


def assigned_attributes(method: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """The instance attributes that `method` assigns (``self.x = ...``, ``self.x: T = ...``, in a
    tuple too), in order: none of a static or class method."""
    if any(_decorator_name(d) in ("staticmethod", "classmethod") for d in method.decorator_list):
        return []
    params = method.args.posonlyargs + method.args.args
    if not params:
        return []
    instance = params[0].arg
    names: list[str] = []
    for node in ast.walk(method):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            for element in target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]:
                if (
                    isinstance(element, ast.Attribute)
                    and isinstance(element.value, ast.Name)
                    and element.value.id == instance
                    and element.attr not in names
                ):
                    names.append(element.attr)
    return names


def instance_attributes(cls: ast.ClassDef) -> list[str]:
    """The instance attributes that the methods of `cls` assign and its body does not declare."""
    declared = set().union(*(bound_names(s) for s in cls.body))
    names: list[str] = []
    for method in _methods(cls.body):
        names += [n for n in assigned_attributes(method) if n not in declared and n not in names]
    return names


def check_attributes(module: str, tree: ast.Module, text: str) -> None:
    """Each public instance attribute of each class of PySpark's module `tree` that the stub of
    text `text` keeps is declared in the stub's class: in its body, or by one of its methods."""
    stubs = dict(_classes(ast.parse(text).body))
    missing = []
    for qualified, cls in _classes(tree.body):
        stub = stubs.get(qualified)
        if stub is None:
            continue
        declared = set().union(*(bound_names(s) for s in stub.body))
        declared |= {a for m in _methods(stub.body) for a in assigned_attributes(m)}
        missing += [
            f"{qualified}.{a}"
            for a in instance_attributes(cls)
            if not _is_private(a) and a not in declared
        ]
    if missing:
        raise GeneratorError(
            f"{module}: the stub declares no instance attribute {missing}, which PySpark's "
            "methods assign; declare them in ATTRIBUTE_TYPES"
        )


def _all_target(node: ast.AST) -> bool:
    """Whether `node` assigns or extends ``__all__``."""
    if isinstance(node, ast.Assign):
        return [_dotted(t) for t in node.targets] == ["__all__"]
    if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        return _dotted(node.target) == "__all__"
    return False


def bound_names(stmt: ast.stmt) -> set[str]:
    """The names that a module-level statement of a stub binds."""
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {stmt.name}
    if isinstance(stmt, ast.Assign):
        return {t.id for t in stmt.targets if isinstance(t, ast.Name)}
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return {stmt.target.id}
    if isinstance(stmt, ast.Import):
        return {(a.asname or a.name.split(".")[0]) for a in stmt.names}
    if isinstance(stmt, ast.ImportFrom):
        return {(a.asname or a.name) for a in stmt.names if a.name != "*"}
    if isinstance(stmt, ast.If):
        return set().union(*(bound_names(s) for s in stmt.body + stmt.orelse))
    return set()


def _is_private(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _dotted(node: ast.expr) -> str | None:
    """`a.b.c` for a Name or an Attribute chain of names, else None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


# ---------------------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------------------


@dataclass
class ImportSite:
    """One name that an import statement binds."""

    stmt: ast.Import | ast.ImportFrom
    alias: ast.alias
    type_checking: bool
    # Whether the stub re-exports the name: decided by `resolve_reexports`.
    reexported: bool = False

    @property
    def name(self) -> str:
        if isinstance(self.stmt, ast.Import):
            return self.alias.asname or self.alias.name.split(".")[0]
        return self.alias.asname or self.alias.name

    @property
    def reexportable(self) -> bool:
        """Whether PySpark's module has the name at run time, imported from ``pyspark`` under
        its own name: a type checker of PySpark's source lets a user import it from the
        module, so the stub re-exports it, when the drop-in has it."""
        return (
            isinstance(self.stmt, ast.ImportFrom)
            and not self.type_checking
            and (self.stmt.module or "").split(".")[0] == "pyspark"
            and self.alias.asname in (None, self.alias.name)
            and not self.alias.name.startswith(("_", "*"))
        )


def resolve_import(module: str, import_from: ast.ImportFrom) -> str:
    """The module of `import_from` of the module `module`: PySpark imports absolutely."""
    if import_from.level:
        raise GeneratorError(f"{module}:{import_from.lineno}: a relative import")
    return import_from.module or ""


# ---------------------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------------------


def check_identity_decorators(pyspark_src: Path) -> None:
    """Each decorator of IDENTITY_DECORATORS returns what it takes, as PySpark annotates it:
    its first (or only) signature is ``(f: T, ...) -> T`` or ``(cls: Type[T], ...) -> Type[T]``
    for a type variable ``T`` of its module."""
    for qualified in sorted(IDENTITY_DECORATORS):
        module, _, name = qualified.rpartition(".")
        tree = ast.parse((pyspark_src / source_path(pyspark_src, module)).read_text())
        type_vars = {
            t.id
            for n in tree.body
            if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call)
            and _dotted(n.value.func) in ("TypeVar", "typing.TypeVar")
            for t in n.targets
            if isinstance(t, ast.Name)
        }
        defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        if not defs:
            raise GeneratorError(f"{qualified}: not defined in PySpark")
        params = defs[0].args.posonlyargs + defs[0].args.args
        param = params[0].annotation if params else None
        returns = defs[0].returns
        identity = (
            param is not None
            and returns is not None
            and ast.dump(param) == ast.dump(returns)
            and (
                (isinstance(param, ast.Name) and param.id in type_vars)
                or (
                    isinstance(param, ast.Subscript)
                    and _dotted(param.value) in ("Type", "typing.Type", "type")
                    and isinstance(param.slice, ast.Name)
                    and param.slice.id in type_vars
                )
            )
        )
        if not identity:
            raise GeneratorError(
                f"{qualified}: not annotated as returning what it takes; keep it in the stubs "
                "(KEPT_DECORATORS) instead of IDENTITY_DECORATORS"
            )


def _decorator_name(decorator: ast.expr) -> str | None:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    return _dotted(target)


# ---------------------------------------------------------------------------------------
# The transform
# ---------------------------------------------------------------------------------------

_ELLIPSIS = ast.Expr(ast.Constant(Ellipsis))


def _condition_kind(test: ast.expr) -> str:
    """How a stub treats an ``if``: "type_checking" (its body), "not_type_checking" (its else),
    "main" (neither: the doctests that run as a script), "static" (kept for the type checker to
    decide) or "both" (both branches)."""
    name = _dotted(test)
    if name in ("TYPE_CHECKING", "typing.TYPE_CHECKING"):
        return "type_checking"
    if (
        isinstance(test, ast.Compare)
        and _dotted(test.left) == "__name__"
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    ):
        return "main"
    if (
        isinstance(test, ast.UnaryOp)
        and isinstance(test.op, ast.Not)
        and _dotted(test.operand) in ("TYPE_CHECKING", "typing.TYPE_CHECKING")
    ):
        return "not_type_checking"
    if any(_dotted(n) in ("sys.version_info", "sys.platform") for n in ast.walk(test)):
        return "static"
    return "both"


@dataclass
class Module:
    """PySpark's module being stubbed."""

    name: str
    imports: dict[str, str] = field(default_factory=dict)  # local name -> qualified name
    local_functions: set[str] = field(default_factory=set)

    def qualified(self, dotted: str) -> str:
        head, _, rest = dotted.partition(".")
        base = self.imports.get(head, head)
        return f"{base}.{rest}" if rest else base


class StubTransformer:
    """The stub statements of one module of PySpark."""

    def __init__(self, module: Module):
        self.module = module
        self.import_sites: list[ImportSite] = []
        # The names that the body of the class being transformed declares, and its qualified
        # name: a method of it that assigns another instance attribute keeps its body.
        self.class_declared: set[str] | None = None
        self.class_path: list[str] = []
        self.attribute_types_used: set[str] = set()

    # Decorators

    def keep_decorator(self, decorator: ast.expr, where: str) -> bool:
        name = _decorator_name(decorator)
        if name is None:
            raise GeneratorError(f"{where}: decorator {ast.unparse(decorator)} is not a name")
        if name in KEPT_DECORATORS or _PROPERTY_ACCESSOR.match(name):
            return True
        if name in self.module.local_functions:
            return True
        if self.module.qualified(name) in IDENTITY_DECORATORS:
            return False
        if self.module.qualified(name) in KEPT_DECORATORS:
            return True
        raise GeneratorError(
            f"{where}: unknown decorator {ast.unparse(decorator)} "
            f"({self.module.qualified(name)}); add it to KEPT_DECORATORS or, if it returns what "
            "it takes, to IDENTITY_DECORATORS"
        )

    # Statements

    def body(self, stmts: list[ast.stmt], scope: str, type_checking: bool = False):
        """The stub statements of `stmts`, in a scope "module" or "class"."""
        out: list[ast.stmt] = []
        overloaded: set[str] = set()
        for stmt in stmts:
            for kept in self.statement(stmt, scope, type_checking):
                if isinstance(kept, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    is_overload = any(
                        _decorator_name(d) in ("overload", "typing.overload")
                        for d in kept.decorator_list
                    )
                    if is_overload:
                        overloaded.add(kept.name)
                    elif kept.name in overloaded:
                        # The implementation of an overloaded function: stubs declare only
                        # the overloads.
                        overloaded.discard(kept.name)
                        continue
                out.append(kept)
        return out

    def statement(self, stmt: ast.stmt, scope: str, type_checking: bool):
        where = f"{self.module.name}:{getattr(stmt, 'lineno', '?')}"
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            if isinstance(stmt, ast.ImportFrom):
                if stmt.module == "__future__":
                    return []
                stmt = copy.copy(stmt)
                stmt.module = resolve_import(self.module.name, stmt)
                stmt.level = 0
            if scope != "module":
                raise GeneratorError(f"{where}: an import in a class")
            for alias in stmt.names:
                self.import_sites.append(ImportSite(stmt, alias, type_checking))
            return [stmt]
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return [self.function(stmt, where)]
        if isinstance(stmt, ast.ClassDef):
            return [self.cls(stmt, where)]
        if isinstance(stmt, ast.Assign):
            if all(isinstance(t, ast.Name) for t in stmt.targets):
                return [copy.deepcopy(stmt)]
            return []  # X.__doc__ = ..., x[k] = ...
        if isinstance(stmt, ast.AnnAssign):
            if not isinstance(stmt.target, ast.Name):
                return []
            kept = copy.deepcopy(stmt)
            if stmt.target.id == "__all__":
                return [kept]  # its names, as a type checker reads them
            # A class keeps that its field has a default (NamedTuple and dataclass fields).
            kept.value = ast.Constant(Ellipsis) if stmt.value and scope == "class" else None
            return [kept]
        if isinstance(stmt, ast.AugAssign):
            if isinstance(stmt.target, ast.Name) and stmt.target.id == "__all__":
                return [copy.deepcopy(stmt)]
            return []
        if isinstance(stmt, ast.If):
            kind = _condition_kind(stmt.test)
            if kind == "type_checking":
                return self.body(stmt.body, scope, True)
            if kind == "not_type_checking":
                return self.body(stmt.orelse, scope, type_checking)
            if kind == "main":
                return []
            if kind == "static":
                body = self.body(stmt.body, scope, type_checking)
                orelse = self.body(stmt.orelse, scope, type_checking)
                if not body and not orelse:
                    return []
                return [ast.If(copy.deepcopy(stmt.test), body or [ast.Pass()], orelse)]
            return self.body(stmt.body, scope, type_checking) + self.body(
                stmt.orelse, scope, type_checking
            )
        if isinstance(stmt, ast.Try):
            parts = stmt.body + [s for h in stmt.handlers for s in h.body]
            return self.body(parts + stmt.orelse + stmt.finalbody, scope, type_checking)
        if isinstance(
            stmt, (ast.Expr, ast.Pass, ast.For, ast.While, ast.With, ast.Delete, ast.Raise)
        ):
            return []
        if isinstance(stmt, (ast.Assert, ast.Global, ast.Nonlocal)):
            return []
        raise GeneratorError(f"{where}: unexpected {type(stmt).__name__} statement")

    def function(self, node, where: str):
        where = f"{where} {node.name}"
        decorators = [d for d in node.decorator_list if self.keep_decorator(d, where)]
        stub = copy.deepcopy(node)
        stub.body = [copy.deepcopy(_ELLIPSIS)]
        if self.class_declared is not None and any(
            name not in self.class_declared for name in assigned_attributes(node)
        ):
            # A type checker infers the instance attributes that the method assigns from its
            # body, as from PySpark's source.
            stub.body = [copy.deepcopy(s) for s in node.body if not _is_docstring(s)]
        stub.decorator_list = copy.deepcopy(decorators)
        stub.type_comment = None
        return stub

    def cls(self, node: ast.ClassDef, where: str):
        where = f"{where} {node.name}"
        decorators = [d for d in node.decorator_list if self.keep_decorator(d, where)]
        declared = set().union(*(bound_names(s) for s in node.body))
        outer = self.class_declared
        self.class_declared = declared
        self.class_path.append(node.name)
        try:
            attributes = self.attribute_types(node, declared)
            body = self.body(node.body, "class")
        finally:
            self.class_declared = outer
            self.class_path.pop()
        stub = copy.copy(node)
        stub.bases = copy.deepcopy(node.bases)
        stub.keywords = copy.deepcopy(node.keywords)
        stub.body = attributes + body or [copy.deepcopy(_ELLIPSIS)]
        stub.decorator_list = copy.deepcopy(decorators)
        return stub

    def attribute_types(self, node: ast.ClassDef, declared: set[str]) -> list[ast.stmt]:
        """The declarations of ATTRIBUTE_TYPES of the class `node`."""
        qualified = ".".join(self.class_path)
        assigned = instance_attributes(node)
        out: list[ast.stmt] = []
        for key, annotation in ATTRIBUTE_TYPES.get(self.module.name, {}).items():
            cls, _, name = key.rpartition(".")
            if cls != qualified:
                continue
            where = f"{self.module.name}: ATTRIBUTE_TYPES {key}"
            if name in declared:
                raise GeneratorError(f"{where}: PySpark's class declares it now; drop it")
            if name not in assigned:
                raise GeneratorError(f"{where}: PySpark's class does not assign it; drop it")
            self.attribute_types_used.add(key)
            out.append(
                ast.AnnAssign(
                    target=ast.Name(id=name, ctx=ast.Store()),
                    annotation=ast.parse(annotation, mode="eval").body,
                    value=None,
                    simple=1,
                )
            )
        return out


def _module_imports(module_name: str, tree: ast.Module) -> dict[str, str]:
    """The local name -> qualified name of every import of `tree`, at any level."""
    imports: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    imports[a.asname] = a.name
                else:
                    imports[a.name.split(".")[0]] = a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            base = resolve_import(module_name, node)
            for a in node.names:
                if a.name != "*":
                    imports[a.asname or a.name] = f"{base}.{a.name}"
    return imports


def prune(stmts: list[ast.stmt], sites: list[ImportSite], extra_refs: set[str]):
    """The statements of the stub: the public declarations, the private ones that they refer
    to, and the imports that they need or that the stub re-exports."""

    def needed(stmt: ast.stmt) -> bool:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            return False  # decided below, per name
        names = bound_names(stmt)
        return any(not _is_private(n) for n in names) or not names

    kept = [s for s in stmts if needed(s)]
    candidates = [
        s for s in stmts if not needed(s) and not isinstance(s, (ast.Import, ast.ImportFrom))
    ]
    refs = referenced_names(kept) | extra_refs
    # Private declarations that a kept statement refers to, to a fixed point.
    changed = True
    while changed:
        changed = False
        for stmt in list(candidates):
            if bound_names(stmt) & refs:
                candidates.remove(stmt)
                kept.append(stmt)
                refs |= referenced_names([stmt])
                changed = True

    seen: set[str] = set()
    kept_aliases: dict[int, list[ast.alias]] = {}
    for site in sites:
        if site.alias.name == "*":
            kept_aliases.setdefault(id(site.stmt), []).append(site.alias)
            continue
        if site.name in seen or not (site.reexported or site.name in refs):
            continue
        seen.add(site.name)
        alias = copy.copy(site.alias)
        if site.reexported:
            alias.asname = alias.name
        kept_aliases.setdefault(id(site.stmt), []).append(alias)

    out: list[ast.stmt] = []
    kept_ids = {id(s) for s in kept}
    for stmt in stmts:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            aliases = kept_aliases.get(id(stmt))
            if aliases:
                new = copy.copy(stmt)
                new.names = aliases
                out.append(new)
        elif id(stmt) in kept_ids:
            out.append(stmt)
    return out, refs


def _top_level(stmts: list[ast.stmt]) -> list[ast.stmt]:
    """The statements of a module that run at its top level, under any condition: those of its
    ``if``, ``try``, ``with`` and ``for`` blocks too, not those of its classes and functions."""
    out: list[ast.stmt] = []
    for stmt in stmts:
        out.append(stmt)
        if isinstance(stmt, ast.Try):
            handlers = [s for h in stmt.handlers for s in h.body]
            out += _top_level(stmt.body + handlers + stmt.orelse + stmt.finalbody)
        elif isinstance(stmt, (ast.If, ast.For, ast.While)):
            out += _top_level(stmt.body + stmt.orelse)
        elif isinstance(stmt, ast.With):
            out += _top_level(stmt.body)
    return out


def source_bindings(stmts: list[ast.stmt]) -> set[str]:
    """The names that PySpark's module binds at its top level or in its classes (not in the
    bodies of its functions), under any condition."""
    names: set[str] = set()
    for stmt in stmts:
        names |= bound_names(stmt)
        if isinstance(stmt, ast.ClassDef):
            names |= source_bindings(stmt.body)
        elif isinstance(stmt, ast.If):
            names |= source_bindings(stmt.body + stmt.orelse)
        elif isinstance(stmt, ast.Try):
            handlers = [s for h in stmt.handlers for s in h.body]
            names |= source_bindings(stmt.body + handlers + stmt.orelse + stmt.finalbody)
    return names


def check_references(module: str, stmts: list[ast.stmt], refs: set[str], source: set[str]):
    """Every name that the stub refers to, that PySpark's module binds (`source`), is bound in
    the stub, at its top level or in one of its classes, unless a star import may bind it. A
    string that parses as a name, such as the default ``"inner"`` of a parameter, is no
    reference, unless PySpark's module binds that name."""
    if any(isinstance(s, ast.ImportFrom) and any(a.name == "*" for a in s.names) for s in stmts):
        return
    bound = set().union(*(bound_names(s) for s in stmts))
    for node in ast.walk(ast.Module(body=stmts, type_ignores=[])):
        if isinstance(node, ast.ClassDef):
            bound.add(node.name)
            for member in node.body:
                bound |= bound_names(member)
    unbound = sorted(n for n in refs - bound if n in source and not hasattr(builtins, n))
    if unbound:
        raise GeneratorError(f"{module}: the stub refers to names that it drops: {unbound}")


# ---------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------


def _verbatim_defaults(function, source: str):
    """`function` whose default values are their text in PySpark's `source`, which editors show
    as PySpark writes it ("any" rather than the 'any' of ``ast.unparse``)."""
    function = copy.deepcopy(function)

    def verbatim(node):
        text = None if node is None else ast.get_source_segment(source, node)
        return node if text is None or "\n" in text else ast.Name(id=text)

    function.args.defaults = [verbatim(d) for d in function.args.defaults]
    function.args.kw_defaults = [verbatim(d) for d in function.args.kw_defaults]
    return function


def _body_source(function, source: str, indent: str) -> list[str]:
    """The lines of the body that the stub keeps of `function`, as PySpark's `source` writes them
    (which ast.unparse of Python 3.9 and 3.10 would print otherwise), indented by `indent`."""
    body = function.body
    first = min([body[0].lineno] + [d.lineno for d in getattr(body[0], "decorator_list", [])])
    if first <= function.lineno:
        raise GeneratorError(f"cannot render the body of {function.name} on the line of its def")
    own = " " * body[0].col_offset
    lines = []
    for line in source.splitlines()[first - 1 : body[-1].end_lineno]:
        if not line.strip():
            lines.append("")
        elif line.startswith(own):
            lines.append(indent + line[len(own) :])
        else:
            lines.append(line)  # the continuation of a string or a bracket, as it is
    return lines


def _render(stmt: ast.stmt, source: str, indent: str = "") -> list[str]:
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        signature = copy.copy(stmt)
        signature.body = [copy.deepcopy(_ELLIPSIS)]
        text = ast.unparse(_verbatim_defaults(signature, source))
        head, _, tail = text.rpartition(":\n")
        if tail.strip() != "...":
            raise GeneratorError(f"cannot render {stmt.name}")
        if not _is_ellipsis(stmt.body):
            # A method that keeps its body.
            lines = [indent + line for line in (head + ":").splitlines()]
            return lines + _body_source(stmt, source, indent + "    ")
        return [indent + line for line in (head + ": ...").splitlines()]
    if isinstance(stmt, ast.ClassDef):
        shell = copy.copy(stmt)
        shell.body = [ast.Pass()]
        header = ast.unparse(shell).rpartition(":\n")[0] + ":"
        lines = [indent + line for line in header.splitlines()]
        if len(stmt.body) == 1 and isinstance(stmt.body[0], ast.Expr):
            lines[-1] += " ..."
            return lines
        for member in stmt.body:
            lines += _render(member, source, indent + "    ")
        return lines
    if isinstance(stmt, ast.If):
        lines = [f"{indent}if {ast.unparse(stmt.test)}:"]
        for member in stmt.body:
            lines += _render(member, source, indent + "    ")
        if stmt.orelse:
            lines.append(f"{indent}else:")
            for member in stmt.orelse:
                lines += _render(member, source, indent + "    ")
        return lines
    return [indent + line for line in ast.unparse(stmt).splitlines()]


def render(stmts: list[ast.stmt], header: str, extra: Extra | None, source: str) -> str:
    """The text of a stub of the statements `stmts` of PySpark's module of text `source`."""
    lines: list[str] = []
    previous = None
    for stmt in stmts:
        kind = type(stmt).__name__
        if kind in ("Import", "ImportFrom"):
            kind = "import"
        if previous is not None and (kind != previous or kind == "ClassDef"):
            lines.append("")
        lines += _render(stmt, source)
        previous = kind
    text = header + "\n" + "\n".join(lines) + "\n"
    if extra is not None:
        text += "\n# Names of the drop-in's module that PySpark's does not have.\n" + extra.source
    return text


# ---------------------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------------------


def _add_to_all(stmts: list[ast.stmt], names: tuple[str, ...], module: str) -> None:
    for stmt in stmts:
        if (
            _all_target(stmt)
            and not isinstance(stmt, ast.AugAssign)
            and isinstance(stmt.value, (ast.List, ast.Tuple))
        ):
            stmt.value.elts += [ast.Constant(n) for n in names]
            return
    raise GeneratorError(f"{module}: has no __all__ for the extra names {list(names)}")


@dataclass
class Stub:
    """The stub of one module, before its imports are pruned."""

    module: str
    source: Path  # PySpark's module, relative to --pyspark-src
    text: str  # its text
    tree: ast.Module
    stmts: list[ast.stmt]
    sites: list[ImportSite]
    extra: Extra | None
    extra_stmts: list[ast.stmt]

    def declared(self) -> set[str]:
        """The names that the stub defines or assigns, its own and those of its extra."""
        stmts = [s for s in self.stmts if not isinstance(s, (ast.Import, ast.ImportFrom))]
        return set().union(*(bound_names(s) for s in stmts + self.extra_stmts))

    def exports(self) -> set[str]:
        """The names that the stub exports itself: not those of its star imports."""
        return self.declared() | {s.name for s in self.sites if s.reexported}

    def star_imports(self) -> list[str]:
        return [s.stmt.module for s in self.sites if s.alias.name == "*"]


def transform(pyspark_src: Path, module: str) -> Stub:
    """The stub statements of PySpark's module `module`, and its extra."""
    relative = source_path(pyspark_src, module)
    text = (pyspark_src / relative).read_text()
    tree = ast.parse(text)
    info = Module(
        name=module,
        imports=_module_imports(module, tree),
        local_functions={
            n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        },
    )
    transformer = StubTransformer(info)
    stmts = transformer.body(tree.body, "module")
    unused = sorted(set(ATTRIBUTE_TYPES.get(module, {})) - transformer.attribute_types_used)
    if unused:
        raise GeneratorError(f"{module}: ATTRIBUTE_TYPES of no class of PySpark's module: {unused}")
    sites = transformer.import_sites
    extra = EXTRAS.get(module)
    extra_stmts: list[ast.stmt] = []
    if extra is not None:
        extra_stmts = ast.parse(extra.source).body
        extra_names = set().union(*(bound_names(s) for s in extra_stmts))
        declared = set().union(
            *(bound_names(s) for s in stmts if not isinstance(s, (ast.Import, ast.ImportFrom)))
        )
        clash = extra_names & (declared | {s.name for s in sites if s.reexportable})
        if clash:
            raise GeneratorError(
                f"{module}: PySpark's module now has {sorted(clash)}; drop them from EXTRAS"
            )
        # A name that PySpark's module imports under TYPE_CHECKING only, and that the
        # drop-in's module has, is imported by the extra instead, which re-exports it.
        sites = [s for s in sites if s.name not in extra_names]
        if extra.all_names:
            _add_to_all(stmts, extra.all_names, module)
    return Stub(module, relative, text, tree, stmts, sites, extra, extra_stmts)


class DropIn:
    """What the drop-in's modules bind at their top level, read from their source (a stub, if
    they have one): whether a stub can re-export a name from one of them."""

    def __init__(self, root: Path, stubs: dict[str, Stub]):
        self.root = root / PACKAGE_ROOT
        self.stubs = stubs
        self._bindings: dict[str, set[str] | None] = {}

    def _file(self, module: str) -> Path | None:
        base = self.root / Path(*module.split("."))
        for candidate in (
            base / "__init__.pyi",
            base / "__init__.py",
            base.with_suffix(".pyi"),
            base.with_suffix(".py"),
        ):
            if candidate.is_file():
                return candidate
        return None

    def _is_module(self, module: str) -> bool:
        return module in self.stubs or self._file(module) is not None

    def bindings(self, module: str) -> set[str] | None:
        """The top-level names of the drop-in's `module`, or None when they are dynamic: it has
        a star import or writes its globals()."""
        if module not in self._bindings:
            self._bindings[module] = set()  # a cycle of star imports binds nothing more
            path = self._file(module)
            names: set[str] | None = set()
            if path is not None:
                tree = ast.parse(path.read_text())
                top = _top_level(tree.body)
                names = set().union(*(bound_names(s) for s in top))
                for stmt in top:
                    if isinstance(stmt, ast.ImportFrom) and any(a.name == "*" for a in stmt.names):
                        starred = self.provides_all(stmt.module or "")
                        names = None if starred is None or names is None else names | starred
                if any(
                    isinstance(n, ast.Call) and _dotted(n.func) == "globals" for n in ast.walk(tree)
                ):
                    names = None
            self._bindings[module] = names
        return self._bindings[module]

    def provides_all(self, module: str, seen: frozenset[str] = frozenset()) -> set[str] | None:
        """The names of the drop-in's `module`, or None when they are dynamic."""
        if module in seen:
            return set()
        if module not in self.stubs:
            return self.bindings(module)
        names: set[str] | None = self.stubs[module].exports()
        for starred in self.stubs[module].star_imports():
            more = self.provides_all(starred, seen | {module})
            names = None if more is None or names is None else names | more
        return names

    def provides(self, module: str, name: str) -> bool:
        """Whether the drop-in's `module` has `name`, so that a type checker resolves it."""
        if self._is_module(f"{module}.{name}"):
            return True
        names = self.provides_all(module)
        return names is None or name in names


def resolve_reexports(stubs: dict[str, Stub], dropin: DropIn) -> None:
    """Re-export each reexportable import of a stub that the drop-in has, to a fixed point (a
    stub re-exports the names that other stubs re-export)."""
    for stub in stubs.values():
        for site in stub.sites:
            site.reexported = site.reexportable
    changed = True
    while changed:
        changed = False
        for stub in stubs.values():
            for site in stub.sites:
                if site.reexported and not dropin.provides(site.stmt.module, site.alias.name):
                    site.reexported = False
                    changed = True


def render_stub(stub: Stub, tag: str) -> str:
    """The text of `stub`."""
    extra_refs = referenced_names(stub.extra_stmts)
    kept, refs = prune(stub.stmts, stub.sites, extra_refs)
    check_references(
        stub.module, kept + stub.extra_stmts, refs - extra_refs, source_bindings(stub.tree.body)
    )
    header = HEADER.format(source=stub.source.as_posix(), tag=tag)
    text = render(kept, header, stub.extra, stub.text)
    check_attributes(stub.module, stub.tree, text)
    return text


def generate(root: Path, pyspark_src: str, log=print) -> dict[Path, str]:
    """The stubs and pyproject.toml with their block of ruff excludes, by path relative to
    `root`."""
    src = Path(pyspark_src).resolve()
    if not (src / "pyspark" / "sql" / "functions" / "builtin.py").is_file():
        raise GeneratorError(f"--pyspark-src {pyspark_src}: not the python/ directory of Spark")
    unknown = sorted(set(EXTRAS) - set(STUB_MODULES))
    if unknown:
        raise GeneratorError(f"EXTRAS of modules without a stub: {unknown}")
    unknown = sorted(set(ATTRIBUTE_TYPES) - set(STUB_MODULES))
    if unknown:
        raise GeneratorError(f"ATTRIBUTE_TYPES of modules without a stub: {unknown}")
    check_identity_decorators(src)
    tag = tracked_tag(root)
    stubs = {module: transform(src, module) for module in STUB_MODULES}
    resolve_reexports(stubs, DropIn(root, stubs))
    outputs = {
        PACKAGE_ROOT / stub.source.with_suffix(".pyi"): render_stub(stub, tag)
        for stub in stubs.values()
    }
    log(f"Generated {len(outputs)} stubs from PySpark {tag}")
    outputs[PYPROJECT] = with_ruff_block((root / PYPROJECT).read_text(), sorted(outputs))
    return outputs


def with_ruff_block(pyproject: str, stubs: list[Path]) -> str:
    """`pyproject` whose block of generated stubs lists `stubs`."""
    head, begin, rest = pyproject.partition(RUFF_BLOCK_BEGIN)
    _, end, tail = rest.partition(RUFF_BLOCK_END)
    if not begin or not end:
        raise GeneratorError(f"{PYPROJECT}: no block of generated type stubs in extend-exclude")
    lines = "".join(f'    "{path.as_posix()}",\n' for path in stubs)
    return head + begin + lines + end + tail


def stale_stubs(root: Path, outputs: dict[Path, str]) -> list[Path]:
    """The stubs that the generator wrote, of modules that it no longer stubs."""
    return [
        stub.relative_to(root)
        for stub in sorted((root / PACKAGE_ROOT).rglob("*.pyi"))
        if stub.relative_to(root) not in outputs and GENERATED in stub.read_text()
    ]


def check(root: Path, outputs: dict[Path, str], log=print) -> bool:
    """Whether the checked-in files equal `outputs`, and no stale stub is checked in; logs a
    unified diff of each that is not."""
    clean = True
    for path, expected in outputs.items():
        target = root / path
        actual = target.read_text() if target.is_file() else ""
        if actual == expected:
            log(f"OK    {path} matches its generator")
            continue
        clean = False
        log(f"DRIFT {path} differs from the output of scripts/gen_type_stubs.py:")
        diff = difflib.unified_diff(
            actual.splitlines(keepends=True),
            expected.splitlines(keepends=True),
            fromfile=f"{path} (checked in)",
            tofile=f"{path} (generated)",
        )
        for line in diff:
            log(line.rstrip("\n"))
    for stale in stale_stubs(root, outputs):
        clean = False
        log(f"STALE {stale} is a generated stub of no module of the generator")
    return clean


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--pyspark-src",
        required=True,
        help="Path to the python/ directory of an Apache Spark checkout at the tracked tag, "
        "whose modules the stubs are made of. It is parsed, not imported.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Write nothing: regenerate in memory, print a diff of every checked-in stub that "
        "differs from its generated text, and exit with status 1 if any does.",
    )
    args = parser.parse_args(argv)

    try:
        outputs = generate(REPO_ROOT, args.pyspark_src)
    except GeneratorError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.check:
        if check(REPO_ROOT, outputs):
            return 0
        print(
            "\nThe type stubs have drifted. Do not edit them by hand: change "
            "scripts/gen_type_stubs.py and regenerate with\n"
            "  python scripts/gen_type_stubs.py --pyspark-src <spark at the tracked tag>/python"
        )
        return 1

    for path, text in outputs.items():
        (REPO_ROOT / path).parent.mkdir(parents=True, exist_ok=True)
        (REPO_ROOT / path).write_text(text)
        print(f"Generated {path}")
    for stale in stale_stubs(REPO_ROOT, outputs):
        (REPO_ROOT / stale).unlink()
        print(f"Removed {stale}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
