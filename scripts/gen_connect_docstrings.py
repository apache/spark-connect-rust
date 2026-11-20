#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generator of the docstrings that PySpark Connect's modules copy onto the drop-in's native classes.

PySpark Connect's modules give the methods of their classes the docstrings of PySpark's classic
classes, of assignments such as ``format.__doc__ = PySparkDataStreamReader.format.__doc__`` of
pyspark/sql/connect/streaming/readwriter.py. The drop-in's classes of those modules that are native,
of the Rust extension, cannot take an assignment of a docstring at run time: their docstrings are
those of the ``#[doc]`` attributes of their Rust definitions, which the extension is built with.
This script generates crates/pyspark-rs/src/connect_docs_generated.rs, whose ``connect_doc!`` gives
the docstring of each object of ``MODULES`` that PySpark's module assigns, as PySpark has it at run
time, which the Rust definitions take (``#[doc = connect_doc!(DataStreamReader.format)]``). It also
gives the docstring of each class of ``CLASS_MODULES``, whose classes are native here too
(``pyspark.sql.types`` and the ``SparkSession`` of ``pyspark.sql.connect.session``), and of each
class of the body of such a class (``SparkSession.Builder``), and of each public member of them, its
own or of a base of it in PySpark, as Python finds it on the class
(``connect_doc!(IntegerType.fromInternal)`` is that of ``DataType.fromInternal``, and
``connect_doc!(SparkSession.sql)`` the classic ``sql`` that PySpark Connect's module assigns); an
object of no docstring has none, of which the Rust definitions take no ``#[doc]``. Last, it gives
each class of ``INHERITING`` and each public member of it the docstring that ``inspect.getdoc`` and
``help()`` find of it in PySpark: PySpark Connect's class is a subclass of the classic one, and
neither has a docstring of its own, so they find that of the classic class
(``connect_doc!(Column.alias)`` is that of the classic ``Column.alias``), which the native class, a
virtual subclass of it, does not inherit. The public base of a class of ``PUBLIC_BASES``, PySpark's
classic class, takes the docstrings of the native class (``DataFrame`` of
python/pyspark/sql/dataframe.py): this script generates too
python/pyspark/sql/_classic_docs_generated.py, of the docstrings of the classic class that differ
from those, which the public base takes in their place. The docstrings are worked out of PySpark's
source as scripts/gen_type_stubs.py works them out for the stubs (its ``Docs``).

Usage, from anywhere:

    python scripts/gen_connect_docstrings.py --pyspark-src <spark>/python            # write
    python scripts/gen_connect_docstrings.py --pyspark-src <spark>/python --check    # verify

``<spark>/python`` is the python/ directory of an Apache Spark checkout at the tracked tag
(SPARK_TAG of scripts/check_vendored_upstream.sh). It is parsed, not imported. ``--check``
regenerates in memory and fails with a diff when a checked-in file differs; CI runs it
(.github/workflows/generated-drift.yml).
"""

from __future__ import annotations

import argparse
import ast
import difflib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from gen_type_stubs import _CACHES, Docs, _decorator_name, tracked_tag  # noqa: E402

OUTPUT = Path("crates/pyspark-rs/src/connect_docs_generated.rs")
PYTHON_OUTPUT = Path("python/pyspark/sql/_classic_docs_generated.py")
# The modules of PySpark Connect whose classes are native in the drop-in, and whose module copies
# the docstrings of PySpark's classic classes onto them.
MODULES = (
    "pyspark.sql.connect.catalog",
    "pyspark.sql.connect.conf",
    "pyspark.sql.connect.datasource",
    "pyspark.sql.connect.group",
    "pyspark.sql.connect.merge",
    "pyspark.sql.connect.observation",
    "pyspark.sql.connect.readwriter",
    "pyspark.sql.connect.streaming.query",
    "pyspark.sql.connect.streaming.readwriter",
    "pyspark.sql.connect.tvf",
)

# The modules of PySpark whose classes are native in the drop-in, of the docstrings of all their
# classes, and of the classes of their classes (`SparkSession.Builder`), and of their public members,
# as Python finds them on each class.
CLASS_MODULES = ("pyspark.sql.types", "pyspark.sql.connect.session")
# Of them, the modules whose classes a macro of the classes forwards by their names as strings
# (`tt`, the abstract_type! of types.rs).
NAMED_CLASS_MODULES = ("pyspark.sql.types",)

# The classes of PySpark Connect, by their modules, that are subclasses of PySpark's classic
# classes and whose classes are native in the drop-in, virtual subclasses of the classic ones: of
# the docstrings that `inspect.getdoc` and `help()` find of them and of their members in PySpark,
# those of the classic class.
INHERITING = {
    "pyspark.sql.connect.column": ("Column",),
    "pyspark.sql.connect.window": ("WindowSpec",),
    "pyspark.sql.connect.table_arg": ("TableArg",),
    "pyspark.sql.connect.dataframe": (
        "DataFrame",
        "DataFrameNaFunctions",
        "DataFrameStatFunctions",
    ),
}
# The class decorators of PySpark that wrap each method of the class apart (`with_origin_to_class`
# of PySpark Connect's Column): an alias of a member (`name = alias`) is then a function of the name
# of the member that the class does not have, of which `inspect.getdoc` finds no docstring.
WRAPPING_CLASS_DECORATORS = frozenset(
    {"with_origin_to_class", "pyspark.errors.utils.with_origin_to_class"}
)
# PySpark's classic classes whose public bases in the drop-in take the docstrings of the native
# Connect classes, by the name of the dictionary of _classic_docs_generated.py of their docstrings
# that differ from those: of PySpark's classic class (module, class) and of its Connect class of
# ``INHERITING``.
PUBLIC_BASES = {
    "DATAFRAME": (
        ("pyspark.sql.dataframe", "DataFrame"),
        ("pyspark.sql.connect.dataframe", "DataFrame"),
    ),
}


def docstrings(pyspark_src: str) -> dict[str, str]:
    """The docstring of each object that a module of ``MODULES`` assigns one, by its qualified
    name (``DataStreamReader.format``, ``DataStreamReader``), in the order of the modules."""
    docs = Docs(Path(pyspark_src))
    found: dict[str, str] = {}
    for module in MODULES:
        source = docs.source(module)
        if source is None:
            raise SystemExit(f"error: {module} not found in {pyspark_src}")
        for qualname in source.doc_assigns:
            if qualname.endswith(".__init__"):
                # The `__init__` of a native class is CPython's slot, whose wrapper has
                # CPython's docstring, which the class cannot set.
                continue
            doc = docs.raw(module, qualname)
            if doc is None:
                raise SystemExit(f"error: {module}.{qualname} has no docstring")
            if qualname in found:
                raise SystemExit(f"error: {qualname} of two modules")
            found[qualname] = doc
        for name, node in source.defs.items():
            if "." in name or not isinstance(node, ast.ClassDef) or name in found:
                continue
            # The docstring of a class of the module of its own, that the module does not assign.
            doc = docs.raw(module, name)
            if doc is not None:
                found[name] = doc
        for name, node in source.defs.items():
            if (
                name.count(".") != 1
                or name in found
                or not isinstance(node, ast.Assign)
                or not isinstance(node.value, ast.Name)
            ):
                continue
            # An alias of a member of a class (`mean = avg` of GroupedData), whose docstring is
            # that of the member.
            target = f"{name.split('.')[0]}.{node.value.id}"
            if target in found:
                found[name] = "@" + target
    return found


def class_docstrings(pyspark_src: str, modules: tuple[str, ...] = CLASS_MODULES) -> dict[str, str]:
    """The docstring of each public class of a module of `modules`, and of each public class of
    such a class (``SparkSession.Builder``), and of each public member of the class, as Python
    finds it on the class: the member of the first class of its MRO in PySpark that defines it, of
    the module. A member of no docstring, as a class of none, is left out. The value of a member
    that the class does not define is the qualified name of the one it finds."""
    docs = Docs(Path(pyspark_src))
    found: dict[str, str] = {}
    for module in modules:
        source = docs.source(module)
        if source is None:
            raise SystemExit(f"error: {module} not found in {pyspark_src}")
        classes = [
            name
            for name, node in source.defs.items()
            if name.count(".") <= 1
            and isinstance(node, ast.ClassDef)
            and not any(part.startswith("_") for part in name.split("."))
            and ("." not in name or isinstance(source.defs.get(name.split(".")[0]), ast.ClassDef))
        ]
        for cls in classes:
            doc = docs.raw(module, cls)
            if doc is not None:
                found[cls] = doc
            seen: set[str] = set()
            for base_module, base in docs.mro(module, cls):
                if base_module != module:
                    continue
                for qualname in docs.source(base_module).defs:
                    member = qualname[len(base) + 1 :]
                    if (
                        not qualname.startswith(base + ".")
                        or "." in member
                        or member.startswith("_")
                        or member in seen
                    ):
                        continue
                    seen.add(member)
                    if docs.raw(base_module, qualname) is None:
                        continue
                    found[f"{cls}.{member}"] = (
                        docs.raw(base_module, qualname) if base == cls else f"@{base}.{member}"
                    )
    return found


def _first_docstring(docs: Docs, classes: list[tuple[str, str]], member: str) -> str | None:
    """The docstring of `member` (the class itself of "") of the first of `classes`, (module,
    class) of an MRO, that defines it with one, as ``inspect.getdoc`` looks the classes up. A
    cache of functools (``functools.cache``, ``functools.cached_property``) is a descriptor whose
    class ``inspect`` does not find: of the first that defines it so, it is its own, or none."""
    for module, cls in classes:
        qualname = f"{cls}.{member}" if member else cls
        if member and qualname not in docs.source(module).defs:
            continue
        doc = docs.raw(module, qualname)
        node = docs.node(module, qualname)
        if doc is not None or (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(_decorator_name(d) in _CACHES for d in node.decorator_list)
        ):
            return doc
    return None


def inherited_docstrings(pyspark_src: str) -> dict[str, str]:
    """The docstring that ``inspect.getdoc`` finds in PySpark of each class of ``INHERITING`` and of
    each public member of it, as Python finds the member on the class: its own, or of the first
    class of the MRO of the class that defines it that has one, of the classic class. An alias of a
    member (``groupby = groupBy`` of PySpark Connect's DataFrame) is the function of that member,
    whose docstring ``inspect.getdoc`` looks up by the name of the member, but of none where a
    decorator of the class (``WRAPPING_CLASS_DECORATORS``) wraps it apart from the member (``name =
    alias`` of PySpark Connect's Column). An object of no docstring is left out."""
    docs = Docs(Path(pyspark_src))
    found: dict[str, str] = {}
    for module, classes in INHERITING.items():
        if docs.source(module) is None:
            raise SystemExit(f"error: {module} not found in {pyspark_src}")
        for cls in classes:
            mro = docs.mro(module, cls)
            wrapped = any(
                _decorator_name(d) in WRAPPING_CLASS_DECORATORS
                for d in docs.node(module, cls).decorator_list
            )
            doc = _first_docstring(docs, mro, "")
            if doc is not None:
                found[cls] = doc
            seen: set[str] = set()
            for i, (base_module, base) in enumerate(mro):
                for qualname, node in docs.source(base_module).defs.items():
                    member = qualname[len(base) + 1 :]
                    if (
                        not qualname.startswith(base + ".")
                        or "." in member
                        or member.startswith("_")
                        or member in seen
                    ):
                        continue
                    seen.add(member)
                    name = member
                    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name):
                        if wrapped and i == 0:
                            continue  # an alias, of no docstring
                        name = node.value.id
                    doc = _first_docstring(docs, mro[i:], name)
                    if doc is not None:
                        found[f"{cls}.{member}"] = doc
    return found


def _public_members(docs: Docs, mro: list[tuple[str, str]]) -> list[str]:
    """The public members of the class of `mro` that the classes of its MRO define, by name."""
    found: set[str] = set()
    for module, cls in mro:
        for qualname in docs.source(module).defs:
            member = qualname[len(cls) + 1 :]
            if qualname.startswith(cls + ".") and "." not in member and not member.startswith("_"):
                found.add(member)
    return sorted(found)


def classic_docstrings(pyspark_src: str) -> dict[str, dict[str, str | None]]:
    """Of each class of ``PUBLIC_BASES``, by the name of its dictionary, the ``__doc__`` of the
    class and of each public member of PySpark's classic class, as Python finds it on the class (of
    the first class of its MRO that defines it), that differs from the docstring of PySpark's
    Connect class of ``inherited_docstrings``, which the native class has: by the name of the
    member ("" of the class)."""
    docs = Docs(Path(pyspark_src))
    inherited = inherited_docstrings(pyspark_src)
    found: dict[str, dict[str, str | None]] = {}
    for name, ((module, cls), (_, connect_cls)) in PUBLIC_BASES.items():
        mro = docs.mro(module, cls)
        differ: dict[str, str | None] = {}
        for member in ["", *_public_members(docs, mro)]:
            definer = next(
                (
                    (base_module, f"{base}.{member}" if member else base)
                    for base_module, base in mro
                    if not member or f"{base}.{member}" in docs.source(base_module).defs
                ),
                None,
            )
            doc = docs.raw(*definer) if definer is not None else None
            if doc != inherited.get(f"{connect_cls}.{member}" if member else connect_cls):
                differ[member] = doc
        found[name] = differ
    return found


def generate_python(pyspark_src: str) -> str:
    tag = tracked_tag(REPO_ROOT)
    lines = [
        f"# Generated by scripts/gen_connect_docstrings.py of Apache Spark {tag}. Do not edit:",
        "# regenerate it with the script.",
        '"""The docstrings of PySpark\'s classic classes whose public bases here take the',
        "docstrings of the native Connect classes (crates/pyspark-rs/src/connect_docs_generated.rs),",
        "which differ from those: of the members that PySpark Connect's class wraps in a cache of",
        "functools (``functools.cache``), of which ``inspect.getdoc`` finds no docstring, and of its",
        "aliases (``groupby = groupBy``), whose docstring is that of the member that they name. The",
        'public base takes them in their place, by the name of the member ("" of the class)."""',
    ]
    for name, docstrings in classic_docstrings(pyspark_src).items():
        lines += ["", f"{name} = {{"]
        lines += [f"    {member!r}: {doc!r}," for member, doc in docstrings.items()]
        lines.append("}")
    return "\n".join(lines) + "\n"


def _raw_string(text: str) -> str:
    """`text` as a Rust raw string literal, of as few hashes as it needs."""
    hashes = 1
    while '"' + "#" * hashes in text:
        hashes += 1
    return f'r{"#" * hashes}"{text}"{"#" * hashes}'


def generate(pyspark_src: str) -> str:
    tag = tracked_tag(REPO_ROOT)
    of_classes = class_docstrings(pyspark_src)
    classes_of_class_modules = {
        name for name in class_docstrings(pyspark_src, NAMED_CLASS_MODULES) if "." not in name
    }
    lines = [
        f"// Generated by scripts/gen_connect_docstrings.py of Apache Spark {tag}. Do not edit:",
        "// regenerate it with the script.",
        "",
        "//! The docstrings that PySpark Connect's modules give the methods of their classes of",
        "//! PySpark's classic classes (`format.__doc__ = PySparkDataStreamReader.format.__doc__`),",
        "//! of the classes of those modules that are native here, which take them of their `#[doc]`",
        "//! attributes; those of the native types of `pyspark.sql.types`; and those that",
        "//! `inspect.getdoc` finds of the classic classes of PySpark Connect's subclasses of them,",
        "//! of no docstrings of their own (`Column.alias`), of which the native classes are virtual",
        "//! subclasses.",
        "",
        "/// The docstring of the object `$name` of PySpark Connect's module, as PySpark has it at",
        "/// run time: `connect_doc!(DataStreamReader.format)`, of the textual scope of the modules",
        "/// that lib.rs declares after this one (`#[macro_use]`), as a macro in an attribute is",
        "/// resolved before the imports are.",
        "#[rustfmt::skip]",
        "macro_rules! connect_doc {",
    ]
    arms = [
        *docstrings(pyspark_src).items(),
        *of_classes.items(),
        *inherited_docstrings(pyspark_src).items(),
    ]
    names = [qualname for qualname, _ in arms]
    if len(set(names)) != len(names):
        twice = sorted({name for name in names if names.count(name) > 1})
        raise SystemExit(f"error: the docstrings of {twice} of two sources")
    for qualname, doc in arms:
        pattern = " . ".join(qualname.split("."))
        if doc.startswith("@"):
            # A member that the class inherits, or an alias, of the docstring of the member that
            # it is.
            lines.append(
                f"    ({pattern}) => {{ connect_doc!({' . '.join(doc[1:].split('.'))}) }};"
            )
            continue
        lines.append(f"    ({pattern}) => {{")
        lines.append(f"        {_raw_string(doc)}")
        lines.append("    };")
        if "." not in qualname and qualname in classes_of_class_modules:
            # Of the name of the class as a string, which a macro of classes forwards (`tt`).
            lines.append(f'    ("{qualname}") => {{ connect_doc!({qualname}) }};')
    lines += ["}", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--pyspark-src",
        required=True,
        help="Path to the python/ directory of an Apache Spark checkout at the tracked tag, "
        "whose modules the docstrings are of. It is parsed, not imported.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Write nothing: regenerate in memory, print a diff if the checked-in file differs "
        "from its generated text, and exit with status 1 if it does.",
    )
    args = parser.parse_args(argv)
    outputs = {
        OUTPUT: generate(args.pyspark_src),
        PYTHON_OUTPUT: generate_python(args.pyspark_src),
    }
    if args.check:
        drift = False
        for output, expected in outputs.items():
            target = REPO_ROOT / output
            actual = target.read_text() if target.is_file() else ""
            if actual == expected:
                print(f"OK    {output} matches its generator")
                continue
            drift = True
            print(f"DRIFT {output} differs from the output of scripts/gen_connect_docstrings.py:")
            for line in difflib.unified_diff(
                actual.splitlines(keepends=True),
                expected.splitlines(keepends=True),
                fromfile=f"{output} (checked in)",
                tofile=f"{output} (generated)",
            ):
                print(line.rstrip("\n"))
        if drift:
            print(
                "\nDo not edit them by hand: regenerate them with\n"
                "  python scripts/gen_connect_docstrings.py --pyspark-src <spark at the tracked tag>/python"
            )
            return 1
        return 0
    for output, expected in outputs.items():
        (REPO_ROOT / output).write_text(expected)
        print(f"Generated {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
