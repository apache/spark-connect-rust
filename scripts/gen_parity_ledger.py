#!/usr/bin/env python3
"""Generate the API parity ledger from the reference PySpark source.

We mirror the Spark Connect Python client (``pyspark.sql.connect.*``) plus the
shared modules it depends on (``pyspark.sql.types``, ``pyspark.errors``, ...).
This script AST-parses each source file and emits, for every public class and
function/method, a row in a CSV ledger. Each row is a checklist item for the
Rust rewrite so nothing is silently dropped.

Usage:
    python scripts/gen_parity_ledger.py \
        --src ~/workspace/origin/spark/python/pyspark \
        --out docs/parity

Status of each item is tracked in ``docs/parity/status.csv`` (hand/tool edited);
this script only (re)generates the *inventory* (``docs/parity/inventory.csv``)
and never clobbers status.
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
from pathlib import Path

# Modules that make up the client surface we must reach full parity on.
# Paths are relative to the pyspark package root.
INCLUDE_DIRS = [
    "sql/connect",  # the Connect client itself
]
INCLUDE_FILES = [
    "sql/types.py",  # shared DataType hierarchy
    "sql/column.py",  # (classic shims referenced by connect)
    "errors/exceptions/connect.py",
    "errors/exceptions/base.py",
]


def is_public(name: str) -> bool:
    # Dunder methods are part of the surface (operators, __getitem__...), keep them.
    if name.startswith("__") and name.endswith("__"):
        return True
    return not name.startswith("_")


def signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    a = fn.args
    parts: list[str] = []
    posonly = getattr(a, "posonlyargs", [])
    for arg in posonly:
        parts.append(arg.arg)
    if posonly:
        parts.append("/")
    for arg in a.args:
        parts.append(arg.arg)
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        parts.append("*")
    for arg in a.kwonlyargs:
        parts.append(arg.arg)
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    return "(" + ", ".join(parts) + ")"


def walk_file(path: Path, rel: str, rows: list[dict]) -> None:
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if is_public(node.name):
                rows.append(
                    dict(
                        module=rel,
                        kind="function",
                        cls="",
                        name=node.name,
                        sig=signature(node),
                        lineno=node.lineno,
                    )
                )
        elif isinstance(node, ast.ClassDef):
            if not is_public(node.name):
                continue
            rows.append(
                dict(module=rel, kind="class", cls="", name=node.name, sig="", lineno=node.lineno)
            )
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if is_public(sub.name):
                        rows.append(
                            dict(
                                module=rel,
                                kind="method",
                                cls=node.name,
                                name=sub.name,
                                sig=signature(sub),
                                lineno=sub.lineno,
                            )
                        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="pyspark package root")
    ap.add_argument("--out", default="docs/parity")
    args = ap.parse_args()

    src = Path(os.path.expanduser(args.src))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for d in INCLUDE_DIRS:
        files += sorted((src / d).rglob("*.py"))
    for f in INCLUDE_FILES:
        p = src / f
        if p.exists():
            files.append(p)

    rows: list[dict] = []
    for f in files:
        rel = str(f.relative_to(src))
        walk_file(f, rel, rows)

    inv = out / "inventory.csv"

    # Preserve any hand-tracked status/notes across regenerations, keyed by
    # (module, cls, name). New items default to "todo".
    prior: dict[tuple[str, str, str], dict] = {}
    if inv.exists():
        for r in csv.DictReader(inv.open()):
            prior[(r["module"], r["cls"], r["name"])] = r
    for r in rows:
        key = (r["module"], r["cls"], r["name"])
        old = prior.get(key)
        r["status"] = old.get("status", "todo") if old else "todo"
        r["notes"] = old.get("notes", "") if old else ""

    with inv.open("w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["module", "kind", "cls", "name", "sig", "lineno", "status", "notes"],
        )
        w.writeheader()
        w.writerows(rows)

    n_mod = len({r["module"] for r in rows})
    n_cls = sum(1 for r in rows if r["kind"] == "class")
    n_fn = sum(1 for r in rows if r["kind"] in ("function", "method"))
    print(
        f"wrote {inv}: {len(rows)} items across {n_mod} modules "
        f"({n_cls} classes, {n_fn} functions/methods)"
    )


if __name__ == "__main__":
    main()
