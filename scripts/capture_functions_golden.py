#!/usr/bin/env python3
"""Auto-capture golden Expression protos for pyspark.sql.connect.functions.

Introspects the functions module, attempts to call each public function with a
small set of heuristic argument tuples, and for each call that yields a Column,
records the normalized expression protobuf. This seeds a broad golden set for
the Rust `functions` builder to test against, and reports how many of the ~576
functions were captured with simple args (the rest need bespoke args and can be
added by hand later).

Usage:
    python3 scripts/capture_functions_golden.py --remote sc://localhost:15002 \
        --out tests/golden/functions.jsonl
"""

from __future__ import annotations

import argparse
import base64
import inspect
import json
from pathlib import Path

# reuse the normalizer from the sibling script
from capture_golden import normalize  # type: ignore
from pyspark.sql import SparkSession
from pyspark.sql.connect import functions as F
from pyspark.sql.connect.column import Column

# Each candidate is a list of typed arg specs. Recorded in the golden so the
# Rust side can reconstruct the exact same arguments deterministically.
#   col:NAME  -> F.col(NAME)            (Rust: col(NAME))
#   int:N     -> Python int N           (Rust: integer-literal Column)
#   str:S     -> Python str S           (Rust: string-literal Column)
#   litint:N  -> F.lit(N)               (Rust: integer-literal Column)
CANDIDATE_SPECS = [
    [],
    ["col:a"],
    ["col:a", "col:b"],
    ["col:a", "col:b", "col:c"],
    ["col:a", "col:b", "col:c", "col:d"],
    ["col:a", "int:1"],
    ["col:a", "str:x"],
    ["col:a", "col:b", "int:1"],
    ["int:1"],
    ["str:x"],
    ["col:a", "litint:1"],
]


def materialize(spec, F):
    out = []
    for tok in spec:
        kind, _, val = tok.partition(":")
        if kind == "col":
            out.append(F.col(val))
        elif kind == "int":
            out.append(int(val))
        elif kind == "str":
            out.append(val)
        elif kind == "litint":
            out.append(F.lit(int(val)))
        else:
            raise ValueError(tok)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default="sc://localhost:15002")
    ap.add_argument("--out", default="tests/golden/functions.jsonl")
    args = ap.parse_args()

    spark = SparkSession.builder.remote(args.remote).getOrCreate()

    names = sorted(
        n
        for n in dir(F)
        if not n.startswith("_") and callable(getattr(F, n)) and not inspect.isclass(getattr(F, n))
    )

    out_records = []
    captured = 0
    skipped = []
    for name in names:
        fn = getattr(F, name)
        got = None
        for spec in CANDIDATE_SPECS:
            try:
                argv = materialize(spec, F)
                res = fn(*argv)
            except Exception:
                continue
            if isinstance(res, Column):
                try:
                    expr = res._expr.to_plan(spark.client)
                except Exception:
                    continue
                normalize(expr)
                got = (spec, expr)
                break
        if got is None:
            skipped.append(name)
            continue
        spec, expr = got
        out_records.append(
            dict(
                name=name,
                kind="function",
                args=spec,
                b64=base64.b64encode(expr.SerializeToString()).decode(),
                text=str(expr),
            )
        )
        captured += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in out_records) + "\n")
    print(f"functions total={len(names)} captured={captured} skipped={len(skipped)}")
    print(f"wrote {out}")
    print("first 25 skipped (need bespoke args):", ", ".join(skipped[:25]))
    spark.stop()


if __name__ == "__main__":
    main()
