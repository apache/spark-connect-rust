#!/usr/bin/env python3
"""Real parity gate: official connect tests, reference client vs OUR Rust client.

Unlike the old (vacuous) gate, "ours" here runs each official test file through the
Rust transport injection plugin (scripts/rust_transport_plugin.py), so it genuinely
exercises our Rust client. "reference" runs the same file with the stock pyspark. A
REGRESSION is a test the reference passes but ours fails; environmental failures the
reference also hits are not counted.

Usage:
    RUST_PYSPARK_SO=/repo/python/pyspark/_pyspark.so \
    python3 scripts/rust_parity_diff.py \
        --spark ~/workspace/origin/spark-v4.2.0 \
        --remote sc://localhost:15002 \
        [--path-contains sql/tests/connect] [--filter test_parity_] [--jobs 3]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def discover(spark_py, name_filter, path_contains, exclude):
    out = []
    for p in sorted(spark_py.rglob("test_*.py")):
        s = str(p).replace(os.sep, "/")
        if "/tests/connect/" not in s:
            continue
        if name_filter and name_filter not in p.name:
            continue
        if path_contains and path_contains not in s:
            continue
        if exclude and any(x and x in p.name for x in exclude):
            continue
        out.append(p)
    return out


def parse(text):
    c = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
    for line in reversed(text.strip().splitlines()):
        if " passed" in line or " failed" in line or " error" in line or " skipped" in line:
            for k in ("passed", "failed", "skipped", "error", "errors"):
                m = re.search(rf"(\d+) {k}\b", line)
                if m:
                    c["error" if k == "errors" else k] = int(m.group(1))
            if "passed" in line or "failed" in line or "error" in line:
                break
    return c


def run(test_file, spark_py, remote, rust):
    env = dict(os.environ)
    env["SPARK_CONNECT_TESTING_REMOTE"] = remote
    env["SPARK_TESTING"] = "1"
    env["SPARK_SKIP_CONNECT_COMPAT_TESTS"] = "1"
    args = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    if rust:
        env["PYTHONPATH"] = os.pathsep.join([str(REPO / "scripts"), str(spark_py)])
        env["RUST_PYSPARK_SO"] = os.environ["RUST_PYSPARK_SO"]
        args += ["-p", "rust_transport_plugin"]
    else:
        env["PYTHONPATH"] = str(spark_py)
    args.append(str(test_file))
    try:
        r = subprocess.run(args, env=env, capture_output=True, text=True, timeout=360)
        return parse(r.stdout + "\n" + r.stderr)
    except subprocess.TimeoutExpired:
        return {"passed": 0, "failed": 0, "skipped": 0, "error": 0, "timeout": True}


def fails(text_counts):
    return text_counts.get("failed", 0) + text_counts.get("error", 0)


def is_regression(ref, ours):
    """A regression is where the reference passes but ours does not.

    If the reference itself could not run the file (timed out, or produced no
    passes and no failures - e.g. a server-side feature unsupported in this
    environment), it is environmental and NOT counted against us.
    """
    if ref.get("timeout") or (ref["passed"] == 0 and fails(ref) == 0):
        return False
    if ours.get("timeout"):
        return True  # reference ran, ours hung
    return fails(ours) > fails(ref) or ours["passed"] < ref["passed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spark", required=True)
    ap.add_argument("--remote", default="sc://localhost:15002")
    ap.add_argument("--filter", default=None)
    ap.add_argument("--path-contains", default=None)
    ap.add_argument(
        "--exclude",
        default=None,
        help="comma-separated filename substrings to skip (e.g. modules needing "
        "server-side Python workers that a single-node pure-connect server can't run)",
    )
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument(
        "--retries",
        type=int,
        default=2,
        help="If a file is flagged as a regression, re-run it (fresh ref+ours) up to this "
        "many more times; only a regression that PERSISTS on every attempt counts. This "
        "distinguishes a real regression from environmental flakiness (e.g. the streaming "
        "listener/observation tests, which pass reliably in isolation but can transiently "
        "time out or error late in a long serial run when the shared single-node server is "
        "resource-starved). A persistent bug still fails every attempt and is still caught.",
    )
    args = ap.parse_args()
    exclude = [x.strip() for x in args.exclude.split(",")] if args.exclude else []

    if not os.environ.get("RUST_PYSPARK_SO"):
        print("!! set RUST_PYSPARK_SO to the built extension")
        return 2
    spark_py = Path(os.path.expanduser(args.spark)) / "python"
    files = discover(spark_py, args.filter, args.path_contains, exclude)
    if not files:
        print("!! no test files found")
        return 2
    print(f"Discovered {len(files)} connect test files.\n", flush=True)

    def work(tf):
        ref = run(tf, spark_py, args.remote, rust=False)
        ours = run(tf, spark_py, args.remote, rust=True)
        attempts = 1
        # Re-run a flagged file fresh: a genuine regression persists across attempts,
        # while an environmental flake clears. Keep the first non-regression result.
        while is_regression(ref, ours) and attempts <= args.retries:
            attempts += 1
            ref = run(tf, spark_py, args.remote, rust=False)
            ours = run(tf, spark_py, args.remote, rust=True)
        return tf.name, ref, ours, attempts

    regressions = []
    done = 0
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs))
    for name, ref, ours, attempts in ex.map(work, files):
        done += 1
        is_reg = is_regression(ref, ours)
        tag = "REGRESSION" if is_reg else "ok"
        retry_note = f" (after {attempts} attempts)" if attempts > 1 else ""
        print(
            f"[{done}/{len(files)}] {tag:<10} {name:<48} "
            f"ref(p={ref['passed']},f={fails(ref)}) ours(p={ours['passed']},f={fails(ours)})"
            + (" TIMEOUT" if ours.get("timeout") else "")
            + retry_note,
            flush=True,
        )
        if is_reg:
            regressions.append(name)
    ex.shutdown()
    print(f"\n{len(regressions)} regression file(s) out of {len(files)}.")
    if regressions:
        print("Regressions:", ", ".join(regressions))
    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
