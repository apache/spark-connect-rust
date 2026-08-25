#!/usr/bin/env python3
"""Run Apache Spark's official Connect test files against a target pyspark package.

This is the incremental parity gate (see docs/design/ACCEPTANCE.md). It points a
chosen official connect test file at our live dev Connect server and runs it with
a *target* `pyspark` package on PYTHONPATH - the real one for a baseline, or our
Rust-backed replacement once it can be imported.

Examples:
    # Baseline: confirm the official test passes with the reference client.
    python3 scripts/run_official_connect_tests.py \
        --spark ~/workspace/origin/spark \
        test_connect_column test_connect_functions

    # Against our package (once built/installed): prepend it to PYTHONPATH.
    python3 scripts/run_official_connect_tests.py \
        --spark ~/workspace/origin/spark \
        --target-pyspark /path/to/our/python \
        test_connect_column

    # Run all modules from a manifest file (as CI does).
    python3 scripts/run_official_connect_tests.py \
        --spark ~/workspace/origin/spark \
        --target-pyspark /path/to/our/python \
        --modules-file tests/official/connect_test_modules.txt

The test files are discovered under
    <spark>/python/pyspark/**/tests/connect/**/test_*.py

See docs/design/OFFICIAL_TESTS.md for CI integration details.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def find_test(spark_py: Path, name: str) -> Path | None:
    if name.endswith(".py"):
        name = name[:-3]
    matches = list((spark_py / "pyspark").rglob(f"{name}.py"))
    matches = [m for m in matches if "tests/connect" in str(m).replace(os.sep, "/")]
    return matches[0] if matches else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spark", required=True, help="Spark checkout root")
    ap.add_argument(
        "--remote",
        default="sc://localhost:15002",
        help="Spark Connect server endpoint (default: sc://localhost:15002)",
    )
    ap.add_argument(
        "--target-pyspark",
        default=None,
        help="dir to prepend to PYTHONPATH (our replacement package)",
    )
    ap.add_argument(
        "--modules-file",
        default=None,
        help="file containing list of test modules (one per line, # for comments)",
    )
    ap.add_argument(
        "names",
        nargs="*",
        help="test module names, e.g. pyspark.sql.tests.connect.test_connect_column",
    )
    args = ap.parse_args()

    spark = Path(os.path.expanduser(args.spark))
    spark_py = spark / "python"

    # Collect test modules to run
    test_modules = []
    if args.modules_file:
        # Load modules from file
        modules_file = Path(os.path.expanduser(args.modules_file))
        if not modules_file.exists():
            print(f"!! modules file not found: {args.modules_file}")
            return 2
        with open(modules_file) as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                test_modules.append(line)
    else:
        # Use positional arguments
        test_modules = args.names

    if not test_modules:
        print("ERROR: no test modules specified (use positional args or --modules-file)")
        ap.print_help()
        return 2

    env = dict(os.environ)
    env["SPARK_CONNECT_TESTING_REMOTE"] = args.remote
    env["SPARK_TESTING"] = "1"
    # Force the pure-client path: no local JVM.
    # This makes is_remote_only() return True, preventing JVM startup.
    env["SPARK_SKIP_CONNECT_COMPAT_TESTS"] = "1"
    pp = []
    if args.target_pyspark:
        pp.append(os.path.expanduser(args.target_pyspark))
    pp.append(str(spark_py))
    pp.append(env.get("PYTHONPATH", ""))
    env["PYTHONPATH"] = os.pathsep.join(p for p in pp if p)

    rc = 0
    passed = 0
    failed = 0
    not_found = 0

    for name in test_modules:
        tf = find_test(spark_py, name)
        if tf is None:
            print(f"!! test module not found: {name}")
            not_found += 1
            rc = 2
            continue
        print(f"==> {name}  ({tf.relative_to(spark_py)})")
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tf)],
            env=env,
        )
        if r.returncode == 0:
            passed += 1
        else:
            failed += 1
        rc = rc or r.returncode

    # Print summary
    total = passed + failed + not_found
    print(f"\n{'=' * 70}")
    print(f"Test Summary: {total} modules")
    print(f"  Passed:    {passed}")
    print(f"  Failed:    {failed}")
    print(f"  Not found: {not_found}")
    print(f"{'=' * 70}")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
