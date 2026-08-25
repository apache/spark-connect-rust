#!/usr/bin/env python3
"""Generate the parity skip manifest of known *environmental* test failures.

The parity gate (scripts/run_official_tests.py) runs the official Spark Connect
Python test suite through our Rust client and requires every test to pass except
those listed in the manifest. A test belongs in the manifest only when the
*reference* pyspark client also fails it in this single-node, pure-Connect CI
environment (e.g. features a real cluster / server-side Python worker setup would
provide but this harness cannot): such failures are environmental, not client
regressions, so they are skipped for both clients.

This script runs the REFERENCE client (stock pyspark from the Spark source tree,
no transport injection) across every official connect test file and records each
FAILED/ERROR test id, with the reason pytest reported, into the manifest. Re-run
it (and commit the result) whenever the pinned Spark version bumps.

Usage:
    SPARK_CONNECT_TESTING_REMOTE=sc://localhost:15002 \
    python3 scripts/gen_parity_skiplist.py --spark ~/spark-source \
        [--out scripts/parity_known_failures.txt] [--timeout 360]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "scripts" / "parity_known_failures.txt"

# A pytest short-summary failure line: "FAILED path::Class::test - AssertionError: ..".
_FAIL_RE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)(?:\s+-\s+(.*))?$")


def discover(spark_py: Path):
    """The official SQL connect test files, in stable order (mirrors run_official_tests).

    Scoped to ``sql/tests/connect`` to match the parity gate: the ml/pandas connect
    suites need extra deps (torch, sklearn, ...) the CI image does not install.
    """
    out = []
    for p in sorted(spark_py.rglob("test_*.py")):
        s = str(p).replace(os.sep, "/")
        if "/sql/tests/connect/" in s:
            out.append(p)
    return out


def reference_failures(test_file: Path, spark_py: Path, remote: str, timeout: int):
    """Run the reference client on one file; return {nodeid: reason} for FAILED/ERROR.

    The nodeid is normalized to be file-relative (path from the spark python root)
    so the manifest is independent of where the Spark source tree lives.
    """
    env = dict(os.environ)
    env["SPARK_CONNECT_TESTING_REMOTE"] = remote
    env["SPARK_TESTING"] = "1"
    env["SPARK_SKIP_CONNECT_COMPAT_TESTS"] = "1"
    env["PYTHONPATH"] = str(spark_py)
    # -rfE prints a short-summary line per failure/error with its reason; --tb=no keeps
    # output compact; -p no:cacheprovider avoids writing a .pytest_cache.
    args = [
        sys.executable, "-m", "pytest", "-q", "-rfE", "--tb=no",
        "-p", "no:cacheprovider", str(test_file),
    ]
    try:
        r = subprocess.run(args, env=env, capture_output=True, text=True, timeout=timeout)
        text = r.stdout + "\n" + r.stderr
    except subprocess.TimeoutExpired:
        # A file the reference can't even finish here is wholly environmental; mark it.
        rel = test_file.relative_to(spark_py).as_posix()
        return {rel: "reference client timed out (environmental)"}, True

    failures = {}
    for line in text.splitlines():
        m = _FAIL_RE.match(line.strip())
        if not m:
            continue
        nodeid = m.group(1)
        reason = (m.group(2) or "reference also fails (environmental)").strip()
        # Normalize the path portion to be relative to the spark python root.
        parts = nodeid.split("::", 1)
        try:
            relpath = Path(parts[0]).resolve().relative_to(spark_py).as_posix()
        except Exception:
            relpath = parts[0]
        nodeid = relpath + ("::" + parts[1] if len(parts) > 1 else "")
        failures[nodeid] = reason
    return failures, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spark", required=True, help="Spark source tree (has python/)")
    ap.add_argument("--remote", default=os.environ.get(
        "SPARK_CONNECT_TESTING_REMOTE", "sc://localhost:15002"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--timeout", type=int, default=360)
    args = ap.parse_args()

    spark_py = Path(os.path.expanduser(args.spark)) / "python"
    files = discover(spark_py)
    if not files:
        print("!! no connect test files found under", spark_py)
        return 2
    print(f"Scanning {len(files)} connect test files with the reference client...\n",
          flush=True)

    all_failures = {}
    for i, tf in enumerate(files, 1):
        fails, timed_out = reference_failures(tf, spark_py, args.remote, args.timeout)
        note = " TIMEOUT" if timed_out else ""
        print(f"[{i}/{len(files)}] {tf.name:<52} ref_fail={len(fails)}{note}", flush=True)
        all_failures.update(fails)

    out = Path(args.out)
    with out.open("w") as f:
        f.write("# Known environmental failures in the official Spark Connect test suite.\n")
        f.write("# These tests FAIL with the *reference* pyspark client too, in this\n")
        f.write("# single-node pure-Connect CI environment, so they are skipped for our\n")
        f.write("# client as well (they are not client regressions). One `nodeid  # reason`\n")
        f.write("# per line; regenerate with scripts/gen_parity_skiplist.py on a version bump.\n")
        for nodeid in sorted(all_failures):
            f.write(f"{nodeid}  # {all_failures[nodeid]}\n")
    print(f"\nWrote {len(all_failures)} known-failure entries to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
