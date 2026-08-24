#!/usr/bin/env python3
"""Reference-vs-ours parity gate for the official Spark Connect test suite.

This encodes the correct definition of "done": our Rust-backed client must match
the *reference* pyspark client's pass/skip/fail profile against the SAME Spark
Connect server. It runs each official connect test file twice - once with the
reference client, once with ours - parses pytest's summary, and reports per-file
diffs. A file is a REGRESSION only when a test the reference client passes fails
(or errors) with our client. Tests that the reference client also skips/fails are
environmental (no local JVM, single-node server, unsupported server type, etc.)
and are NOT counted against us.

Discovery: all `test_*.py` under <spark>/python/pyspark/**/tests/connect/**.

Exit code: 0 if there are no regressions, 1 otherwise. `--baseline-only` and
`--ours-only` record one side to JSON without judging.

Usage:
    python3 scripts/parity_baseline_diff.py \
        --spark ~/workspace/origin/spark \
        --target-pyspark ./python \
        --remote sc://localhost:15002 \
        [--filter test_parity_] [--jobs 1] [--out /tmp/parity_diff.json]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# pytest summary line, e.g. "5 passed, 2 skipped, 1 failed in 3.2s"
_SUMMARY_RE = re.compile(
    r"(?:(\d+) failed)?.*?(?:(\d+) passed)?.*?(?:(\d+) skipped)?.*?(?:(\d+) error)?",
)


def discover(spark_py: Path, name_filter: str | None, path_contains: str | None) -> list[Path]:
    files = []
    for p in sorted(spark_py.rglob("test_*.py")):
        s = str(p).replace(os.sep, "/")
        if "/tests/connect/" not in s:
            continue
        if name_filter and name_filter not in p.name:
            continue
        if path_contains and path_contains not in s:
            continue
        files.append(p)
    return files


def parse_summary(text: str) -> dict[str, int]:
    """Extract counts from the last pytest summary line."""
    counts = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
    # Look at the tail; pytest prints the summary on the final non-empty lines.
    for line in reversed(text.strip().splitlines()):
        line = line.strip().strip("=").strip()
        if not line:
            continue
        found = False
        for key in ("passed", "failed", "skipped", "error", "errors"):
            m = re.search(rf"(\d+) {key}\b", line)
            if m:
                counts["error" if key == "errors" else key] = int(m.group(1))
                found = True
        if found and ("passed" in line or "failed" in line or "error" in line or "skipped" in line):
            break
    return counts


def run_one(test_file: Path, spark_py: Path, remote: str, target_pyspark: str | None) -> dict:
    env = dict(os.environ)
    env["SPARK_CONNECT_TESTING_REMOTE"] = remote
    env["SPARK_TESTING"] = "1"
    env["SPARK_SKIP_CONNECT_COMPAT_TESTS"] = "1"  # force remote-only (no local JVM)
    pp = []
    if target_pyspark:
        pp.append(os.path.abspath(os.path.expanduser(target_pyspark)))
    pp.append(str(spark_py))
    if env.get("PYTHONPATH"):
        pp.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pp)

    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(test_file)],
        env=env,
        capture_output=True,
        text=True,
        # A single connect test file should finish in well under this. If it does not,
        # treat it as a hang (recorded as a timeout) rather than blocking the whole sweep
        # for 20 minutes on one file.
        timeout=int(os.environ.get("PARITY_FILE_TIMEOUT", "300")),
    )
    out = r.stdout + "\n" + r.stderr
    counts = parse_summary(out)
    counts["returncode"] = r.returncode
    return counts


def collect_side(files, spark_py, remote, target, jobs) -> dict[str, dict]:
    results: dict[str, dict] = {}

    def work(tf: Path):
        try:
            return tf.name, run_one(tf, spark_py, remote, target)
        except subprocess.TimeoutExpired:
            return tf.name, {
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "error": 0,
                "returncode": -1,
                "timeout": True,
            }

    if jobs <= 1:
        for tf in files:
            name, res = work(tf)
            results[name] = res
            print(f"  {name:<55} {_short(res)}")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
            for name, res in ex.map(work, files):
                results[name] = res
                print(f"  {name:<55} {_short(res)}")
    return results


def _short(res: dict) -> str:
    if res.get("timeout"):
        return "TIMEOUT"
    return f"pass={res['passed']} fail={res['failed']} skip={res['skipped']} err={res['error']}"


def _is_regression(r: dict, o: dict) -> bool:
    """A file regresses when ours fails/errors more than reference, passes fewer, or times out."""
    if o.get("timeout"):
        return True
    return (o["failed"] + o["error"]) > (r["failed"] + r["error"]) or o["passed"] < r["passed"]


def run_interleaved(files, spark_py, remote, target, jobs) -> dict:
    """Run reference AND ours per file, diffing immediately so regressions stream out.

    This gives incremental, actionable results (fix the first regression without waiting
    for the whole suite), unlike running one full side and then the other.
    """

    def work(tf: Path):
        def one(t):
            try:
                return run_one(tf, spark_py, remote, t)
            except subprocess.TimeoutExpired:
                return {
                    "passed": 0,
                    "failed": 0,
                    "skipped": 0,
                    "error": 0,
                    "returncode": -1,
                    "timeout": True,
                }

        return tf.name, one(None), one(target)

    payload = {"reference": {}, "ours": {}, "regressions": []}
    n_reg = 0
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs))
    futures = [executor.submit(work, tf) for tf in files]
    done = 0
    for fut in concurrent.futures.as_completed(futures):
        name, r, o = fut.result()
        payload["reference"][name] = r
        payload["ours"][name] = o
        done += 1
        if _is_regression(r, o):
            n_reg += 1
            payload["regressions"].append(name)
            print(f"[{done}/{len(files)}] REGRESSION {name}")
            print(f"      reference: {_short(r)}")
            print(f"      ours     : {_short(o)}")
        else:
            print(f"[{done}/{len(files)}] ok {name:<45} ours={_short(o)}", flush=True)
    executor.shutdown()
    print(f"\n{n_reg} regression file(s) out of {len(files)}.")
    if payload["regressions"]:
        print("Regressions:", ", ".join(sorted(payload["regressions"])))
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spark", required=True)
    ap.add_argument("--target-pyspark", default=None)
    ap.add_argument("--remote", default="sc://localhost:15002")
    ap.add_argument(
        "--filter", default=None, help="only run test files whose name contains this substring"
    )
    ap.add_argument(
        "--path-contains",
        default=None,
        help="only run test files whose full path contains this substring "
        "(e.g. 'sql/tests/connect' for just the SQL connect tests)",
    )
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--baseline-only", action="store_true", help="run only the reference client and record it"
    )
    ap.add_argument("--ours-only", action="store_true", help="run only our client and record it")
    args = ap.parse_args()

    spark_py = Path(os.path.expanduser(args.spark)) / "python"
    files = discover(spark_py, args.filter, args.path_contains)
    if not files:
        print(f"!! no connect test files found under {spark_py} (filter={args.filter})")
        return 2
    print(f"Discovered {len(files)} connect test files (filter={args.filter or 'none'}).\n")

    payload: dict = {"remote": args.remote, "files": len(files)}
    rc = 0

    if args.baseline_only or args.ours_only:
        # Single-side recording (no diff).
        target = args.target_pyspark if args.ours_only else None
        side = "ours" if args.ours_only else "reference"
        print(f"== {side} ==")
        payload[side] = collect_side(files, spark_py, args.remote, target, args.jobs)
    else:
        print(
            f"{'=' * 82}\nPARITY DIFF (streaming; regression = reference passes but ours does not)\n{'=' * 82}"
        )
        result = run_interleaved(files, spark_py, args.remote, args.target_pyspark, args.jobs)
        payload.update(result)
        rc = 1 if result["regressions"] else 0

    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {args.out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
