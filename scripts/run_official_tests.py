#!/usr/bin/env python3
"""Parity gate: run the official Spark Connect test suite through OUR Rust client.

Now that the client is at full parity, we no longer run the reference client on every
file to compute a per-file baseline (that doubled CI time). Instead we run ONLY our
client (via the Rust transport-injection plugin, scripts/rust_transport_plugin.py) and
require every test to pass, except a checked-in manifest of *known environmental
failures* (scripts/parity_known_failures.txt) - tests the reference client also fails
in this single-node pure-Connect CI environment. Those are deselected for our run too.

Any test that fails and is NOT in the manifest is a genuine problem: either a client
regression, or a new environmental failure. Regenerate the manifest with
scripts/gen_parity_skiplist.py (which runs the reference client) to reclassify, and
commit the update if it is environmental.

Usage:
    RUST_PYSPARK_SO=/repo/python/pyspark/_pyspark.so \
    SPARK_CONNECT_TESTING_REMOTE=sc://localhost:15002 \
    python3 scripts/run_official_tests.py --spark ~/spark-source [--jobs 1]
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
DEFAULT_MANIFEST = REPO / "scripts" / "parity_known_failures.txt"

# The only files whose failures are genuinely event-timing-driven: they wait on
# real server-pushed events (streaming query progress/listener callbacks, observed
# metrics) and can transiently exceed the per-file cap late in a long serial run on
# a resource-starved single-node server, yet pass reliably in isolation. ONLY these
# get retried; every other file is expected to be deterministic, so a
# non-deterministic client bug there fails the gate on its first occurrence instead
# of having to reproduce on every one of N attempts (which would let a 50%-flaky bug
# through most runs). Basenames, matched exactly.
FLAKY_FILES = {
    "test_parity_streaming.py",
    "test_parity_foreach_batch.py",
    "test_parity_foreach.py",
    "test_parity_listener.py",
    "test_parity_observation.py",
}

_FAIL_RE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)")
_COUNT_RE = {k: re.compile(rf"(\d+) {k}\b") for k in ("passed", "failed", "error", "skipped")}


def discover(spark_py: Path):
    # Scoped to sql/tests/connect (mirrors gen_parity_skiplist and the old gate): the
    # ml/pandas connect suites need extra deps (torch, sklearn, ...) CI does not install.
    out = []
    for p in sorted(spark_py.rglob("test_*.py")):
        if "/sql/tests/connect/" in str(p).replace(os.sep, "/"):
            out.append(p)
    return out


def load_manifest(path: Path):
    """Return (per_file_deselect, whole_file_skip).

    per_file_deselect maps a file-relative path -> list of "Class::test" suffixes to
    deselect; whole_file_skip is the set of file-relative paths to skip entirely
    (manifest entries with no "::", i.e. the reference could not even run the file).
    """
    per_file, whole_file = {}, set()
    if not path.exists():
        return per_file, whole_file
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "::" in line:
            relpath, suffix = line.split("::", 1)
            per_file.setdefault(relpath, []).append(suffix)
        else:
            whole_file.add(line)
    return per_file, whole_file


def parse_counts(text: str):
    c = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
    for line in reversed(text.strip().splitlines()):
        if any(w in line for w in (" passed", " failed", " error", " skipped")):
            for k, rx in _COUNT_RE.items():
                m = rx.search(line)
                if m:
                    c[k] = int(m.group(1))
            if any(w in line for w in ("passed", "failed", "error")):
                break
    return c


def run_ours(
    test_file: Path, spark_py: Path, remote: str, deselect, timeout: int, select_only=False
):
    env = dict(os.environ)
    env["SPARK_CONNECT_TESTING_REMOTE"] = remote
    env["SPARK_TESTING"] = "1"
    env["SPARK_SKIP_CONNECT_COMPAT_TESTS"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(REPO / "scripts"), str(spark_py)])
    env["RUST_PYSPARK_SO"] = os.environ["RUST_PYSPARK_SO"]
    # Deselection is done by the transport plugin (pytest_collection_modifyitems) via
    # this env var - matching node-id *suffixes* - rather than pytest's own --deselect.
    # --deselect compares against a node id relative to pytest's rootdir, which differs
    # between a Spark *dist* (rootdir = python/, id = `pyspark/...`) and a *source clone*
    # (rootdir may be the repo root, id = `python/pyspark/...`); an absolute or wrong-
    # prefix path silently matches nothing and voids the whole manifest. Suffix matching
    # in the plugin is rootdir-independent. Run from spark_py (as Apache's ./python/
    # run-tests does) with the file passed relative to it.
    rel_file = test_file.relative_to(spark_py).as_posix()
    # select_only: run ONLY the listed tests (the drift check); otherwise deselect them.
    env["RUST_PARITY_SELECT_ONLY" if select_only else "RUST_PARITY_DESELECT"] = "\n".join(deselect)
    args = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-rfE",
        "--tb=short",
        "-p",
        "no:cacheprovider",
        "-p",
        "rust_transport_plugin",
    ]
    args.append(rel_file)
    try:
        r = subprocess.run(
            args, env=env, cwd=str(spark_py), capture_output=True, text=True, timeout=timeout
        )
        text = r.stdout + "\n" + r.stderr
    except subprocess.TimeoutExpired:
        return {"passed": 0, "failed": 0, "error": 0, "skipped": 0, "timeout": True}, []
    counts = parse_counts(text)
    failed_ids = [
        m.group(1) for m in map(_FAIL_RE.match, (ln.strip() for ln in text.splitlines())) if m
    ]
    return counts, failed_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spark", required=True, help="Spark source tree (has python/)")
    ap.add_argument(
        "--remote", default=os.environ.get("SPARK_CONNECT_TESTING_REMOTE", "sc://localhost:15002")
    )
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=360)
    ap.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Max extra attempts for the event-timing-driven files ONLY (see "
        "FLAKY_FILES): the streaming-query-listener/observation tests wait on real "
        "server-pushed events and can transiently exceed the per-file cap late in a "
        "long serial run on a resource-starved single-node server, yet pass reliably "
        "in isolation. All other files are treated as deterministic and get zero "
        "retries, so a non-deterministic client bug there is not masked by re-running.",
    )
    ap.add_argument(
        "--no-drift-check",
        action="store_true",
        help="Skip the post-run check that re-runs skiplisted tests to report any that "
        "now pass (candidates to remove from the manifest).",
    )
    args = ap.parse_args()

    if not os.environ.get("RUST_PYSPARK_SO"):
        print("!! set RUST_PYSPARK_SO to the built extension")
        return 2
    spark_py = Path(os.path.expanduser(args.spark)) / "python"
    files = discover(spark_py)
    if not files:
        print("!! no connect test files found under", spark_py)
        return 2

    per_file, whole_file = load_manifest(Path(args.manifest))
    n_skip = sum(len(v) for v in per_file.values())
    print(
        f"Discovered {len(files)} connect test files; manifest skips "
        f"{n_skip} tests + {len(whole_file)} whole files.\n",
        flush=True,
    )

    def work(tf: Path):
        rel = tf.relative_to(spark_py).as_posix()
        if rel in whole_file:
            return tf.name, None, [], True, 1  # skipped whole file
        deselect = per_file.get(rel, [])
        counts, failed_ids = run_ours(tf, spark_py, args.remote, deselect, args.timeout)
        # Retry ONLY the event-timing-driven files (see FLAKY_FILES): for them a flake
        # clears on a fresh run while a genuine failure persists. Deterministic files
        # are NOT retried, so a non-deterministic bug there surfaces immediately rather
        # than needing to reproduce on every attempt.
        retries = args.retries if tf.name in FLAKY_FILES else 0
        attempts = 1
        while (
            counts.get("failed", 0) + counts.get("error", 0) or counts.get("timeout")
        ) and attempts <= retries:
            attempts += 1
            counts, failed_ids = run_ours(tf, spark_py, args.remote, deselect, args.timeout)
        return tf.name, counts, failed_ids, False, attempts

    failures = []  # (file, [nodeids])
    done = 0
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs))
    for name, counts, failed_ids, skipped_file, attempts in ex.map(work, files):
        done += 1
        if skipped_file:
            print(f"[{done}/{len(files)}] skip(env)  {name}", flush=True)
            continue
        bad = counts.get("failed", 0) + counts.get("error", 0) or counts.get("timeout")
        tag = "FAIL" if bad else "ok"
        retry_note = f" (after {attempts} attempts)" if attempts > 1 else ""
        print(
            f"[{done}/{len(files)}] {tag:<10} {name:<48} "
            f"p={counts['passed']} f={counts['failed']} e={counts['error']} "
            f"skip={counts['skipped']}"
            + (" TIMEOUT" if counts.get("timeout") else "")
            + retry_note,
            flush=True,
        )
        if bad:
            failures.append((name, failed_ids or ["<timeout>"]))
    ex.shutdown()

    # Drift check: re-run the skiplisted tests (selecting ONLY them) and report any
    # that now PASS. Without this the manifest can only rot toward less coverage - a
    # test stays skipped forever and a later regression in a skiplisted area is
    # invisible. This never fails the gate (a genuine environmental skip keeps
    # failing here); it only surfaces entries to remove so the list stays honest.
    if not args.no_drift_check:
        now_passing = []
        files_by_rel = {tf.relative_to(spark_py).as_posix(): tf for tf in files}
        for rel, suffixes in sorted(per_file.items()):
            tf = files_by_rel.get(rel)
            if tf is None:
                continue
            counts, _ = run_ours(
                tf, spark_py, args.remote, suffixes, args.timeout, select_only=True
            )
            # All selected tests passed (none failed/errored, and some ran).
            if counts.get("passed", 0) and not (
                counts.get("failed", 0) + counts.get("error", 0) or counts.get("timeout")
            ):
                now_passing.append((rel, counts.get("passed", 0), counts.get("skipped", 0)))
        if now_passing:
            print(
                "\nDrift check: skiplisted tests that now PASS (candidates to remove "
                "from\nscripts/parity_known_failures.txt so the gate exercises them):"
            )
            for rel, npass, nskip in now_passing:
                print(f"  {rel}: {npass} passed, {nskip} skipped")

    print(f"\n{len(failures)} file(s) with unexpected failures out of {len(files)}.")
    if failures:
        print(
            "\nUnexpected failures (a regression, or a new environmental failure to add\n"
            "to scripts/parity_known_failures.txt via scripts/gen_parity_skiplist.py):"
        )
        for _name, ids in failures:
            for nid in ids:
                print(f"  {nid}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
