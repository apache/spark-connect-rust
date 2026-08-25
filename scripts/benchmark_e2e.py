#!/usr/bin/env python3
"""End-to-end performance benchmark: our Rust-backed client vs the reference pyspark client.

Both clients talk to the SAME Spark Connect server (default sc://localhost:15002),
so the only variable is the client implementation (plan building, gRPC transport,
Arrow decoding). We time a set of representative operations, run each many times,
and report median/mean latency plus the speedup ratio.

Usage:
    # Benchmark the REFERENCE client (pyspark on PYTHONPATH, our package absent):
    PYTHONPATH=/path/to/spark/python \
        python3 scripts/benchmark_e2e.py --label reference --out /tmp/bench_reference.json

    # Benchmark OUR client (our python/ prepended so it shadows pyspark):
    PYTHONPATH=./python:/path/to/spark/python \
        python3 scripts/benchmark_e2e.py --label ours --out /tmp/bench_ours.json

    # Compare two result files:
    python3 scripts/benchmark_e2e.py --compare /tmp/bench_reference.json /tmp/bench_ours.json

The two-process design is deliberate: each client is imported in a clean interpreter
so there is no import-order ambiguity about which `pyspark` package wins.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from typing import Callable


def _fmt_ms(x: float) -> str:
    return f"{x * 1000:.2f}ms"


def make_workloads(spark):
    """Return a dict {name: (setup, run)} of benchmark cases.

    `setup` runs once (not timed) and returns state; `run(state)` is the timed body
    and must force materialization (collect/count/toPandas) so we measure e2e latency.
    """
    from pyspark.sql import functions as F

    def w_range_collect_setup():
        return spark.range(0, 100_000)

    def w_range_collect(df):
        return df.collect()

    def w_select_filter_setup():
        return spark.range(0, 500_000)

    def w_select_filter(df):
        return df.select((F.col("id") * 2).alias("x")).filter(F.col("x") % 3 == 0).count()

    def w_groupby_agg_setup():
        return spark.range(0, 500_000).select((F.col("id") % 100).alias("k"), F.col("id"))

    def w_groupby_agg(df):
        return df.groupBy("k").agg(F.sum("id"), F.avg("id"), F.count("id")).collect()

    def w_withcolumns_setup():
        return spark.range(0, 200_000)

    def w_withcolumns(df):
        return (
            df.withColumn("a", F.col("id") + 1)
            .withColumn("b", F.col("a") * 2)
            .withColumn("c", F.sqrt(F.col("b").cast("double")))
            .filter(F.col("c") > 1.0)
            .count()
        )

    def w_join_setup():
        left = spark.range(0, 50_000).select(F.col("id").alias("k"), F.col("id").alias("lv"))
        right = spark.range(0, 50_000).select(F.col("id").alias("k"), (F.col("id") * 3).alias("rv"))
        return (left, right)

    def w_join(state):
        left, right = state
        return left.join(right, on="k").select("k", "lv", "rv").count()

    def w_collect_topandas_setup():
        return spark.range(0, 100_000).select(
            F.col("id"), (F.col("id") * 1.5).alias("d"), F.col("id").cast("string").alias("s")
        )

    def w_collect_topandas(df):
        # toPandas() is not yet implemented in our client; collect() exercises the
        # same Arrow-decode e2e path and works on both clients.
        return df.collect()

    def w_many_small_setup():
        # Latency-bound: many tiny queries in a row.
        return spark

    def w_many_small(sp):
        total = 0
        for i in range(50):
            total += sp.range(0, 10).filter(F.col("id") > i % 5).count()
        return total

    return {
        "range_collect_100k": (w_range_collect_setup, w_range_collect),
        "select_filter_count_500k": (w_select_filter_setup, w_select_filter),
        "groupby_agg_500k": (w_groupby_agg_setup, w_groupby_agg),
        "withcolumns_chain_200k": (w_withcolumns_setup, w_withcolumns),
        "join_count_50k": (w_join_setup, w_join),
        "collect_wide_100k": (w_collect_topandas_setup, w_collect_topandas),
        "many_small_queries_50x": (w_many_small_setup, w_many_small),
    }


def time_case(setup: Callable, run: Callable, iters: int, warmup: int) -> list[float]:
    state = setup()
    for _ in range(warmup):
        run(state)
    samples = []
    for _ in range(iters):
        gc.collect()
        t0 = time.perf_counter()
        run(state)
        samples.append(time.perf_counter() - t0)
    return samples


def run_benchmark(
    remote: str, label: str, iters: int, warmup: int, out: str | None, only: str | None = None
) -> int:
    # Report which pyspark actually got imported - proves which client is under test.
    import pyspark
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.remote(remote).getOrCreate()

    workloads = make_workloads(spark)
    if only:
        if only not in workloads:
            print(f"!! unknown workload: {only}")
            return 2
        workloads = {only: workloads[only]}
    results = {}
    print(f"# client label : {label}")
    print(f"# pyspark from : {os.path.dirname(pyspark.__file__)}")
    print(f"# remote       : {remote}")
    print(f"# iters={iters} warmup={warmup}\n")
    print(f"{'workload':<30} {'median':>10} {'mean':>10} {'min':>10} {'stdev':>10}")
    print("-" * 74)
    for name, (setup, run) in workloads.items():
        try:
            samples = time_case(setup, run, iters, warmup)
        except Exception as e:  # noqa: BLE001
            print(f"{name:<30} ERROR: {type(e).__name__}: {e}")
            results[name] = {"error": f"{type(e).__name__}: {e}"}
            continue
        median = statistics.median(samples)
        mean = statistics.fmean(samples)
        mn = min(samples)
        stdev = statistics.pstdev(samples)
        results[name] = {
            "median_s": median,
            "mean_s": mean,
            "min_s": mn,
            "stdev_s": stdev,
            "samples": samples,
        }
        print(
            f"{name:<30} {_fmt_ms(median):>10} {_fmt_ms(mean):>10} "
            f"{_fmt_ms(mn):>10} {_fmt_ms(stdev):>10}"
        )

    spark.stop()

    payload = {
        "label": label,
        "remote": remote,
        "pyspark_path": os.path.dirname(pyspark.__file__),
        "iters": iters,
        "warmup": warmup,
        "results": results,
    }
    if out:
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {out}")
    return 0


def compare(ref_path: str, ours_path: str) -> int:
    with open(ref_path) as f:
        ref = json.load(f)
    with open(ours_path) as f:
        ours = json.load(f)

    print(f"\n{'=' * 82}")
    print(f"COMPARISON  reference={ref['label']}  vs  ours={ours['label']}")
    print(f"  reference pyspark: {ref.get('pyspark_path')}")
    print(f"  ours      pyspark: {ours.get('pyspark_path')}")
    print(f"{'=' * 82}")
    print(f"{'workload':<30} {'ref median':>12} {'our median':>12} {'speedup':>10} {'':>6}")
    print("-" * 82)
    names = list(ref["results"].keys())
    speedups = []
    for name in names:
        r = ref["results"].get(name, {})
        o = ours["results"].get(name, {})
        if "error" in r or "error" in o or "median_s" not in r or "median_s" not in o:
            note = r.get("error") or o.get("error") or "missing"
            print(f"{name:<30} {'--':>12} {'--':>12} {'--':>10}  {note}")
            continue
        rm, om = r["median_s"], o["median_s"]
        speedup = rm / om if om > 0 else float("inf")
        speedups.append(speedup)
        flag = "faster" if speedup >= 1.0 else "SLOWER"
        print(f"{name:<30} {_fmt_ms(rm):>12} {_fmt_ms(om):>12} {speedup:>9.2f}x  {flag}")
    print("-" * 82)
    if speedups:
        gm = statistics.geometric_mean(speedups)
        print(f"{'geomean speedup (ours vs reference)':<30} {'':>12} {'':>12} {gm:>9.2f}x")
        print("\n  >1.0x means our Rust client is faster; <1.0x means slower.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--remote", default=os.environ.get("SPARK_CONNECT_TESTING_REMOTE", "sc://localhost:15002")
    )
    ap.add_argument("--label", default="client")
    ap.add_argument("--iters", type=int, default=25)
    # Several workloads (join, aggregates) have a noisy warmup curve on the server;
    # 3 warmups was too few and produced misleading medians. 8 stabilizes them.
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--only", default=None, help="run just this one workload (for per-process isolation)"
    )
    ap.add_argument(
        "--isolate",
        action="store_true",
        help="run EACH workload in its own fresh subprocess, then aggregate "
        "into --out (avoids cross-workload heap/GC contamination)",
    )
    ap.add_argument(
        "--compare",
        nargs=2,
        metavar=("REFERENCE_JSON", "OURS_JSON"),
        help="compare two result files instead of running a benchmark",
    )
    args = ap.parse_args()

    if args.compare:
        return compare(args.compare[0], args.compare[1])
    if args.isolate:
        return run_isolated(args)
    return run_benchmark(args.remote, args.label, args.iters, args.warmup, args.out, args.only)


def run_isolated(args) -> int:
    """Run each workload in a fresh subprocess so no workload's heap state biases another."""
    import tempfile

    names = list(make_workloads(_DummySpark()).keys())
    merged = {
        "label": args.label,
        "remote": args.remote,
        "iters": args.iters,
        "warmup": args.warmup,
        "results": {},
        "isolated": True,
    }
    for name in names:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            tmp = tf.name
        cmd = [
            sys.executable,
            __file__,
            "--remote",
            args.remote,
            "--label",
            args.label,
            "--iters",
            str(args.iters),
            "--warmup",
            str(args.warmup),
            "--only",
            name,
            "--out",
            tmp,
        ]
        subprocess.run(cmd, check=False)
        try:
            with open(tmp) as f:
                sub = json.load(f)
            merged["results"].update(sub.get("results", {}))
            merged["pyspark_path"] = sub.get("pyspark_path")
        except (OSError, json.JSONDecodeError):
            merged["results"][name] = {"error": "subprocess produced no result"}
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    if args.out:
        with open(args.out, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"\nwrote {args.out} (isolated)")
    return 0


class _DummySpark:
    """Lets make_workloads() build its dict without a live session (we only need keys)."""

    def __getattr__(self, _):
        return lambda *a, **k: self


if __name__ == "__main__":
    raise SystemExit(main())
