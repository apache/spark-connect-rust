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
"""Comprehensive Spark Connect client benchmark: Rust drop-in vs reference PySpark.

Both clients talk to the SAME Spark Connect server, so the only variable is the
client implementation (plan building, gRPC transport, Arrow decoding, Row/pandas
materialization). The harness measures the metrics called for in the benchmark
spec, to the extent a single-host loopback setup can measure them honestly:

  core metrics : rows/sec, GB/sec, CPU/GB, p50/p99/p999 latency, client CPU
                 (== "Rust CPU" for ours, "Python CPU" for reference), server
                 JVM CPU (local proxy for "executor CPU"), memory (peak RSS),
                 cold-start latency
  transport    : payload-size sweep 64 KB .. 16 MB (rows x cols x 8B)
  execution    : row-count sweep, concurrent client-process counts, UDF classes

Which client runs is decided entirely by how THIS process is launched:
  ours : RUST_PYSPARK_SO=<repo>/python/pyspark/_pyspark.so PYTHONPATH=<repo>/python
  ref  : neither set (installed pyspark-client wins)
Subprocesses spawned for cold-start / concurrency inherit the same environment,
so they exercise the same client.

Run one client end-to-end and write JSON:
  <env for client> python scripts/benchmark_suite.py --label ours --out /tmp/ours.json
Then compare + render the report section:
  python scripts/benchmark_suite.py --report /tmp/ref.json /tmp/ours.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import time

REMOTE_DEFAULT = os.environ.get("SPARK_CONNECT_TESTING_REMOTE", "sc://localhost:15002")

# Transport payload sweep. Schema is NCOLS 64-bit longs, so one row is NCOLS*8
# logical Arrow body bytes; row counts are chosen to hit each target size.
NCOLS = 8
BYTES_PER_ROW = NCOLS * 8
SIZE_TARGETS = [
    ("64KB", 64 * 1024),
    ("256KB", 256 * 1024),
    ("1MB", 1024 * 1024),
    ("4MB", 4 * 1024 * 1024),
    ("8MB", 8 * 1024 * 1024),
    ("16MB", 16 * 1024 * 1024),
]
ROWCOUNTS = [10_000, 100_000, 1_000_000]
CONCURRENCY_LEVELS = [1, 2, 4, 8]


def pct(samples: list[float], q: float) -> float:
    """Percentile via linear interpolation (q in [0,100])."""
    if not samples:
        return float("nan")
    s = sorted(samples)
    if len(s) == 1:
        return s[0]
    r = (q / 100.0) * (len(s) - 1)
    lo = int(r)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (r - lo)


def summarize(samples: list[float]) -> dict:
    return {
        "n": len(samples),
        "p50_s": pct(samples, 50),
        "p95_s": pct(samples, 95),
        "p99_s": pct(samples, 99),
        "p999_s": pct(samples, 99.9),
        "min_s": min(samples),
        "mean_s": statistics.fmean(samples),
        "stdev_s": statistics.pstdev(samples),
    }


class CpuSampler:
    """Samples user+system CPU seconds for this process and (optionally) the
    Spark server JVM, so we can attribute client CPU and a local proxy for
    executor CPU across a timed region."""

    def __init__(self, remote: str):
        self.server_pid = _find_server_pid(remote)
        try:
            import psutil  # noqa: F401

            self.psutil = __import__("psutil")
        except Exception:
            self.psutil = None
        self.self_proc = self.psutil.Process() if self.psutil else None
        self.srv_proc = None
        if self.psutil and self.server_pid:
            try:
                self.srv_proc = self.psutil.Process(self.server_pid)
            except Exception:
                self.srv_proc = None

    @staticmethod
    def _cpu(proc):
        if proc is None:
            return None
        try:
            t = proc.cpu_times()
            return t.user + t.system
        except Exception:
            return None

    def start(self):
        self._c0 = self._cpu(self.self_proc)
        self._s0 = self._cpu(self.srv_proc)
        self._w0 = time.perf_counter()

    def stop(self) -> dict:
        w = time.perf_counter() - self._w0
        c1 = self._cpu(self.self_proc)
        s1 = self._cpu(self.srv_proc)
        return {
            "wall_s": w,
            "client_cpu_s": (c1 - self._c0) if (c1 is not None and self._c0 is not None) else None,
            "server_cpu_s": (s1 - self._s0) if (s1 is not None and self._s0 is not None) else None,
        }


def _find_server_pid(remote: str) -> int | None:
    """Best-effort: the java process listening on the Connect port."""
    port = remote.rsplit(":", 1)[-1]
    try:
        out = subprocess.run(
            ["lsof", "-iTCP:" + port, "-sTCP:LISTEN", "-n", "-P"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.splitlines()
        for line in out[1:]:
            parts = line.split()
            if len(parts) > 1 and parts[1].isdigit():
                return int(parts[1])
    except Exception:
        pass
    return None


def peak_rss_mb() -> float:
    import resource

    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kilobytes.
    return ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024


# --------------------------------------------------------------------------- #
# Workload bodies
# --------------------------------------------------------------------------- #
def _wide_df(spark, rows: int):
    from pyspark.sql import functions as F

    cols = [(F.col("id") + i).alias(f"c{i}") for i in range(NCOLS)]
    return spark.range(0, rows).select(*cols)


def bench_collect(spark, rows: int, iters: int, warmup: int, remote: str) -> dict:
    """Time df.collect() for a fixed-width result of `rows` rows."""
    df = _wide_df(spark, rows)
    for _ in range(warmup):
        df.collect()
    sampler = CpuSampler(remote)
    samples = []
    gc.collect()
    sampler.start()
    for _ in range(iters):
        t0 = time.perf_counter()
        df.collect()
        samples.append(time.perf_counter() - t0)
    agg = sampler.stop()
    logical_bytes = rows * BYTES_PER_ROW
    s = summarize(samples)
    gb = logical_bytes / 1e9
    rows_per_s = rows / s["p50_s"] if s["p50_s"] > 0 else float("nan")
    gb_per_s = gb / s["p50_s"] if s["p50_s"] > 0 else float("nan")
    # CPU per GB across the whole timed region (iters * gb transferred).
    total_gb = gb * iters
    client_cpu_per_gb = (agg["client_cpu_s"] / total_gb) if agg["client_cpu_s"] and total_gb else None
    server_cpu_per_gb = (agg["server_cpu_s"] / total_gb) if agg["server_cpu_s"] and total_gb else None
    return {
        "rows": rows,
        "cols": NCOLS,
        "logical_bytes": logical_bytes,
        "latency": s,
        "rows_per_s_p50": rows_per_s,
        "gb_per_s_p50": gb_per_s,
        "client_cpu_s_total": agg["client_cpu_s"],
        "server_cpu_s_total": agg["server_cpu_s"],
        "client_cpu_per_gb": client_cpu_per_gb,
        "server_cpu_per_gb": server_cpu_per_gb,
        "peak_rss_mb": peak_rss_mb(),
    }


def run_transport(spark, iters: int, warmup: int, remote: str) -> dict:
    out = {}
    for name, target in SIZE_TARGETS:
        rows = max(1, target // BYTES_PER_ROW)
        out[name] = bench_collect(spark, rows, iters, warmup, remote)
        r = out[name]
        print(
            f"  transport {name:>6}: rows={rows:>9} p50={r['latency']['p50_s']*1000:8.2f}ms "
            f"p99={r['latency']['p99_s']*1000:8.2f}ms "
            f"{r['gb_per_s_p50']:6.3f} GB/s {r['rows_per_s_p50']/1e6:7.2f} Mrows/s",
            flush=True,
        )
    return out


def run_rowcount(spark, iters: int, warmup: int, remote: str) -> dict:
    out = {}
    for rows in ROWCOUNTS:
        # Fewer iters for the very large results to keep runtime sane.
        it = iters if rows <= 1_000_000 else max(5, iters // 3)
        out[str(rows)] = bench_collect(spark, rows, it, warmup, remote)
        r = out[str(rows)]
        print(
            f"  rowcount {rows:>10}: p50={r['latency']['p50_s']*1000:9.2f}ms "
            f"{r['gb_per_s_p50']:6.3f} GB/s {r['rows_per_s_p50']/1e6:7.2f} Mrows/s",
            flush=True,
        )
    return out


# --------------------------------------------------------------------------- #
# UDF classes  (executed on the SERVER python worker for both clients)
# --------------------------------------------------------------------------- #
def run_udf(spark, iters: int, warmup: int, remote: str) -> dict:
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType, IntegerType, StringType

    rows = 100_000
    base = spark.range(0, rows)

    def define():
        cases = {}

        # identity
        cases["identity"] = F.udf(lambda x: x, IntegerType())
        # simple arithmetic
        cases["arithmetic"] = F.udf(lambda x: (x * 2 + 1) % 7, IntegerType())
        # strings
        cases["strings"] = F.udf(lambda x: (str(x) * 3)[:16].upper(), StringType())

        # expensive pure python
        def _expensive(x):
            t = 0
            for i in range(200):
                t += (x + i) % 13
            return t

        cases["expensive_py"] = F.udf(_expensive, IntegerType())

        # numpy
        def _np(x):
            import numpy as np

            return float(np.sqrt(np.arange(x % 64 + 1, dtype="float64")).sum())

        cases["numpy"] = F.udf(_np, DoubleType())
        return cases

    cases = define()
    out = {}
    for name, udf in cases.items():
        df = base.select(udf(F.col("id")).alias("y"))
        try:
            for _ in range(warmup):
                df.count()
            samples = []
            for _ in range(iters):
                t0 = time.perf_counter()
                df.count()
                samples.append(time.perf_counter() - t0)
            s = summarize(samples)
            out[name] = {"rows": rows, "latency": s, "rows_per_s_p50": rows / s["p50_s"]}
            print(f"  udf {name:>14}: p50={s['p50_s']*1000:9.2f}ms  {rows/s['p50_s']/1e6:6.3f} Mrows/s", flush=True)
        except Exception as e:  # noqa: BLE001
            out[name] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
            print(f"  udf {name:>14}: ERROR {out[name]['error']}", flush=True)

    # pandas_udf (vectorized, Arrow) - separate because signature differs
    try:
        from pyspark.sql.functions import pandas_udf

        @pandas_udf(DoubleType())
        def _pdf(s):
            return s * 1.5 + 1.0

        df = base.select(_pdf(F.col("id")).alias("y"))
        for _ in range(warmup):
            df.count()
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            df.count()
            samples.append(time.perf_counter() - t0)
        s = summarize(samples)
        out["pandas_udf"] = {"rows": rows, "latency": s, "rows_per_s_p50": rows / s["p50_s"]}
        print(f"  udf {'pandas_udf':>14}: p50={s['p50_s']*1000:9.2f}ms  {rows/s['p50_s']/1e6:6.3f} Mrows/s", flush=True)
    except Exception as e:  # noqa: BLE001
        out["pandas_udf"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
        print(f"  udf {'pandas_udf':>14}: ERROR {out['pandas_udf']['error']}", flush=True)

    return out


# --------------------------------------------------------------------------- #
# Cold start  (spawn fresh interpreters; inherit this process's client env)
# --------------------------------------------------------------------------- #
COLDSTART_SNIPPET = (
    "import time,os;"
    "t0=time.perf_counter();"
    "from pyspark.sql import SparkSession;"
    "t1=time.perf_counter();"
    "s=SparkSession.builder.remote(os.environ['BENCH_REMOTE']).getOrCreate();"
    "t2=time.perf_counter();"
    "s.range(0,1).collect();"
    "t3=time.perf_counter();"
    "s.stop();"
    "import json,sys;"
    "sys.stdout.write('COLDJSON'+json.dumps({'import_s':t1-t0,'session_s':t2-t1,'firstq_s':t3-t2,'total_s':t3-t0}))"
)


def run_coldstart(remote: str, runs: int) -> dict:
    env = dict(os.environ)
    env["BENCH_REMOTE"] = remote
    imp, sess, fq, tot = [], [], [], []
    for _ in range(runs):
        w0 = time.perf_counter()
        p = subprocess.run(
            [sys.executable, "-c", COLDSTART_SNIPPET],
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        wall = time.perf_counter() - w0
        marker = p.stdout.find("COLDJSON")
        if marker == -1:
            print(f"  coldstart: FAILED\n{p.stderr[-400:]}", flush=True)
            continue
        d = json.loads(p.stdout[marker + len("COLDJSON") :])
        imp.append(d["import_s"])
        sess.append(d["session_s"])
        fq.append(d["firstq_s"])
        tot.append(wall)  # full process wall incl. interpreter spawn
        print(
            f"  coldstart run: import={d['import_s']*1000:7.1f}ms session={d['session_s']*1000:7.1f}ms "
            f"firstq={d['firstq_s']*1000:7.1f}ms proc_total={wall*1000:7.1f}ms",
            flush=True,
        )
    if not tot:
        return {"error": "all cold-start runs failed"}
    return {
        "runs": len(tot),
        "import": summarize(imp),
        "session": summarize(sess),
        "first_query": summarize(fq),
        "process_total": summarize(tot),
    }


# --------------------------------------------------------------------------- #
# Concurrency  (N worker processes hammer the server for a fixed duration)
# --------------------------------------------------------------------------- #
WORKER_SNIPPET = """
import time, os, json, sys
from pyspark.sql import SparkSession, functions as F
remote = os.environ['BENCH_REMOTE']
dur = float(os.environ['BENCH_DUR'])
rows = int(os.environ['BENCH_ROWS'])
spark = SparkSession.builder.remote(remote).getOrCreate()
cols = [(F.col('id') + i).alias('c%d' % i) for i in range(8)]
df = spark.range(0, rows).select(*cols)
df.collect()  # warm
lat = []
end = time.perf_counter() + dur
n = 0
while time.perf_counter() < end:
    t0 = time.perf_counter()
    df.collect()
    lat.append(time.perf_counter() - t0)
    n += 1
spark.stop()
sys.stdout.write('WJSON' + json.dumps({'ops': n, 'lat': lat, 'rows': rows}))
"""


def run_concurrency(remote: str, duration: float) -> dict:
    out = {}
    rows = 100_000  # ~ 6.4 MB logical per collect
    for p in CONCURRENCY_LEVELS:
        env = dict(os.environ)
        env["BENCH_REMOTE"] = remote
        env["BENCH_DUR"] = str(duration)
        env["BENCH_ROWS"] = str(rows)
        procs = []
        t0 = time.perf_counter()
        for _ in range(p):
            procs.append(
                subprocess.Popen(
                    [sys.executable, "-c", WORKER_SNIPPET],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        total_ops = 0
        all_lat = []
        failed = 0
        for pr in procs:
            so, se = pr.communicate(timeout=int(duration) + 180)
            m = so.find("WJSON")
            if m == -1:
                failed += 1
                continue
            d = json.loads(so[m + len("WJSON") :])
            total_ops += d["ops"]
            all_lat.extend(d["lat"])
        wall = time.perf_counter() - t0
        agg_rows_per_s = total_ops * rows / wall if wall > 0 else float("nan")
        agg_gb_per_s = total_ops * rows * BYTES_PER_ROW / 1e9 / wall if wall > 0 else float("nan")
        out[str(p)] = {
            "procs": p,
            "failed_procs": failed,
            "total_ops": total_ops,
            "wall_s": wall,
            "agg_rows_per_s": agg_rows_per_s,
            "agg_gb_per_s": agg_gb_per_s,
            "latency": summarize(all_lat) if all_lat else {},
        }
        lp = out[str(p)]["latency"]
        print(
            f"  concurrency P={p}: ops={total_ops:>6} {agg_gb_per_s:6.3f} GB/s "
            f"{agg_rows_per_s/1e6:6.2f} Mrows/s  p50={lp.get('p50_s',float('nan'))*1000:8.2f}ms "
            f"p99={lp.get('p99_s',float('nan'))*1000:8.2f}ms (failed={failed})",
            flush=True,
        )
    return out


# --------------------------------------------------------------------------- #
def run_all(args) -> int:
    import pyspark
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    client_path = os.path.dirname(pyspark.__file__)
    print(f"# label={args.label} client={client_path} arch={platform.machine()} remote={args.remote}")
    print(f"# pyspark_version={pyspark.__version__} iters={args.iters} warmup={args.warmup}\n", flush=True)

    payload = {
        "label": args.label,
        "remote": args.remote,
        "client_path": client_path,
        "pyspark_version": pyspark.__version__,
        "arch": platform.machine(),
        "python": sys.version.split()[0],
        "iters": args.iters,
        "warmup": args.warmup,
        "results": {},
    }
    modes = args.modes.split(",")
    if "transport" in modes:
        print("[transport payload sweep]", flush=True)
        payload["results"]["transport"] = run_transport(spark, args.iters, args.warmup, args.remote)
    if "rowcount" in modes:
        print("[row-count sweep]", flush=True)
        payload["results"]["rowcount"] = run_rowcount(spark, args.iters, args.warmup, args.remote)
    if "udf" in modes:
        print("[udf classes]", flush=True)
        payload["results"]["udf"] = run_udf(spark, max(6, args.iters // 3), 2, args.remote)
    spark.stop()

    # These spawn subprocesses; do them after stopping our own session.
    if "coldstart" in modes:
        print("[cold start]", flush=True)
        payload["results"]["coldstart"] = run_coldstart(args.remote, args.coldstart_runs)
    if "concurrency" in modes:
        print("[concurrency]", flush=True)
        payload["results"]["concurrency"] = run_concurrency(args.remote, args.concurrency_dur)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.out}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default=REMOTE_DEFAULT)
    ap.add_argument("--label", default="client")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--coldstart-runs", type=int, default=7)
    ap.add_argument("--concurrency-dur", type=float, default=10.0)
    ap.add_argument(
        "--modes",
        default="transport,rowcount,udf,coldstart,concurrency",
        help="comma list of modes to run",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", nargs=2, metavar=("REF_JSON", "OURS_JSON"))
    args = ap.parse_args()
    if args.report:
        from bench_report import render  # local import; see scripts/bench_report.py

        return render(args.report[0], args.report[1])
    return run_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
