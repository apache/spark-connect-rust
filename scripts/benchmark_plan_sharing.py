#!/usr/bin/env python3
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Measure native PySpark plan/expression construction and execution separately.

Run this same script with the baseline and patched Python environments, then
compare their JSON files. See dev/design/plan-sharing-benchmark.md. No transport
plugin, pytest injection, installation, or building is performed by this script.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

PHASES = ("construction_s", "action_s", "total_s")
PLAN_WORKLOADS = ("linear", "branched")
EXPRESSION_WORKLOADS = ("expression_chain", "expression_reuse")
WORKLOADS = PLAN_WORKLOADS + EXPRESSION_WORKLOADS


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_result(name: str, rows: int, depth: int, branches: int) -> dict:
    checksum = rows * (rows - 1) // 2 + rows * depth
    if name in ("branched", "expression_reuse"):
        checksum = branches * checksum + rows * branches * (branches + 1) // 2
        rows *= branches
    elif name not in ("linear", "expression_chain"):
        raise ValueError(f"Unknown workload: {name}")
    if max(rows, checksum) > (1 << 63) - 1:
        raise ValueError("Workload would overflow Spark's signed 64-bit aggregate")
    return {"checksum": checksum, "count": rows}


def build_query(spark, functions, name: str, config: dict):
    """Retain every intermediate DataFrame until AFTER both measured phases."""
    if name in EXPRESSION_WORKLOADS:
        return build_expression_query(spark, functions, name, config)
    retained = []

    def keep(frame):
        retained.append(frame)
        return frame

    frame = keep(spark.range(0, config["rows"], 1, config["partitions"]))
    frame = keep(frame.select(functions.col("id").alias("value")))
    for _ in range(config["depth"]):
        frame = keep(frame.select((functions.col("value") + 1).alias("value")))
    if name == "branched":
        base = frame
        frames = [
            keep(base.select((functions.col("value") + offset).alias("value")))
            for offset in range(1, config["branches"] + 1)
        ]
        frame = frames[0]
        for branch in frames[1:]:
            frame = keep(frame.union(branch))
    elif name != "linear":
        raise ValueError(f"Unknown workload: {name}")
    query = keep(
        frame.select(
            functions.sum("value").alias("checksum"),
            functions.count("value").alias("count"),
        )
    )
    return query, retained


def build_expression_query(spark, functions, name: str, config: dict):
    """Build a deep Column, retaining all Columns and DataFrames through collect."""
    retained = []

    def keep(handle):
        retained.append(handle)
        return handle

    source = keep(spark.range(0, config["rows"], 1, config["partitions"]))
    value = keep(functions.col("id"))
    for _ in range(config["depth"]):
        value = keep(value + 1)
    if name == "expression_chain":
        frame = keep(source.select(keep(value.alias("value"))))
    elif name == "expression_reuse":
        frames = []
        for offset in range(1, config["branches"] + 1):
            shifted = keep(value + offset)
            frames.append(keep(source.select(keep(shifted.alias("value")))))
        frame = frames[0]
        for branch in frames[1:]:
            frame = keep(frame.union(branch))
    else:
        raise ValueError(f"Unknown expression workload: {name}")
    checksum = keep(keep(functions.sum("value")).alias("checksum"))
    count = keep(keep(functions.count("value")).alias("count"))
    query = keep(frame.select(checksum, count))
    return query, retained


def check_result(result, expected: dict) -> dict:
    if len(result) != 1 or len(result[0]) != 2:
        raise ValueError(f"Expected one aggregate row with two fields, got {result!r}")
    actual = {"checksum": result[0][0], "count": result[0][1]}
    if any(type(value) is not int for value in actual.values()) or actual != expected:
        raise ValueError(f"Incorrect aggregate: expected {expected}, got {actual}")
    return actual


def measure(spark, functions, name: str, config: dict) -> dict:
    expected = expected_result(name, config["rows"], config["depth"], config["branches"])
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    query = retained = result = None
    try:
        start = time.perf_counter()
        query, retained = build_query(spark, functions, name, config)
        constructed = time.perf_counter()
        result = query.collect()
        finished = time.perf_counter()
        # Validation and destruction are intentionally outside the measured interval.
        actual = check_result(result, expected)
        return {
            "construction_s": constructed - start,
            "action_s": finished - constructed,
            "total_s": finished - start,
            "result": actual,
        }
    finally:
        result = query = retained = None
        if gc_was_enabled:
            gc.enable()


def client_provenance(spark, functions, expected_client: str, expected_sha256: str | None):
    import pyspark

    if "rust_transport_plugin" in sys.modules:
        raise RuntimeError("Transport-injected official PySpark is not a valid benchmark client")
    probe = spark.range(1)
    column = functions.col("id")
    extension = None
    if expected_client == "rust":
        native = importlib.import_module("pyspark._pyspark")
        if (
            type(spark) is not native.SparkSession
            or type(probe) is not native.DataFrame
            or type(column) is not native.Column
        ):
            raise RuntimeError("Expected the actual Rust-backed Session, DataFrame, and Column")
        path = Path(native.__file__).resolve()
        if not any(str(path).endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES):
            raise RuntimeError(f"Expected a compiled extension, got {path}")
        checksum = file_sha256(path)
        if expected_sha256 and checksum != expected_sha256.lower():
            raise RuntimeError("Loaded extension SHA-256 does not match --extension-sha256")
        extension = {"path": str(path), "sha256": checksum, "size_bytes": path.stat().st_size}
    else:
        from pyspark.sql.connect.dataframe import DataFrame
        from pyspark.sql.connect.session import SparkSession

        if type(spark) is not SparkSession or type(probe) is not DataFrame:
            raise RuntimeError("Expected official PySpark Connect classes")
        if "pyspark._pyspark" in sys.modules or "_pyspark" in sys.modules:
            raise RuntimeError("Official control must not load the Rust extension")
        if not hasattr(DataFrame.collect, "__code__"):
            raise RuntimeError("Official control must use Python's DataFrame implementation")
    versions = {}
    for distribution in ("pyarrow", "numpy", "pandas"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return {
        "client": expected_client,
        "pyspark_path": str(Path(pyspark.__file__).resolve()),
        "pyspark_version": pyspark.__version__,
        "functions_path": str(Path(functions.__file__).resolve()),
        "dataframe_class": f"{type(probe).__module__}.{type(probe).__qualname__}",
        "extension": extension,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "dependencies": versions,
        "script_sha256": file_sha256(Path(__file__)),
    }


def run(args) -> dict:
    import pyspark.sql.functions as functions
    from pyspark.sql import SparkSession

    if Path(args.out).exists():
        raise FileExistsError(f"Refusing to overwrite benchmark results: {args.out}")
    config = {
        key: getattr(args, key)
        for key in (
            "rows", "depth", "branches", "partitions", "warmups", "repetitions", "order_seed"
        )
    }
    names = list(WORKLOADS) if args.workload == "all" else [args.workload]
    for name in names:
        expected_result(name, args.rows, args.depth, args.branches)
    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    try:
        provenance = client_provenance(spark, functions, args.expect_client, args.extension_sha256)
        server_version = spark.version
        server_configs = {
            key: spark.conf.get(key)
            for key in ("spark.sql.shuffle.partitions", "spark.sql.adaptive.enabled")
        }
        print(
            json.dumps({"label": args.label, "build_id": args.build_id, **provenance}), flush=True
        )
        results = {name: {"warmups": [], "samples": []} for name in names}
        schedule = []
        rng = random.Random(args.order_seed)
        for phase, repetitions in (("warmups", args.warmups), ("samples", args.repetitions)):
            for iteration in range(repetitions):
                order = list(names)
                rng.shuffle(order)
                for name in order:
                    sample = measure(spark, functions, name, config)
                    results[name][phase].append(sample)
                    schedule.append({"phase": phase, "iteration": iteration, "workload": name})
                    print(f"{args.label} {name} {phase}[{iteration}] {sample}", flush=True)
        for name, result in results.items():
            result["expected"] = expected_result(name, args.rows, args.depth, args.branches)
            result["median_s"] = {
                phase: statistics.median(sample[phase] for sample in result["samples"])
                for phase in PHASES
            }
    finally:
        spark.stop()
    # Do not persist credentials in the endpoint; compare the full URL by its digest.
    endpoint = urlsplit(args.remote.split(";", 1)[0])
    return {
        "format_version": 1,
        "status": "ok",
        "label": args.label,
        "build_id": args.build_id,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "endpoint": f"{endpoint.scheme}://{endpoint.hostname}:{endpoint.port or 15002}",
        "endpoint_sha256": hashlib.sha256(args.remote.encode()).hexdigest(),
        "server_version": server_version,
        "server_configs": server_configs,
        "config": config,
        "provenance": provenance,
        "schedule": schedule,
        "results": results,
    }


def validate_report(report: dict) -> None:
    if report["format_version"] != 1 or report["status"] != "ok" or not report["results"]:
        raise ValueError("Not a successful plan-sharing benchmark report")
    config = report["config"]
    for name, result in report["results"].items():
        expected = expected_result(name, config["rows"], config["depth"], config["branches"])
        if result["expected"] != expected:
            raise ValueError(f"Incorrect expected result for {name}")
        for phase, count in (("samples", config["repetitions"]), ("warmups", config["warmups"])):
            if len(result[phase]) != count or (phase == "samples" and count <= 0):
                raise ValueError(f"Missing {phase} for {name}")
            for sample in result[phase]:
                if sample["result"] != expected:
                    raise ValueError(f"Unverified result for {name}")
                if any(not math.isfinite(sample[p]) or sample[p] <= 0 for p in PHASES):
                    raise ValueError(f"Invalid timings for {name}")
                if not math.isclose(
                    sample["total_s"], sample["construction_s"] + sample["action_s"]
                ):
                    raise ValueError(f"Inconsistent phase timings for {name}")


def compare(baseline_path: str, candidate_path: str) -> None:
    baseline = json.loads(Path(baseline_path).read_text())
    candidate = json.loads(Path(candidate_path).read_text())
    for report in (baseline, candidate):
        validate_report(report)
    for key in ("config", "endpoint_sha256", "server_version", "server_configs"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"Incomparable reports: different {key}")
    if set(baseline["results"]) != set(candidate["results"]):
        raise ValueError("Incomparable reports: different workloads")
    for key in ("python_version", "platform", "hostname", "dependencies", "script_sha256"):
        if baseline["provenance"][key] != candidate["provenance"][key]:
            raise ValueError(f"Incomparable reports: different {key}")
    left = baseline["provenance"]["extension"]
    right = candidate["provenance"]["extension"]
    if left and right and left["sha256"] == right["sha256"]:
        raise ValueError("Baseline and candidate loaded the same extension binary")
    print(f"Baseline: {baseline['label']} ({baseline['build_id']})")
    print(f"Candidate: {candidate['label']} ({candidate['build_id']})")
    print("workload          phase          baseline ms  candidate ms  baseline/candidate")
    for name in baseline["results"]:
        for phase in PHASES:
            before = statistics.median(s[phase] for s in baseline["results"][name]["samples"])
            after = statistics.median(s[phase] for s in candidate["results"][name]["samples"])
            print(
                f"{name:17} {phase:14} {before * 1000:11.3f} "
                f"{after * 1000:13.3f} {before / after:18.3f}x"
            )
    print("Ratios above 1 favor the candidate; medians alone do not establish significance.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE_JSON", "CANDIDATE_JSON"))
    parser.add_argument("--remote", default="sc://localhost:15002")
    parser.add_argument("--label", help="Run label, for example baseline or patched")
    parser.add_argument("--build-id", help="Caller-supplied source revision/build description")
    parser.add_argument("--expect-client", choices=("rust", "official"), default="rust")
    parser.add_argument("--extension-sha256", help="Optionally pin the loaded native binary")
    parser.add_argument("--out", help="New JSON output path; existing files are not overwritten")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--partitions", type=int, default=1)
    parser.add_argument("--workload", choices=("all", *WORKLOADS), default="all")
    parser.add_argument("--order-seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not args.compare:
        if not all((args.label, args.build_id, args.out)):
            parser.error("--label, --build-id, and --out are required for a benchmark run")
        if args.warmups < 0 or args.depth < 0:
            parser.error("--warmups and --depth must be nonnegative")
        if min(args.repetitions, args.rows, args.branches, args.partitions) <= 0:
            parser.error("--repetitions, --rows, --branches, and --partitions must be positive")
        if args.extension_sha256 and args.expect_client != "rust":
            parser.error("--extension-sha256 requires --expect-client rust")
    try:
        if args.compare:
            compare(*args.compare)
        else:
            report = run(args)
            validate_report(report)
            with Path(args.out).open("x") as output:
                json.dump(report, output, indent=2)
                output.write("\n")
            print(f"Wrote {args.out}")
        return 0
    except Exception as error:
        print(f"Benchmark failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
