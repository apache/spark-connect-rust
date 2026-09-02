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
"""Render a Markdown benchmark report comparing reference vs ours result JSON."""

from __future__ import annotations

import json


def _ms(x):
    return f"{x * 1000:.2f}" if isinstance(x, (int, float)) else "-"


def _load(p):
    with open(p) as f:
        return json.load(f)


def render(ref_path: str, ours_path: str) -> int:
    ref = _load(ref_path)
    ours = _load(ours_path)
    L = []
    p = L.append

    p("## Benchmark results\n")
    p(
        f"- Reference client: pyspark {ref['pyspark_version']} ({ref['arch']}, py{ref['python']}) at `{ref['client_path']}`"
    )
    p(
        f"- Rust client: pyspark-client-rust {ours['pyspark_version']} ({ours['arch']}, py{ours['python']})"
    )
    p(
        f"- Both clients → same Spark Connect server `{ref['remote']}`; iters={ref['iters']}, warmup={ref['warmup']}\n"
    )

    # Transport sweep
    rt, ot = ref["results"].get("transport", {}), ours["results"].get("transport", {})
    if rt and ot:
        p("### Transport: payload-size sweep (df.collect)\n")
        p(
            "| payload | rows | ref p50 (ms) | ours p50 (ms) | speedup | ref GB/s | ours GB/s | ref Mrows/s | ours Mrows/s |"
        )
        p("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
        for k in rt:
            r, o = rt[k], ot.get(k, {})
            if not o:
                continue
            sp = r["latency"]["p50_s"] / o["latency"]["p50_s"]
            p(
                f"| {k} | {r['rows']:,} | {_ms(r['latency']['p50_s'])} | {_ms(o['latency']['p50_s'])} | "
                f"{sp:.2f}x | {r['gb_per_s_p50']:.3f} | {o['gb_per_s_p50']:.3f} | "
                f"{r['rows_per_s_p50'] / 1e6:.2f} | {o['rows_per_s_p50'] / 1e6:.2f} |"
            )
        p("")
        p("| payload | ref p99 (ms) | ours p99 (ms) | ref p999 (ms) | ours p999 (ms) |")
        p("|---|--:|--:|--:|--:|")
        for k in rt:
            r, o = rt[k], ot.get(k, {})
            if not o:
                continue
            p(
                f"| {k} | {_ms(r['latency']['p99_s'])} | {_ms(o['latency']['p99_s'])} | "
                f"{_ms(r['latency']['p999_s'])} | {_ms(o['latency']['p999_s'])} |"
            )
        p("")
        # CPU/GB & memory (from the largest payload)
        big = list(rt.keys())[-1]
        rb, ob = rt[big], ot[big]
        p(f"### CPU & memory (at {big} payload)\n")
        p("| metric | reference (Python client) | Rust client |")
        p("|---|--:|--:|")

        def _f(x, u=""):
            return f"{x:.3f}{u}" if isinstance(x, (int, float)) else "n/a"

        p(
            f"| client CPU / GB decoded | {_f(rb.get('client_cpu_per_gb'), ' s')} | {_f(ob.get('client_cpu_per_gb'), ' s')} |"
        )
        p(
            f"| server (JVM) CPU / GB | {_f(rb.get('server_cpu_per_gb'), ' s')} | {_f(ob.get('server_cpu_per_gb'), ' s')} |"
        )
        p(
            f"| client peak RSS | {_f(rb.get('peak_rss_mb'), ' MB')} | {_f(ob.get('peak_rss_mb'), ' MB')} |"
        )
        p("")

    # Row-count sweep
    rr, orr = ref["results"].get("rowcount", {}), ours["results"].get("rowcount", {})
    if rr and orr:
        p("### Execution: row-count sweep (df.collect)\n")
        p("| rows | ref p50 (ms) | ours p50 (ms) | speedup | ours Mrows/s |")
        p("|--:|--:|--:|--:|--:|")
        for k in rr:
            r, o = rr[k], orr.get(k, {})
            if not o:
                continue
            sp = r["latency"]["p50_s"] / o["latency"]["p50_s"]
            p(
                f"| {int(k):,} | {_ms(r['latency']['p50_s'])} | {_ms(o['latency']['p50_s'])} | {sp:.2f}x | {o['rows_per_s_p50'] / 1e6:.2f} |"
            )
        p("")

    # Cold start
    rc, oc = ref["results"].get("coldstart", {}), ours["results"].get("coldstart", {})
    if rc and oc and "error" not in rc and "error" not in oc:
        p("### Cold-start latency (fresh interpreter → first result)\n")
        p("| phase | reference p50 (ms) | Rust client p50 (ms) | speedup |")
        p("|---|--:|--:|--:|")
        for phase, key in [
            ("import pyspark", "import"),
            ("build session", "session"),
            ("first query", "first_query"),
            ("process total (incl. spawn)", "process_total"),
        ]:
            rp, op = rc[key]["p50_s"], oc[key]["p50_s"]
            sp = rp / op if op > 0 else float("inf")
            p(f"| {phase} | {_ms(rp)} | {_ms(op)} | {sp:.2f}x |")
        p("")

    # Concurrency
    rcc, occ = ref["results"].get("concurrency", {}), ours["results"].get("concurrency", {})
    if rcc and occ:
        p("### Concurrency: N client processes, aggregate throughput\n")
        p("| processes | ref agg GB/s | ours agg GB/s | ref p99 (ms) | ours p99 (ms) |")
        p("|--:|--:|--:|--:|--:|")
        for k in rcc:
            r, o = rcc[k], occ.get(k, {})
            if not o:
                continue
            rp99 = r["latency"].get("p99_s", float("nan"))
            op99 = o["latency"].get("p99_s", float("nan"))
            p(
                f"| {k} | {r['agg_gb_per_s']:.3f} | {o['agg_gb_per_s']:.3f} | {_ms(rp99)} | {_ms(op99)} |"
            )
        p("")

    # UDF
    ru, ou = ref["results"].get("udf", {}), ours["results"].get("udf", {})
    if ru and ou:
        p("### UDF classes (executed on the server Python worker — client-neutral)\n")
        p("| UDF class | ref p50 (ms) | ours p50 (ms) | ratio |")
        p("|---|--:|--:|--:|")
        for k in ru:
            r, o = ru[k], ou.get(k, {})
            if "error" in r or "error" in o or "latency" not in r or "latency" not in o:
                p(f"| {k} | {r.get('error', '-')} | {o.get('error', '-')} | - |")
                continue
            rp, op = r["latency"]["p50_s"], o["latency"]["p50_s"]
            p(f"| {k} | {_ms(rp)} | {_ms(op)} | {rp / op:.2f}x |")
        p("")

    print("\n".join(L))
    return 0
