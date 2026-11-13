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
"""Compile the ``rust`` code snippets in the docs against the crate.

The published guides drifted out of sync with the API (examples that used `lit`
without importing it, or `SparkSessionBuilder` without a `use`), and nothing in CI
caught it because the docs workflow only *deploys*. This type-checks every
```rust``` block so a doc example can't reference a symbol the crate doesn't export.

Each block is wrapped in a small harness that binds ``spark`` and ``df`` (so the
common "continue from an earlier block" fragments type-check) and is run through
``rustc`` against the already-built ``apache-spark-connect`` rlib. Only compilation
(type-checking) is performed - snippets are never executed, so no server is needed.
Emit metadata only so checking each snippet does not link and retain a separate
native executable containing the crate and its dependencies.

A block that must be skipped (pseudo-code, a deliberate compile-error example, or
a snippet that needs a feature/macro the snippet harness doesn't build - e.g. the
`wasm-udf` `#[spark_wasm_udf]` examples) should be fenced as ```{.rust .no-run}```.
That brace form still renders as a Rust code block in mkdocs/superfences (unlike
```rust,ignore```, whose comma makes superfences drop the fence entirely and print
the backticks literally), and it is not matched by the checker below, so it is not
compiled.

Usage:
    cargo build -p apache-spark-connect
    python scripts/check_doc_snippets.py
"""

import glob
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ```rust``` (checked) vs ```{.rust .no-run}``` (rendered but skipped). We only check
# plain ```rust``` fences; any info string with a comma/space/brace is left alone.
_BLOCK = re.compile(r"^```rust\s*$\n(.*?)^```\s*$", re.DOTALL | re.MULTILINE)

# Harness: bind the identifiers the fragments assume from earlier blocks. `use`
# statements are legal inside a fn body, so a block can carry its own imports; a
# block that re-binds `spark`/`df` just shadows these (harmless).
HARNESS = """#![allow(unused, unused_imports, dead_code, clippy::all)]
fn main() -> Result<(), Box<dyn std::error::Error>> {{
    let spark = spark_connect::SparkSession::builder()
        .remote("sc://localhost:15002")
        .get_or_create()?;
    // The ambient names guide fragments assume from a preceding block.
    let df = spark.range(10)?;
    let df1 = spark.range(10)?;
    let df2 = spark.range(10)?;
    {{
{body}
    }}
    Ok(())
}}
"""


def check_block(
    code: str,
    tmpdir: Path,
    idx: int,
    rlib: Path | None = None,
    deps: Sequence[Path] = (),
) -> tuple[bool, str]:
    """Compile the snippet ``code`` (the ``idx``-th) in ``tmpdir`` against ``rlib``, the rlib of
    ``apache-spark-connect``, whose dependencies are in the directories ``deps``. Without
    ``rlib``, the snippet is compiled without the crate."""
    indented = "\n".join("        " + ln if ln.strip() else ln for ln in code.splitlines())
    src = tmpdir / f"snippet_{idx}.rs"
    src.write_text(HARNESS.format(body=indented))
    out = tmpdir / f"snippet_{idx}.rmeta"
    r = subprocess.run(
        [
            "rustc",
            "--edition",
            "2021",
            "--crate-type",
            "bin",
            "--emit=metadata",
            *(arg for dep in deps for arg in ("-L", f"dependency={dep}")),
            *(("--extern", f"spark_connect={rlib}") if rlib is not None else ()),
            "-o",
            str(out),
            str(src),
        ],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0, r.stderr


def crate_artifacts() -> tuple[Path | None, list[Path]]:
    """The rlib of ``apache-spark-connect`` and the directories of the artifacts of its
    dependencies (arrow, prost, ...), which rustc finds them in with ``-L dependency``, as cargo
    reports them for ``cargo build -p apache-spark-connect``, of the units that are fresh too.

    Cargo's layout of them changed: before Rust 1.100, the artifacts of all the dependencies are
    in ``target/debug/deps``, and from Rust 1.100, those of each unit are in a directory of its
    own, ``target/debug/build/<package>/<hash>/out``.
    """
    r = subprocess.run(
        ["cargo", "build", "-p", "apache-spark-connect", "--message-format=json"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(r.stderr[-4000:])
        return None, []
    rlib = None
    deps: list[Path] = []
    for line in r.stdout.splitlines():
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if msg.get("reason") != "compiler-artifact":
            continue
        for name in msg.get("filenames", []):
            path = Path(name)
            if (
                path.suffix in (".rlib", ".rmeta", ".so", ".dylib", ".dll")
                and path.parent not in deps
            ):
                deps.append(path.parent)
            if msg.get("target", {}).get("name") == "spark_connect" and path.suffix == ".rlib":
                rlib = path
    return rlib, deps


def main() -> int:
    rlib, deps = crate_artifacts()
    if rlib is None or not rlib.exists():
        print("!! no rlib of apache-spark-connect - run `cargo build -p apache-spark-connect`")
        return 2

    failures = []
    checked = 0
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        idx = 0
        for md in sorted(glob.glob(str(REPO / "docs" / "**" / "*.md"), recursive=True)):
            text = Path(md).read_text()
            for m in _BLOCK.finditer(text):
                idx += 1
                checked += 1
                ok, err = check_block(m.group(1), tmpdir, idx, rlib, deps)
                if not ok:
                    rel = Path(md).relative_to(REPO)
                    # First rustc error line is the most informative.
                    first = next(
                        (ln for ln in err.splitlines() if ln.startswith("error")), err[:200]
                    )
                    failures.append((str(rel), first))

    print(f"Checked {checked} rust snippet(s); {len(failures)} failed to compile.")
    for rel, err in failures:
        print(f"  {rel}: {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
