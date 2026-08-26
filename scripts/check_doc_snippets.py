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
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# cargo places the primary lib without a hash in target/debug; its dependencies
# (arrow, prost, ...) live in target/debug/deps and are found via -L dependency.
RLIB = REPO / "target" / "debug" / "libspark_connect.rlib"
DEPS = REPO / "target" / "debug" / "deps"

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


def check_block(code: str, tmpdir: Path, idx: int) -> tuple[bool, str]:
    indented = "\n".join("        " + ln if ln.strip() else ln for ln in code.splitlines())
    src = tmpdir / f"snippet_{idx}.rs"
    src.write_text(HARNESS.format(body=indented))
    out = tmpdir / f"snippet_{idx}.out"
    r = subprocess.run(
        [
            "rustc",
            "--edition",
            "2021",
            "--crate-type",
            "bin",
            "-L",
            f"dependency={DEPS}",
            "--extern",
            f"spark_connect={RLIB}",
            "-o",
            str(out),
            str(src),
        ],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0, r.stderr


def main() -> int:
    if not RLIB.exists():
        print(f"!! {RLIB} not found - run `cargo build -p apache-spark-connect` first")
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
                ok, err = check_block(m.group(1), tmpdir, idx)
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
