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
```rust``` block of the README and of the docs, including the indented ones of list
items, admonitions and content tabs, so a doc example can't reference a symbol the
crate doesn't export.

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
the backticks literally). A plain ```rust``` fence can be skipped too, by putting the
HTML comment `<!-- snippet: no-run -->` on the line directly before it. GitHub and PyPI
do not show that comment, and they keep the Rust highlighting that the brace form loses
there, so the README uses it. Only a marker on the line directly before a fence counts.
A fence whose info string starts with `rust` (or `{.rust`) and is neither ```rust```,
nor that brace form, nor marked, is reported as a failure, not skipped. A fence the
checker cannot parse (never closed, or in a blockquote or nested in another fence) is
reported too, rather than swallowing the fences after it.

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
from typing import NamedTuple

REPO = Path(__file__).resolve().parent.parent

# A fence line: any indent, optional `>` (blockquote) prefixes, then three or more backticks
# or tildes. Every such line is counted, so a fence that is not parsed is noticed.
_FENCE_LINE = re.compile(r"^[ \t>]*(?:`{3,}|~{3,})")
# An opening fence of the top level, with its indentation and info string. A fence may be
# indented, as in a list item, an admonition or a content tab, and its closing fence with it.
_OPEN = re.compile(r"^(?P<indent>[ \t]*)(?P<marker>`{3,}|~{3,})(?P<info>.*)$")
# A closing fence: the marker's character, at least as long as the opening marker, and nothing
# else on the line.
_CLOSE = re.compile(r"^[ \t]*(?P<marker>`{3,}|~{3,})[ \t]*$")

# The info string of a snippet that is compiled, and of one that is rendered but not compiled.
CHECKED_INFO = "rust"
NO_RUN_INFO = "{.rust .no-run}"
# A marker on the line directly before a fence, which skips the fence as NO_RUN_INFO does.
_NO_RUN_MARKER = re.compile(r"^[ \t]*<!--[ \t]*snippet:[ \t]*no-run[ \t]*-->[ \t]*$")


class Fence(NamedTuple):
    """A fenced block: the line of its opening fence, its info string, its body, and whether
    a no-run marker is on the line before it."""

    line: int
    info: str
    code: str
    no_run: bool = False


def _closes(line: str, marker: str) -> bool:
    m = _CLOSE.match(line)
    return (
        m is not None
        and m.group("marker")[0] == marker[0]
        and len(m.group("marker")) >= len(marker)
    )


def fences(text: str) -> tuple[list[Fence], list[tuple[int, str]]]:
    """The fenced blocks of the Markdown ``text``, and the problems found in its fences as
    (line, message) pairs. The body of an indented block loses up to the indentation of its
    opening fence, as Markdown strips it."""
    lines = text.replace("\r\n", "\n").split("\n")
    found: list[Fence] = []
    problems: list[tuple[int, str]] = []
    parsed: set[int] = set()  # the lines of the opening and closing fences of the blocks
    i = 0
    while i < len(lines):
        opened = _OPEN.match(lines[i])
        if opened is None:
            i += 1
            continue
        marker = opened.group("marker")
        close = next((j for j in range(i + 1, len(lines)) if _closes(lines[j], marker)), None)
        if close is None:
            problems.append((i + 1, "this fence is never closed"))
            return found, problems
        width = len(opened.group("indent"))
        body = [ln[min(width, len(ln) - len(ln.lstrip(" \t"))) :] for ln in lines[i + 1 : close]]
        info = " ".join(opened.group("info").split())
        marked = i > 0 and _NO_RUN_MARKER.match(lines[i - 1]) is not None
        found.append(Fence(i + 1, info, "\n".join(body), marked))
        parsed.update((i + 1, close + 1))
        i = close + 1
    # Every fence line must belong to a block: a fence in a blockquote, or nested in another
    # fence, is not parsed and would otherwise be skipped silently.
    fence_lines = [n for n, ln in enumerate(lines, 1) if _FENCE_LINE.match(ln)]
    if len(fence_lines) != len(parsed):
        unparsed = next(n for n in fence_lines if n not in parsed)
        problems.append(
            (
                unparsed,
                f"{len(fence_lines) - len(parsed)} fence line(s) are not parsed as blocks "
                "(a fence in a blockquote or nested in another fence is not checked)",
            )
        )
    return found, problems


def snippets(text: str) -> tuple[list[Fence], list[tuple[int, str]]]:
    """The ```rust``` blocks of the Markdown ``text`` to compile, and the problems to report:
    the problems of ``fences``, and each fence whose info string starts with ``rust`` or
    ``{.rust`` but is neither of the two forms that are checked or skipped."""
    blocks, problems = fences(text)
    checked = []
    for block in blocks:
        if block.no_run:
            continue
        if block.info == CHECKED_INFO:
            checked.append(block)
        elif block.info.lower().startswith(("rust", "{.rust")) and block.info != NO_RUN_INFO:
            problems.append(
                (
                    block.line,
                    f'fence info "{block.info}" is neither "{CHECKED_INFO}" nor "{NO_RUN_INFO}", '
                    "so it would be skipped",
                )
            )
    return checked, problems


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
        # With --message-format=json, cargo prints the errors of rustc on stdout, as JSON.
        for line in r.stdout.splitlines():
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            message = msg.get("message") or {}
            if msg.get("reason") == "compiler-message" and message.get("level") == "error":
                print(message.get("rendered", ""), end="")
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


def markdown_files() -> list[Path]:
    """The Markdown files to check: the README, which GitHub and PyPI show, then the docs."""
    docs = sorted(glob.glob(str(REPO / "docs" / "**" / "*.md"), recursive=True))
    return [REPO / "README.md", *(Path(md) for md in docs)]


def main() -> int:
    rlib, deps = crate_artifacts()
    if rlib is None or not rlib.exists():
        print("!! no rlib of apache-spark-connect - run `cargo build -p apache-spark-connect`")
        return 2

    # Each failure is (file, line of the fence, message).
    failures: list[tuple[str, int, str]] = []
    checked = 0
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        idx = 0
        for md in markdown_files():
            rel = str(md.relative_to(REPO))
            blocks, problems = snippets(md.read_text())
            failures.extend((rel, line, message) for line, message in problems)
            for block in blocks:
                idx += 1
                checked += 1
                ok, err = check_block(block.code, tmpdir, idx, rlib, deps)
                if not ok:
                    # First rustc error line is the most informative.
                    first = next(
                        (ln for ln in err.splitlines() if ln.startswith("error")), err[:200]
                    )
                    failures.append((rel, block.line, first))

    print(f"Checked {checked} rust snippet(s); {len(failures)} failed to compile or were refused.")
    for rel, line, message in sorted(failures):
        print(f"  {rel}:{line}: {message}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
