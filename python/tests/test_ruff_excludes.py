"""The vendored paths that ruff excludes are those of the drift gate.

``pyproject.toml`` keeps the files vendored from Apache Spark out of ruff, and
``scripts/check_vendored_upstream.sh`` keeps them byte-identical to the pinned tag. A TOML
file cannot read a shell array, so the list exists twice; this check fails when the two
differ, in either direction. It needs only the standard library, so the lint workflow runs
it as a script: ``python python/tests/test_ruff_excludes.py``.
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BEGIN = "# --- vendored from Apache Spark: BEGIN ---"
END = "# --- vendored from Apache Spark: END ---"


def _quoted(text):
    """The double-quoted strings of ``text``, with the comments of its lines left out."""
    found = []
    for line in text.splitlines():
        found += re.findall(r'"([^"]+)"', line.split("#", 1)[0])
    return found


def _drift_gate_paths():
    script = (REPO_ROOT / "scripts" / "check_vendored_upstream.sh").read_text()

    def array(name):
        body = re.search(rf"^{name}=\(\n(.*?)^\)", script, re.S | re.M)
        assert body is not None, f"{name} not found in check_vendored_upstream.sh"
        return _quoted(body.group(1))

    # Ruff reads only Python sources, so e.g. errors/error-conditions.json has no entry.
    files = [f for f in array("VENDORED_FILES") if f.endswith((".py", ".pyi"))]
    return [f"python/pyspark/{p}" for p in array("VENDORED_DIRS") + files]


def _ruff_excluded_paths():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    assert pyproject.count(BEGIN) == 1 and pyproject.count(END) == 1
    return _quoted(pyproject.split(BEGIN, 1)[1].split(END, 1)[0])


def test_ruff_excludes_match_the_drift_gate():
    gate, ruff = _drift_gate_paths(), _ruff_excluded_paths()
    assert len(set(ruff)) == len(ruff), "duplicate entries in the vendored block"
    only_gate = sorted(set(gate) - set(ruff))
    only_ruff = sorted(set(ruff) - set(gate))
    assert not only_gate and not only_ruff, (
        "the vendored block of [tool.ruff] extend-exclude in pyproject.toml and "
        "VENDORED_DIRS/VENDORED_FILES in scripts/check_vendored_upstream.sh differ.\n"
        f"  only in the drift gate (add to pyproject.toml): {only_gate}\n"
        f"  only in pyproject.toml (remove, or add to the drift gate): {only_ruff}"
    )
    missing = [p for p in ruff if not (REPO_ROOT / p).exists()]
    assert not missing, f"excluded paths that do not exist: {missing}"


if __name__ == "__main__":
    test_ruff_excludes_match_the_drift_gate()
    print(f"OK: {len(_ruff_excluded_paths())} vendored paths excluded from ruff")
    sys.exit(0)
