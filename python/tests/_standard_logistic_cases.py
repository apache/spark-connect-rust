# SPDX-License-Identifier: Apache-2.0
"""Run a standard ML example without optional Python transport or ML dependencies."""

import subprocess
import sys

_BLOCK_OPTIONAL_DEPENDENCIES = """
import importlib.abc
import sys

blocked = (
    "google.protobuf", "google.rpc", "grpc", "grpc_status", "py4j", "zstandard",
    "scipy", "sklearn", "torch",
)

class BlockOptionalDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise ModuleNotFoundError("Optional dependency blocked: " + fullname, name=fullname)

sys.meta_path.insert(0, BlockOptionalDependencies())
"""

def without_optional_dependencies(code):
    result = subprocess.run(
        [sys.executable, "-c", _BLOCK_OPTIONAL_DEPENDENCIES + code + """
assert not any(
    module == name or module.startswith(name + ".")
    for module in sys.modules for name in blocked
)
"""],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
