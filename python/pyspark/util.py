"""PySpark utilities for Rust-backed client."""
import os
import sys
import traceback
from typing import TextIO

_is_remote_only = None


def print_exec(stream: TextIO) -> None:
    """Print the current exception traceback to ``stream`` (used by serializers)."""
    ei = sys.exc_info()
    traceback.print_exception(ei[0], ei[1], ei[2], None, stream)


def is_remote_only() -> bool:
    """
    Returns if the current running environment is only for Spark Connect.
    If users install pyspark-client alone, RDD API does not exist.
    """
    global _is_remote_only

    # Always return True for Spark Connect-only environment
    # In our case, we don't have the full Spark infrastructure
    if "SPARK_SKIP_CONNECT_COMPAT_TESTS" in os.environ:
        return True

    # For the Rust-backed client, we're always connect-only
    return True


def _parse_memory(s: str) -> int:
    """
    Parse a memory string in the format supported by Java (e.g. 1g, 200m) and
    return the value in MiB

    Examples
    --------
    >>> _parse_memory("256m")
    256
    >>> _parse_memory("2g")
    2048
    """
    units = {"g": 1024, "m": 1, "t": 1 << 20, "k": 1.0 / 1024}
    if s[-1].lower() not in units:
        raise ValueError("invalid format: " + s)
    return int(float(s[:-1]) * units[s[-1].lower()])
