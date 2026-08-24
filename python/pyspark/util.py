"""PySpark utilities for Rust-backed client."""
import os

_is_remote_only = None


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
