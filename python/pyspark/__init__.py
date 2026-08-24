#!pyspark-rs Drop-in replacement for pyspark client
#
# Exposes the Rust-backed Spark Connect client as a pure-Python package.

__version__ = "0.1.0"

from pyspark.sql import (
    SparkSession,
    DataFrame,
    Column,
    Row,
)

# For compatibility with testing harness, provide a stub SparkConf that works with
# Spark Connect. This is imported by the testing utils but we don't need a full
# implementation since we're connect-only.
class SparkConf:
    """Stub SparkConf for Spark Connect testing.

    For pure Connect mode, we don't need a full SparkConf with JVM support.
    This stub accepts configuration as upstream would.
    """
    def __init__(self, loadDefaults=True, _jvm=None, _jconf=None):
        self._jconf = None
        self.settings = {}

    def set(self, key, value):
        """Set a configuration value."""
        self.settings[key] = value
        return self

    def get(self, key, defaultValue=None):
        """Get a configuration value."""
        return self.settings.get(key, defaultValue)

    def remove(self, key):
        """Remove a configuration key."""
        if key in self.settings:
            del self.settings[key]
        return self

    def setAll(self, settings):
        """Set multiple configuration values."""
        for key, value in settings.items():
            self.set(key, value)
        return self

    def getAll(self):
        """Get all configuration values."""
        return list(self.settings.items())


__all__ = [
    "SparkSession",
    "DataFrame",
    "Column",
    "Row",
    "SparkConf",
]
