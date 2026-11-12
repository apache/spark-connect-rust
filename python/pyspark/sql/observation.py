"""pyspark.sql.observation - Observation, backed by the Rust _pyspark extension."""
from pyspark._pyspark import Observation

__all__ = ["Observation"]

# The module of the class, pyspark.sql.connect.observation, that of PySpark's Observation of
# Spark Connect, is imported with this one, after the class, which it re-exports: cloudpickle
# finds a module only in sys.modules, so it then pickles the class by reference to it, as it
# pickles PySpark's, and a Python worker of the server unpickles PySpark's Observation.
import pyspark.sql.connect.observation  # noqa: E402,F401
