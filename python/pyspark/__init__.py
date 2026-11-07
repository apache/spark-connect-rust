#!pyspark-rs Drop-in replacement for pyspark client
#
# Exposes the Rust-backed Spark Connect client as a pure-Python package.

from pyspark.version import __version__  # noqa: F401  (tracks the Spark version we target)

# --- Client identity ---------------------------------------------------------
# This distribution is ``pyspark-client-rust``: the Spark Connect client backed by
# the native Rust engine (tonic), installed under the ``pyspark`` import name as a
# drop-in for the reference ``pyspark-client``. These markers let user code and bug
# reports tell, at runtime, which client is in use -- this one, not the reference
# Python client. See https://apache.github.io/spark-connect-rust/which-client/
__rust_client__: bool = True
__engine__: str = "rust"  # native Rust core via tonic -- not grpcio/py4j

from functools import wraps
from typing import cast, Any, Callable, TypeVar, Union

# As the PySpark client's, of a Connect-only client: the names of pyspark but those of the
# classic SparkContext (RDD, SparkContext, SparkFiles, ...), which need a JVM.
from pyspark.conf import SparkConf
from pyspark.storagelevel import StorageLevel
from pyspark.accumulators import Accumulator, AccumulatorParam
from pyspark.serializers import MarshalSerializer, CPickleSerializer
from pyspark.taskcontext import TaskContext, BarrierTaskContext, BarrierTaskInfo
from pyspark.profiler import Profiler, BasicProfiler
from pyspark._globals import _NoValue  # noqa: F401

_F = TypeVar("_F", bound=Callable)


def since(version: Union[str, float]) -> Callable[[_F], _F]:
    """
    A decorator that annotates a function to append the version of Spark the function was added.
    """
    import re

    indent_p = re.compile(r"\n( +)")

    def deco(f: _F) -> _F:
        assert f.__doc__ is not None

        indents = indent_p.findall(f.__doc__)
        indent = " " * (min(len(m) for m in indents) if indents else 0)
        f.__doc__ = f.__doc__.rstrip() + "\n\n%s.. versionadded:: %s" % (indent, version)
        return f

    return deco


def keyword_only(func: _F) -> _F:
    """
    A decorator that forces keyword arguments in the wrapped method
    and saves actual input keyword arguments in `_input_kwargs`.

    Notes
    -----
    Should only be used to wrap a method where first arg is `self`
    """

    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if len(args) > 0:
            raise TypeError("Method %s forces keyword arguments." % func.__name__)
        self._input_kwargs = kwargs
        return func(self, **kwargs)

    return cast(_F, wrapper)


from pyspark.sql import (
    SparkSession,
    DataFrame,
    Column,
    Row,
)
from pyspark.util import InheritableThread, inheritable_thread_target

__all__ = [
    "SparkSession",
    "DataFrame",
    "Column",
    "Row",
    "SparkConf",
    "StorageLevel",
    "Accumulator",
    "AccumulatorParam",
    "MarshalSerializer",
    "CPickleSerializer",
    "Profiler",
    "BasicProfiler",
    "TaskContext",
    "BarrierTaskContext",
    "BarrierTaskInfo",
    "InheritableThread",
    "inheritable_thread_target",
    "__version__",
]
