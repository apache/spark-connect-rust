"""PySpark utilities for Rust-backed client."""
import contextlib
import faulthandler
import functools
import itertools
import os
import re
import sys
import threading
import traceback
from contextlib import contextmanager
from types import TracebackType
from typing import (
    Any,
    Callable,
    IO,
    Iterator,
    Optional,
    TextIO,
    Tuple,
)

from pyspark.serializers import (
    write_int,
    read_int,
    write_with_length,
    SpecialLengths,
)

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

# Evaluation-type constants (Rust-backed), mirroring pyspark.util.PythonEvalType.
from pyspark._pyspark import PythonEvalType  # noqa: E402,F401


class VersionUtils:
    """
    Provides utility method to determine Spark versions with given input string.
    """

    @staticmethod
    def majorMinorVersion(sparkVersion: str) -> Tuple[int, int]:
        """
        Given a Spark version string, return the (major version number, minor version number).
        E.g., for 2.0.1-SNAPSHOT, return (2, 0).

        Examples
        --------
        >>> sparkVersion = "2.4.0"
        >>> VersionUtils.majorMinorVersion(sparkVersion)
        (2, 4)
        >>> sparkVersion = "2.3.0-SNAPSHOT"
        >>> VersionUtils.majorMinorVersion(sparkVersion)
        (2, 3)
        """
        m = re.search(r"^(\d+)\.(\d+)(\..*)?$", sparkVersion)
        if m is not None:
            return (int(m.group(1)), int(m.group(2)))
        else:
            raise ValueError(
                "Spark tried to parse '%s' as a Spark" % sparkVersion
                + " version string, but it could not find the major and minor"
                + " version numbers."
            )


def walk_tb(tb: Optional[TracebackType]) -> Iterator[TracebackType]:
    """Walk through traceback frames."""
    while tb is not None:
        yield tb
        tb = tb.tb_next


def try_simplify_traceback(tb: TracebackType) -> Optional[TracebackType]:
    """
    Simplify the traceback. It removes the tracebacks in the current package, and only
    shows the traceback that is related to the thirdparty and user-specified codes.
    """
    import pyspark

    root = os.path.dirname(pyspark.__file__)
    tb_next = None
    new_tb = None
    pairs = zip(walk_tb(tb), traceback.extract_tb(tb))
    last_seen = []

    for cur_tb, cur_frame in pairs:
        if not cur_frame.filename.startswith(root):
            # Filter the stacktrace from the PySpark source itself.
            last_seen = [(cur_tb, cur_frame)]
            break

    for cur_tb, cur_frame in reversed(list(itertools.chain(last_seen, pairs))):
        # Once we have seen the file names outside, don't skip.
        new_tb = TracebackType(
            tb_next=tb_next,
            tb_frame=cur_tb.tb_frame,
            tb_lasti=cur_tb.tb_frame.f_lasti,
            tb_lineno=cur_tb.tb_frame.f_lineno if cur_tb.tb_frame.f_lineno is not None else -1,
        )
        tb_next = new_tb
    return new_tb


def handle_worker_exception(
    e: BaseException, outfile: IO, hide_traceback: Optional[bool] = None
) -> None:
    """
    Handles exception for Python worker which writes SpecialLengths.PYTHON_EXCEPTION_THROWN (-2)
    and exception traceback info to outfile. JVM could then read from the outfile and perform
    exception handling there.

    Parameters
    ----------
    e : BaseException
        Exception handled
    outfile : IO
        IO object to write the exception info
    hide_traceback : bool, optional
        Whether to hide the traceback in the output.
        By default, hides the traceback if environment variable SPARK_HIDE_TRACEBACK is set.
    """

    if hide_traceback is None:
        hide_traceback = bool(os.environ.get("SPARK_HIDE_TRACEBACK", False))

    def format_exception() -> str:
        if hide_traceback:
            return "".join(traceback.format_exception_only(type(e), e))
        if os.environ.get("SPARK_SIMPLIFIED_TRACEBACK", False):
            tb = try_simplify_traceback(sys.exc_info()[-1])  # type: ignore[arg-type]
            if tb is not None:
                e.__cause__ = None
                return "".join(traceback.format_exception(type(e), e, tb))
        return traceback.format_exc()

    try:
        exc_info = format_exception()
        write_int(SpecialLengths.PYTHON_EXCEPTION_THROWN, outfile)
        write_with_length(exc_info.encode("utf-8"), outfile)
    except IOError:
        # JVM close the socket
        pass
    except BaseException:
        # Write the error to stderr if it happened while serializing
        print("PySpark worker failed with exception:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)


class _FaulthandlerHelper:
    """Helper class for managing faulthandler operations."""

    def __init__(self) -> None:
        self._log_path: Optional[str] = None
        self._log_file: Optional[TextIO] = None
        self._periodic_traceback = False
        self._reentry_depth = 0

    def start(self) -> None:
        self._reentry_depth += 1
        if self._log_path:
            # faulthandler is already enabled
            return
        self._log_path = os.environ.get("PYTHON_FAULTHANDLER_DIR", None)
        if self._log_path:
            self._log_path = os.path.join(self._log_path, str(os.getpid()))
            self._log_file = open(self._log_path, "w")

            faulthandler.enable(file=self._log_file)

    def stop(self) -> None:
        self._reentry_depth -= 1
        if self._reentry_depth > 0:
            return
        if self._log_path:
            faulthandler.disable()
            if self._log_file:
                self._log_file.close()
                self._log_file = None
            try:
                os.remove(self._log_path)
            finally:
                self._log_path = None
        if self._periodic_traceback:
            faulthandler.cancel_dump_traceback_later()
            self._periodic_traceback = False

    def start_periodic_traceback(self) -> None:
        # If the registration is already done - do nothing
        if self._periodic_traceback:
            return

        traceback_dump_interval_seconds = os.environ.get(
            "PYTHON_TRACEBACK_DUMP_INTERVAL_SECONDS", None
        )
        if traceback_dump_interval_seconds is not None and int(traceback_dump_interval_seconds) > 0:
            self._periodic_traceback = True
            faulthandler.dump_traceback_later(int(traceback_dump_interval_seconds), repeat=True)

    def with_faulthandler(self, func: Callable) -> Callable:
        """
        Registers fault handler for the duration of function execution.
        After function execution is over the faulthandler registration is cleaned as well,
        including any files created for the integration.
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                self.start()
                return func(*args, **kwargs)
            finally:
                self.stop()

        return wrapper

    @contextmanager
    def enable_faulthandler(self, start_periodic_traceback: bool = True) -> Iterator[None]:
        try:
            self.start()
            if start_periodic_traceback:
                self.start_periodic_traceback()
            yield
        finally:
            self.stop()


_faulthandler_helper = _FaulthandlerHelper()
with_faulthandler = _faulthandler_helper.with_faulthandler
start_faulthandler_periodic_traceback = _faulthandler_helper.start_periodic_traceback
enable_faulthandler = _faulthandler_helper.enable_faulthandler


def _print_missing_jar(lib_name: str, pkg_name: str, jar_name: str, spark_version: str) -> None:
    print(
        """
________________________________________________________________________________________________

  Spark %(lib_name)s libraries not found in class path. Try one of the following.

  1. Include the %(lib_name)s library and its dependencies with in the
     spark-submit command as

     $ bin/spark-submit --packages org.apache.spark:spark-%(pkg_name)s:%(spark_version)s ...

  2. Download the JAR of the artifact from Maven Central http://search.maven.org/,
     Group Id = org.apache.spark, Artifact Id = spark-%(jar_name)s, Version = %(spark_version)s.
     Then, include the jar in the spark-submit command as

     $ bin/spark-submit --jars <spark-%(jar_name)s.jar> ...

________________________________________________________________________________________________

"""
        % {
            "lib_name": lib_name,
            "pkg_name": pkg_name,
            "jar_name": jar_name,
            "spark_version": spark_version,
        }
    )
