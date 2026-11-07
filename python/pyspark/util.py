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
    Set,
    TextIO,
    Tuple,
    Union,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

from pyspark.serializers import (
    write_int,
    read_int,
    write_with_length,
    SpecialLengths,
)

_is_remote_only = None

# The ranges of the integral types of the JVM, as PySpark's.
JVM_BYTE_MIN: int = -(1 << 7)
JVM_BYTE_MAX: int = (1 << 7) - 1
JVM_SHORT_MIN: int = -(1 << 15)
JVM_SHORT_MAX: int = (1 << 15) - 1
JVM_INT_MIN: int = -(1 << 31)
JVM_INT_MAX: int = (1 << 31) - 1
JVM_LONG_MIN: int = -(1 << 63)
JVM_LONG_MAX: int = (1 << 63) - 1


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

def inheritable_thread_target(f: Union[Callable, "SparkSession"]) -> Callable:
    """
    Return thread target wrapper which is recommended to be used in PySpark when the
    pinned thread mode is enabled. The wrapper function, before calling original
    thread target, it inherits the inheritable properties specific
    to JVM thread such as ``InheritableThreadLocal``, or thread local such as tags
    with Spark Connect.

    When the pinned thread mode is off, the target wrapper is a no-op.

    .. versionadded:: 3.2.0

    .. versionchanged:: 3.5.0
        Supports Spark Connect.

    Parameters
    ----------
    f : function, or :class:`SparkSession`
        the original thread target, or :class:`SparkSession` if Spark Connect is being used.
        See the examples below.

    Notes
    -----
    This API is experimental.

    It is important to know that it captures the local properties or tags when you
    decorate it whereas :class:`InheritableThread` captures when the thread is started.
    Therefore, it is encouraged to decorate it when you want to capture the local
    properties.

    For example, the local properties or tags from the current Spark context or Spark
    session is captured when you define a function here instead of the invocation:

    >>> @inheritable_thread_target
    ... def target_func():
    ...     pass  # your codes.

    If you have any updates on local properties or tags afterwards, it would not be
    reflected to the Spark context in ``target_func()``.

    The example below mimics the behavior of JVM threads as close as possible:

    >>> Thread(target=inheritable_thread_target(target_func)).start()  # doctest: +SKIP

    If you're using Spark Connect or if you want to inherit the tags properly,
    you should explicitly provide Spark session as follows:

    >>> @inheritable_thread_target(session)  # doctest: +SKIP
    ... def target_func():
    ...     pass  # your codes.

    >>> Thread(target=inheritable_thread_target(session)(target_func)).start()  # doctest: +SKIP
    """
    # As PySpark's of Spark Connect, the only mode of this client: `f` is the session, whose
    # tags, those of this thread, the thread inherits, with the session as its active one.
    from pyspark.sql import SparkSession

    session = f

    def outer(ff: Callable) -> Callable:
        assert isinstance(session, SparkSession), (
            "f is expected to be SparkSession for spark connect"
        )
        tags = set(session.getTags())

        @functools.wraps(ff)
        def inner(*args: Any, **kwargs: Any) -> Any:
            # Propagates the active remote spark session to the current thread.
            assert isinstance(session, SparkSession), (
                "f is expected to be SparkSession for spark connect"
            )

            SparkSession._set_default_and_active_session(session)
            # Set the tags of the session in the child thread.
            session.clearTags()
            for tag in tags:
                session.addTag(tag)
            return ff(*args, **kwargs)

        return inner

    return outer


class InheritableThread(threading.Thread):
    """
    Thread that is recommended to be used in PySpark when the pinned thread mode is
    enabled. The wrapper function, before calling original thread target, it
    inherits the inheritable properties specific to JVM thread such as
    ``InheritableThreadLocal``, or thread local such as tags
    with Spark Connect.

    When the pinned thread mode is off, this works as :class:`threading.Thread`.

    .. versionadded:: 3.1.0

    .. versionchanged:: 3.5.0
        Supports Spark Connect.

    Notes
    -----
    This API is experimental.
    """

    _props: Any
    _tags: Set[str]
    _session: "SparkSession"

    def __init__(
        self, target: Callable, *args: Any, session: Optional["SparkSession"] = None, **kwargs: Any
    ):
        # As PySpark's of Spark Connect: the thread takes the tags of `session` of the thread
        # that starts it.
        assert session is not None, "Spark Connect must be provided."
        self._session = session

        def copy_local_properties(*a: Any, **k: Any) -> Any:
            assert hasattr(self, "_tags")
            assert session is not None
            session.clearTags()
            for tag in self._tags:
                session.addTag(tag)
            return target(*a, **k)

        super().__init__(target=copy_local_properties, *args, **kwargs)  # type: ignore[misc]

    def start(self) -> None:
        assert hasattr(self, "_session")
        self._tags = set(self._session.getTags())
        return super().start()

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


# Verbatim from PySpark: the local socket helpers of the JVM gateway. Spark Connect does not
# use them, but the modules of PySpark that the client vendors, such as
# pyspark.sql.pandas.conversion, import them.
import io  # noqa: E402
import socket  # noqa: E402
from typing import Union  # noqa: E402

from pyspark.errors import PySparkRuntimeError  # noqa: E402
from pyspark.serializers import UTF8Deserializer  # noqa: E402


def _create_local_socket(sock_info: "JavaArray") -> Tuple["io.BufferedRWPair", "socket.socket"]:
    """
    Create a local socket that can be used to load deserialized data from the JVM

    Parameters
    ----------
    sock_info : tuple
        Tuple containing port number and authentication secret for a local socket.

    Returns
    -------
    sockfile file descriptor of the local socket
    """
    sockfile: "io.BufferedRWPair"
    sock: "socket.socket"
    conn_info: int = sock_info[0]
    auth_secret: str = sock_info[1]
    sockfile, sock = local_connect_and_auth(conn_info, auth_secret)
    # The RDD materialization time is unpredictable, if we set a timeout for socket reading
    # operation, it will very possibly fail. See SPARK-18281.
    sock.settimeout(None)
    return sockfile, sock


@contextmanager
def _load_from_socket(sock_info: "JavaArray", serializer: "Serializer") -> Iterator[Any]:
    """
    Connect to a local socket described by sock_info and use the given serializer to yield data

    Parameters
    ----------
    sock_info : tuple
        Tuple containing port number and authentication secret for a local socket.
    serializer : class:`Serializer`
        The PySpark serializer to use

    Returns
    -------
    result of meth:`Serializer.load_stream`,
    usually a generator that yields deserialized data
    """
    try:
        sockfile, sock = _create_local_socket(sock_info)
        yield serializer.load_stream(sockfile)
    finally:
        sockfile.close()
        sock.close()


def _local_iterator_from_socket(sock_info: "JavaArray", serializer: "Serializer") -> Iterator[Any]:
    class PyLocalIterable:
        """Create a synchronous local iterable over a socket"""

        def __init__(self, _sock_info: "JavaArray", _serializer: "Serializer"):
            port: int
            auth_secret: str
            self.jsocket_auth_server: "JavaObject"
            port, auth_secret, self.jsocket_auth_server = _sock_info
            self._sockfile, self._sock = _create_local_socket((port, auth_secret))
            self._serializer = _serializer
            self._read_iter: Iterator[Any] = iter([])  # Initialize as empty iterator
            self._read_status = 1

        def __iter__(self) -> Iterator[Any]:
            while self._read_status == 1:
                # Request next partition data from Java
                write_int(1, self._sockfile)
                self._sockfile.flush()

                # If response is 1 then there is a partition to read, if 0 then fully consumed
                self._read_status = read_int(self._sockfile)
                if self._read_status == 1:
                    # Load the partition data as a stream and read each item
                    self._read_iter = self._serializer.load_stream(self._sockfile)
                    for item in self._read_iter:
                        yield item

                # An error occurred, join serving thread and raise any exceptions from the JVM
                elif self._read_status == -1:
                    self.jsocket_auth_server.getResult()

        def __del__(self) -> None:
            # If local iterator is not fully consumed,
            if self._read_status == 1:
                try:
                    # Finish consuming partition data stream
                    for _ in self._read_iter:
                        pass
                    # Tell Java to stop sending data and close connection
                    write_int(0, self._sockfile)
                    self._sockfile.flush()
                except Exception:
                    # Ignore any errors, socket is automatically closed when garbage-collected
                    pass
            self._sockfile.close()
            self._sock.close()

    return iter(PyLocalIterable(sock_info, serializer))


def local_connect_and_auth(
    conn_info: Optional[Union[str, int]], auth_secret: Optional[str]
) -> Tuple:
    """
    Connect to local host, authenticate with it, and return a (sockfile,sock) for that connection.
    Handles IPV4 & IPV6, does some error handling.

    Parameters
    ----------
    port : str or int, optional
    auth_secret : str, optional

    Returns
    -------
    tuple
        with (sockfile, sock)
    """
    is_unix_domain_socket = isinstance(conn_info, str) and auth_secret is None
    if is_unix_domain_socket:
        sock_path = conn_info
        assert isinstance(sock_path, str)
        sock = None
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(int(os.environ.get("SPARK_AUTH_SOCKET_TIMEOUT", 15)))
            sock.connect(sock_path)
            sockfile = sock.makefile("rwb", int(os.environ.get("SPARK_BUFFER_SIZE", 65536)))
            return (sockfile, sock)
        except socket.error as e:
            if sock is not None:
                sock.close()
            raise PySparkRuntimeError(
                errorClass="CANNOT_OPEN_SOCKET",
                messageParameters={
                    "errors": "tried to connect to %s, but an error occurred: %s"
                    % (sock_path, str(e)),
                },
            )

    sock = None
    errors = []
    # Support for both IPv4 and IPv6.
    addr = "127.0.0.1"
    if os.environ.get("SPARK_PREFER_IPV6", "false").lower() == "true":
        addr = "::1"
    for res in socket.getaddrinfo(addr, conn_info, socket.AF_UNSPEC, socket.SOCK_STREAM):
        af, socktype, proto, _, sa = res
        try:
            sock = socket.socket(af, socktype, proto)
            sock.settimeout(int(os.environ.get("SPARK_AUTH_SOCKET_TIMEOUT", 15)))
            sock.connect(sa)
            sockfile = sock.makefile("rwb", int(os.environ.get("SPARK_BUFFER_SIZE", 65536)))
            assert isinstance(auth_secret, str)
            _do_server_auth(sockfile, auth_secret)
            return (sockfile, sock)
        except socket.error as e:
            emsg = str(e)
            errors.append("tried to connect to %s, but an error occurred: %s" % (sa, emsg))
            if sock is not None:
                sock.close()
                sock = None
    raise PySparkRuntimeError(
        errorClass="CANNOT_OPEN_SOCKET",
        messageParameters={
            "errors": str(errors),
        },
    )


def _do_server_auth(conn: "io.IOBase", auth_secret: str) -> None:
    """
    Performs the authentication protocol defined by the SocketAuthHelper class on the given
    file-like object 'conn'.
    """
    write_with_length(auth_secret.encode("utf-8"), conn)
    conn.flush()
    reply = UTF8Deserializer().loads(conn)
    if reply != "ok":
        conn.close()
        raise PySparkRuntimeError(
            errorClass="UNEXPECTED_RESPONSE_FROM_SERVER",
            messageParameters={},
        )


# The names of PySpark's pyspark.util that its other modules and tools use, from Apache Spark
# v4.3.0.
import gc  # noqa: E402
from typing import Generator, List, TypeVar  # noqa: E402

__all__: List[str] = []

FuncT = TypeVar("FuncT", bound=Callable[..., Any])


def fail_on_stopiteration(f: Callable) -> Callable:
    """
    Wraps the input function to fail on 'StopIteration' by raising a 'RuntimeError'
    prevents silent loss of data when 'f' is used in a for loop in Spark code
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return f(*args, **kwargs)
        except StopIteration as exc:
            raise PySparkRuntimeError(
                errorClass="STOP_ITERATION_OCCURRED",
                messageParameters={
                    "exc": str(exc),
                },
            )

    return wrapper


@contextlib.contextmanager
def disable_gc() -> Generator[None, None, None]:
    gc_enabled_originally = gc.isenabled()
    if gc_enabled_originally:
        gc.disable()
    try:
        yield
    finally:
        if gc_enabled_originally:
            gc.enable()


# This function will be called in `pyspark` script as well,
# so this should be available without a running Spark.
# If move or rename, please update the script too.
def spark_connect_mode() -> str:
    """
    Return the env var SPARK_CONNECT_MODE; otherwise "1" if `pyspark_connect` is available.
    """
    connect_by_default = os.environ.get("SPARK_CONNECT_MODE")
    if connect_by_default is not None:
        return connect_by_default
    try:
        import pyspark_connect  # noqa: F401

        return "1"
    except ImportError:
        return "0"


def default_api_mode() -> str:
    """
    Return the default API mode.
    """
    if spark_connect_mode() == "1":
        return "connect"
    else:
        return "classic"
