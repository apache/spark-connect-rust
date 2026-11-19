"""The setup of the processes of the tests of the drop-in."""

import signal

import pytest

# A pytest that a non-interactive shell starts in the background (`pytest ... &`, `nohup ... &`)
# inherits SIGINT as ignored, and Python then leaves it ignored: the SIGINT that the tests of the
# interrupts send would be dropped, and each of them would wait out its fallback (the retry
# budget, the timeout of its subprocess) before it fails. Python's own handler is installed
# instead, and the subprocesses of the tests start with the default too, as exec resets a
# signal that a handler catches.
if signal.getsignal(signal.SIGINT) == signal.SIG_IGN:
    signal.signal(signal.SIGINT, signal.default_int_handler)


@pytest.fixture(autouse=True)
def _debugging_cache_of_each_test():
    # PySpark's is_debugging_enabled keeps its first answer of
    # spark.python.sql.dataFrameDebugging.enabled for the interpreter
    # (pyspark.errors.utils._enable_debugging_cache), and the first Column method of a test that
    # has an active session reads it: True of a mock server, which takes the default. Each test
    # leaves the cache as it found it, so that a test does not decide the call-site origins that
    # the expressions of the tests after it carry, whatever the order of the files (see
    # _debugging_read_without_a_session of test_dropin_offline.py).
    import pyspark.errors.utils as errors_utils

    cached = errors_utils._enable_debugging_cache
    yield
    errors_utils._enable_debugging_cache = cached


@pytest.fixture(autouse=True, scope="module")
def _debugging_cache_of_each_module():
    # As _debugging_cache_of_each_test, of the fixtures of a module's scope: pytest sets them up
    # before the fixtures of each test, so a session that such a fixture makes reads the cache
    # before _debugging_cache_of_each_test records it (the objects of
    # test_dropin_class_modules_offline.py). Each module leaves the cache as it found it too.
    import pyspark.errors.utils as errors_utils

    cached = errors_utils._enable_debugging_cache
    yield
    errors_utils._enable_debugging_cache = cached


def pytest_sessionfinish(session, exitstatus):
    # The exit of this process sends the ML cache clean of each session that is not stopped, and
    # waits on its retries when the server is gone, as PySpark's exit does. A session that the
    # tests dropped sends nothing, but they keep sessions open to servers that are stopped or
    # never listened, as module globals, fixtures and clients of sessions of their own. Closing
    # them here, without a release, makes their exit hooks return at once, as PySpark's skip of a
    # closed client does; the product's exit is unchanged for a live session. Official PySpark,
    # which runs the tests as the reference, has no such seam.
    try:
        from pyspark._pyspark import _close_sessions_at_exit
    except ImportError:
        return
    _close_sessions_at_exit()
