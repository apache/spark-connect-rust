"""The setup of the processes of the tests of the drop-in."""

import signal
import sys

import pytest

# A pytest that a non-interactive shell starts in the background (`pytest ... &`, `nohup ... &`)
# inherits SIGINT as ignored, and Python then leaves it ignored: the SIGINT that the tests of the
# interrupts send would be dropped, and each of them would wait out its fallback (the retry
# budget, the timeout of its subprocess) before it fails. Python's own handler is installed
# instead, and the subprocesses of the tests start with the default too, as exec resets a
# signal that a handler catches.
if signal.getsignal(signal.SIGINT) == signal.SIG_IGN:
    signal.signal(signal.SIGINT, signal.default_int_handler)


# The take of the violations of the check of the clones of the parts of plans and expressions, of
# the drop-in's extension, which the configuration of the session turns on; None of official
# PySpark, which runs the tests as the reference and has no extension. The count of the violations
# that are reported so far, as of the end of the last test, and that test; and the reports of the
# violations between tests, which fail the session at its end.
_take_part_clone_violations = None
_violations_reported = 0
_last_test = None
_violations_between_tests = []


def pytest_configure(config):
    # The extension reports the parts of plans and expressions to Python's garbage collector while
    # no call that runs without the GIL may clone them: of each such call it knows the parts that
    # it may clone (see `gc` of crates/pyspark-rs). Its check of the clones that the core makes
    # finds a clone without the GIL, outside a counted call, of a part that no call marks busy:
    # one that a collection could see in one pass and not in the next. It is on from the start of
    # the session, so that the collection of the tests, the fixtures of every scope and the
    # threads that run between the tests are checked too; it cannot be on Windows, where a thread
    # cannot tell whether it holds the GIL.
    global _take_part_clone_violations
    try:
        from pyspark._pyspark import _check_part_clones
        from pyspark._pyspark import _take_part_clone_violations as take
    except ImportError:
        return
    on = _check_part_clones()
    if sys.platform == "win32":
        return
    assert on, "the check of the clones of the parts of plans and expressions does not turn on"
    _take_part_clone_violations = take


def _violations(count, logs, where):
    # Of the clones of parts, and of the locks of the states of options (`crate::gc::GilMutex`),
    # which a thread holds only while no Python code runs and the GIL is not released.
    return (
        f"{count} clone(s) of parts of plans and expressions without the GIL that no call of the "
        f"extension counts or marks busy, or lock(s) of the states of options held across Python "
        f"code or a release of the GIL, {where}:\n\n" + "\n\n".join(logs)
    )


# Where a violation outside a test's call can come from.
_OUTSIDE_A_TEST = (
    "in the setup or the teardown of fixtures of a module, a class or the session, in the "
    "collection, or in a thread"
)


@pytest.fixture(autouse=True)
def _no_clone_of_a_part_that_no_call_counts(request):
    # A violation in a test fails it. One between tests fails none, as the next test is not its
    # cause, which then runs: it fails the session at its end, of the tests around it (see
    # pytest_sessionfinish).
    global _violations_reported, _last_test
    take = _take_part_clone_violations
    if take is None:
        yield
        return
    _, _, before, logs = take()
    if before > _violations_reported:
        where = (
            f"before the first test, {request.node.nodeid}"
            if _last_test is None
            else f"between the tests {_last_test} and {request.node.nodeid}"
        )
        _violations_between_tests.append(
            _violations(before - _violations_reported, logs, f"{where} ({_OUTSIDE_A_TEST})")
        )
        _violations_reported = before
    yield
    _, _, after, logs = take()
    _violations_reported = after
    _last_test = request.node.nodeid
    if after > before:
        pytest.fail(_violations(after - before, logs, "in the test"), pytrace=False)


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
    # The clones between tests, and after the last test, of the teardown of the fixtures of the
    # last module and of the session, fail the session; and so does a check that went off
    # meanwhile, as a subinterpreter turns off CPython's check of the state of the GIL that it
    # reads.
    take = _take_part_clone_violations
    if take is not None:
        from pyspark._pyspark import _part_clones_checked

        _, _, count, logs = take()
        failures = list(_violations_between_tests)
        if count > _violations_reported:
            where = "before any test" if _last_test is None else f"after the last test, {_last_test}"
            failures.append(
                _violations(count - _violations_reported, logs, f"{where} ({_OUTSIDE_A_TEST})")
            )
        if not _part_clones_checked():
            failures.append("the check of the clones of the parts went off during the session")
        if failures:
            reporter = session.config.pluginmanager.get_plugin("terminalreporter")
            if reporter is not None:
                reporter.write_sep("=", "the check of the clones of the parts", red=True)
            for failure in failures:
                if reporter is not None:
                    reporter.write_line(failure, red=True)
                else:
                    print(failure, file=sys.stderr)
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
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
    # The threads that the tests left and that are not daemons end first, for a while, as the exit
    # waits for them anyway: their calls end as they would, rather than fail of a closed session.
    # The calls of a daemon thread fail then; once the exit runs the extension's `atexit` hook, the
    # thread builds no exception of them and waits for the end of the process (see
    # `deliver::python_work` of crates/pyspark-rs).
    import threading
    import time

    deadline = time.monotonic() + 10
    for thread in threading.enumerate():
        if thread is not threading.current_thread() and not thread.daemon:
            thread.join(max(0.0, deadline - time.monotonic()))
    _close_sessions_at_exit()
