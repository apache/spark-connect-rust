#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
The exit hook of a session: the ML cache of a session that is not stopped is cleaned when the
interpreter exits, and the session is released there if SPARK_CONNECT_RELEASE_SESSION_ON_EXIT
asks for it, as PySpark 4.3.0's client does. A session that the program dropped sends nothing, as
PySpark's that its `__del__` closed. Each case runs a child interpreter that connects to a mock
server and exits; the requests that the server received show what the exit sent.
"""

import os
import signal
import subprocess
import sys
import textwrap
import time

import pyspark.sql.connect.proto as pb2
import pytest
from _mock_connect_server import mock_connect_session, parse_message

_VARIABLE = "SPARK_CONNECT_RELEASE_SESSION_ON_EXIT"

# The child: it connects, prints the IDs of its sessions, then does what its mode asks for. The
# modes "stop_wait" and "new_session_dropped_wait" stop the session, then print "ready" and hold
# the child until a line on its stdin, for a test that stops the server before the child exits.
# The mode "live_wait" does so with its session live, and prints "exiting" when it exits, just
# before the exit hook of its session runs. The modes "client_*" print the ID of the session of
# the client that they drop.
_CHILD = textwrap.dedent("""
    import os
    import sys

    from pyspark.sql import SparkSession

    remote, mode = sys.argv[1], sys.argv[2]
    spark = SparkSession.builder.remote(remote).getOrCreate()
    sessions = [spark]
    if mode == "two" or mode.startswith("new_session"):
        sessions.append(spark.newSession())
    print(" ".join(session.session_id for session in sessions), flush=True)
    if mode in ("new_session_dropped", "new_session_dropped_wait"):
        del sessions[1]
    if mode == "new_session_dataframe":
        # Only a DataFrame of the new session is left, which holds the session.
        dataframe = sessions.pop().range(3)
    if mode in (
        "new_session_catalog_dropped",
        "new_session_catalog_collected",
        "new_session_streams_dropped",
        "new_session_streams_collected",
    ):
        # The catalog, or the StreamingQueryManager, of the new session holds the session, which
        # keeps it: a reference cycle, which its drop does not free, and which a collection frees.
        if "_catalog_" in mode:
            sessions[1].catalog
        else:
            sessions[1].streams
        del sessions[1]
        if mode.endswith("_collected"):
            import gc

            gc.collect()
    if mode == "new_session_dropped_wait":
        spark.stop()
        print("ready", flush=True)
        sys.stdin.readline()
    if mode == "client_dropped":
        from pyspark.sql.connect.client import SparkConnectClient

        # A client of a session of its own, which the child drops at once.
        print(SparkConnectClient(remote)._session_id, flush=True)
    if mode == "client_new_session_dropped":
        print(spark.client.newSession()._session_id, flush=True)
    if mode == "client_clone_dropped":
        print(spark.client.clone()._session_id, flush=True)
    if mode == "stop":
        spark.stop()
    if mode == "bad_global_extension":
        # An extension that is not an Any, which fails the clean_cache of the exit, whose request
        # carries the extensions, and not the release, whose request carries none.
        spark.client.global_user_context_extensions.append(("bad", 5))
    if mode == "set_after_creation":
        os.environ["SPARK_CONNECT_RELEASE_SESSION_ON_EXIT"] = "true"
    if mode == "unset_after_creation":
        del os.environ["SPARK_CONNECT_RELEASE_SESSION_ON_EXIT"]
    if mode == "release_attribute_set":
        spark.client._release_session_on_exit = True
    if mode == "release_attribute_unset":
        spark.client._release_session_on_exit = False
    if mode == "closed_attribute":
        spark.client._closed = True
    if mode == "closed_copy":
        import copy

        # A copy of the client, closed, whose request raises NO_ACTIVE_SESSION.
        copied = copy.copy(spark.client)
        copied._closed = True
        try:
            copied.get_configs("spark.a")
        except Exception as error:
            assert error.getCondition() == "NO_ACTIVE_SESSION", error
    if mode == "stop_wait":
        spark.stop()
        print("ready", flush=True)
        sys.stdin.readline()
    if mode in ("live_wait", "live_handler_wait"):
        import atexit

        if mode == "live_handler_wait":
            import signal

            class HandlerError(Exception):
                pass

            def handler(signum, frame):
                print("signal", flush=True)
                raise HandlerError("of the handler")

            signal.signal(signal.SIGINT, handler)
        print("ready", flush=True)
        sys.stdin.readline()
        # Registered after the session, so it runs before the session's exit hook.
        atexit.register(lambda: print("exiting", flush=True))
    if mode == "close_all":
        from pyspark._pyspark import _close_sessions_at_exit

        _close_sessions_at_exit()
    if mode == "atexit_execute":
        import atexit

        # Registered after the session, so it runs before the session's exit hook.
        def execute_at_exit():
            try:
                spark.range(3).collect()
                print("returned", flush=True)
            except BaseException as error:
                print(type(error).__module__, type(error).__name__, error, flush=True)

        atexit.register(execute_at_exit)
""")


@pytest.fixture(scope="module")
def _exit_mock_server():
    pytest.importorskip("grpc")
    with mock_connect_session() as session:
        yield session


@pytest.fixture
def exit_service(_exit_mock_server):
    _, service = _exit_mock_server
    service.reset()
    return service


def _child_env(variable=None):
    # The environment of the child, with the variable as given: unset when it is None.
    env = {key: value for key, value in os.environ.items() if key != _VARIABLE}
    if variable is not None:
        env[_VARIABLE] = variable
    return env


def _run_child_lines(remote, mode, variable=None, timeout=120):
    # Runs the child to its exit. Returns the IDs of its sessions and the lines it printed after
    # them.
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, remote, mode],
        capture_output=True,
        text=True,
        env=_child_env(variable),
        timeout=timeout,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    first, *rest = result.stdout.splitlines()
    return first.split(), rest


def _run_child(remote, mode, variable=None, timeout=120):
    # Runs the child to its exit. Returns its exit status and the IDs of its sessions.
    session_ids, _ = _run_child_lines(remote, mode, variable, timeout)
    return 0, session_ids


def _exit_requests(service, session_id):
    # The requests of the session that the server received, in order, of the kinds that the exit
    # sends: "clean_cache" for the ML cache, "release_session" for ReleaseSession, "execute" for
    # any other ExecutePlan, and "release_execute" for a ReleaseExecute, which the exit sends
    # none of, as PySpark's does not.
    requests = []
    for kind, data in service.recorded_order:
        if parse_message(data).get(1, [b""])[0].decode() != session_id:
            continue
        if kind == "execute":
            command = pb2.ExecutePlanRequest.FromString(data).plan.command
            if command.ml_command.HasField("clean_cache"):
                requests.append("clean_cache")
            else:
                requests.append("execute")
        elif kind in ("release_session", "release_execute"):
            requests.append(kind)
    return requests


def test_a_live_session_cleans_its_ml_cache_at_exit(exit_service):
    # (a) A session that is not stopped sends one clean_cache at exit, and nothing else.
    _, (session_id,) = _run_child(exit_service.remote, "plain")
    assert _exit_requests(exit_service, session_id) == ["clean_cache"]


def test_b_stopped_session_sends_no_clean_cache_at_exit(exit_service):
    # (b) A stopped session sends none at exit. Its stop releases it, once, as before.
    _, (session_id,) = _run_child(exit_service.remote, "stop")
    assert _exit_requests(exit_service, session_id) == ["release_session"]


def test_c_release_variable_releases_the_session_after_the_clean_cache(exit_service):
    # (c) With SPARK_CONNECT_RELEASE_SESSION_ON_EXIT=true, clean_cache, then ReleaseSession.
    _, (session_id,) = _run_child(exit_service.remote, "plain", variable="true")
    assert _exit_requests(exit_service, session_id) == ["clean_cache", "release_session"]


@pytest.mark.parametrize(
    ("value", "released"),
    [("1", True), ("TRUE", True), ("false", False), ("yes", False), ("", False)],
)
def test_c_release_variable_values_as_pyspark(exit_service, value, released):
    # As PySpark's: the variable is true for "true" or "1", in any case, and for no other value.
    _, (session_id,) = _run_child(exit_service.remote, "plain", variable=value)
    expected = ["clean_cache", "release_session"] if released else ["clean_cache"]
    assert _exit_requests(exit_service, session_id) == expected


def test_c_release_at_exit_carries_no_user_context_extensions(exit_service):
    # As PySpark's `_on_exit`, whose `_cleanup_ml_cache` fails to build its request of an extension
    # that is not an Any, and whose `release_session` builds one of no extensions: the release is
    # sent all the same.
    _, (session_id,) = _run_child(exit_service.remote, "bad_global_extension", variable="true")
    assert _exit_requests(exit_service, session_id) == ["release_session"]


def test_d_variable_is_read_when_the_session_is_made(exit_service):
    # (d) The variable is read at the creation of the session, not at exit: one set after the
    # creation does not release the session, and one that is unset after it does.
    _, (session_id,) = _run_child(exit_service.remote, "set_after_creation")
    assert _exit_requests(exit_service, session_id) == ["clean_cache"]

    _, (session_id,) = _run_child(exit_service.remote, "unset_after_creation", variable="true")
    assert _exit_requests(exit_service, session_id) == ["clean_cache", "release_session"]


def test_d_the_attribute_of_the_client_is_read_at_exit_as_pyspark(exit_service):
    # (d) As PySpark's `_on_exit`, which reads the `_release_session_on_exit` of the client, which
    # the variable sets when the client is made: set True, the exit releases the session, and set
    # False, it does not.
    _, (session_id,) = _run_child(exit_service.remote, "release_attribute_set")
    assert _exit_requests(exit_service, session_id) == ["clean_cache", "release_session"]

    _, (session_id,) = _run_child(exit_service.remote, "release_attribute_unset", variable="true")
    assert _exit_requests(exit_service, session_id) == ["clean_cache"]


def test_b_a_client_set_closed_sends_nothing_at_exit_as_pyspark(exit_service):
    # (b) As PySpark's `_on_exit`, of the `_closed` of the client of the session: set True, the
    # exit sends nothing, and a copy of the client that is closed, whose request raised, leaves
    # the client open, which cleans its ML cache at exit.
    _, (session_id,) = _run_child(exit_service.remote, "closed_attribute", variable="true")
    assert _exit_requests(exit_service, session_id) == []

    _, (session_id,) = _run_child(exit_service.remote, "closed_copy")
    assert _exit_requests(exit_service, session_id) == ["clean_cache"]


def test_g_closing_the_registered_sessions_makes_their_exit_send_nothing(exit_service):
    # The seam of the test suites, _close_sessions_at_exit: it closes each registered session
    # without a release, so the exit of a live session that it closed sends no request, as the
    # exit of a closed client does, and the session's ML cache is not cleaned at exit.
    _, (session_id,) = _run_child(exit_service.remote, "close_all")
    assert _exit_requests(exit_service, session_id) == []


def test_e_two_live_sessions_send_one_clean_cache_each(exit_service):
    # (e) Each live session of the process sends its own clean_cache at exit, once.
    _, session_ids = _run_child(exit_service.remote, "two")
    assert len(session_ids) == 2 and session_ids[0] != session_ids[1]
    for session_id in session_ids:
        assert _exit_requests(exit_service, session_id) == ["clean_cache"]


@pytest.mark.parametrize("variable", [None, "true"])
def test_i_a_new_session_that_the_child_dropped_sends_nothing_at_exit(exit_service, variable):
    # (i) PySpark's SparkSession.__del__ closes the client of a session object that is freed, and
    # the exit hook of a closed client sends nothing: a newSession() that the child dropped,
    # whose objects are all freed, sends no clean_cache and no release at exit, while the
    # session that it keeps sends them, with and without the release variable.
    _, (kept_id, dropped_id) = _run_child(exit_service.remote, "new_session_dropped", variable)
    kept = ["clean_cache", "release_session"] if variable else ["clean_cache"]
    assert _exit_requests(exit_service, kept_id) == kept
    assert _exit_requests(exit_service, dropped_id) == []


@pytest.mark.parametrize("variable", [None, "true"])
def test_i_a_new_session_that_a_dataframe_holds_cleans_its_ml_cache_at_exit(
    exit_service, variable
):
    # (i) A DataFrame holds its session, as PySpark's holds the session object, so a newSession()
    # of which the child keeps only a DataFrame is alive at exit and sends its exit requests.
    _, (_, session_id) = _run_child(exit_service.remote, "new_session_dataframe", variable)
    expected = ["clean_cache", "release_session"] if variable else ["clean_cache"]
    assert _exit_requests(exit_service, session_id) == expected


@pytest.mark.parametrize("variable", [None, "true"])
def test_i_a_dropped_new_session_whose_catalog_was_read_is_open_until_a_collection(
    exit_service, variable
):
    # (i) As PySpark's: once its catalog is read, a session and its Catalog, which holds the
    # session as its `_sparkSession`, are a reference cycle, which the drop of the session does not
    # free. So a newSession() that the child dropped after it read its catalog is open until a
    # collection frees it: an exit before that sends its clean_cache (and its release, with the
    # release variable), as PySpark's exit does, and after a collection, which closes it, nothing.
    _, (_, dropped_id) = _run_child(exit_service.remote, "new_session_catalog_dropped", variable)
    expected = ["clean_cache", "release_session"] if variable else ["clean_cache"]
    assert _exit_requests(exit_service, dropped_id) == expected
    _, (_, collected_id) = _run_child(
        exit_service.remote, "new_session_catalog_collected", variable
    )
    assert _exit_requests(exit_service, collected_id) == []


@pytest.mark.parametrize("variable", [None, "true"])
def test_i_a_dropped_new_session_whose_streams_were_read_is_open_until_a_collection(
    exit_service, variable
):
    # (i) As PySpark's: once its streams are read, a session and its StreamingQueryManager, which
    # holds the session as its `_session`, are a reference cycle, which the drop of the session
    # does not free. So a newSession() that the child dropped after it read its streams is open
    # until a collection frees it: an exit before that sends its clean_cache (and its release, with
    # the release variable), as PySpark's exit does, and after a collection, which closes it,
    # nothing.
    _, (_, dropped_id) = _run_child(exit_service.remote, "new_session_streams_dropped", variable)
    expected = ["clean_cache", "release_session"] if variable else ["clean_cache"]
    assert _exit_requests(exit_service, dropped_id) == expected
    _, (_, collected_id) = _run_child(
        exit_service.remote, "new_session_streams_collected", variable
    )
    assert _exit_requests(exit_service, collected_id) == []


@pytest.mark.parametrize(
    "mode", ["client_dropped", "client_new_session_dropped", "client_clone_dropped"]
)
def test_i_a_dropped_client_of_a_session_of_its_own_cleans_its_ml_cache_at_exit(
    exit_service, mode
):
    # (i) The atexit registration of PySpark's SparkConnectClient keeps the client until the exit,
    # and no session object owns a client of SparkConnectClient(connection), or of the newSession
    # or the clone of a client, which none closes: its exit hook sends its clean_cache, even
    # though the child dropped the client, as the session that the child keeps does. The requests
    # are told apart by the sessions that the child printed, as each other test's are: a request
    # of another session can reach the server too, such as a request of another process that is
    # retried after its server stopped (the release of an execution, or the execution), once this
    # server has the port of that one.
    (session_id,), (client_session_id,) = _run_child_lines(exit_service.remote, mode)
    assert client_session_id != session_id
    assert _exit_requests(exit_service, client_session_id) == ["clean_cache"]
    assert _exit_requests(exit_service, session_id) == ["clean_cache"]


def test_h_an_execution_once_the_interpreter_shuts_down_raises_pysparks_runtime_error(
    exit_service,
):
    # (h) PySpark's client sends its releases through a ThreadPoolExecutor, which refuses new work
    # once the interpreter has begun to shut down: an execution in an atexit handler raises its
    # RuntimeError after the first response, and sends no ReleaseExecute. So does the
    # clean_cache of the exit hook, which runs after the handler.
    (session_id,), lines = _run_child_lines(exit_service.remote, "atexit_execute")
    assert lines == [
        "builtins RuntimeError cannot schedule new futures after interpreter shutdown"
    ]
    assert _exit_requests(exit_service, session_id) == ["execute", "clean_cache"]


def _exit_once_the_server_is_gone(mode, variable, interrupt=False):
    # Runs the child in a mode that waits once it stopped its session, until the server of its
    # own stops: then the child exits. Returns the requests of each session of the child up to its
    # stop, the exit status of the child, the seconds that its exit took, and its stderr. With
    # `interrupt`, the child is sent SIGINT once its exit hooks run, and the seconds are those of
    # its exit after the signal.
    with mock_connect_session() as (spark, service):
        # The session of the test is released while the server is up, so that its stop, which
        # the fixture makes, does not wait on a server that is gone.
        spark.stop()
        child = subprocess.Popen(
            [sys.executable, "-c", _CHILD, service.remote, mode],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_child_env(variable),
        )
        try:
            session_ids = child.stdout.readline().split()
            assert child.stdout.readline().strip() == "ready"
            requests = [_exit_requests(service, session_id) for session_id in session_ids]
            # The server stops, so that nothing listens at the exit of the child.
            service.connection.send(("stop",))
            child.stdin.write("\n")
            child.stdin.close()
            if interrupt:
                assert child.stdout.readline().strip() == "exiting"
                # The exit hook of the session waits on the retries of its request by now.
                time.sleep(1)
                child.send_signal(signal.SIGINT)
            started = time.monotonic()
            status = child.wait(timeout=60)
            elapsed = time.monotonic() - started
            stderr = child.stderr.read()
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
    return requests, status, elapsed, stderr


@pytest.mark.parametrize("variable", [None, "true"])
def test_f_dropped_new_session_exits_promptly_when_the_server_is_gone(variable):
    # (f) A newSession() that the child dropped sends nothing at exit, as PySpark's session that
    # its __del__ closed: when the server is gone before the child exits, the child exits promptly
    # with status 0, where the clean_cache of the session would wait on its retries, with and
    # without the release variable.
    requests, status, elapsed, stderr = _exit_once_the_server_is_gone(
        "new_session_dropped_wait", variable
    )
    # The requests of the child up to the stop of the session that it kept: its release.
    assert requests == [["release_session"], []]
    assert status == 0, stderr
    assert elapsed < 30, elapsed


@pytest.mark.parametrize("variable", [None, "true"])
def test_f_stopped_session_exits_promptly_when_the_server_is_gone(variable):
    # (f) A session that is stopped while the server is up sends nothing at exit, as PySpark's
    # exit hook skips a closed client: when the server is gone before the child exits, the child
    # exits promptly with status 0 and sends no request, with and without the release variable.
    requests, status, elapsed, stderr = _exit_once_the_server_is_gone("stop_wait", variable)
    # The requests of the child up to its stop: its release, and nothing at exit.
    assert requests == [["release_session"]]
    assert status == 0, stderr
    assert elapsed < 30, elapsed


@pytest.mark.parametrize("variable", [None, "true"])
def test_f_a_signal_ends_the_wait_of_the_exit_hook_when_the_server_is_gone(variable):
    # (f) The clean_cache of a live session waits on its retries at exit when the server is gone,
    # as PySpark's does, and a signal ends that wait, as it does PySpark's, whose
    # `_cleanup_ml_cache` catches `Exception` only: the KeyboardInterrupt of SIGINT is raised out
    # of the exit hook, which atexit reports, and the child exits at once with status 0, without
    # waiting on a release either, with and without the release variable.
    requests, status, elapsed, stderr = _exit_once_the_server_is_gone(
        "live_wait", variable, interrupt=True
    )
    # The live session sent nothing up to the stop of the server.
    assert requests == [[]]
    assert status == 0, stderr
    assert elapsed < 30, elapsed
    assert "Exception ignored in atexit callback" in stderr, stderr
    assert "KeyboardInterrupt" in stderr, stderr


def test_f_the_plan_compression_lookup_of_the_exit_hook_catches_the_exception_of_a_handler(
        exit_service):
    # (f) The clean_cache of the exit hook of a session that ran no plan looks the plan compression
    # up first, as PySpark's `_set_command_in_plan` does, whose `except Exception` catches the
    # exception that a signal handler raises while the lookup waits: the lookup ends, and the exit
    # hook goes on to send its clean_cache, as PySpark's keeps waiting, and the child exits with
    # status 0, of no exception reported by atexit.
    exit_service.plan_compression_lookup_delay = 10.0
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD, exit_service.remote, "live_handler_wait"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_child_env(None),
    )
    try:
        (session_id,) = child.stdout.readline().split()
        assert child.stdout.readline().strip() == "ready"
        child.stdin.write("\n")
        child.stdin.close()
        assert child.stdout.readline().strip() == "exiting"
        # The exit hook of the session waits on its lookup by now.
        time.sleep(1)
        child.send_signal(signal.SIGINT)
        started = time.monotonic()
        status = child.wait(timeout=60)
        elapsed = time.monotonic() - started
        stdout, stderr = child.stdout.read(), child.stderr.read()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    assert status == 0, stderr
    assert stdout.split() == ["signal"]
    assert elapsed < 8, elapsed
    assert "Exception ignored" not in stderr, stderr
    assert _exit_requests(exit_service, session_id) == ["clean_cache"]
