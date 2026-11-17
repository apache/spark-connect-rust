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
asks for it, as PySpark 4.3.0's client does. Each case runs a child interpreter that connects to
a mock server and exits; the requests that the server received show what the exit sent.
"""

import os
import subprocess
import sys
import textwrap
import time

import pyspark.sql.connect.proto as pb2
import pytest
from _mock_connect_server import mock_connect_session, parse_message

_VARIABLE = "SPARK_CONNECT_RELEASE_SESSION_ON_EXIT"

# The child: it connects, prints the IDs of its sessions, then does what its mode asks for. The
# mode "stop_wait" stops the session, then prints "ready" and holds the child until a line on its
# stdin, for a test that stops the server before the child exits.
_CHILD = textwrap.dedent("""
    import os
    import sys

    from pyspark.sql import SparkSession

    remote, mode = sys.argv[1], sys.argv[2]
    spark = SparkSession.builder.remote(remote).getOrCreate()
    sessions = [spark]
    if mode == "two":
        sessions.append(spark.newSession())
    print(" ".join(session.session_id for session in sessions), flush=True)
    if mode == "stop":
        spark.stop()
    if mode == "set_after_creation":
        os.environ["SPARK_CONNECT_RELEASE_SESSION_ON_EXIT"] = "true"
    if mode == "unset_after_creation":
        del os.environ["SPARK_CONNECT_RELEASE_SESSION_ON_EXIT"]
    if mode == "stop_wait":
        spark.stop()
        print("ready", flush=True)
        sys.stdin.readline()
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


def test_d_variable_is_read_when_the_session_is_made(exit_service):
    # (d) The variable is read at the creation of the session, not at exit: one set after the
    # creation does not release the session, and one that is unset after it does.
    _, (session_id,) = _run_child(exit_service.remote, "set_after_creation")
    assert _exit_requests(exit_service, session_id) == ["clean_cache"]

    _, (session_id,) = _run_child(exit_service.remote, "unset_after_creation", variable="true")
    assert _exit_requests(exit_service, session_id) == ["clean_cache", "release_session"]


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


@pytest.mark.parametrize("variable", [None, "true"])
def test_f_stopped_session_exits_promptly_when_the_server_is_gone(variable):
    # (f) A session that is stopped while the server is up sends nothing at exit, as PySpark's
    # exit hook skips a closed client: when the server is gone before the child exits, the child
    # exits promptly with status 0 and sends no request, with and without the release variable.
    with mock_connect_session() as (spark, service):
        # The session of the test is released while the server is up, so that its stop, which
        # the fixture makes, does not wait on a server that is gone.
        spark.stop()
        child = subprocess.Popen(
            [sys.executable, "-c", _CHILD, service.remote, "stop_wait"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_child_env(variable),
        )
        try:
            (session_id,) = child.stdout.readline().split()
            assert child.stdout.readline().strip() == "ready"
            # The requests of the child up to its stop: its release, and nothing at exit.
            assert _exit_requests(service, session_id) == ["release_session"]
            # The server stops, so that nothing listens at the exit of the child.
            service.connection.send(("stop",))
            child.stdin.write("\n")
            child.stdin.close()
            started = time.monotonic()
            status = child.wait(timeout=60)
            elapsed = time.monotonic() - started
            stderr = child.stderr.read()
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
    assert status == 0, stderr
    assert elapsed < 30, elapsed
