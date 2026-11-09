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
"""Streaming listener lifecycle tests against the separately hosted mock service."""

import subprocess
import sys
import tempfile
import threading
import time
import warnings
from pathlib import Path

import pytest

from pyspark.sql.streaming.listener import StreamingQueryListener


class IdleListener(StreamingQueryListener):
    def __init__(self):
        super().__init__()
        self.timestamps = []
        self.condition = threading.Condition()

    def onQueryStarted(self, event):
        pass

    def onQueryProgress(self, event):
        pass

    def onQueryIdle(self, event):
        with self.condition:
            self.timestamps.append(event.timestamp)
            self.condition.notify_all()

    def onQueryTerminated(self, event):
        pass

    def wait_for(self, timestamp):
        with self.condition:
            assert self.condition.wait_for(lambda: timestamp in self.timestamps, timeout=5)


def emit_idle(service, session, timestamp):
    service.listener_event(session.session_id, 3, {
        "id": "00000000-0000-4000-8000-000000000001",
        "runId": "00000000-0000-4000-8000-000000000002",
        "timestamp": timestamp,
    })


def run_listener_case(*arguments):
    # A deadlocked native callback must fail without hanging the test runner. A file
    # keeps a killed child's mock-service process from holding a capture pipe open.
    with tempfile.TemporaryFile(mode="w+") as output:
        try:
            result = subprocess.run(
                [sys.executable, str(Path(__file__)), *arguments],
                stdout=output, stderr=subprocess.STDOUT, timeout=30,
            )
        except subprocess.TimeoutExpired:
            output.seek(0)
            pytest.fail("Concurrent listener shutdown timed out:\n" + output.read())
        output.seek(0)
        assert result.returncode == 0, output.read()


def listener_warnings(caught):
    prefixes = ("Failed to add the listener", "Failed to remove the listener")
    return [str(item.message) for item in caught
            if issubclass(item.category, UserWarning) and str(item.message).startswith(prefixes)]


class TestStreamingListenerLifecycle:
    @pytest.mark.parametrize("remove_failure, callback_action", [
        (False, "remove"), (True, "remove"), (False, "stop"), (True, "stop"), (False, "add"),
        (False, "remove_duplicate"), (True, "remove_duplicate"),
    ])
    def test_streaming_listener_callback_lifecycle_calls_during_external_close(
        self, remove_failure, callback_action,
    ):
        run_listener_case(
            "--concurrent-close-case", str(int(remove_failure)), callback_action,
        )

    @pytest.mark.parametrize("method", ["stop", "client_close"])
    def test_streaming_listener_terminal_teardown_cancels_established_stream_registration(
        self, method,
    ):
        run_listener_case("--pending-registration-case", method)

    def test_streaming_listener_stop_releases_a_session_with_a_callback_rpc(self):
        run_listener_case("--callback-rpc-case")

    @pytest.mark.parametrize("method", ["stop", "client_close"])
    @pytest.mark.parametrize("blocker", ["event", "lock"])
    def test_streaming_listener_terminal_teardown_does_not_wait_for_a_callback(
        self, method, blocker,
    ):
        run_listener_case("--blocked-callback-case", method, blocker)

    def test_streaming_listener_started_waits_require_the_requested_rpc(self):
        run_listener_case("--second-rpc-case")

    @pytest.mark.parametrize("termination", ["end", "error"])
    def test_streaming_listener_ended_stream_clears_registry_and_reopens(self, termination):
        run_listener_case("--ended-stream-case", termination)

    def test_streaming_listener_managers_share_a_session_registry(self, mock_connect):
        spark, service = mock_connect
        service.listener_bus_enabled = True
        first, second = IdleListener(), IdleListener()
        assert spark.streams is spark.streams
        alias = spark.range(1).sparkSession.streams
        try:
            spark.streams.addListener(first)
            spark.streams.addListener(second)
            assert service.listener_subscriptions == 1
            emit_idle(service, spark, "both")
            first.wait_for("both")
            second.wait_for("both")

            # A handle obtained through a DataFrame can remove a listener registered
            # through the original session, without stopping the other listener.
            alias.removeListener(first)
            assert service.listener_removals == 0
            emit_idle(service, spark, "second")
            second.wait_for("second")
            assert first.timestamps == ["both"]
            spark.streams.removeListener(first)
            assert service.listener_removals == 0
            spark.streams.removeListener(second)
            assert service.listener_removals == 1
            assert service.wait_listener_streams_finished(1)
            spark.streams.close()
            assert service.listener_removals == 1
        finally:
            spark.streams.close()

    def test_streaming_listener_close_reopens_the_same_manager(self, mock_connect):
        spark, service = mock_connect
        service.listener_bus_enabled = True
        manager = spark.streams
        first, second = IdleListener(), IdleListener()
        try:
            manager.addListener(first)
            spark.streams.close()
            assert service.listener_removals == 1
            assert service.wait_listener_streams_finished(1)
            spark.streams.addListener(second)
            assert service.listener_subscriptions == 2
            emit_idle(service, spark, "reopened")
            second.wait_for("reopened")
            assert first.timestamps == []
        finally:
            manager.close()
        assert service.listener_removals == 2
        assert service.wait_listener_streams_finished(2)

    def test_streaming_listener_new_sessions_have_separate_buses(self, mock_connect):
        spark, service = mock_connect
        service.listener_bus_enabled = True
        other = spark.newSession()
        first, second = IdleListener(), IdleListener()
        try:
            spark.streams.addListener(first)
            other.streams.addListener(second)
            assert service.listener_subscriptions == 2
            other.streams.close()
            emit_idle(service, spark, "original")
            first.wait_for("original")
            assert second.timestamps == []
            assert service.listener_removals == 1
        finally:
            other.stop()
            spark.streams.close()
        assert service.listener_removals == 2

    def test_streaming_listener_duplicate_ids_are_removed_only_from_the_matching_bus(
        self, mock_connect,
    ):
        spark, service = mock_connect
        service.listener_bus_enabled = True
        other = spark.newSession()
        listener = IdleListener()
        try:
            spark.streams.addListener(listener)
            spark.streams.addListener(listener)
            first, second = listener._rust_listener_ids
            other.streams.addListener(listener)
            third = listener._rust_listener_ids[-1]
            assert len({first, second, third}) == 3
            assert spark.streams._sqlb._listener_bus == [listener, listener]
            assert other.streams._sqlb._listener_bus == [listener]
            assert service.listener_subscriptions == 2

            other.streams.removeListener(listener)
            assert listener._rust_listener_ids == [first, second]
            assert other.streams._sqlb._listener_bus == []
            assert spark.streams._sqlb._listener_bus == [listener, listener]
            assert service.listener_removals == 1
            assert service.wait_listener_streams_finished(1)

            spark.streams.removeListener(listener)
            assert listener._rust_listener_ids == [second]
            assert spark.streams._sqlb._listener_bus == [listener]
            assert service.listener_removals == 1
            spark.streams.removeListener(listener)
            assert not hasattr(listener, "_rust_listener_ids")
            assert spark.streams._sqlb._listener_bus == []
            assert service.listener_removals == 2
            assert service.wait_listener_streams_finished(2)
        finally:
            other.stop()
            spark.streams.close()

    def test_streaming_listener_registration_failure_can_be_retried(self, mock_connect):
        spark, service = mock_connect
        service.listener_bus_enabled = True
        service.listener_register_failure = True
        listener = IdleListener()
        with pytest.warns(UserWarning, match="listener registration failed") as caught:
            assert spark.streams.addListener(listener) is None
        assert "The listener is not added, please add it again." in str(caught[0].message)
        assert not hasattr(listener, "_rust_listener_ids")
        service.listener_register_failure = False
        try:
            spark.streams.addListener(listener)
            emit_idle(service, spark, "retried")
            listener.wait_for("retried")
            assert service.listener_subscriptions == 2
        finally:
            spark.streams.close()

    @pytest.mark.parametrize("method", ["removeListener", "close"])
    def test_streaming_listener_removal_failure_can_be_retried(self, mock_connect, method):
        spark, service = mock_connect
        service.listener_bus_enabled = True
        listener = IdleListener()
        spark.streams.addListener(listener)
        listener_ids = list(listener._rust_listener_ids)

        def remove():
            if method == "removeListener":
                return spark.streams.removeListener(listener)
            return spark.streams.close()

        service.listener_remove_failure = True
        try:
            with pytest.warns(UserWarning, match="listener removal failed") as caught:
                assert remove() is None
            assert (
                "The listener is not removed, please remove it again."
                in str(caught[0].message)
            )
            assert listener._rust_listener_ids == listener_ids
            assert service.listener_streams_finished == 0
            emit_idle(service, spark, "still-registered")
            listener.wait_for("still-registered")

            service.listener_remove_failure = False
            assert remove() is None
            assert service.listener_removals == 2
            assert service.wait_listener_streams_finished(1)
            # A completed remove consumes its id; manager close leaves stale private ids,
            # which must not prevent registering and removing the same object again.
            remaining_ids = listener_ids if method == "close" else []
            assert getattr(listener, "_rust_listener_ids", []) == remaining_ids
            # A fresh registration after removal opens a new server subscription.
            spark.streams.addListener(listener)
            assert listener._rust_listener_ids[:-1] == remaining_ids
            assert listener._rust_listener_ids[-1] not in listener_ids
            assert service.listener_subscriptions == 2
            emit_idle(service, spark, "registered-again")
            listener.wait_for("registered-again")
            spark.streams.removeListener(listener)
            assert getattr(listener, "_rust_listener_ids", []) == remaining_ids
            assert service.listener_removals == 3
            assert service.wait_listener_streams_finished(2)
        finally:
            service.listener_remove_failure = False
            spark.streams.close()

    @pytest.mark.parametrize("release_failure", [False, True])
    def test_streaming_listener_stop_releases_without_a_listener_removal_rpc(
        self, mock_connect, caplog, release_failure,
    ):
        spark, service = mock_connect
        service.listener_bus_enabled = True
        spark.streams.addListener(IdleListener())
        # An ordinary close would block/fail here; terminal stop must use only ReleaseSession.
        service.listener_remove_blocked = True
        service.listener_remove_failure = True
        service.release_session_failure = release_failure
        try:
            with caplog.at_level("WARNING", logger="pyspark"):
                assert spark.stop() is None
            assert spark.is_stopped
            assert service.listener_removals == 0
            assert service.release_session_calls == 1
            assert spark.streams._sqlb._listener_bus == []
            assert service.wait_listener_streams_finished(1)
            messages = [record.message for record in caplog.records if record.name == "pyspark"]
            if release_failure:
                assert len(messages) == 1
                assert messages[0].startswith(
                    "session.stop(): Session could not be released. Error: "
                )
                assert messages[0].endswith("session release failed")
            else:
                assert messages == []
            spark.stop()
            assert service.listener_removals == 0
            assert service.release_session_calls == 1
        finally:
            service.listener_remove_failure = False
            service.listener_remove_blocked = False
            service.release_session_failure = False
            spark.streams.close()

    def test_streaming_listener_client_close_cancels_without_releasing(self, mock_connect):
        spark, service = mock_connect
        service.listener_bus_enabled = True
        spark.streams.addListener(IdleListener())
        spark.client.close()
        assert spark.is_stopped
        assert spark.streams._sqlb._listener_bus == []
        assert service.wait_listener_streams_finished(1)
        assert service.listener_removals == 0
        assert service.release_session_calls == 0
        spark.stop()
        assert service.release_session_calls == 0


def concurrent_close_case(remove_failure, callback_action):
    from _mock_connect_server import mock_connect_session

    with mock_connect_session() as (spark, service):
        service.listener_bus_enabled = True
        service.listener_remove_blocked = True
        service.listener_remove_failure = remove_failure
        entered, release, removed = (threading.Event() for _ in range(3))
        callback_threads, errors = [], []
        registrations = 2 if callback_action == "remove_duplicate" else 1
        caller_thread = threading.get_ident()
        attempted = IdleListener()

        class RemovingListener(IdleListener):
            def onQueryIdle(self, event):
                if event.timestamp != "closing":
                    return super().onQueryIdle(event)
                callback_threads.append(threading.get_ident())
                entered.set()
                assert release.wait(5)
                if callback_action == "stop":
                    spark.stop()
                elif callback_action == "add":
                    spark.streams.addListener(attempted)
                else:
                    spark.streams.removeListener(self)
                if len(callback_threads) == registrations:
                    removed.set()

        def close():
            try:
                assert spark.streams.close() is None
            except Exception as error:
                errors.append(error)

        listener = RemovingListener()
        for _ in range(registrations):
            spark.streams.addListener(listener)
        listener_ids = list(listener._rust_listener_ids)
        closing = threading.Thread(target=close, daemon=True)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                emit_idle(service, spark, "closing")
                assert entered.wait(5)
                closing.start()
                try:
                    assert service.wait_listener_removal()
                    release.set()
                    # The RPC stays gated until the callback's remove/stop has returned.
                    assert removed.wait(5)
                    assert len(callback_threads) == registrations
                    assert all(thread not in (caller_thread, closing.ident)
                               for thread in callback_threads)
                    assert closing.is_alive()
                    assert listener._rust_listener_ids == listener_ids
                finally:
                    release.set()
                    service.release_listener_removal()
                    closing.join(5)
            assert not closing.is_alive()
            assert errors == []
            relevant = listener_warnings(caught)
            add_warnings = [message for message in relevant
                            if message.startswith("Failed to add the listener")]
            close_warnings = [message for message in relevant
                              if message.startswith("Failed to remove the listener")]
            if callback_action == "add":
                assert len(add_warnings) == 1
                assert "cannot change the listener bus from its own callback" in add_warnings[0]
                assert "The listener is not added, please add it again." in add_warnings[0]
                assert not hasattr(attempted, "_rust_listener_ids")
                assert attempted.timestamps == []
            else:
                assert add_warnings == []
            if remove_failure:
                assert len(close_warnings) == 1
                assert "listener removal failed" in close_warnings[0]
            else:
                assert close_warnings == []
            if callback_action == "stop":
                assert spark.is_stopped
                assert spark.streams._sqlb._listener_bus == []
                assert service.release_session_calls == 1
                assert service.listener_removals == 1
                assert service.wait_listener_streams_finished(1)
                return
            if remove_failure:
                assert spark.streams._sqlb._listener_bus == [listener] * registrations
                assert service.listener_streams_finished == 0
                emit_idle(service, spark, "still-registered")
                listener.wait_for("still-registered")
                service.listener_remove_failure = False
                service.listener_remove_blocked = False
                for remaining in range(registrations - 1, -1, -1):
                    spark.streams.removeListener(listener)
                    assert spark.streams._sqlb._listener_bus == [listener] * remaining
                    if remaining:
                        assert listener._rust_listener_ids == listener_ids[-remaining:]
                    else:
                        assert not hasattr(listener, "_rust_listener_ids")
                assert service.listener_removals == 2
            else:
                assert service.listener_removals == 1
            assert service.wait_listener_streams_finished(1)
            # Successful external close leaves stale ids. A fresh add appends a new id
            # and opens exactly one subscription; failed-close retries consumed their ids.
            remaining_ids = list(getattr(listener, "_rust_listener_ids", []))
            spark.streams.addListener(listener)
            assert listener._rust_listener_ids[:-1] == remaining_ids
            assert listener._rust_listener_ids[-1] not in listener_ids
            assert service.listener_subscriptions == 2
            emit_idle(service, spark, "registered-again")
            listener.wait_for("registered-again")
        finally:
            release.set()
            service.release_listener_removal()
            if closing.ident is not None:
                closing.join(5)
            service.listener_remove_failure = False
            service.listener_remove_blocked = False
            spark.streams.close()


def callback_rpc_case():
    from _mock_connect_server import mock_connect_session

    with mock_connect_session() as (spark, service):
        service.listener_bus_enabled = True
        finished, returned = threading.Event(), threading.Event()
        callback_errors, stopping_errors = [], []

        class RpcListener(IdleListener):
            def onQueryIdle(self, event):
                try:
                    spark.range(1).collect()
                except Exception as error:
                    callback_errors.append(error)
                finally:
                    finished.set()

        def stop():
            try:
                assert spark.stop() is None
                returned.set()
            except Exception as error:
                stopping_errors.append(error)

        spark.streams.addListener(RpcListener())
        service.block_execute = True
        service.release_session_unblocks_execute = True
        stopping = threading.Thread(target=stop, daemon=True)
        try:
            emit_idle(service, spark, "callback-rpc")
            assert service.wait_execute_started(1)
            assert not finished.is_set()
            stopping.start()
            # The mock's five-second fallback must not rescue a join-before-release deadlock.
            assert returned.wait(2), "stop waited for the callback RPC before releasing its session"
            # No explicit response release has run: only ReleaseSession can open the gate.
            assert finished.wait(2), "ReleaseSession did not unblock the callback RPC"
            assert service.release_session_calls == 1
            assert service.execute_gate_timeouts == 0
            assert spark.is_stopped
            assert spark.streams._sqlb._listener_bus == []
            assert service.listener_removals == 0
        finally:
            service.release_response()
            if stopping.ident is not None:
                stopping.join(5)
            assert finished.wait(5), "callback RPC did not finish after its response was released"
        assert not stopping.is_alive()
        assert stopping_errors == []
        # A callback already in flight can finish normally or observe terminal session closure.
        assert all("NO_ACTIVE_SESSION" in str(error) for error in callback_errors)
        assert service.execute_gate_timeouts == 0
        assert service.wait_listener_streams_finished(1)


def blocked_callback_case(method, blocker):
    from _mock_connect_server import mock_connect_session

    with mock_connect_session() as (spark, service):
        service.listener_bus_enabled = True
        caller_ready, requested, returned = (threading.Event() for _ in range(3))
        entered, finished = threading.Event(), threading.Event()
        release_caller, release_callback = threading.Event(), threading.Event()
        callback_lock, errors = threading.Lock(), []

        class BlockingListener(IdleListener):
            def onQueryIdle(self, event):
                try:
                    if event.timestamp == "blocked":
                        entered.set()
                        if blocker == "event":
                            assert release_callback.wait(5), "callback Event was not released"
                        else:
                            assert callback_lock.acquire(timeout=5), "callback lock timed out"
                            callback_lock.release()
                    super().onQueryIdle(event)
                except Exception as error:
                    errors.append(error)
                finally:
                    if event.timestamp == "blocked":
                        finished.set()

        def teardown():
            # The same caller holds the lock and invokes terminal teardown. Joining its
            # callback would wait for a lock that this caller cannot release until return.
            with callback_lock:
                caller_ready.set()
                try:
                    assert requested.wait(5), "terminal teardown was not requested"
                    if method == "stop":
                        assert spark.stop() is None
                    else:
                        assert spark.client.close() is None
                    returned.set()
                    assert release_caller.wait(5), "terminal caller was not released"
                except Exception as error:
                    errors.append(error)

        listener = BlockingListener()
        spark.streams.addListener(listener)
        stopping = threading.Thread(target=teardown, daemon=True)
        try:
            stopping.start()
            assert caller_ready.wait(5)
            emit_idle(service, spark, "blocked")
            assert entered.wait(5)
            # Queue another event while the callback still owns the worker, before
            # cancellation can retire the server queue. It must never start a dispatch.
            emit_idle(service, spark, "queued-before-terminal")
            requested.set()
            assert returned.wait(2), "terminal teardown waited for an in-progress callback"
            assert not finished.is_set()
            assert listener.timestamps == []
            assert spark.is_stopped
            assert spark.streams._sqlb._listener_bus == []
            assert service.listener_removals == 0
            assert service.release_session_calls == (1 if method == "stop" else 0)
            # The current event's listener snapshot may finish after terminal return.
            # A fresh event must not start another dispatch from the canceled stream.
            emit_idle(service, spark, "after-terminal")
        finally:
            requested.set()
            release_callback.set()
            release_caller.set()
            stopping.join(5)
            if entered.is_set():
                assert finished.wait(5), "in-progress callback did not finish during cleanup"
        assert not stopping.is_alive()
        assert errors == []
        assert service.wait_listener_streams_finished(1)
        assert listener.timestamps == ["blocked"]


def pending_registration_case(method):
    from _mock_connect_server import mock_connect_session

    with mock_connect_session() as (spark, service):
        service.listener_bus_enabled = True
        # Early mock headers establish a stream before its acknowledgement. Establishing
        # the RPC itself remains subject to the transport's deadline/retry policy.
        service.listener_register_blocked = True
        listener, errors = IdleListener(), []

        def add():
            try:
                assert spark.streams.addListener(listener) is None
            except Exception as error:
                errors.append(error)

        def teardown():
            try:
                if method == "stop":
                    assert spark.stop() is None
                else:
                    assert spark.client.close() is None
            except Exception as error:
                errors.append(error)

        starting = threading.Thread(target=add, daemon=True)
        stopping = threading.Thread(target=teardown, daemon=True)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                # Unrelated process-wide warnings must not change lifecycle assertions.
                warnings.warn("unrelated warning used to check filtering", DeprecationWarning)
                starting.start()
                assert service.wait_listener_registration()
                stopping.start()
                stopping.join(5)
                assert not stopping.is_alive(), (
                    "terminal teardown waited for stream acknowledgement"
                )
                starting.join(5)
                assert not starting.is_alive(), "registration was not canceled"
            assert errors == []
            relevant = listener_warnings(caught)
            assert len(relevant) == 1
            assert "NO_ACTIVE_SESSION" in relevant[0]
            assert "The listener is not added, please add it again." in relevant[0]
            assert not hasattr(listener, "_rust_listener_ids")
            assert spark.is_stopped
            assert spark.streams._sqlb._listener_bus == []
            assert service.listener_removals == 0
            assert service.release_session_calls == (1 if method == "stop" else 0)
            assert service.wait_listener_streams_finished(1)
        finally:
            service.release_listener_registration()
            starting.join(5)
            if stopping.ident is not None:
                stopping.join(5)


def ended_stream_case(termination):
    from _mock_connect_server import mock_connect_session

    with mock_connect_session() as (spark, service):
        service.listener_bus_enabled = True
        manager = spark.streams
        listener = IdleListener()
        try:
            manager.addListener(listener)
            stale_ids = list(listener._rust_listener_ids)
            emit_idle(service, spark, "before-end")
            listener.wait_for("before-end")
            # End the subscription without a client removal RPC, either at ResultComplete
            # or with a nonretryable server error. Server completion precedes native cleanup.
            assert service.end_listener_stream(spark.session_id, failure=termination == "error")
            assert service.wait_listener_streams_finished(1)
            deadline = time.monotonic() + 5
            while manager._sqlb._listener_bus:
                assert time.monotonic() < deadline, "ended stream kept its registrations"
                time.sleep(0.01)
            assert listener._rust_listener_ids == stale_ids
            manager.removeListener(listener)
            assert listener._rust_listener_ids == stale_ids
            assert service.listener_removals == 0

            # Reuse the same manager and object: stale tokens must not match the new
            # generation or prevent removal of its one active registration.
            manager.addListener(listener)
            assert listener._rust_listener_ids[:-1] == stale_ids
            assert listener._rust_listener_ids[-1] not in stale_ids
            assert service.listener_subscriptions == 2
            assert manager._sqlb._listener_bus == [listener]
            emit_idle(service, spark, "after-end")
            listener.wait_for("after-end")
            assert listener.timestamps == ["before-end", "after-end"]
            manager.removeListener(listener)
            assert manager._sqlb._listener_bus == []
            assert listener._rust_listener_ids == stale_ids
            assert service.listener_removals == 1
            assert service.wait_listener_streams_finished(2)
            manager.removeListener(listener)
            assert listener._rust_listener_ids == stale_ids
            assert service.listener_removals == 1
        finally:
            manager.close()


def second_rpc_case():
    from _mock_connect_server import mock_connect_session

    with mock_connect_session() as (spark, service):
        service.listener_bus_enabled = True
        spark.streams.addListener(IdleListener())
        assert service.wait_listener_registration(1)
        spark.streams.close()
        assert service.wait_listener_removal(1)
        assert service.wait_listener_streams_finished(1)
        # Earlier RPCs must not satisfy a wait for the next registration/removal.
        assert not service.wait_listener_registration(2, timeout=0.05)
        assert not service.wait_listener_removal(2, timeout=0.05)
        service.listener_register_blocked = True
        service.listener_remove_blocked = True
        listener, errors = IdleListener(), []

        def add():
            try:
                spark.streams.addListener(listener)
            except Exception as error:
                errors.append(error)

        def close():
            try:
                spark.streams.close()
            except Exception as error:
                errors.append(error)

        starting = threading.Thread(target=add, daemon=True)
        closing = threading.Thread(target=close, daemon=True)
        try:
            starting.start()
            assert service.wait_listener_registration(2)
            assert starting.is_alive()
            service.release_listener_registration()
            starting.join(5)
            assert not starting.is_alive()
            assert hasattr(listener, "_rust_listener_ids")
            closing.start()
            assert service.wait_listener_removal(2)
            assert closing.is_alive()
            service.release_listener_removal()
            closing.join(5)
            assert not closing.is_alive()
            assert errors == []
            assert service.wait_listener_streams_finished(2)
        finally:
            service.release_listener_registration()
            service.release_listener_removal()
            starting.join(5)
            if closing.ident is not None:
                closing.join(5)
            service.listener_register_blocked = False
            service.listener_remove_blocked = False
            spark.streams.close()


if __name__ == "__main__":
    if sys.argv[1] == "--concurrent-close-case":
        concurrent_close_case(bool(int(sys.argv[2])), sys.argv[3])
    elif sys.argv[1] == "--pending-registration-case":
        pending_registration_case(sys.argv[2])
    elif sys.argv[1] == "--callback-rpc-case":
        callback_rpc_case()
    elif sys.argv[1] == "--blocked-callback-case":
        blocked_callback_case(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "--ended-stream-case":
        ended_stream_case(sys.argv[2])
    else:
        assert sys.argv[1] == "--second-rpc-case"
        second_rpc_case()
