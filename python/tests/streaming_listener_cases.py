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

import json
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from pathlib import Path

import pytest
from pyspark.sql.streaming.listener import QueryIdleEvent, StreamingQueryListener


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
            pytest.fail(f"Listener case {arguments} timed out:\n" + output.read())
        output.seek(0)
        assert result.returncode == 0, output.read()


_LIFECYCLE_WARNING_PREFIXES = ("Failed to add the listener", "Failed to remove the listener")
_HANDLER_WARNING_PREFIX = (
    "StreamingQueryListenerBus Handler thread received exception, all client side "
    "listeners are removed and handler thread is terminated. The error is: "
)


def warnings_with_prefixes(caught, prefixes):
    return [str(item.message) for item in caught
            if issubclass(item.category, UserWarning) and str(item.message).startswith(prefixes)]


def listener_warnings(caught):
    return warnings_with_prefixes(caught, _LIFECYCLE_WARNING_PREFIXES)


def listener_event_warnings(caught):
    return warnings_with_prefixes(
        caught, (_HANDLER_WARNING_PREFIX, "Listener ", *_LIFECYCLE_WARNING_PREFIXES)
    )


def emit_unrelated_warnings():
    warnings.warn("unrelated streaming test deprecation", DeprecationWarning)
    warnings.warn("unrelated streaming test user warning", UserWarning)


def wait_for_empty_bus(manager, message="failed stream kept its registrations"):
    deadline = time.monotonic() + 5
    while manager._sqlb._listener_bus:
        assert time.monotonic() < deadline, message
        time.sleep(0.01)


def start_response(event_json):
    from _mock_connect_server import message_field

    # WriteStreamOperationStartResult: query_id=1, query_started_event_json=3.
    return message_field(
        8,
        message_field(1, message_field(1, "q") + message_field(2, "r"))
        + message_field(3, event_json),
    )


class TestStreamingListenerLifecycle:
    @pytest.mark.parametrize("method", ["start", "toTable"])
    @pytest.mark.parametrize("registered", [False, True])
    @pytest.mark.parametrize("event_json, exception", [
        ("", json.JSONDecodeError), ("{", json.JSONDecodeError), ("{}", KeyError),
    ], ids=["empty-json", "invalid-json", "missing-fields"])
    def test_streaming_started_event_decode_error_propagates(
        self, mock_connect, method, registered, event_json, exception,
    ):
        spark, service = mock_connect
        service.listener_bus_enabled = True
        listener = IdleListener()
        try:
            if registered:
                spark.streams.addListener(listener)
            service.execute_payload = start_response(event_json)
            writer = spark.readStream.format("rate").load().writeStream
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                emit_unrelated_warnings()
                with pytest.raises(exception):
                    getattr(writer, method)(*(["table"] if method == "toTable" else []))
            assert not listener_event_warnings(caught)
            assert spark.streams._sqlb._listener_bus == ([listener] if registered else [])
        finally:
            spark.streams.close()

    @pytest.mark.parametrize("method", ["start", "toTable"])
    def test_streaming_started_callbacks_share_typed_event_and_isolate_exceptions(
        self, mock_connect, method,
    ):
        spark, service = mock_connect
        service.listener_bus_enabled = True
        events = []

        class StartedListener(IdleListener):
            def onQueryStarted(self, event):
                events.append(event)
                if len(events) == 1:
                    raise ValueError("started callback failed")

        listener = StartedListener()
        try:
            # Duplicate registrations receive the same deserialized event, synchronously.
            spark.streams.addListener(listener)
            spark.streams.addListener(listener)
            service.execute_payload = start_response(json.dumps({
                "id": "00000000-0000-4000-8000-000000000001",
                "runId": "00000000-0000-4000-8000-000000000002",
                "name": "query", "timestamp": "started", "jobTags": ["tag"],
            }))
            writer = spark.readStream.format("rate").load().writeStream
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                emit_unrelated_warnings()
                query = getattr(writer, method)(*(["table"] if method == "toTable" else []))
            assert query.id == "q"
            relevant = listener_event_warnings(caught)
            assert len(relevant) == 1 and "started callback failed" in relevant[0]
            assert len(events) == 2 and events[0] is events[1]
            assert events[0].timestamp == "started" and events[0].jobTags == {"tag"}
            assert spark.streams._sqlb._listener_bus == [listener, listener]
        finally:
            spark.streams.close()

    @pytest.mark.parametrize("event_type, event_json", [
        (None, ""),
        (3, ""),
        (3, "{"),
        (3, "{}"),
        (99, "{}"),
        (0, "{}"),
        (-1, "{}"),
    ], ids=["transport", "empty-json", "invalid-json", "missing-fields", "unknown-type",
            "unspecified-type", "async-started-type"])
    def test_streaming_bus_errors_warn_once_clear_all_and_allow_reregistration(
        self, mock_connect, event_type, event_json,
    ):
        spark, service = mock_connect
        service.listener_bus_enabled = True
        manager = spark.streams
        first, second = IdleListener(), IdleListener()
        try:
            manager.addListener(first)
            manager.addListener(first)
            manager.addListener(second)
            emit_idle(service, spark, "before-error")
            first.wait_for("before-error")
            second.wait_for("before-error")
            if event_type is None:
                from pyspark.errors.exceptions.connect import SparkConnectGrpcException

                # Compare the entire error rendering with a normal RPC abort carrying
                # the same status and details as the listener-stream abort.
                previous_failure = service.execute_failure
                service.execute_failure = "listener event stream failed"
                try:
                    with pytest.raises(SparkConnectGrpcException) as error:
                        spark.range(1).collect()
                    expected_error = str(error.value)
                finally:
                    service.execute_failure = previous_failure
            elif event_type == 3:
                # Derive the complete reference string from the vendored event decoder,
                # independently of the native bus and its exception round trip.
                with pytest.raises((json.JSONDecodeError, KeyError)) as error:
                    QueryIdleEvent.fromJson(json.loads(event_json))
                expected_error = str(error.value)
            else:
                from pyspark.errors import PySparkValueError

                expected_error = str(PySparkValueError(
                    errorClass="UNKNOWN_VALUE_FOR",
                    messageParameters={"var": f"proto.StreamingQueryEventType: {event_type}"},
                ))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                emit_unrelated_warnings()
                if event_type is None:
                    assert service.end_listener_stream(spark.session_id, failure=True)
                else:
                    service.listener_event_json(spark.session_id, event_type, event_json)
                wait_for_empty_bus(manager)
            messages = listener_event_warnings(caught)
            assert len(messages) == 1
            assert messages[0] == _HANDLER_WARNING_PREFIX + expected_error
            assert first.timestamps == ["before-error", "before-error"]
            assert second.timestamps == ["before-error"]
            assert service.wait_listener_streams_finished(1)
            assert service.listener_removals == 0

            # The failed stream leaves the server's registration held, so the add removes it first.
            manager.addListener(second)
            assert service.listener_removals == 1
            assert service.listener_subscriptions == 2
            emit_idle(service, spark, "after-error")
            second.wait_for("after-error")
            assert manager._sqlb._listener_bus == [second]
            assert first.timestamps == ["before-error", "before-error"]
            manager.removeListener(second)
            assert service.wait_listener_streams_finished(2)
        finally:
            manager.close()

    def test_streaming_callback_exceptions_warn_and_leave_the_bus_running(self, mock_connect):
        spark, service = mock_connect
        service.listener_bus_enabled = True

        class FailingListener(IdleListener):
            def onQueryIdle(self, event):
                super().onQueryIdle(event)
                raise ValueError("idle callback failed")

        first, second = FailingListener(), IdleListener()
        try:
            spark.streams.addListener(first)
            spark.streams.addListener(second)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                emit_unrelated_warnings()
                for timestamp in ("first-callback", "second-callback"):
                    emit_idle(service, spark, timestamp)
                    second.wait_for(timestamp)
            relevant = listener_event_warnings(caught)
            assert len(relevant) == 2
            assert relevant == [f"Listener {first} threw an exception\nidle callback failed"] * 2
            assert first.timestamps == second.timestamps == ["first-callback", "second-callback"]
            assert spark.streams._sqlb._listener_bus == [first, second]
            assert service.listener_subscriptions == 1
        finally:
            spark.streams.close()

    def test_streaming_duplicate_async_listeners_share_one_decoded_event(
        self, mock_connect, monkeypatch,
    ):
        from pyspark.sql.streaming import query as query_module

        spark, service = mock_connect
        service.listener_bus_enabled = True
        decoded, received = [], []
        deserialize = query_module._deserialize_listener_event

        def count_decode(*args, **kwargs):
            event = deserialize(*args, **kwargs)
            decoded.append(event)
            return event

        monkeypatch.setattr(query_module, "_deserialize_listener_event", count_decode)

        class SharedListener(IdleListener):
            def onQueryIdle(self, event):
                with self.condition:
                    received.append(event)
                    self.condition.notify_all()

        listener = SharedListener()
        try:
            spark.streams.addListener(listener)
            spark.streams.addListener(listener)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                emit_unrelated_warnings()
                emit_idle(service, spark, "shared-event")
                with listener.condition:
                    assert listener.condition.wait_for(lambda: len(received) == 2, timeout=5)
            assert not listener_event_warnings(caught)
            assert len(decoded) == 1
            assert received[0] is received[1] is decoded[0]
            assert received[0].timestamp == "shared-event"
        finally:
            spark.streams.close()

    @pytest.mark.parametrize("failure", ["base-exception", "warning-as-error"])
    def test_streaming_unisolated_callback_errors_clear_the_bus(self, mock_connect, failure):
        spark, service = mock_connect
        service.listener_bus_enabled = True

        class CallbackExit(BaseException):
            pass

        class FailingListener(IdleListener):
            def onQueryIdle(self, event):
                super().onQueryIdle(event)
                if failure == "base-exception":
                    raise CallbackExit("uncaught callback exit")
                raise ValueError("callback warning becomes an error")

        first, second = FailingListener(), IdleListener()
        try:
            spark.streams.addListener(first)
            spark.streams.addListener(second)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                warnings.filterwarnings("error", message="Listener ", category=UserWarning)
                emit_unrelated_warnings()
                emit_idle(service, spark, "unisolated-error")
                wait_for_empty_bus(spark.streams)
            relevant = listener_event_warnings(caught)
            assert len(relevant) == 1
            expected_error = (
                "uncaught callback exit" if failure == "base-exception"
                else f"Listener {first} threw an exception\ncallback warning becomes an error"
            )
            assert relevant[0] == _HANDLER_WARNING_PREFIX + expected_error
            assert first.timestamps == ["unisolated-error"]
            assert second.timestamps == []
            assert service.wait_listener_streams_finished(1)
        finally:
            spark.streams.close()

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
            other.streams.addListener(listener)
            assert spark.streams._sqlb._listener_bus == [listener, listener]
            assert other.streams._sqlb._listener_bus == [listener]
            assert service.listener_subscriptions == 2

            other.streams.removeListener(listener)
            assert other.streams._sqlb._listener_bus == []
            assert spark.streams._sqlb._listener_bus == [listener, listener]
            assert service.listener_removals == 1
            assert service.wait_listener_streams_finished(1)

            spark.streams.removeListener(listener)
            assert spark.streams._sqlb._listener_bus == [listener]
            assert service.listener_removals == 1
            spark.streams.removeListener(listener)
            assert spark.streams._sqlb._listener_bus == []
            assert service.listener_removals == 2
            assert service.wait_listener_streams_finished(2)
        finally:
            other.stop()
            spark.streams.close()

    def test_streaming_listener_registrations_are_not_kept_on_the_listener(self, mock_connect):
        # As PySpark's bus, which keeps its listeners in its own list: adding a listener leaves
        # nothing on the object but the session that _set_spark_session gives it, so a listener
        # that has no room for another attribute, of __slots__, is added and removed.
        # (PySpark's client-side bus takes any object with the callbacks.)
        spark, service = mock_connect
        service.listener_bus_enabled = True

        class SlottedListener:
            # Not a StreamingQueryListener, whose instances have a __dict__.
            __slots__ = ("spark",)

            def _set_spark_session(self, session):
                self.spark = session

        slotted, plain = SlottedListener(), IdleListener()
        assert not hasattr(slotted, "__dict__")
        try:
            spark.streams.addListener(plain)
            attributes = set(vars(plain))
            spark.streams.addListener(plain)
            assert set(vars(plain)) == attributes
            spark.streams.addListener(slotted)
            assert spark.streams._sqlb._listener_bus == [plain, plain, slotted]
            spark.streams.removeListener(slotted)
            assert spark.streams._sqlb._listener_bus == [plain, plain]
            spark.streams.removeListener(plain)
            spark.streams.removeListener(plain)
            assert spark.streams._sqlb._listener_bus == []
            # Not registered any more: nothing to remove, as PySpark's remove does nothing.
            spark.streams.removeListener(plain)
            assert service.listener_removals == 1
        finally:
            spark.streams.close()

    def test_streaming_listener_is_removed_by_equality_as_pyspark(self, mock_connect):
        # As the list.remove of PySpark's bus: the first listener that is the one given, or
        # equal to it, is removed.
        spark, service = mock_connect
        service.listener_bus_enabled = True

        class NamedListener(IdleListener):
            def __init__(self, name):
                super().__init__()
                self.name = name

            def __eq__(self, other):
                return isinstance(other, NamedListener) and self.name == other.name

            __hash__ = None

        first, second = NamedListener("a"), NamedListener("b")
        try:
            spark.streams.addListener(first)
            spark.streams.addListener(second)
            spark.streams.removeListener(NamedListener("b"))
            [left] = spark.streams._sqlb._listener_bus
            assert left is first
        finally:
            spark.streams.close()

    def test_streaming_listener_added_and_removed_by_many_threads_at_once(self, mock_connect):
        # The registrations of one listener that several threads add and remove at once are
        # each removed once: none is lost, and none is left in the bus.
        spark, service = mock_connect
        service.listener_bus_enabled = True
        listener, keeper = IdleListener(), IdleListener()
        threads, rounds = 8, 25
        barrier = threading.Barrier(threads)
        failures = []

        def work():
            try:
                barrier.wait(10)
                for _ in range(rounds):
                    spark.streams.addListener(listener)
                    spark.streams.removeListener(listener)
            except BaseException as error:  # noqa: BLE001
                failures.append(error)

        try:
            # A listener that stays keeps the bus open, so that the rounds make no RPC.
            spark.streams.addListener(keeper)
            workers = [threading.Thread(target=work) for _ in range(threads)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(60)
            assert failures == []
            assert spark.streams._sqlb._listener_bus == [keeper]
            assert not hasattr(listener, "_rust_listener_ids")
            assert service.listener_subscriptions == 1
        finally:
            spark.streams.close()

    def test_streaming_listener_registration_failure_can_be_retried(self, mock_connect):
        spark, service = mock_connect
        service.listener_bus_enabled = True
        service.listener_register_failure = True
        listener = IdleListener()
        with pytest.warns(UserWarning, match="listener registration failed") as caught:
            assert spark.streams.addListener(listener) is None
        assert "The listener is not added, please add it again." in str(caught[0].message)
        assert spark.streams._sqlb._listener_bus == []
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
            assert spark.streams._sqlb._listener_bus == [listener]
            assert service.listener_streams_finished == 0
            emit_idle(service, spark, "still-registered")
            listener.wait_for("still-registered")

            service.listener_remove_failure = False
            assert remove() is None
            assert service.listener_removals == 2
            assert service.wait_listener_streams_finished(1)
            # A completed remove, or close, leaves no registration of the listener, so the
            # same object registers and is removed again.
            assert spark.streams._sqlb._listener_bus == []
            # A fresh registration after removal opens a new server subscription.
            spark.streams.addListener(listener)
            assert spark.streams._sqlb._listener_bus == [listener]
            assert service.listener_subscriptions == 2
            emit_idle(service, spark, "registered-again")
            listener.wait_for("registered-again")
            spark.streams.removeListener(listener)
            assert spark.streams._sqlb._listener_bus == []
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

    def test_streaming_listener_stop_delivers_the_events_sent_while_releasing(self, mock_connect):
        # As PySpark's stop(): the session is released before the listener bus is cancelled, so
        # the terminated events that the server sends while it releases the session, of the
        # queries still running, reach the listeners. The mock holds the release's response
        # until the listener has the event; a bus cancelled before the release drops it.
        spark, service = mock_connect
        service.listener_bus_enabled = True
        terminated = threading.Event()

        class TerminatedListener(IdleListener):
            def onQueryTerminated(self, event):
                terminated.set()

        spark.streams.addListener(TerminatedListener())
        assert service.wait_listener_registration(1)
        service.release_session_events = [(2, json.dumps({
            "id": "00000000-0000-4000-8000-000000000001",
            "runId": "00000000-0000-4000-8000-000000000002",
            "exception": None,
            "errorClassOnException": None,
        }))]
        service.release_session_blocked = True
        stopper = threading.Thread(target=spark.stop, daemon=True)
        stopper.start()
        try:
            assert terminated.wait(timeout=15)
        finally:
            service.release_session_release()
            stopper.join(timeout=15)
            service.release_session_blocked = False
            service.release_session_events = []
        assert not stopper.is_alive()
        assert spark.is_stopped
        assert service.release_session_calls == 1

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


    def test_streaming_listener_start_removal_waits_for_the_started_event_in_delivery(
        self, mock_connect,
    ):
        # As PySpark's bus, whose delivery of the started event of start() holds the lock of
        # its removal: a listener removed by another thread gets that event before
        # removeListener returns. A removal from the callback that receives the event does
        # not wait for its own delivery.
        spark, service = mock_connect
        service.listener_bus_enabled = True
        service.execute_payload = start_response(json.dumps({
            "id": "00000000-0000-4000-8000-000000000001",
            "runId": "00000000-0000-4000-8000-000000000002",
            "name": "query", "timestamp": "started", "jobTags": [],
        }))
        order = []
        outcome = {}
        entered, release = threading.Event(), threading.Event()

        class Blocking(IdleListener):
            def onQueryStarted(self, event):
                order.append("first started")
                entered.set()
                assert release.wait(5)

        class Recording(IdleListener):
            def onQueryStarted(self, event):
                order.append("second started")

        def start_query():
            try:
                outcome["query"] = spark.readStream.format("rate").load().writeStream.start()
            except BaseException as error:
                outcome["error"] = error

        def remove(listener):
            spark.streams.removeListener(listener)
            order.append("removed")

        first, second = Blocking(), Recording()
        try:
            spark.streams.addListener(first)
            spark.streams.addListener(second)
            starter = threading.Thread(target=start_query, daemon=True)
            starter.start()
            assert entered.wait(5)
            removal = threading.Thread(target=remove, args=(second,), daemon=True)
            removal.start()
            # The started event is in delivery to the first listener, so the removal waits.
            removal.join(0.5)
            assert removal.is_alive()
            release.set()
            removal.join(5)
            starter.join(5)
            assert not removal.is_alive() and not starter.is_alive()
            assert "error" not in outcome
            assert order == ["first started", "second started", "removed"]
            emit_idle(service, spark, "after")
            first.wait_for("after")
            assert second.timestamps == []
        finally:
            release.set()
            spark.streams.close()

        # A removal from the callback that receives the started event returns at once.
        removed_listener = Recording()
        order.clear()

        class Removing(IdleListener):
            def onQueryStarted(self, event):
                order.append("removing started")
                spark.streams.removeListener(removed_listener)
                order.append("removed in callback")

        try:
            spark.streams.addListener(Removing())
            spark.streams.addListener(removed_listener)
            starter = threading.Thread(target=start_query, daemon=True)
            starter.start()
            starter.join(5)
            assert not starter.is_alive()
            assert "error" not in outcome
            assert order[:2] == ["removing started", "removed in callback"]
        finally:
            spark.streams.close()

    def test_streaming_listener_removal_waits_for_the_event_in_delivery(self, mock_connect):
        # As PySpark's bus, whose delivery of an event holds the lock of its removal: a
        # listener removed by another thread while an event is delivered gets that event
        # before removeListener returns, and no event after it.
        spark, service = mock_connect
        service.listener_bus_enabled = True
        order = []
        entered, release = threading.Event(), threading.Event()

        class Blocking(IdleListener):
            def onQueryIdle(self, event):
                order.append("first " + event.timestamp)
                entered.set()
                assert release.wait(5)
                super().onQueryIdle(event)

        class Recording(IdleListener):
            def onQueryIdle(self, event):
                order.append("second " + event.timestamp)
                super().onQueryIdle(event)

        def remove():
            spark.streams.removeListener(second)
            order.append("removed")

        first, second = Blocking(), Recording()
        try:
            spark.streams.addListener(first)
            spark.streams.addListener(second)
            emit_idle(service, spark, "in-delivery")
            assert entered.wait(5)
            removal = threading.Thread(target=remove)
            removal.start()
            # The first listener holds the delivery, so the removal waits for it.
            removal.join(0.5)
            assert removal.is_alive()
            release.set()
            removal.join(5)
            assert not removal.is_alive()
            assert order == ["first in-delivery", "second in-delivery", "removed"]
            emit_idle(service, spark, "after")
            first.wait_for("after")
            assert second.timestamps == ["in-delivery"]
        finally:
            release.set()
            spark.streams.close()


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
                    if callback_action != "stop":
                        # The callback's own removal did not consume a registration.
                        assert spark.streams._sqlb._listener_bus == [listener] * registrations
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
                assert attempted not in spark.streams._sqlb._listener_bus
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
                assert service.listener_removals == 2
            else:
                assert service.listener_removals == 1
            assert service.wait_listener_streams_finished(1)
            # A close, or the retried removals, leaves no registration. A fresh add is the
            # only one, and opens exactly one subscription.
            assert spark.streams._sqlb._listener_bus == []
            spark.streams.addListener(listener)
            assert spark.streams._sqlb._listener_bus == [listener]
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
            emit_idle(service, spark, "before-end")
            listener.wait_for("before-end")
            # End the subscription without a client removal RPC, either at ResultComplete
            # or with a nonretryable server error. Server completion precedes native cleanup.
            assert service.end_listener_stream(spark.session_id, failure=termination == "error")
            assert service.wait_listener_streams_finished(1)
            wait_for_empty_bus(manager, "ended stream kept its registrations")
            manager.removeListener(listener)
            assert service.listener_removals == 0

            # Reuse the same manager and object: the registrations that the end cleared
            # must not prevent removal of its one active registration. A failed stream
            # leaves the server's registration held, so the add removes it first; a normal
            # end leaves none.
            removals = 1 if termination == "error" else 0
            manager.addListener(listener)
            assert service.listener_removals == removals
            assert service.listener_subscriptions == 2
            assert manager._sqlb._listener_bus == [listener]
            emit_idle(service, spark, "after-end")
            listener.wait_for("after-end")
            assert listener.timestamps == ["before-end", "after-end"]
            manager.removeListener(listener)
            assert manager._sqlb._listener_bus == []
            assert service.listener_removals == removals + 1
            assert service.wait_listener_streams_finished(2)
            manager.removeListener(listener)
            assert service.listener_removals == removals + 1
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
            assert spark.streams._sqlb._listener_bus == [listener]
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
