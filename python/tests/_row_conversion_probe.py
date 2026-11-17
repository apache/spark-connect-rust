"""Observe tuple allocation and contain late signals without changing thread cadence."""

import _thread
import gc
import signal
import threading
import time


class RowConversionProbe:
    def __init__(self, row_type, field_name, expected_rows, *, cancel=False,
                 observe_progress=True, freeze_gc=True, on_materialization=None,
                 observe_materialization=True):
        self.row_type = row_type
        self.field_name = field_name
        self.expected_rows = expected_rows
        self.cancel = cancel
        self.observe_progress = observe_progress
        self.freeze_gc = freeze_gc
        self.on_materialization = on_materialization
        self.observe_materialization = observe_materialization
        self.signal_guard = None
        self.observed = {}
        self.failures = []
        self.finished = threading.Event()
        self.first_seen = threading.Event()
        self.collecting = False
        self.in_callback = False
        self.worker = None

    def matches(self, value):
        return type(value) is self.row_type and self.field_name in getattr(value, "__fields__", ())

    def snapshot(self):
        return self.count(*self.capture())

    def capture(self):
        return time.monotonic(), gc.get_objects()

    def count(self, captured_at, objects):
        return {
            "captured_at": captured_at,
            "rows": sum(self.matches(value) for value in objects),
            "counted_at": time.monotonic(),
        }

    def materializing(self, phase, info):
        self.in_callback = True
        try:
            if phase == "start" and "first_rows_observed_at" not in self.observed:
                captured_at = time.monotonic()
                if any(self.matches(value) for value in gc.get_objects(0)):
                    self.observed["first_rows_observed_at"] = captured_at
                    self.observed["first_rows_observed_generation"] = info["generation"]
            elif phase == "stop" and "first_rows_observed_at" in self.observed:
                gc.callbacks.remove(self.materializing)
                if self.freeze_gc:
                    gc.disable()
                if self.on_materialization is not None:
                    self.on_materialization()
                self.first_seen.set()
        except BaseException as error:
            self.failures.append(error)
        finally:
            self.in_callback = False

    def observe(self):
        try:
            self.first_seen.wait()
            while self.in_callback and not self.finished.is_set():
                time.sleep(0)
            if self.finished.is_set():
                return
            captured = self.capture()
            if self.cancel:
                # Request the interrupt before counting: under a tracer such as coverage, the
                # count can outlast the rest of the native materialization, and the request
                # then arrives after collect returned. The handler counts again and raises
                # only while the conversion is still partial.
                _thread.interrupt_main()
            self.observed["progress_snapshot"] = self.count(*captured)
        except BaseException as error:
            self.failures.append(error)

    def interrupted(self, signum, frame):
        self.observed["signal_handler_at"] = time.monotonic()
        self.observed["signal_during_probe_callback"] = self.in_callback
        guarded = self.signal_guard is not None and self.signal_guard()
        self.observed["signal_during_row_hook"] = guarded
        if not self.collecting or self.in_callback or guarded:
            self.observed["late_or_unproven_signal"] = True
            return
        snapshot = self.snapshot()
        self.observed["signal_snapshot"] = snapshot
        if 0 < snapshot["rows"] < self.expected_rows:
            raise KeyboardInterrupt
        self.observed["late_or_unproven_signal"] = True

    def __enter__(self):
        if gc.callbacks:
            raise RuntimeError("foreign GC callbacks would contaminate the negative control")
        self.gc_was_enabled = gc.isenabled()
        gc.collect()
        self.previous_handler = signal.signal(signal.SIGINT, self.interrupted)
        try:
            gc.enable()
            if self.observe_materialization:
                gc.callbacks.append(self.materializing)
            if self.observe_progress:
                self.worker = threading.Thread(target=self.observe)
                self.worker.start()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def collect(self, action):
        self.collecting = True
        try:
            return action()
        finally:
            self.collecting = False
            self.observed["collect_ended_at"] = time.monotonic()

    def __exit__(self, error_type, error, traceback):
        self.collecting = False
        try:
            self.finished.set()
            self.first_seen.set()
            if self.worker is not None and self.worker.ident is not None:
                self.worker.join()
        finally:
            try:
                if self.materializing in gc.callbacks:
                    gc.callbacks.remove(self.materializing)
                if self.gc_was_enabled:
                    gc.enable()
                else:
                    gc.disable()
            finally:
                signal.signal(signal.SIGINT, self.previous_handler)
