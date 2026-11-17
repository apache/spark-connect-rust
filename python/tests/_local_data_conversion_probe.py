"""Observe the progress of a native conversion of local data by the references that it holds to
the objects of the data, and contain late signals, without changing thread cadence."""

import _thread
import gc
import signal
import threading
import time


class LocalDataConversionProbe:
    """Run a conversion while another Python thread waits for the GIL to read its progress.

    `in_part()` reads the references that the conversion holds, with `sys.getrefcount` of a few
    objects of the data, or of many of them in one `list(map(sys.getrefcount, objects))`, which
    runs in C, without a switch to another thread. It is true only while the conversion is in the
    part of it under test, such as the collection of the rows. The data of a case ends with a value
    that fails the conversion without running Python code, so that no part of it that follows,
    such as the release of the rows, can be observed. Automatic garbage collection is disabled
    meanwhile, as a collection, which an allocation can start, runs the Python code of finalizers
    and weakref callbacks, at which the other thread could run too. The thread takes the first
    true reading; with `cancel`, it then interrupts the main thread, whose handler raises
    KeyboardInterrupt only while the conversion has not ended, as `held()`, of the references
    that the conversion holds until it ends, tells.
    """

    def __init__(self, in_part, held, *, cancel=False):
        self.in_part = in_part
        self.held = held
        self.cancel = cancel
        self.observed = {}
        self.failures = []
        self.converting = False
        self.finished = threading.Event()
        self.worker = None

    def observe(self):
        try:
            while not self.finished.is_set():
                if not self.converting:
                    # Before the conversion, let the main thread start it.
                    time.sleep(0.0001)
                    continue
                captured_at = time.monotonic()
                if self.in_part():
                    self.observed["progress"] = {"captured_at": captured_at}
                    if self.cancel:
                        _thread.interrupt_main()
                    return
        except BaseException as error:
            self.failures.append(error)

    def interrupted(self, signum, frame):
        self.observed["signal_handler_at"] = time.monotonic()
        if self.converting and "progress" in self.observed and self.held():
            self.observed["signal_in_conversion"] = True
            raise KeyboardInterrupt
        self.observed["late_or_unproven_signal"] = True

    def __enter__(self):
        self.gc_was_enabled = gc.isenabled()
        gc.collect()
        gc.disable()
        try:
            self.previous_handler = signal.signal(signal.SIGINT, self.interrupted)
        except BaseException:
            if self.gc_was_enabled:
                gc.enable()
            raise
        try:
            self.worker = threading.Thread(target=self.observe)
            self.worker.start()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def convert(self, action):
        self.converting = True
        try:
            return action()
        finally:
            self.converting = False
            self.observed["conversion_ended_at"] = time.monotonic()

    def __exit__(self, error_type, error, traceback):
        self.converting = False
        try:
            self.finished.set()
            if self.worker is not None and self.worker.ident is not None:
                self.worker.join()
        finally:
            try:
                signal.signal(signal.SIGINT, self.previous_handler)
            finally:
                if self.gc_was_enabled:
                    gc.enable()
