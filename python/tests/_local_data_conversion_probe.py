"""Observe the progress of a native conversion of local data by the references that it holds to
the objects of the data, at its checkpoints, and contain late signals."""

import _thread
import gc
import signal
import sys
import threading
import time

# How long either side waits for the other to take its turn, which only a failure of the other
# side, as a crash of the probe's thread, makes it wait so long.
_TURN_TIMEOUT = 60.0


class LocalDataConversionProbe:
    """Run a conversion while another Python thread reads its progress at its checkpoints.

    At each checkpoint of the conversion, the call of a Python function at which the interpreter
    may hand the GIL to another thread, the conversion's thread waits, of a profile function, until
    the probe's thread has taken one reading: so the reading falls at each checkpoint, of no
    scheduling of the operating system. The checkpoints of a native conversion are the calls of
    the Python function that it calls every 1,024 steps (see `crate::interrupt::ConversionBudget`);
    those of an operation of Python code, as `MutableMapping.update`, the calls that it makes, of
    Python functions and of builtins. `in_part()` reads the references that the conversion
    holds, with `sys.getrefcount` of a few objects of the data, or of many of them in one
    `list(map(sys.getrefcount, objects))`. It is true only while the conversion is in the part of
    it under test, such as the collection of the rows: so a checkpoint in that part, and none
    elsewhere, gives a true reading. The data of a case ends with a value that fails the conversion
    without running Python code, so that no part of it that follows, such as the release of the
    rows, can be observed. Automatic garbage collection is disabled meanwhile, as a collection,
    which an allocation can start, runs the Python code of finalizers and weakref callbacks.

    The thread takes the first true reading, after which the conversion runs on without turns;
    with `cancel`, it then interrupts the main thread, whose handler, which runs before the
    checkpoint returns to the conversion, raises KeyboardInterrupt only while the conversion has
    not ended, as `held()`, of the references that the conversion holds until it ends, tells.
    """

    def __init__(self, in_part, held, *, cancel=False):
        self.in_part = in_part
        self.held = held
        self.cancel = cancel
        self.observed = {}
        self.failures = []
        self.converting = False
        self.finished = threading.Event()
        # The turn of the probe's thread, which a checkpoint gives it, and that of the
        # conversion, which the probe gives back once it read.
        self.probe_turn = threading.Event()
        self.conversion_turn = threading.Event()
        self.checkpoints = 0
        self.worker = None

    def observe(self):
        try:
            while True:
                if not self.probe_turn.wait(_TURN_TIMEOUT):
                    raise AssertionError("no checkpoint gave the probe its turn")
                self.probe_turn.clear()
                if self.finished.is_set():
                    return
                captured_at = time.monotonic()
                if self.in_part():
                    self.observed["progress"] = {"captured_at": captured_at}
                    if self.cancel:
                        _thread.interrupt_main()
                    return
                self.conversion_turn.set()
        except BaseException as error:
            self.failures.append(error)
        finally:
            self.conversion_turn.set()

    def checkpoint(self, frame, event, arg):
        """The profile function of the conversion's thread: at a checkpoint of the conversion, a
        call of a Python function or of a builtin, the probe's turn, until it read. The handler
        of a signal takes no turn, and nor does a checkpoint once the probe read the part, or
        failed."""
        if event not in ("call", "c_call") or frame.f_code is _HANDLER_CODE:
            return
        if "progress" in self.observed or self.failures or not self.worker.is_alive():
            return
        self.checkpoints += 1
        self.conversion_turn.clear()
        self.probe_turn.set()
        if not self.conversion_turn.wait(_TURN_TIMEOUT):
            self.failures.append(AssertionError("the probe did not give the conversion its turn"))

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
        # A profile function is called for the calls of Python functions of this thread only.
        previous_profile = sys.getprofile()
        sys.setprofile(self.checkpoint)
        try:
            return action()
        finally:
            # A KeyboardInterrupt of the handler, which runs in the profile function, unsets it.
            sys.setprofile(previous_profile)
            self.converting = False
            self.observed["conversion_ended_at"] = time.monotonic()
            self.observed["checkpoints"] = self.checkpoints

    def __exit__(self, error_type, error, traceback):
        self.converting = False
        try:
            self.finished.set()
            self.probe_turn.set()
            if self.worker is not None and self.worker.ident is not None:
                self.worker.join()
        finally:
            try:
                signal.signal(signal.SIGINT, self.previous_handler)
            finally:
                if self.gc_was_enabled:
                    gc.enable()


# The code of the handler of SIGINT, whose call takes no turn.
_HANDLER_CODE = LocalDataConversionProbe.interrupted.__code__
