"""The setup of the processes of the tests of the drop-in."""

import signal

# A pytest that a non-interactive shell starts in the background (`pytest ... &`, `nohup ... &`)
# inherits SIGINT as ignored, and Python then leaves it ignored: the SIGINT that the tests of the
# interrupts send would be dropped, and each of them would wait out its fallback (the retry
# budget, the timeout of its subprocess) before it fails. Python's own handler is installed
# instead, and the subprocesses of the tests start with the default too, as exec resets a
# signal that a handler catches.
if signal.getsignal(signal.SIGINT) == signal.SIG_IGN:
    signal.signal(signal.SIGINT, signal.default_int_handler)
