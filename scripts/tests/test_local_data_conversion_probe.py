"""Exercise the probe of local data conversions in fresh Python processes, without the Rust
extension: a conversion of Python code, which holds a reference to the same object for each row
that it took, stands for the native one."""

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROGRAM = """
import gc, os, signal, sys, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / 'python/tests'))
from _local_data_conversion_probe import LocalDataConversionProbe

case = sys.argv[2]
original_handler = signal.getsignal(signal.SIGINT)
original_interval = sys.getswitchinterval()
if case == 'disabled_gc':
    gc.disable()
original_gc = gc.isenabled()
target = object()
base = sys.getrefcount(target)
rows = 20000

def taken():
    return sys.getrefcount(target) - base

probe = LocalDataConversionProbe(lambda: 0 < taken() < rows, lambda: taken() > 0,
                                 cancel=case == 'cancel')
interrupted = False

def convert():
    taken_rows = []
    for index in range(rows):
        taken_rows.append(target)
        if index % 128 == 0:
            time.sleep(0.001)
    return taken_rows

try:
    with probe:
        if gc.isenabled():
            raise RuntimeError('automatic GC ran during the conversion')
        if case == 'exception':
            def fail():
                raise ValueError('fixture failure')
            probe.convert(fail)
        elif case == 'late':
            probe.convert(lambda: [])
            def send_late():
                time.sleep(0.01)
                os.kill(os.getpid(), signal.SIGINT)
            sender = threading.Thread(target=send_late)
            sender.start()
            sender.join()
        elif case == 'completed':
            # The result still holds the references, but the conversion has ended.
            converted = probe.convert(convert)
            probe.interrupted(signal.SIGINT, None)
            if not probe.observed.get('late_or_unproven_signal'):
                raise RuntimeError('a completed conversion accepted as interrupted')
            del converted
        else:
            try:
                converted = probe.convert(convert)
            except KeyboardInterrupt:
                interrupted = True
            else:
                if taken() != rows:
                    raise RuntimeError(f'incomplete conversion: {taken()}')
                del converted
except ValueError as error:
    if case != 'exception' or str(error) != 'fixture failure':
        raise
if signal.getsignal(signal.SIGINT) is not original_handler:
    raise RuntimeError('handler was not restored')
if gc.isenabled() != original_gc:
    raise RuntimeError('GC state was not restored')
if sys.getswitchinterval() != original_interval:
    raise RuntimeError('switch interval changed')
if (probe.worker is not None and probe.worker.is_alive()) or probe.failures:
    raise RuntimeError(f'worker cleanup failed: {probe.failures}')
if taken() != 0:
    raise RuntimeError(f'references of the conversion left: {taken()}')
if case in ('progress', 'cancel', 'disabled_gc'):
    if 'progress' not in probe.observed:
        raise RuntimeError(f'no partial reading: {probe.observed}')
    if interrupted != (case == 'cancel'):
        raise RuntimeError(f'incorrect cancellation: {probe.observed}')
    if case == 'cancel' and not probe.observed.get('signal_in_conversion'):
        raise RuntimeError(f'signal not raised in the conversion: {probe.observed}')
if case == 'late' and not probe.observed.get('late_or_unproven_signal'):
    raise RuntimeError('late SIGINT was not recorded')
print(case, 'PASS')
"""


class LocalDataConversionProbeTests(unittest.TestCase):
    def test_probe_subprocess_cases(self):
        for case in ("progress", "cancel", "late", "exception", "disabled_gc", "completed"):
            with self.subTest(case=case):
                result = subprocess.run(
                    [sys.executable, "-S", "-c", PROGRAM, str(ROOT), case],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_a_signal_before_any_reading_is_late(self):
        # Without a reading of the part, a signal during the conversion is not one of it: the
        # handler raises nothing, and the conversion completes.
        program = """
import signal, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / 'python/tests'))
from _local_data_conversion_probe import LocalDataConversionProbe
probe = LocalDataConversionProbe(lambda: False, lambda: True)
with probe:
    def convert():
        probe.interrupted(signal.SIGINT, None)
        return 'converted'
    if probe.convert(convert) != 'converted':
        raise RuntimeError('conversion did not complete')
if 'progress' in probe.observed or not probe.observed.get('late_or_unproven_signal'):
    raise RuntimeError(f'unproven signal accepted: {probe.observed}')
"""
        result = subprocess.run(
            [sys.executable, "-S", "-c", program, str(ROOT)],
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
