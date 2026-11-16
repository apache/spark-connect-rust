"""Exercise probe cleanup in fresh Python processes, without the Rust extension."""

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROGRAM = """
import gc, os, signal, sys, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / 'python/tests'))
from _row_conversion_probe import RowConversionProbe

class ProbeRow(tuple):
    def __new__(cls, value):
        row = tuple.__new__(cls, (value,))
        row.__fields__ = ['probe_field']
        return row

case = sys.argv[2]
original_handler = signal.getsignal(signal.SIGINT)
original_callbacks = list(gc.callbacks)
original_interval = sys.getswitchinterval()
if case == 'disabled_gc':
    gc.disable()
original_gc = gc.isenabled()
probe = RowConversionProbe(ProbeRow, 'probe_field', 20000, cancel=case == 'cancel',
                           observe_progress=case != 'completed')
interrupted = False

def build():
    rows = []
    for index in range(20000):
        rows.append(ProbeRow(index))
        if index % 128 == 0:
            time.sleep(0.001)
    return rows

try:
    with probe:
        if case == 'exception':
            def fail():
                raise ValueError('fixture failure')
            probe.collect(fail)
        elif case == 'late':
            probe.collect(lambda: [])
            def send_late():
                time.sleep(0.01)
                os.kill(os.getpid(), signal.SIGINT)
            sender = threading.Thread(target=send_late)
            sender.start()
            sender.join()
        elif case == 'completed':
            def complete_before_return():
                rows = build()
                probe.interrupted(signal.SIGINT, None)
                return rows
            rows = probe.collect(complete_before_return)
            if probe.observed.get('signal_snapshot', {}).get('rows') != 20000:
                raise RuntimeError(f'completed result falsely partial: {probe.observed}')
            if not probe.observed.get('late_or_unproven_signal'):
                raise RuntimeError('completed result accepted as native interruption')
        else:
            try:
                rows = probe.collect(build)
            except KeyboardInterrupt:
                interrupted = True
            if not interrupted:
                snapshot = probe.snapshot()
                if snapshot['rows'] != 20000:
                    raise RuntimeError(f'completed snapshot coverage: {snapshot}')
                probe.interrupted(signal.SIGINT, None)
                if not probe.observed.get('late_or_unproven_signal'):
                    raise RuntimeError('completed result accepted as interrupted conversion')
except ValueError as error:
    if case != 'exception' or str(error) != 'fixture failure':
        raise
if signal.getsignal(signal.SIGINT) is not original_handler:
    raise RuntimeError('handler was not restored')
if gc.callbacks != original_callbacks or gc.isenabled() != original_gc:
    raise RuntimeError('GC state was not restored')
if sys.getswitchinterval() != original_interval:
    raise RuntimeError('switch interval changed')
if (probe.worker is not None and probe.worker.is_alive()) or probe.failures:
    raise RuntimeError(f'worker/callback cleanup failed: {probe.failures}')
if case in ('progress', 'cancel', 'disabled_gc'):
    snapshot = probe.observed.get('progress_snapshot', {})
    if not 0 < snapshot.get('rows', 0) < 20000:
        raise RuntimeError(f'no partial snapshot: {probe.observed}')
    if interrupted != (case == 'cancel'):
        raise RuntimeError(f'incorrect cancellation: {probe.observed}')
if case == 'late' and not probe.observed.get('late_or_unproven_signal'):
    raise RuntimeError('late SIGINT was not recorded')
print(case, 'PASS')
"""


class RowConversionProbeTests(unittest.TestCase):
    def test_foreign_callbacks_are_rejected_without_changing_state(self):
        program = """
import gc, signal, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / 'python/tests'))
from _row_conversion_probe import RowConversionProbe
callback = lambda phase, info: None
gc.callbacks.append(callback)
handler = signal.getsignal(signal.SIGINT)
gc_enabled = gc.isenabled()
try:
    with RowConversionProbe(tuple, 'unused', 1):
        raise AssertionError('foreign GC callback accepted')
except RuntimeError as error:
    if 'foreign GC callbacks' not in str(error):
        raise
if gc.callbacks != [callback] or signal.getsignal(signal.SIGINT) is not handler:
    raise RuntimeError('rejected probe changed callback/handler state')
if gc.isenabled() != gc_enabled:
    raise RuntimeError('rejected probe changed GC state')
"""
        result = subprocess.run(
            [sys.executable, "-S", "-c", program, str(ROOT)],
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

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

    def test_optimized_benchmark_checks_and_heartbeat_accounting(self):
        program = """
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location(
    'benchmark', Path(sys.argv[1]) / 'scripts/benchmark_row_responsiveness.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
stats = benchmark.heartbeat_stats([], 1.0, 0.0, 2.0)
if stats['materialization_heartbeat_ticks'] != 0:
    raise RuntimeError('synthetic end counted as heartbeat')
if stats['collect_heartbeat_ticks'] != 0 or stats['max_collect_heartbeat_gap_s'] != 2.0:
    raise RuntimeError('incorrect whole-collect heartbeat boundaries')
observed = dict(row_hook_count=1, row_hook_attributes_restored=True,
                row_hook_entered_at=1.0, row_hook_exited_at=2.0,
                sent_at=3.0, signal_handler_at=4.0, signal_snapshot={'rows': 100})
if benchmark.observation_errors(observed, 5.0, use_hook=True, signal_requested=True):
    raise RuntimeError('valid delayed full-result negative control rejected')
for changed in ({'sent_at': 6.0}, {'row_hook_count': 2}, {'signal_during_row_hook': True},
                {'row_hook_attributes_restored': False}):
    invalid = observed | changed
    if not benchmark.observation_errors(invalid, 5.0, use_hook=True, signal_requested=True):
        raise RuntimeError(f'invalid observation accepted: {changed}')
try:
    benchmark.validate_rows([], 1, 1, 8, 'numeric', tuple)
except ValueError:
    pass
else:
    raise RuntimeError('optimized mode removed correctness validation')
"""
        result = subprocess.run(
            [sys.executable, "-S", "-O", "-c", program, str(ROOT)],
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_one_shot_row_notification_restores_class_attributes(self):
        program = """
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location(
    'benchmark', Path(sys.argv[1]) / 'scripts/benchmark_row_responsiveness.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
case = sys.argv[2]
class HookRow(tuple):
    def __setattr__(self, name, value):
        if case == 'action-error':
            raise ValueError('action failed')
        self.__dict__[name] = value
missing = object()
names = ('__getattribute__', '__setattr__')
saved = {name: HookRow.__dict__.get(name, missing) for name in names}
observed = {}
notifications = []
def notify():
    if not all(HookRow.__dict__.get(name, missing) is value for name, value in saved.items()):
        raise RuntimeError('notification preceded restoration')
    if not hook.active:
        raise RuntimeError('notification was not guarded')
    notifications.append(True)
    if case == 'notify-error':
        raise ValueError('notification failed')
hook = benchmark.FirstRowNotification(HookRow, lambda row: row[0] == 1000, notify, observed)
try:
    with hook:
        if case == 'unused':
            pass
        elif case == 'fast':
            row = tuple.__new__(HookRow, (1000,))
            row.__dict__['__fields__'] = ['value']
            later = tuple.__new__(HookRow, (1000,))
            later.__dict__['__fields__'] = ['value']
        else:
            row = HookRow((1000,))
            row.__fields__ = ['value']
            later = HookRow((1000,))
            later.__fields__ = ['value']
except ValueError:
    if case not in ('action-error', 'notify-error'):
        raise
else:
    if case in ('action-error', 'notify-error'):
        raise RuntimeError('expected error did not propagate')
if not all(HookRow.__dict__.get(name, missing) is value for name, value in saved.items()):
    raise RuntimeError('class attributes changed')
if hook.active or not observed['row_hook_attributes_restored']:
    raise RuntimeError('hook leaked')
expected = 0 if case in ('unused', 'action-error') else 1
if len(notifications) != expected or observed.get('row_hook_count', 0) != expected:
    raise RuntimeError('notification did not fire exactly once')
"""
        for case in ("original", "fast", "unused", "action-error", "notify-error"):
            with self.subTest(case=case):
                result = subprocess.run(
                    [sys.executable, "-S", "-c", program, str(ROOT), case],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_signal_inside_row_hook_is_guarded(self):
        program = """
import importlib.util, os, signal, sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'python/tests'))
from _row_conversion_probe import RowConversionProbe
spec = importlib.util.spec_from_file_location('benchmark', root / 'scripts/benchmark_row_responsiveness.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
class HookRow(tuple):
    pass
missing = object()
saved = {name: HookRow.__dict__.get(name, missing) for name in ('__getattribute__', '__setattr__')}
previous_handler = signal.getsignal(signal.SIGINT)
probe = RowConversionProbe(HookRow, 'value', 2, observe_progress=False, observe_materialization=False)
hook = benchmark.FirstRowNotification(HookRow, lambda row: True,
    lambda: os.kill(os.getpid(), signal.SIGINT), probe.observed)
probe.signal_guard = lambda: hook.active
def build():
    with hook:
        row = tuple.__new__(HookRow, (1000,))
        row.__dict__['__fields__'] = ['value']
        return [row]
with probe:
    rows = probe.collect(build)
if not probe.observed.get('signal_during_row_hook') or 'signal_snapshot' in probe.observed:
    raise RuntimeError('hook-time signal was accepted')
if signal.getsignal(signal.SIGINT) is not previous_handler or hook.active:
    raise RuntimeError('hook-time signal leaked state')
if not all(HookRow.__dict__.get(name, missing) is value for name, value in saved.items()):
    raise RuntimeError('hook-time signal changed Row class attributes')
"""
        result = subprocess.run(
            [sys.executable, "-S", "-c", program, str(ROOT)],
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
