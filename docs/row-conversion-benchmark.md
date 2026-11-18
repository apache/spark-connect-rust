# Row conversion throughput and responsiveness

Build the extension from each source revision using the commands in
[Contributing](contributing.md#build-the-extension). Keep three separate builds:
the Python `_create_row` constructor, the native tuple constructor without
checkpoints (a negative control), and the native constructor with checkpoints.
Use the same release profile, Python version, dependencies, allocator, CPU
affinity and host for all three. Do not run another build or benchmark alongside
the measurements. Record source and wheel hashes separately when preparing wheels.

Run one fresh process per sample with the corresponding build on `PYTHONPATH`.
The script uses the existing mock Connect server in a separate process; no remote
Spark server is required. Fixture setup, imports and server startup are outside
the measured collection. `--source-root` must identify the source used to build
the loaded extension, not just the checkout containing the benchmark script.

```sh
python scripts/benchmark_row_responsiveness.py --label original --kind numeric \
  --build-note 'toolchain/profile/dependency identity from the build log' \
  --source-root /path/to/original --uninstrumented --output numeric-original.json
python scripts/benchmark_row_responsiveness.py --label original --kind numeric \
  --build-note 'toolchain/profile/dependency identity from the build log' \
  --source-root /path/to/original --first-row-hook --output heartbeat-original.json
python scripts/benchmark_row_responsiveness.py --label original --kind numeric \
  --build-note 'toolchain/profile/dependency identity from the build log' \
  --source-root /path/to/original --signal-delay 0.05 --output signal-original.json
python scripts/benchmark_row_responsiveness.py --label original --kind nested \
  --build-note 'toolchain/profile/dependency identity from the build log' \
  --source-root /path/to/original --output nested-original.json
python scripts/benchmark_row_responsiveness.py --label original --kind wide \
  --build-note 'toolchain/profile/dependency identity from the build log' \
  --source-root /path/to/original --output wide-original.json
```

Repeat for the fast-only and checkpoint builds. Interleave the build order rather
than finishing every sample of one build first; for example, repeat
original / fast-only / checkpoints / checkpoints / fast-only / original.
Keep throughput and SIGINT runs separate. Use `--uninstrumented` for speed
comparisons: it installs no heartbeat, phase hook, GC callback or signal handler.
Other modes include instrumentation cost and must not be presented as uninstrumented
throughput. Use `--first-row-hook` for phase-marked heartbeat-only samples; signal
samples always enable that hook.

The numeric default is 524,288 rows with eight longs. The nested control has
262,144 rows containing a struct and an array of two structs. The wide control
has 8,192 rows with 256 longs. Values repeat across Arrow batches, so input size
and batch size remain identical across builds. Successful collections validate
the result length, representative values, tuple-backed Row class and shared
top-level field-name list.

## Interpreting samples

The default, GC-observed mode uses a one-shot callback that looks in generation zero for actual result
Rows with a unique field name and records the first observation. It removes itself
at the end of that collection. Native allocations may not service automatic GC
until native conversion has returned, so this observation can be too late to
measure conversion. Foreign callbacks are rejected: their Python frames
would introduce scheduling points even in the fast-only negative control.
Whole-collect gaps include execution/decoding and are reported separately.
Phase gaps and real tick counts cover the observed window until collection
returns or raises. The end boundary contributes to gaps, not to the tick count.
Ticks alone are not proof of mid-conversion progress: the observation callback
itself enters Python, and a tick can also land after native return but before the
Python end timestamp. Establish regression-test sensitivity with the fast-only
control rather than interpreting a nonzero benchmark tick count as a pass.

For `--first-row-hook` and signal samples, two temporary Row methods intercept
the first matching result tuple: the original constructor assigns `__fields__`,
and the native fast constructor reads `__dict__`. Both original class attributes
are restored exactly before notifying the signal helper, including removing an
override of an inherited method. There is no per-Row Python hook after the first
notification. The tuple predicate identifies the benchmark's first numeric tuple
(or first nested child), not a setup/network event. Require exactly one real
notification from every build and restored attributes. The hook adds a Python
frame and may change scheduling/GC: these are instrumented responsiveness results.
Phase metrics start after hook exit, excluding its interval. The signal handler
does not raise during the hook. Context cleanup restores methods on errors too.

For `--signal-delay`, an already-started independent process waits for that hook
notification before delaying and sending `SIGINT`. It does not need the client's
GIL to send the signal. Compare handler-entry delay and collection-return delay:
the latter also includes error unwinding, object cleanup and the handler's live
Row count. A signal with `signal_sent_in_materialization_window=false` did not
exercise conversion and must not be used as a cancellation-latency sample.
Require hook exit before send, and send before collection end. Late or hook-time
samples are invalid. The send-time window alone is necessary but not sufficient to establish that
conversion was still in progress. `signal_snapshot` captures the live Rows at
main-thread handler entry; its all-generations count supports
`handler_saw_partial_rows`. The handler raises only
for a partial result during collection, never during the probe callback or teardown.
Probe callbacks, worker threads and the previous handler are cleaned up in nested
`finally` blocks. Handler dispatch on worker-thread collections is not supported.
Both processes use Python's process-comparable monotonic clock and record its
implementation and resolution.

`observation_valid` and `observation_invalid_reasons` check the hook count,
restoration, send/handler timestamps, instrumentation guards and presence of a
live-Row signal snapshot. Validity is separate from responsiveness: a delayed
handler that sees the entire completed result is a valid negative-control
observation, not successful cancellation. Reject invalid samples before comparing
latency; do not discard valid samples merely because their responsiveness is poor.

Report raw samples and their spread, not only a minimum or median. Check whether
ordering, host load, CPU affinity or the observation overhead explain outliers.
No fixed heartbeat or wall-time threshold is a compatibility guarantee. Interpreter
switch interval, CPU contention and cyclic garbage collection affect gaps; none is
a universal latency floor. Output records the unchanged switch interval, GC
thresholds/counts/stats, foreign callbacks, CPU/load, Python executable/hash/compiler,
package versions and operator-reported build provenance. For attribution, separately
repeat with `--freeze-gc-after-observation`; do not mix that controlled variant with
ordinary-GC samples. Instrumentation and handler heap snapshots have their own cost.

The checkpoint implementation re-enters a cached trivial Python function so that
the interpreter can service pending thread handoffs, then checks Python signals.
It counts both Rows and nested values. Counting Rows
also handles zero-column results; counting values reaches inside wide Rows and
single large arrays or maps. The checkpoint interval is a work budget, not a
millisecond deadline: one large scalar allocation, Python UDT callback or cleanup
can still take longer. The offline regression cases check partial materialization
and cancellation for rows, arrays and maps without a wall-time pass threshold.
Those cases preserve the normal switch interval. After their one-shot observation,
they disable automatic GC and take a single all-generations snapshot from the
Python observer. This avoids repeated polling scans; including all generations
prevents promotion from hiding completed Rows. The initial snapshot still has
whole-heap capture cost. A snapshot
records both capture and count-completion times: counting can finish after native
conversion, without invalidating a partial snapshot captured earlier. Timestamp
ordering alone is not proof: the observer may run after the native call returns
but before Python records its end. The retained completed result must count in
full; the progress discriminator is
the strictly partial count, not just the timestamp. Cancellation also needs an
independent all-generations partial snapshot inside the main-thread signal handler.
The fast-only negative-control run remains necessary to exclude progress caused
by the one-shot callback's own interpreter entry or teardown. GC state,
callbacks and handlers are restored even on errors or late signals.

The same budget (`ConversionBudget` in `crates/pyspark-rs/src/interrupt.rs`) checkpoints the
native conversion of the local data of `createDataFrame`, whose PySpark counterpart is
Python code: the collection of the rows, the sorting of dict rows, inference, taking the
rows apart, the converters and their values in arrays, maps and structs, the integer columns
of an explicit schema, the cells of the native pandas row path (of no PyArrow), the strings of
the columns of objects of a plain pandas frame of a struct of strings (`_pandas_string_ipc`, in
its check and its encoding of the values), and the release of the rows.
The rows are assembled into the relation without the GIL. Its offline regression cases
(`python/tests/local_data_conversion_cases.py`, with the probe of
`python/tests/_local_data_conversion_probe.py`) observe each of these parts but the integer
columns, whose scan of a few hundred thousand rows is too short to observe, and the release,
by the references that the conversion holds to the objects of the data, read with
`sys.getrefcount`. The data of each case ends with a value that fails the conversion right
after that part, without running Python code, so that no later part can be mistaken for it; of
the strings, PySpark's conversion that follows the native one fails for it, without holding the
values that the case observes.
They disable automatic GC during the conversion, whose finalizers and weakref callbacks run
Python code, and warm up the caches of the conversion first. A build without these
checkpoints fails every case.

`python -m unittest discover -s scripts/tests -p test_row_conversion_probe.py` checks
the instrumentation in isolated standard-library-only subprocesses, including late
SIGINT during a join, error cleanup and optimized-mode correctness validation.
These synthetic Python fixtures do not execute the native candidate or establish
its scheduling fairness. Required native runs still include the fast-only negative
control and a CPU-contention experiment before claiming speed retention.
