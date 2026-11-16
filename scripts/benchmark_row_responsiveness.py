"""Measure instrumented collection, Python progress and external SIGINT delivery."""

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
import uuid
from contextlib import nullcontext
from pathlib import Path


def file_hash(filename):
    return hashlib.sha256(Path(filename).read_bytes()).hexdigest()


class FirstRowNotification:
    def __init__(self, row_type, matches, notify, observed):
        self.row_type = row_type
        self.matches = matches
        self.notify = notify
        self.observed = observed
        self.active = False
        self.missing = object()
        self.saved = {}

    def restore(self):
        for name, value in self.saved.items():
            if value is self.missing:
                if name in self.row_type.__dict__:
                    delattr(self.row_type, name)
            else:
                setattr(self.row_type, name, value)

    def run(self, path, action):
        self.active = True
        self.observed["row_hook_entered_at"] = time.monotonic()
        try:
            self.restore()
            result = action()
            self.observed["row_hook_count"] = self.observed.get("row_hook_count", 0) + 1
            self.observed["row_hook_path"] = path
            self.notify()
            return result
        finally:
            self.restore()
            self.observed["row_hook_exited_at"] = time.monotonic()
            self.active = False

    def __enter__(self):
        self.saved = {
            name: self.row_type.__dict__.get(name, self.missing)
            for name in ("__getattribute__", "__setattr__")
        }
        original_getattribute = self.row_type.__getattribute__
        original_setattr = self.row_type.__setattr__

        def getattribute(row, name):
            if name == "__dict__" and self.matches(row):
                return self.run("__getattribute__", lambda: original_getattribute(row, name))
            return original_getattribute(row, name)

        def setattribute(row, name, value):
            if name == "__fields__" and self.matches(row):
                return self.run("__setattr__", lambda: original_setattr(row, name, value))
            return original_setattr(row, name, value)

        try:
            self.row_type.__getattribute__ = getattribute
            self.row_type.__setattr__ = setattribute
        except BaseException:
            self.restore()
            raise
        return self

    def __exit__(self, error_type, error, traceback):
        self.restore()
        self.observed["row_hook_attributes_restored"] = all(
            self.row_type.__dict__.get(name, self.missing) is value
            for name, value in self.saved.items()
        )


def heartbeat_stats(ticks, first_observed, started, ended):
    actual_ticks = [tick for tick in ticks if started <= tick < ended]
    boundaries = [started, *actual_ticks, ended]
    gaps = [
        min(later, ended) - max(earlier, first_observed)
        for earlier, later in zip(boundaries, boundaries[1:])
        if first_observed is not None and later >= first_observed and earlier < ended
    ]
    return {
        "max_collect_heartbeat_gap_s": max(
            (later - earlier for earlier, later in zip(boundaries, boundaries[1:])),
            default=None,
        ),
        "collect_heartbeat_ticks": len(actual_ticks),
        "max_materialization_heartbeat_gap_s": max(gaps, default=None),
        "materialization_heartbeat_ticks": sum(
            first_observed <= tick < ended for tick in actual_ticks
        )
        if first_observed is not None
        else 0,
    }


def observation_errors(observed, ended, *, use_hook, signal_requested):
    errors = []
    if use_hook:
        entered = observed.get("row_hook_entered_at")
        exited = observed.get("row_hook_exited_at")
        if observed.get("row_hook_count") != 1:
            errors.append("Row hook did not fire exactly once")
        if not observed.get("row_hook_attributes_restored"):
            errors.append("Row class attributes were not restored")
        if entered is None or exited is None or not entered <= exited < ended:
            errors.append("Row hook did not finish before collection ended")
    if signal_requested:
        sent = observed.get("sent_at")
        exited = observed.get("row_hook_exited_at")
        handled = observed.get("signal_handler_at")
        if sent is None or exited is None or not exited <= sent < ended:
            errors.append("SIGINT was not sent after the Row hook and before collection ended")
        if handled is None or sent is None or handled < sent:
            errors.append("SIGINT handler timestamp is missing or precedes send")
        if observed.get("signal_during_row_hook") or observed.get("signal_during_probe_callback"):
            errors.append("SIGINT handler ran inside instrumentation")
        if "signal_snapshot" not in observed:
            errors.append("SIGINT handler did not capture live Rows during collection")
    return errors


def validate_rows(rows, row_count, batch_rows, width, kind, row_type):
    if len(rows) != row_count:
        raise ValueError(f"expected {row_count} rows, got {len(rows)}")
    for index in (0, row_count // 2, row_count - 1):
        row = rows[index]
        expected = 1000 + index % batch_rows
        if type(row) is not row_type or not isinstance(row, tuple):
            raise TypeError(f"row {index} is not the public tuple-backed Row")
        if kind == "nested":
            matches = row[0].value == expected and [value.value for value in row[1]] == [
                expected,
                expected,
            ]
        else:
            matches = tuple(row) == (expected,) * width
        if not matches:
            raise ValueError(f"incorrect value in row {index}")
    if rows[0].__fields__ is not rows[-1].__fields__:
        raise ValueError("top-level field lists are not shared")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("numeric", "nested", "wide"), default="numeric")
    parser.add_argument("--rows", type=int)
    parser.add_argument("--batch-rows", type=int, default=16384)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--build-note",
        required=True,
        help="operator-reported toolchain, profile and build/dependency identity",
    )
    parser.add_argument("--signal-delay", type=float)
    parser.add_argument("--first-row-hook", action="store_true")
    parser.add_argument("--uninstrumented", action="store_true")
    parser.add_argument("--freeze-gc-after-observation", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    row_count = (
        args.rows
        if args.rows is not None
        else {
            "numeric": 524288,
            "nested": 262144,
            "wide": 8192,
        }[args.kind]
    )
    batch_rows = min(row_count, args.batch_rows)
    if row_count <= 0 or batch_rows <= 0 or row_count % batch_rows:
        parser.error("rows must be positive and divisible by batch-rows")
    if args.signal_delay is not None and (args.signal_delay < 0 or os.name != "posix"):
        parser.error("signal-delay requires POSIX and a nonnegative delay")
    if args.uninstrumented and (
        args.signal_delay is not None or args.first_row_hook or args.freeze_gc_after_observation
    ):
        parser.error("uninstrumented mode cannot use signals, Row hooks or GC controls")
    use_hook = args.first_row_hook or args.signal_delay is not None
    if use_hook and args.freeze_gc_after_observation:
        parser.error("Row hook mode cannot use the GC-observation control")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "tests"))
    import pyarrow as pa
    import pyspark
    from _mock_connect_server import (
        data_type_proto,
        message_field,
        mock_connect_session,
        varint_field,
    )
    from _row_conversion_probe import RowConversionProbe
    from pyspark import _pyspark
    from pyspark.sql import Row
    from pyspark.sql import types as T

    field_name = f"conversion_{uuid.uuid4().hex}"
    numbers = list(range(1000, 1000 + batch_rows))
    width = 256 if args.kind == "wide" else 8
    if args.kind == "nested":
        child = pa.struct([("value", pa.int64())])
        values = [{"value": number} for number in numbers]
        table = pa.table(
            {
                field_name: pa.array(values, child),
                "items": pa.array([[value, value] for value in values], pa.list_(child)),
            }
        )
        child_schema = T.StructType([T.StructField("value", T.LongType())])
        schema = T.StructType(
            [
                T.StructField(field_name, child_schema),
                T.StructField("items", T.ArrayType(child_schema)),
            ]
        )
    else:
        names = [field_name, *[f"column_{index}" for index in range(1, width)]]
        table = pa.table({name: pa.array(numbers, pa.int64()) for name in names})
        schema = T.StructType([T.StructField(name, T.LongType()) for name in names])
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    payload = message_field(7, data_type_proto(schema.jsonValue())) + message_field(
        2,
        varint_field(1, batch_rows) + message_field(2, sink.getvalue().to_pybytes()),
    )
    ticks = []
    finished = threading.Event()

    def heartbeat():
        while not finished.is_set():
            ticks.append(time.monotonic())
            finished.wait(0.001)

    helper = None
    reader = None
    trigger = None
    with mock_connect_session() as (spark, service):
        service.execute_payloads = [payload] * (row_count // batch_rows)
        frame = spark.range(row_count)
        try:
            if args.signal_delay is not None:
                reader, trigger = os.pipe()
                helper_code = (
                    "import json,os,signal,sys,time; print('ready',flush=True); "
                    "triggered=bool(os.read(int(sys.argv[1]),1)); "
                    "time.sleep(float(sys.argv[3]) if triggered else 0); "
                    "sent=time.monotonic() if triggered else None; "
                    "os.kill(int(sys.argv[2]),signal.SIGINT) if triggered else None; "
                    "print(json.dumps({'sent_at':sent}),flush=True)"
                )
                helper = subprocess.Popen(
                    [
                        sys.executable,
                        "-S",
                        "-c",
                        helper_code,
                        str(reader),
                        str(os.getpid()),
                        str(args.signal_delay),
                    ],
                    pass_fds=(reader,),
                    stdout=subprocess.PIPE,
                    text=True,
                )
                os.close(reader)
                reader = None
                if helper.stdout.readline().strip() != "ready":
                    raise RuntimeError("SIGINT helper did not become ready")
            provenance = {
                "switch_interval_s": sys.getswitchinterval(),
                "gc_threshold": gc.get_threshold(),
                "gc_enabled_before_probe": gc.isenabled(),
                "foreign_gc_callbacks": [
                    f"{type(callback).__module__}.{type(callback).__qualname__}"
                    for callback in gc.callbacks
                ],
                "load_average": os.getloadavg() if hasattr(os, "getloadavg") else None,
                "cpu_count": os.cpu_count(),
            }
            probe = (
                None
                if args.uninstrumented
                else RowConversionProbe(
                    Row,
                    field_name,
                    row_count,
                    observe_progress=False,
                    freeze_gc=args.freeze_gc_after_observation,
                    observe_materialization=not use_hook,
                )
            )
            observed = probe.observed if probe is not None else {}

            def notify():
                if trigger is not None:
                    os.write(trigger, b"1")

            hook = (
                FirstRowNotification(
                    Row,
                    lambda row: len(row) == (1 if args.kind == "nested" else width)
                    and tuple.__getitem__(row, 0) == 1000,
                    notify,
                    observed,
                )
                if use_hook
                else None
            )
            if hook is not None:
                probe.signal_guard = lambda: hook.active
            with probe if probe is not None else nullcontext():
                worker = None if args.uninstrumented else threading.Thread(target=heartbeat)
                try:
                    if worker is not None:
                        worker.start()
                    gc_before = {"stats": gc.get_stats(), "count": gc.get_count()}
                    started = time.monotonic()
                    cpu_started = time.process_time()
                    rows = None
                    was_interrupted = False
                    try:
                        with hook if hook is not None else nullcontext():
                            rows = (
                                frame.collect() if probe is None else probe.collect(frame.collect)
                            )
                    except KeyboardInterrupt:
                        was_interrupted = True
                    finally:
                        ended = time.monotonic() if probe is None else observed["collect_ended_at"]
                        cpu_seconds = time.process_time() - cpu_started
                        gc_after = {"stats": gc.get_stats(), "count": gc.get_count()}
                finally:
                    try:
                        finished.set()
                        if worker is not None and worker.ident is not None:
                            worker.join()
                    finally:
                        if trigger is not None:
                            os.close(trigger)
                            trigger = None
                        if helper is not None:
                            try:
                                output, _ = helper.communicate(timeout=10)
                            except BaseException:
                                if helper.poll() is None:
                                    helper.kill()
                                helper.communicate()
                                raise
                            if helper.returncode:
                                raise RuntimeError("SIGINT helper failed")
                            observed.update(json.loads(output))
        finally:
            if reader is not None:
                os.close(reader)
            if trigger is not None:
                os.close(trigger)
            if helper is not None and helper.poll() is None:
                helper.kill()
                helper.communicate()

        if probe is not None and probe.failures:
            raise RuntimeError(f"materialization probe failed: {probe.failures}")
        if rows is not None:
            validate_rows(rows, row_count, batch_rows, width, args.kind, Row)
        first_observed = observed.get("row_hook_exited_at", observed.get("first_rows_observed_at"))
        sent_at = observed.get("sent_at")
        handler_at = observed.get("signal_handler_at")
        invalid_reasons = observation_errors(
            observed, ended, use_hook=use_hook, signal_requested=args.signal_delay is not None
        )
        result = {
            "label": args.label,
            "build_note_operator_reported": args.build_note,
            "kind": args.kind,
            "instrumentation": "none"
            if args.uninstrumented
            else (
                "one-shot-row-hook-and-heartbeat" if use_hook else "gc-observation-and-heartbeat"
            ),
            "rows": row_count,
            "batch_rows": batch_rows,
            "python": sys.version,
            "python_executable": sys.executable,
            "python_executable_sha256": file_hash(sys.executable),
            "python_implementation": platform.python_implementation(),
            "python_compiler": platform.python_compiler(),
            "platform": platform.platform(),
            "clock": vars(time.get_clock_info("monotonic")),
            "pyspark": pyspark.__version__,
            "pyarrow": pa.__version__,
            "affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            "extension_sha256": file_hash(_pyspark.__file__),
            "harness_sha256": {
                "benchmark": file_hash(__file__),
                "probe": file_hash(sys.modules[RowConversionProbe.__module__].__file__),
            },
            "source_root_operator_reported": str(args.source_root.resolve()),
            "source_sha256": {
                name: file_hash(args.source_root / name)
                for name in (
                    "Cargo.lock",
                    "Cargo.toml",
                    "crates/pyspark-rs/src/row.rs",
                    "crates/pyspark-rs/src/row.py",
                    "crates/pyspark-rs/src/dataframe.rs",
                )
            },
            "gc_before": gc_before,
            "gc_after": gc_after,
            "gc_frozen_after_observation": args.freeze_gc_after_observation,
            "wall_s": ended - started,
            "cpu_s": cpu_seconds,
            **(
                heartbeat_stats(ticks, first_observed, started, ended)
                if not args.uninstrumented
                else {}
            ),
            "interrupted": was_interrupted,
            "observation_valid": not invalid_reasons,
            "observation_invalid_reasons": invalid_reasons,
            "signal_sent_in_materialization_window": (
                first_observed is not None
                and sent_at is not None
                and first_observed <= sent_at < ended
            ),
            "handler_saw_partial_rows": (
                0 < observed.get("signal_snapshot", {}).get("rows", 0) < row_count
            ),
            "signal_handler_delay_s": handler_at - sent_at
            if handler_at is not None and sent_at is not None
            else None,
            "signal_return_delay_s": ended - sent_at
            if sent_at is not None and sent_at < ended
            else None,
            **provenance,
            **observed,
        }
        encoded = json.dumps(result, indent=2)
        if args.output:
            args.output.write_text(encoded + "\n")
        print(encoded)


if __name__ == "__main__":
    main()
