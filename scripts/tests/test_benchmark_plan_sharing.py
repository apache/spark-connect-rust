#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline benchmark-harness tests; no PySpark installation or server required."""

import contextlib
import copy
import gc
import importlib.util
import io
import json
import tempfile
import types
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "benchmark_plan_sharing.py"
SPEC = importlib.util.spec_from_file_location("benchmark_plan_sharing", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class Expression:
    def __init__(self, evaluate, aggregate=None, child=None):
        self.evaluate = evaluate
        self.aggregate = aggregate
        self.child = child
        self.name = None

    def alias(self, name):
        result = Expression(self.evaluate, aggregate=self.aggregate, child=self)
        result.name = name
        return result

    def __add__(self, other):
        return Expression(lambda row: self.evaluate(row) + other, child=self)


class Functions:
    @staticmethod
    def col(name):
        return Expression(lambda row: row[name])

    @staticmethod
    def sum(name):
        return Expression(lambda rows: sum(row[name] for row in rows), aggregate=True)

    @staticmethod
    def count(name):
        return Expression(lambda rows: sum(row[name] is not None for row in rows), aggregate=True)


class Frame:
    def __init__(self, rows, inputs=(), expressions=()):
        self.rows = rows
        self.inputs = inputs
        self.expressions = expressions

    def select(self, *expressions):
        if expressions[0].aggregate:
            rows = [{expr.name: expr.evaluate(self.rows) for expr in expressions}]
        else:
            rows = [{expr.name: expr.evaluate(row) for expr in expressions} for row in self.rows]
        return Frame(rows, inputs=(self,), expressions=expressions)

    def union(self, other):
        return Frame(self.rows + other.rows, inputs=(self, other))

    def collect(self):
        return [tuple(row.values()) for row in self.rows]


class Session:
    @staticmethod
    def range(start, end=None, step=1, partitions=None):
        if end is None:
            start, end = 0, start
        return Frame([{"id": value} for value in range(start, end, step)])


class BenchmarkPlanSharingTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "rows": 5,
            "depth": 3,
            "branches": 4,
            "partitions": 1,
            "warmups": 0,
            "repetitions": 1,
            "order_seed": 0,
        }

    def report(self):
        expected = benchmark.expected_result("linear", 5, 3, 4)
        return {
            "format_version": 1,
            "status": "ok",
            "label": "baseline",
            "build_id": "before",
            "config": self.config,
            "endpoint_sha256": "endpoint",
            "server_version": "test",
            "server_configs": {},
            "provenance": {
                "python_version": "test",
                "platform": "test",
                "hostname": "test",
                "dependencies": {},
                "script_sha256": "script",
                "extension": {"sha256": "a"},
            },
            "results": {
                "linear": {
                    "expected": expected,
                    "warmups": [],
                    "samples": [
                        {
                            "construction_s": 1.0,
                            "action_s": 2.0,
                            "total_s": 3.0,
                            "result": expected,
                        }
                    ],
                }
            },
        }

    def test_workload_values_and_retained_handles(self):
        for name in benchmark.PLAN_WORKLOADS:
            with self.subTest(name=name):
                query, retained = benchmark.build_query(Session(), Functions, name, self.config)
                expected = benchmark.expected_result(name, 5, 3, 4)
                self.assertEqual(benchmark.check_result(query.collect(), expected), expected)
                # range, rename, three projections, aggregate; branches add four
                # projections and three unions. Intermediate frames are not discarded.
                self.assertEqual(len(retained), 6 if name == "linear" else 13)
                self.assertIs(retained[-1], query)

    def test_expression_values_and_retained_handles(self):
        for name in benchmark.EXPRESSION_WORKLOADS:
            with self.subTest(name=name):
                query, retained = benchmark.build_query(Session(), Functions, name, self.config)
                expected = benchmark.expected_result(name, 5, 3, 4)
                self.assertEqual(benchmark.check_result(query.collect(), expected), expected)
                columns = [handle for handle in retained if isinstance(handle, Expression)]
                frames = [handle for handle in retained if isinstance(handle, Frame)]
                # Initial Column + three increments; final aggregate adds two raw
                # Columns and two aliases. Branches add one shifted Column + alias each.
                self.assertEqual(len(columns), 9 if name == "expression_chain" else 16)
                self.assertEqual(len(frames), 3 if name == "expression_chain" else 9)
                for index in range(1, self.config["depth"] + 1):
                    self.assertIs(columns[index].child, columns[index - 1])
                self.assertIs(retained[-1], query)
                for expr in query.expressions:
                    self.assertIn(expr, retained)
                    self.assertIn(expr.child, retained)

    def test_expression_reuse_uses_the_same_source_and_column(self):
        _, retained = benchmark.build_query(Session(), Functions, "expression_reuse", self.config)
        columns = [handle for handle in retained if isinstance(handle, Expression)]
        frames = [handle for handle in retained if isinstance(handle, Frame)]
        shared_column = columns[self.config["depth"]]
        for branch in frames[1 : self.config["branches"] + 1]:
            self.assertIs(branch.inputs[0], frames[0])
            aliased = branch.expressions[0]
            self.assertIs(aliased.child.child, shared_column)
            self.assertIn(aliased, retained)
            self.assertIn(aliased.child, retained)

    def test_expression_chain_has_one_projection_of_the_deep_column(self):
        _, retained = benchmark.build_query(Session(), Functions, "expression_chain", self.config)
        columns = [handle for handle in retained if isinstance(handle, Expression)]
        frames = [handle for handle in retained if isinstance(handle, Frame)]
        self.assertIs(frames[1].inputs[0], frames[0])
        self.assertIs(frames[1].expressions[0].child, columns[self.config["depth"]])

    def test_zero_depth_and_one_branch(self):
        config = {**self.config, "depth": 0, "branches": 1}
        for name in benchmark.WORKLOADS:
            with self.subTest(name=name):
                query, _ = benchmark.build_query(Session(), Functions, name, config)
                checksum = 10 if name in ("linear", "expression_chain") else 15
                self.assertEqual(query.collect(), [(checksum, 5)])

    def test_split_timings_and_gc_restoration(self):
        before = gc.isenabled()
        with patch.object(benchmark.time, "perf_counter", side_effect=[10.0, 12.0, 17.0]):
            sample = benchmark.measure(Session(), Functions, "linear", self.config)
        self.assertEqual(sample["construction_s"], 2.0)
        self.assertEqual(sample["action_s"], 5.0)
        self.assertEqual(sample["total_s"], 7.0)
        self.assertEqual(gc.isenabled(), before)

    def test_action_failure_propagates_and_restores_gc(self):
        before = gc.isenabled()
        with patch.object(Frame, "collect", side_effect=RuntimeError("server failed")):
            with self.assertRaisesRegex(RuntimeError, "server failed"):
                benchmark.measure(Session(), Functions, "linear", self.config)
        self.assertEqual(gc.isenabled(), before)

    def test_measure_retains_handles_until_action_completes(self):
        class Handle:
            pass

        for fails in (False, True):
            with self.subTest(fails=fails):
                references = []

                def build_query(*args, references=references):
                    handle = Handle()
                    references.append(weakref.ref(handle))
                    return Frame([]), [handle]

                def collect(references=references, fails=fails):
                    self.assertIsNotNone(references[0]())
                    if fails:
                        raise RuntimeError("server failed")
                    return [(25, 5)]

                with (
                    patch.object(benchmark, "build_query", side_effect=build_query),
                    patch.object(Frame, "collect", side_effect=collect),
                ):
                    if fails:
                        with self.assertRaisesRegex(RuntimeError, "server failed"):
                            benchmark.measure(Session(), Functions, "linear", self.config)
                    else:
                        benchmark.measure(Session(), Functions, "linear", self.config)
                self.assertIsNone(references[0]())

    def test_incorrect_checksum_or_type_fails(self):
        for rows in ([], [(25,)], [(24, 5)], [(25.0, 5)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                benchmark.check_result(rows, {"checksum": 25, "count": 5})

    def test_overflow_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "overflow"):
            benchmark.expected_result("branched", 1 << 40, 16, 4)

    def test_native_provenance_rejects_wrong_classes_and_hash(self):
        pyspark = types.ModuleType("pyspark")
        native = types.ModuleType("pyspark._pyspark")
        native.SparkSession = object
        native.DataFrame = Frame
        native.Column = Expression
        with patch.dict(benchmark.sys.modules, {"pyspark": pyspark, "pyspark._pyspark": native}):
            with self.assertRaisesRegex(RuntimeError, "actual Rust-backed"):
                benchmark.client_provenance(Session(), Functions, "rust", None)
            native.SparkSession = Session
            with tempfile.TemporaryDirectory() as directory:
                suffix = benchmark.importlib.machinery.EXTENSION_SUFFIXES[0]
                extension = Path(directory) / f"_pyspark{suffix}"
                extension.write_bytes(b"offline identity test")
                native.__file__ = str(extension)
                with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                    benchmark.client_provenance(Session(), Functions, "rust", "incorrect")

    def test_transport_plugin_is_rejected(self):
        with patch.dict(
            benchmark.sys.modules,
            {
                "pyspark": types.ModuleType("pyspark"),
                "rust_transport_plugin": types.ModuleType("rust_transport_plugin"),
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "Transport-injected"):
                benchmark.client_provenance(Session(), Functions, "rust", None)

    def test_report_rejects_missing_samples_and_bad_timings(self):
        report = self.report()
        benchmark.validate_report(report)
        for replacement in ([], [{**report["results"]["linear"]["samples"][0], "total_s": 4.0}]):
            broken = copy.deepcopy(report)
            broken["results"]["linear"]["samples"] = replacement
            with self.assertRaises(ValueError):
                benchmark.validate_report(broken)

    def test_compare_rejects_same_binary_and_different_settings(self):
        before = self.report()
        after = copy.deepcopy(before)
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.json"
            candidate = Path(directory) / "candidate.json"
            baseline.write_text(json.dumps(before))
            candidate.write_text(json.dumps(after))
            with self.assertRaisesRegex(ValueError, "same extension"):
                benchmark.compare(str(baseline), str(candidate))
            after["provenance"]["extension"]["sha256"] = "b"
            candidate.write_text(json.dumps(after))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                benchmark.compare(str(baseline), str(candidate))
            self.assertIn("baseline/candidate", output.getvalue())
            after["config"]["order_seed"] = 1
            candidate.write_text(json.dumps(after))
            with self.assertRaisesRegex(ValueError, "different config"):
                benchmark.compare(str(baseline), str(candidate))

    def test_cli_failure_returns_nonzero_without_a_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            with (
                patch.object(benchmark, "run", side_effect=RuntimeError("wrong client")),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                status = benchmark.main(
                    [
                        "--label",
                        "test",
                        "--build-id",
                        "test",
                        "--out",
                        str(output),
                    ]
                )
            self.assertEqual(status, 1)
            self.assertFalse(output.exists())

    def test_expression_workload_cli_choices(self):
        for name in benchmark.EXPRESSION_WORKLOADS:
            with (
                self.subTest(name=name),
                patch.object(benchmark, "run", side_effect=RuntimeError("offline CLI test")) as run,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(
                    benchmark.main(
                        [
                            "--label",
                            "test",
                            "--build-id",
                            "test",
                            "--out",
                            "unused.json",
                            "--workload",
                            name,
                        ]
                    ),
                    1,
                )
                self.assertEqual(run.call_args.args[0].workload, name)


if __name__ == "__main__":
    unittest.main()
