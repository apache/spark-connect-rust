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

"""The Rust half of gen_fn_dispatch.py: parsing, the override tables and --check.

None of this needs a reference PySpark; the generated-drift workflow checks
functions_generated.py against one.
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "gen_fn_dispatch.py"
SPEC = importlib.util.spec_from_file_location("gen_fn_dispatch", SCRIPT)
gen = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gen
SPEC.loader.exec_module(gen)

RUST = """
pub fn broadcast(df: DataFrame) -> DataFrame {
    df.broadcast()
}

/// Follows a function that does not return a Column.
pub fn cume_dist() -> Column {
    func("cume_dist", vec![])
}

pub fn round(col1: Column) -> Column {
    todo!()
}

pub fn round_scale(col1: Column, scale: Column) -> Column {
    todo!()
}

pub fn r#struct<C: Into<Column>>(cols: impl IntoIterator<Item = C>) -> Column {
    todo!()
}

pub fn concat_ws<C: Into<Column>>(cols: impl IntoIterator<Item = C>) -> Column {
    todo!()
}

pub fn instr(col1: Column, col2: Column) -> Column {
    todo!()
}

pub fn instr_with_start(col1: Column, col2: Column, start: Column) -> Column {
    todo!()
}

pub fn instr_with_start_and_occurrence(
    col1: Column,
    col2: Column,
    start: Column,
    occurrence: Column,
) -> Column {
    todo!()
}

pub fn count_distinct(col1: Column) -> Column {
    todo!()
}

pub fn count_distinct_columns(col1: Column, cols: impl IntoIterator<Item = Column>) -> Column {
    todo!()
}

pub fn transform<F>(col: Column, f: F) -> Column
where
    F: FnOnce(Vec<Column>) -> Column,
{
    todo!()
}

pub fn rand_with_seed(seed: i64) -> Column {
    todo!()
}
"""

TABLES = {
    "skip": {"count_distinct"},
    "dispatch_only": {"count_distinct"},
    "optional_arg_variants": {
        "round": ["round_scale"],
        "instr": ["instr_with_start", "instr_with_start_and_occurrence"],
    },
    "head_rest_arms": {"count_distinct": "count_distinct_columns"},
    "not_dispatched": {"transform": "closure", "rand_with_seed": "i64 seed"},
    "section_overrides": {"concat_ws": "single"},
    "order_as": {},
}


def plan(src=RUST, **overrides):
    return gen.plan_dispatch(gen.parse_rust_functions(src), **{**TABLES, **overrides})


def arm_text(arms, name):
    (arm,) = [a for a in arms if a["name"] == name]
    return "\n".join(line.strip() for line in gen.render_arm(arm))


class ParseTests(unittest.TestCase):
    def test_parses_generics_raw_identifiers_and_multiline_parameters(self):
        functions = gen.parse_rust_functions(RUST)
        self.assertNotIn("broadcast", functions)  # Returns a DataFrame.
        # The function after a non-Column one is not swallowed into its signature.
        self.assertEqual(functions["cume_dist"]["params"], [])
        self.assertEqual(functions["struct"]["full_name"], "r#struct")
        self.assertEqual(functions["struct"]["generics"], "C: Into<Column>")
        self.assertEqual(
            [p["name"] for p in functions["instr_with_start_and_occurrence"]["params"]],
            ["col1", "col2", "start", "occurrence"],
        )
        self.assertEqual(functions["transform"]["params"][1], {"name": "f", "type": "F"})
        self.assertEqual(
            list(functions)[:3], ["cume_dist", "round", "round_scale"], "kept in source order"
        )


class PlanTests(unittest.TestCase):
    def test_sections_then_source_order(self):
        self.assertEqual(
            [(a["name"], a["section"]) for a in plan()],
            [
                ("cume_dist", "no_args"),
                ("round", "single"),
                ("concat_ws", "single"),  # Variadic, moved by SECTION_OVERRIDES.
                ("count_distinct", "single"),
                ("struct", "variadic"),
                ("instr", "multiple"),
            ],
        )

    def test_optional_argument_picks_the_longer_function(self):
        self.assertEqual(
            arm_text(plan(), "round").splitlines()[2],
            "if args.len() >= 2 { Ok(spark_funcs::round_scale(args[0].clone(), args[1].clone())) }"
            " else { Ok(spark_funcs::round(args[0].clone())) }",
        )

    def test_several_optional_arguments_match_on_the_count(self):
        self.assertEqual(
            arm_text(plan(), "instr").splitlines()[2:7],
            [
                "match args.len() {",
                "2 => Ok(spark_funcs::instr(args[0].clone(), args[1].clone())),",
                "3 => Ok(spark_funcs::instr_with_start"
                "(args[0].clone(), args[1].clone(), args[2].clone())),",
                "_ => Ok(spark_funcs::instr_with_start_and_occurrence"
                "(args[0].clone(), args[1].clone(), args[2].clone(), args[3].clone())),",
                "}",
            ],
        )

    def test_head_rest_and_variadic_arms(self):
        arms = plan()
        self.assertIn(
            "Ok(spark_funcs::count_distinct_columns(args[0].clone(), args[1..].to_vec()))",
            arm_text(arms, "count_distinct"),
        )
        self.assertEqual(
            arm_text(arms, "struct"), '"struct" => Ok(spark_funcs::r#struct(args.clone())),'
        )

    def test_order_as_places_an_arm_at_another_function(self):
        arms = plan(order_as={"concat_ws": "cume_dist"})
        self.assertEqual([a["name"] for a in arms if a["section"] == "single"][0], "concat_ws")

    def test_wrappers_are_the_arms_less_the_hand_written_plus_upstream(self):
        names = gen.wrapper_names(plan(), skip=TABLES["skip"], upstream_extra={"sha2"})
        self.assertEqual(names, ["concat_ws", "cume_dist", "instr", "round", "sha2", "struct"])

    def test_unaccounted_signature_fails_instead_of_being_dropped(self):
        with self.assertRaisesRegex(gen.GeneratorError, r"rand_with_seed\(seed: i64\)"):
            plan(not_dispatched={"transform": "closure"})

    def test_stale_or_ill_fitting_table_entries_fail(self):
        cases = {
            "names gone": {"not_dispatched": {**TABLES["not_dispatched"], "gone": "removed"}},
            "must take 2 columns": {"optional_arg_variants": {"round": ["instr_with_start"]}},
            "not f\\(Column, columns\\)": {"head_rest_arms": {"count_distinct": "round_scale"}},
            "not a variadic": {"section_overrides": {"round": "multiple"}},
            "not in SKIP_FUNCTIONS": {"dispatch_only": {"count_distinct", "round"}},
        }
        for message, overrides in cases.items():
            with self.subTest(message), self.assertRaisesRegex(gen.GeneratorError, message):
                plan(**overrides)


class CheckedInTests(unittest.TestCase):
    """The checked-in Rust dispatch is what the generator writes."""

    def test_dispatch_generated_rs_is_reproducible(self):
        outputs = gen.generate(gen.REPO_ROOT, log=lambda *_: None)
        self.assertEqual(list(outputs), [gen.DISPATCH_FILE])
        self.assertEqual(
            outputs[gen.DISPATCH_FILE],
            (gen.REPO_ROOT / gen.DISPATCH_FILE).read_text(),
            "run scripts/gen_fn_dispatch.py --check for the diff",
        )

    def test_check_reports_a_perturbed_file(self):
        outputs = gen.generate(gen.REPO_ROOT, log=lambda *_: None)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root / gen.DISPATCH_FILE
            target.parent.mkdir(parents=True)
            target.write_text(outputs[gen.DISPATCH_FILE])
            self.assertTrue(gen.check(root, outputs, log=lambda *_: None))

            target.write_text(outputs[gen.DISPATCH_FILE].replace("round_scale", "round", 1))
            logged = []
            self.assertFalse(gen.check(root, outputs, log=logged.append))
            self.assertTrue(logged[0].startswith(f"DRIFT {gen.DISPATCH_FILE}"))
            self.assertTrue(any(line.startswith("+") and "round_scale" in line for line in logged))


if __name__ == "__main__":
    unittest.main()
