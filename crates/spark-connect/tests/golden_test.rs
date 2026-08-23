//! Golden parity test: every expression built via the Rust Column/Expression API
//! must serialize to the exact same protobuf the reference PySpark client produces.
//!
//! Goldens live in `tests/golden/exprs.jsonl` (captured by
//! `scripts/capture_golden.py`, base64-encoded `spark.connect.Expression`). We
//! normalize out non-deterministic noise (`common`/origin, attribute `plan_id`)
//! on BOTH sides, then require byte-equality. A required case that is missing
//! from the goldens or mismatches FAILS the test - no silent skips.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

use base64::{engine::general_purpose::STANDARD, Engine};
use prost::Message;
use spark_connect::column::{col, lit_string, when};
use spark_connect::expression::{Expression, LiteralExpression};
use spark_connect::types::DataType;
use spark_connect_proto as proto;

/// An integer literal Column (pyspark infers 32-bit `integer` for small ints).
fn cint(v: i32) -> spark_connect::column::Column {
    spark_connect::column::Column::new(Expression::Literal(LiteralExpression::int(v)))
}

/// Recursively clear fields that vary run-to-run and are not client-authored:
/// `common` (holds Python origin) everywhere, and attribute/regex `plan_id`.
fn normalize(e: &mut proto::Expression) {
    use proto::expression::ExprType as T;
    e.common = None;
    if let Some(t) = e.expr_type.as_mut() {
        match t {
            T::UnresolvedAttribute(a) => a.plan_id = None,
            T::UnresolvedRegex(r) => r.plan_id = None,
            T::UnresolvedFunction(f) => {
                for a in f.arguments.iter_mut() {
                    normalize(a);
                }
            }
            T::Alias(a) => {
                if let Some(x) = a.expr.as_deref_mut() {
                    normalize(x);
                }
            }
            T::Cast(c) => {
                if let Some(x) = c.expr.as_deref_mut() {
                    normalize(x);
                }
            }
            T::SortOrder(s) => {
                if let Some(x) = s.child.as_deref_mut() {
                    normalize(x);
                }
            }
            T::UnresolvedExtractValue(v) => {
                if let Some(x) = v.child.as_deref_mut() {
                    normalize(x);
                }
                if let Some(x) = v.extraction.as_deref_mut() {
                    normalize(x);
                }
            }
            _ => {}
        }
    }
}

fn load_goldens() -> HashMap<String, proto::Expression> {
    // Goldens live at the repo root; tests run with CWD at the crate dir.
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/golden/exprs.jsonl"
    );
    let file = File::open(path).expect("golden file exprs.jsonl missing");
    let mut out = HashMap::new();
    for line in BufReader::new(file).lines() {
        let line = line.unwrap();
        if line.trim().is_empty() {
            continue;
        }
        let obj: serde_json::Value = serde_json::from_str(&line).unwrap();
        let name = obj["name"].as_str().unwrap().to_string();
        let b64 = obj["b64"].as_str().unwrap();
        let bytes = STANDARD.decode(b64).unwrap();
        let mut expr = proto::Expression::decode(&bytes[..]).unwrap();
        normalize(&mut expr);
        out.insert(name, expr);
    }
    out
}

/// Build the Rust expression for each golden case name we are responsible for.
/// (Pure `functions.*` cases like upper/concat/coalesce/sum_agg/count_star are
/// validated in the functions phase, not here.)
fn build(name: &str) -> Option<Expression> {
    let e = match name {
        "col" => col("x").expression().clone(),
        "lit_int" => Expression::Literal(LiteralExpression::int(5)),
        "lit_str" => Expression::Literal(LiteralExpression::string("hello")),
        "lit_double" => Expression::Literal(LiteralExpression::double(3.14)),
        "lit_bool" => Expression::Literal(LiteralExpression::boolean(true)),
        "lit_null" => Expression::Literal(LiteralExpression::Null(DataType::Null)),
        "add" => (col("id") + cint(1)).expression().clone(),
        "sub" => (col("id") - cint(1)).expression().clone(),
        "mul" => (col("id") * cint(2)).expression().clone(),
        "truediv" => (col("id") / cint(2)).expression().clone(),
        "mod" => (col("id") % cint(3)).expression().clone(),
        "eq" => col("id").eq(cint(5)).expression().clone(),
        "gt" => col("id").gt(cint(5)).expression().clone(),
        "and" => col("id")
            .gt(cint(1))
            .and(col("id").lt(cint(9)))
            .expression()
            .clone(),
        "or" => col("id")
            .gt(cint(1))
            .or(col("id").lt(cint(9)))
            .expression()
            .clone(),
        "not" => col("id").gt(cint(1)).not().expression().clone(),
        "alias" => col("id").alias("y").expression().clone(),
        "cast" => col("id").cast_str("string").expression().clone(),
        "isnull" => col("id").is_null().expression().clone(),
        "when" => when(col("id").gt(cint(1)), lit_string("a"))
            .when(col("id").gt(cint(2)), lit_string("b"))
            .otherwise(lit_string("c"))
            .expression()
            .clone(),
        "substr" => col("s").substr(cint(1), cint(3)).expression().clone(),
        "getitem" => col("m").get_item(lit_string("k")).expression().clone(),
        "getfield" => col("st").get_field("f").expression().clone(),
        _ => return None,
    };
    Some(e)
}

/// Cases this test is responsible for (must all be present & match).
const REQUIRED: &[&str] = &[
    "col",
    "lit_int",
    "lit_str",
    "lit_double",
    "lit_bool",
    "lit_null",
    "add",
    "sub",
    "mul",
    "truediv",
    "mod",
    "eq",
    "gt",
    "and",
    "or",
    "not",
    "alias",
    "cast",
    "isnull",
    "when",
    "substr",
    "getitem",
    "getfield",
];

#[test]
fn expressions_match_reference_golden() {
    let goldens = load_goldens();
    let mut failures: Vec<String> = Vec::new();

    for &name in REQUIRED {
        let expected = match goldens.get(name) {
            Some(e) => e.clone(),
            None => {
                failures.push(format!("{name}: MISSING from golden file"));
                continue;
            }
        };
        let expr = build(name).unwrap_or_else(|| panic!("no builder for required case {name}"));
        let mut actual = expr.to_proto();
        normalize(&mut actual);
        if actual != expected {
            failures.push(format!(
                "{name}: MISMATCH\n  expected: {expected:?}\n  actual:   {actual:?}"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} golden expression cases failed:\n{}",
        failures.len(),
        REQUIRED.len(),
        failures.join("\n")
    );
    println!("all {} golden expression cases passed", REQUIRED.len());
}
