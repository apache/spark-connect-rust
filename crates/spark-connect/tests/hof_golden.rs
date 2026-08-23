//! Golden parity test for higher-order (lambda) functions.
//!
//! Tests that all lambda-based functions serialize to exactly the same protobuf
//! as the reference PySpark client. Lambda variable names are non-deterministic
//! across runs (depend on counter state), so we canonicalize them on both sides
//! before comparing: rename every distinct UnresolvedNamedLambdaVariable name_parts
//! to v0, v1, ... in first-appearance order.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

use base64::{engine::general_purpose::STANDARD, Engine};
use prost::Message;
use spark_connect::column::col;
use spark_connect::expression::Expression;
use spark_connect::functions::{
    aggregate, aggregate_with_finish, exists, filter, forall, map_filter, map_zip_with, transform,
    transform_idx, transform_keys, transform_values, zip_with,
};
use spark_connect_proto as proto;

/// An integer literal Column.
fn cint(v: i32) -> spark_connect::column::Column {
    spark_connect::column::Column::new(Expression::Literal(
        spark_connect::expression::LiteralExpression::int(v),
    ))
}

/// Recursively canonicalize lambda variable names in an expression.
/// On first appearance of each unique name_parts, assign it a canonical name v0, v1, etc.
fn canonicalize_lambda_vars(
    e: &mut proto::Expression,
    name_map: &mut HashMap<String, String>,
    counter: &mut usize,
) {
    use proto::expression::ExprType as T;

    // Clear non-deterministic fields
    e.common = None;

    if let Some(t) = e.expr_type.as_mut() {
        match t {
            T::UnresolvedAttribute(a) => a.plan_id = None,
            T::UnresolvedRegex(r) => r.plan_id = None,
            T::UnresolvedFunction(f) => {
                for a in f.arguments.iter_mut() {
                    canonicalize_lambda_vars(a, name_map, counter);
                }
            }
            T::Alias(a) => {
                if let Some(x) = a.expr.as_deref_mut() {
                    canonicalize_lambda_vars(x, name_map, counter);
                }
            }
            T::Cast(c) => {
                if let Some(x) = c.expr.as_deref_mut() {
                    canonicalize_lambda_vars(x, name_map, counter);
                }
            }
            T::SortOrder(s) => {
                if let Some(x) = s.child.as_deref_mut() {
                    canonicalize_lambda_vars(x, name_map, counter);
                }
            }
            T::UnresolvedExtractValue(v) => {
                if let Some(x) = v.child.as_deref_mut() {
                    canonicalize_lambda_vars(x, name_map, counter);
                }
                if let Some(x) = v.extraction.as_deref_mut() {
                    canonicalize_lambda_vars(x, name_map, counter);
                }
            }
            T::LambdaFunction(lf) => {
                if let Some(func) = lf.function.as_deref_mut() {
                    canonicalize_lambda_vars(func, name_map, counter);
                }
                // Rename lambda variable arguments
                for arg in lf.arguments.iter_mut() {
                    for name_part in arg.name_parts.iter_mut() {
                        let canonical = name_map
                            .entry(name_part.clone())
                            .or_insert_with(|| {
                                let c = *counter;
                                *counter += 1;
                                format!("v{}", c)
                            })
                            .clone();
                        *name_part = canonical;
                    }
                }
            }
            T::UnresolvedNamedLambdaVariable(var) => {
                for name_part in var.name_parts.iter_mut() {
                    let canonical = name_map
                        .entry(name_part.clone())
                        .or_insert_with(|| {
                            let c = *counter;
                            *counter += 1;
                            format!("v{}", c)
                        })
                        .clone();
                    *name_part = canonical;
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
        "/../../tests/golden/functions_hof.jsonl"
    );
    let file = File::open(path).expect("golden file functions_hof.jsonl missing");
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
        // Canonicalize reference golden
        let mut name_map = HashMap::new();
        let mut counter = 0;
        canonicalize_lambda_vars(&mut expr, &mut name_map, &mut counter);
        out.insert(name, expr);
    }
    out
}

fn build(name: &str) -> Option<Expression> {
    let e = match name {
        "transform" => transform(col("a"), |args| args[0].clone() + cint(1))
            .expression()
            .clone(),
        "transform_idx" => transform_idx(col("a"), |args| args[0].clone() + args[1].clone())
            .expression()
            .clone(),
        "filter" => filter(col("a"), |args| args[0].clone().gt(cint(0)))
            .expression()
            .clone(),
        "exists" => exists(col("a"), |args| args[0].clone().gt(cint(0)))
            .expression()
            .clone(),
        "forall" => forall(col("a"), |args| args[0].clone().gt(cint(0)))
            .expression()
            .clone(),
        "aggregate" => aggregate(col("a"), cint(0), |args| args[0].clone() + args[1].clone())
            .expression()
            .clone(),
        "aggregate_finish" => aggregate_with_finish(
            col("a"),
            cint(0),
            |args| args[0].clone() + args[1].clone(),
            |args| args[0].clone() * cint(2),
        )
        .expression()
        .clone(),
        "zip_with" => zip_with(col("a"), col("b"), |args| args[0].clone() + args[1].clone())
            .expression()
            .clone(),
        "transform_keys" => transform_keys(col("m"), |args| args[0].clone())
            .expression()
            .clone(),
        "transform_values" => transform_values(col("m"), |args| args[1].clone() + cint(1))
            .expression()
            .clone(),
        "map_filter" => map_filter(col("m"), |args| args[1].clone().gt(cint(0)))
            .expression()
            .clone(),
        "map_zip_with" => map_zip_with(col("m1"), col("m2"), |args| {
            args[1].clone() + args[2].clone()
        })
        .expression()
        .clone(),
        _ => return None,
    };
    Some(e)
}

const REQUIRED: &[&str] = &[
    "transform",
    "transform_idx",
    "filter",
    "exists",
    "forall",
    "aggregate",
    "aggregate_finish",
    "zip_with",
    "transform_keys",
    "transform_values",
    "map_filter",
    "map_zip_with",
];

#[test]
fn hof_expressions_match_reference_golden() {
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
        // Canonicalize actual expression
        let mut name_map = HashMap::new();
        let mut counter = 0;
        canonicalize_lambda_vars(&mut actual, &mut name_map, &mut counter);

        if actual != expected {
            failures.push(format!(
                "{name}: MISMATCH\n  expected: {expected:?}\n  actual:   {actual:?}"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} golden HOF expression cases failed:\n{}",
        failures.len(),
        REQUIRED.len(),
        failures.join("\n")
    );
    println!("all {} golden HOF expression cases passed", REQUIRED.len());
}
