//! Golden parity test: window expressions built via the Rust Column/WindowSpec API
//! must serialize to the exact same protobuf the reference PySpark client produces.
//!
//! Goldens live in `tests/golden/window.jsonl` (captured by
//! `scripts/capture_window_golden.py`).

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

use base64::{engine::general_purpose::STANDARD, Engine};
use prost::Message;
use spark_connect::column::{col, lit};
use spark_connect::expression::{Expression, LiteralExpression};
use spark_connect::functions;
use spark_connect::window::{FrameBound, FrameType, Window, WindowSpec};
use spark_connect_proto as proto;

/// An integer literal Column (pyspark infers 32-bit `integer` for small ints).
fn cint(v: i32) -> spark_connect::column::Column {
    spark_connect::column::Column::new(Expression::Literal(LiteralExpression::int(v)))
}

/// Recursively clear fields that vary run-to-run and are not client-authored:
/// `common` (holds Python origin) everywhere, window-specific frame_spec handling.
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
            T::Window(w) => {
                if let Some(x) = w.window_function.as_deref_mut() {
                    normalize(x);
                }
                for x in w.partition_spec.iter_mut() {
                    normalize(x);
                }
                for s in w.order_spec.iter_mut() {
                    if let Some(x) = s.child.as_deref_mut() {
                        normalize(x);
                    }
                }
            }
            _ => {}
        }
    }
}

fn load_goldens() -> HashMap<String, proto::Expression> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/golden/window.jsonl"
    );
    let file = File::open(path).expect("golden file window.jsonl missing");
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

/// Build the Rust window expression for each golden case name.
fn build(name: &str) -> Option<Expression> {
    let e = match name {
        "window_row_number_partition" => functions::row_number()
            .over(WindowSpec::new().partition_by(vec![col("a").expression().clone()]))
            .expression()
            .clone(),

        "window_row_number_orderby" => functions::row_number()
            .over(WindowSpec::new().order_by(vec![
                spark_connect::expression::SortOrder::asc_nulls_first(
                    col("b").expression().clone(),
                ),
            ]))
            .expression()
            .clone(),

        "window_row_number_full" => functions::row_number()
            .over(
                WindowSpec::new()
                    .partition_by(vec![col("a").expression().clone()])
                    .order_by(vec![spark_connect::expression::SortOrder::asc_nulls_first(
                        col("b").expression().clone(),
                    )]),
            )
            .expression()
            .clone(),

        "window_rank" => functions::rank()
            .over(WindowSpec::new().order_by(vec![
                spark_connect::expression::SortOrder::asc_nulls_first(
                    col("b").expression().clone(),
                ),
            ]))
            .expression()
            .clone(),

        "window_dense_rank" => functions::dense_rank()
            .over(
                WindowSpec::new()
                    .partition_by(vec![col("a").expression().clone()])
                    .order_by(vec![spark_connect::expression::SortOrder::asc_nulls_first(
                        col("b").expression().clone(),
                    )]),
            )
            .expression()
            .clone(),

        "window_sum_basic" => functions::sum(col("x"))
            .over(WindowSpec::new().partition_by(vec![col("a").expression().clone()]))
            .expression()
            .clone(),

        "window_count_basic" => functions::count(col("x"))
            .over(WindowSpec::new().partition_by(vec![col("a").expression().clone()]))
            .expression()
            .clone(),

        "window_avg_basic" => functions::avg(col("x"))
            .over(WindowSpec::new().order_by(vec![
                spark_connect::expression::SortOrder::asc_nulls_first(
                    col("b").expression().clone(),
                ),
            ]))
            .expression()
            .clone(),

        "window_sum_rows_unbounded_to_current" => functions::sum(col("x"))
            .over(
                WindowSpec::new()
                    .partition_by(vec![col("a").expression().clone()])
                    .order_by(vec![spark_connect::expression::SortOrder::asc_nulls_first(
                        col("b").expression().clone(),
                    )])
                    .rows_between(FrameBound::UnboundedPreceding, FrameBound::CurrentRow),
            )
            .expression()
            .clone(),

        "window_sum_rows_unbounded_to_unbounded" => functions::sum(col("x"))
            .over(
                WindowSpec::new()
                    .partition_by(vec![col("a").expression().clone()])
                    .order_by(vec![spark_connect::expression::SortOrder::asc_nulls_first(
                        col("b").expression().clone(),
                    )])
                    .rows_between(
                        FrameBound::UnboundedPreceding,
                        FrameBound::UnboundedFollowing,
                    ),
            )
            .expression()
            .clone(),

        "window_sum_range_unbounded_to_current" => functions::sum(col("x"))
            .over(
                WindowSpec::new()
                    .partition_by(vec![col("a").expression().clone()])
                    .order_by(vec![spark_connect::expression::SortOrder::asc_nulls_first(
                        col("b").expression().clone(),
                    )])
                    .range_between(FrameBound::UnboundedPreceding, FrameBound::CurrentRow),
            )
            .expression()
            .clone(),

        "window_rank_desc" => functions::rank()
            .over(WindowSpec::new().order_by(vec![
                spark_connect::expression::SortOrder::desc_nulls_last(
                    col("b").expression().clone(),
                ),
            ]))
            .expression()
            .clone(),

        "window_row_number_multi_partition" => functions::row_number()
            .over(
                WindowSpec::new()
                    .partition_by(vec![
                        col("a").expression().clone(),
                        col("b").expression().clone(),
                    ])
                    .order_by(vec![spark_connect::expression::SortOrder::asc_nulls_first(
                        col("c").expression().clone(),
                    )]),
            )
            .expression()
            .clone(),

        "window_lag" => functions::lag(col("x"))
            .over(
                WindowSpec::new()
                    .partition_by(vec![col("a").expression().clone()])
                    .order_by(vec![spark_connect::expression::SortOrder::asc_nulls_first(
                        col("b").expression().clone(),
                    )]),
            )
            .expression()
            .clone(),

        "window_lead" => functions::lead(col("x"))
            .over(
                WindowSpec::new()
                    .partition_by(vec![col("a").expression().clone()])
                    .order_by(vec![spark_connect::expression::SortOrder::asc_nulls_first(
                        col("b").expression().clone(),
                    )]),
            )
            .expression()
            .clone(),

        _ => return None,
    };
    Some(e)
}

#[test]
fn test_window_golden() {
    let goldens = load_goldens();
    let mut matched = 0;

    for (name, expected) in &goldens {
        let built = build(name).expect(&format!("missing build case for {}", name));
        let mut actual = built.to_proto();
        normalize(&mut actual);

        if actual.encode_to_vec() != expected.encode_to_vec() {
            panic!(
                "window expression {} mismatch:\nexpected: {:?}\nactual:   {:?}",
                name, expected, actual
            );
        }
        matched += 1;
    }

    assert!(
        matched > 0,
        "no window golden cases matched - is window.jsonl empty?"
    );
    println!("✓ {} window expressions passed golden parity", matched);
}
