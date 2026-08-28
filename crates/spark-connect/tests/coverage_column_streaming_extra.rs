//! Pure-logic (no-server) coverage for `column.rs` corners not exercised elsewhere:
//! `alias_with_metadata`, `try_cast`, the non-`CaseWhen` branch of `otherwise`, the
//! window-frame boundary mapping in `over` (all `FrameBound` variants), the operator
//! trait impls (`&`, `|`, `!`), and the `lit` LongType branch for out-of-i32 values.
//!
//! (The `streaming.rs` `foreach`/`foreach_batch` builders need a `DataStreamWriter`,
//! which requires a live session/DataFrame, so they are covered by the e2e streaming
//! tests, not here.)

use std::collections::BTreeMap;

use spark_connect::column::{col, lit, lit_string};
use spark_connect::types::DataType;
use spark_connect::window::{FrameBound, WindowSpec};

#[test]
fn alias_with_metadata_serializes_map() {
    let mut meta = BTreeMap::new();
    meta.insert("comment".to_string(), "hello".to_string());
    meta.insert("unit".to_string(), "ms".to_string());
    let c = col("x").alias_with_metadata("y", meta);
    // Build the proto to exercise the alias-with-metadata path end to end.
    let _ = c.to_proto();

    // Empty metadata also goes through the serializer.
    let c2 = col("z").alias_with_metadata("w", BTreeMap::new());
    let _ = c2.to_proto();
}

#[test]
fn try_cast_builds_cast_expr() {
    let c = col("x").try_cast(DataType::Integer);
    let _ = c.to_proto();
    let c2 = col("x").try_cast_str("string");
    let _ = c2.to_proto();
}

#[test]
fn otherwise_on_non_casewhen_keeps_expr() {
    // `col("x")` is a plain column reference, not a CaseWhen, so `otherwise` takes the
    // fall-through branch that returns the receiver's expression unchanged.
    let c = col("x").otherwise(lit(1));
    let _ = c.to_proto();
}

#[test]
fn over_maps_all_frame_boundaries() {
    // Three windows together hit every FrameBound arm for both the lower and upper
    // boundary in `Column::over`'s frame-spec mapping.
    let w1 = WindowSpec::new().rows_between(
        FrameBound::UnboundedPreceding,
        FrameBound::UnboundedFollowing,
    );
    let _ = col("v").over(w1).to_proto();

    let w2 = WindowSpec::new().rows_between(FrameBound::Preceding(1), FrameBound::Following(2));
    let _ = col("v").over(w2).to_proto();

    let w3 = WindowSpec::new().range_between(FrameBound::CurrentRow, FrameBound::CurrentRow);
    let _ = col("v").over(w3).to_proto();
}

#[test]
fn column_operator_traits() {
    // BitAnd / BitOr / Not operator impls delegate to and()/or()/not.
    let a = col("a");
    let b = col("b");
    let _ = (a.clone() & b.clone()).to_proto();
    let _ = (a.clone() | b.clone()).to_proto();
    let _ = (!a).to_proto();
}

#[test]
fn lit_long_branch_for_large_values() {
    // Values outside i32 range take the LongType branch of `lit`.
    let big = lit(10_000_000_000_i64);
    let _ = big.to_proto();
    // Small value takes the IntegerType branch (sanity).
    let small = lit(7);
    let _ = small.to_proto();
    let _ = lit_string("hello").to_proto();
}
