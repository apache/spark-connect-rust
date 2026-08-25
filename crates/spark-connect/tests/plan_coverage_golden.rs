//! Coverage smoke test: build every `plan::*` builder and serialize it.
//!
//! Each builder constructs a `LogicalPlan` variant; `to_proto()` walks it into a
//! `spark.connect.Relation`. This exercises every builder body and its matching
//! `to_proto` arm without a live server. Parity of the resulting proto is covered
//! by `plan_arg_regression_golden.rs` and the live e2e tests; this only guards
//! that no builder/arm panics or is left untested.

use std::collections::HashMap;

use spark_connect::column::col;
use spark_connect::expression::Expression;
use spark_connect::plan::{self, AggregateGroupType, JoinType, LogicalPlan, SetOpType};
use spark_connect::row::Value;
use spark_connect::types::DataType;

fn base() -> LogicalPlan {
    plan::range(0, 10, 1)
}

fn expr() -> Expression {
    col("c").expression().clone()
}

fn ser(lp: LogicalPlan) {
    // Serializing forces the full to_proto walk of this plan (and its inputs).
    use prost::Message;
    let _ = lp.to_proto().encode_to_vec();
}

#[test]
fn every_plan_builder_serializes() {
    ser(plan::range(0, 10, 1));
    ser(plan::range_with_partitions(0, 10, 1, 4));
    ser(plan::sql("SELECT 1"));
    ser(plan::project(base(), vec![col("c")]));
    ser(plan::filter(base(), col("c")));
    ser(plan::aggregate(
        base(),
        AggregateGroupType::GroupBy,
        vec![expr()],
        vec![expr()],
    ));
    ser(plan::aggregate(
        base(),
        AggregateGroupType::Rollup,
        vec![expr()],
        vec![expr()],
    ));
    ser(plan::aggregate(
        base(),
        AggregateGroupType::Cube,
        vec![expr()],
        vec![expr()],
    ));
    ser(plan::aggregate_with_pivot(
        base(),
        AggregateGroupType::Pivot,
        vec![expr()],
        vec![expr()],
        expr(),
        vec![expr()],
    ));
    ser(plan::aggregate_with_grouping_sets(
        base(),
        vec![expr()],
        vec![expr()],
        vec![vec![expr()]],
    ));
    for jt in [
        JoinType::Inner,
        JoinType::LeftOuter,
        JoinType::RightOuter,
        JoinType::FullOuter,
        JoinType::LeftSemi,
        JoinType::LeftAnti,
        JoinType::Cross,
    ] {
        ser(plan::join(base(), base(), jt, Some(col("c")), vec![]));
    }
    ser(plan::join(
        base(),
        base(),
        JoinType::Inner,
        None,
        vec!["c".to_string()],
    ));
    for so in [SetOpType::Union, SetOpType::Intersect, SetOpType::Except] {
        ser(plan::set_operation(base(), base(), so, true, false, false));
        ser(plan::set_operation(base(), base(), so, false, true, true));
    }
    ser(plan::limit(base(), 5));
    ser(plan::offset(base(), 2));
    ser(plan::tail(base(), 3));
    ser(plan::deduplicate(base(), true, vec![], false));
    ser(plan::deduplicate(
        base(),
        false,
        vec!["c".to_string()],
        true,
    ));
    ser(plan::sort(base(), vec![expr()], true));
    ser(plan::sort(base(), vec![expr()], false));
    ser(plan::sample(base(), 0.0, 0.5, false, Some(42)));
    ser(plan::sample(base(), 0.0, 0.5, true, None));
    ser(plan::repartition(base(), 8, true));
    ser(plan::repartition(base(), 8, false));
    ser(plan::repartition_by_expression(base(), 8, vec![expr()]));
    ser(plan::with_columns(
        base(),
        vec!["c2".to_string()],
        vec![col("c")],
    ));
    let mut renames = HashMap::new();
    renames.insert("c".to_string(), "c2".to_string());
    ser(plan::with_columns_renamed(base(), renames));
    ser(plan::drop(base(), vec!["c".to_string()]));
    ser(plan::to_df(base(), vec!["a".to_string(), "b".to_string()]));
    ser(plan::to_schema(base(), DataType::Integer));
    ser(plan::hint(base(), "broadcast", vec!["c".to_string()]));
    ser(plan::unpivot(
        base(),
        vec![col("c")],
        Some(vec![col("c")]),
        "var",
        "val",
    ));
    ser(plan::unpivot(base(), vec![col("c")], None, "var", "val"));
    ser(plan::na_fill(
        base(),
        Value::Integer(0),
        vec!["c".to_string()],
    ));
    ser(plan::na_fill(
        base(),
        Value::Double(1.5),
        vec!["c".to_string()],
    ));
    ser(plan::na_fill(
        base(),
        Value::String("x".to_string()),
        vec![],
    ));
    ser(plan::na_drop(base(), "any", None, vec![]));
    ser(plan::na_drop(base(), "all", Some(1), vec!["c".to_string()]));
    ser(plan::na_replace(
        base(),
        vec![("a".to_string(), "b".to_string())],
        vec!["c".to_string()],
    ));
    ser(plan::describe(base(), vec!["c".to_string()]));
    ser(plan::summary(
        base(),
        vec!["25%".to_string(), "50%".to_string()],
    ));
    ser(plan::col_regex(base(), "c.*"));
    ser(plan::subquery_alias(base(), "t"));
    ser(plan::local_relation(DataType::Integer, None));
    ser(plan::local_relation(DataType::Integer, Some(vec![1, 2, 3])));
    ser(plan::cached_remote_relation("rel-123"));
}
