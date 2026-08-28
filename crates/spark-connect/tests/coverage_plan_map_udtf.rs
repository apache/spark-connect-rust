//! Golden coverage for the `LogicalPlan::to_proto()` map/UDTF arms that
//! `plans_golden.rs` and `coverage_plan_extra.rs` do not build: `MapPartitions`,
//! `GroupMap` (both the plain and the fully-stateful shapes), `CoGroupMap`,
//! `NearestByJoin`, and `CommonInlineUdtf` (both with and without a return type).
//! No server: each test constructs the `LogicalPlan` variant and calls `to_proto()`
//! so the proto-building lines run. Assertions are intentionally light.

use spark_connect::expression::{ColumnReference, Expression};
use spark_connect::plan::{self, LogicalPlan, TransformWithStateInfo};
use spark_connect::types::DataType;
use spark_connect::udf::{eval_type, CommonInlineUserDefinedFunctionExpression, PythonUDFPayload};

fn col_expr(name: &str) -> Expression {
    Expression::ColumnReference(ColumnReference::new(name))
}

fn base() -> LogicalPlan {
    plan::range(0, 10, 1)
}

fn udf() -> CommonInlineUserDefinedFunctionExpression {
    let payload = PythonUDFPayload::new(
        DataType::Integer,
        eval_type::SQL_BATCHED_UDF,
        vec![1, 2, 3],
        "3.11".to_string(),
    );
    CommonInlineUserDefinedFunctionExpression::new(
        "f".to_string(),
        true,
        vec![col_expr("x")],
        payload,
    )
}

#[test]
fn map_partitions_to_proto() {
    let p = LogicalPlan::MapPartitions {
        input: Box::new(base()),
        func: udf(),
        is_barrier: true,
    };
    assert!(p.to_proto().rel_type.is_some());
}

#[test]
fn group_map_to_proto_plain_and_stateful() {
    // Plain applyInPandas shape: every optional/stateful field empty (the `None` branches).
    let plain = LogicalPlan::GroupMap {
        input: Box::new(base()),
        grouping_expressions: vec![col_expr("g")],
        func: udf(),
        sorting_expressions: vec![],
        initial_input: None,
        initial_grouping_expressions: vec![],
        is_map_groups_with_state: None,
        output_mode: None,
        timeout_conf: None,
        state_schema: None,
        transform_with_state_info: None,
    };
    assert!(plain.to_proto().rel_type.is_some());

    // Fully-stateful shape: exercise the `Some(...)` branches for initial_input,
    // state_schema, and transform_with_state_info.
    let stateful = LogicalPlan::GroupMap {
        input: Box::new(base()),
        grouping_expressions: vec![col_expr("g")],
        func: udf(),
        sorting_expressions: vec![col_expr("s")],
        initial_input: Some(Box::new(base())),
        initial_grouping_expressions: vec![col_expr("ig")],
        is_map_groups_with_state: Some(true),
        output_mode: Some("append".to_string()),
        timeout_conf: Some("NoTimeout".to_string()),
        state_schema: Some(DataType::Struct { fields: vec![] }),
        transform_with_state_info: Some(TransformWithStateInfo {
            time_mode: "None".to_string(),
            event_time_column_name: Some("ts".to_string()),
            output_schema: Some(DataType::Struct { fields: vec![] }),
        }),
    };
    assert!(stateful.to_proto().rel_type.is_some());
}

#[test]
fn co_group_map_to_proto() {
    let p = LogicalPlan::CoGroupMap {
        input: Box::new(base()),
        input_grouping_expressions: vec![col_expr("a")],
        other: Box::new(base()),
        other_grouping_expressions: vec![col_expr("b")],
        func: udf(),
    };
    assert!(p.to_proto().rel_type.is_some());
}

#[test]
fn nearest_by_join_to_proto() {
    let p = LogicalPlan::NearestByJoin {
        left: Box::new(base()),
        right: Box::new(base()),
        ranking_expression: col_expr("dist"),
        num_results: 5,
        join_type: "inner".to_string(),
        mode: "brute_force".to_string(),
        direction: "ascending".to_string(),
    };
    assert!(p.to_proto().rel_type.is_some());
}

#[test]
fn common_inline_udtf_to_proto_with_and_without_return_type() {
    let with_rt = LogicalPlan::CommonInlineUdtf {
        function_name: "my_udtf".to_string(),
        deterministic: true,
        arguments: vec![col_expr("x")],
        return_type: Some(DataType::Struct { fields: vec![] }),
        eval_type: 200,
        command: vec![1, 2, 3],
        python_ver: "3.11".to_string(),
    };
    assert!(with_rt.to_proto().rel_type.is_some());

    let no_rt = LogicalPlan::CommonInlineUdtf {
        function_name: "u2".to_string(),
        deterministic: false,
        arguments: vec![],
        return_type: None,
        eval_type: 201,
        command: vec![],
        python_ver: "3.12".to_string(),
    };
    assert!(no_rt.to_proto().rel_type.is_some());
}
