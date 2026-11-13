//! Golden coverage for the `LogicalPlan::to_proto()` map/UDTF arms that
//! `plans_golden.rs` and `coverage_plan_extra.rs` do not build: `MapPartitions`,
//! `GroupMap` (both the plain and the fully-stateful shapes), `CoGroupMap`,
//! `NearestByJoin`, and `CommonInlineUdtf` (both with and without a return type).
//! No server: each test constructs the plan kind and inputs and calls `to_proto()`
//! so the proto-building lines run. Assertions are intentionally light.

use std::sync::Arc;

use spark_connect::expression::Expression;
use spark_connect::plan::{self, LogicalPlan, LogicalPlanKind, TransformWithStateInfo};
use spark_connect::types::DataType;
use spark_connect::udf::{eval_type, CommonInlineUserDefinedFunctionExpression, PythonUDFPayload};

fn col_expr(name: &str) -> Expression {
    Expression::column(name)
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
    let p = LogicalPlan::try_new(
        LogicalPlanKind::MapPartitions {
            func: udf(),
            is_barrier: true,
            profile_id: None,
        },
        vec![Arc::new(base())],
    )
    .unwrap();
    assert!(p.to_proto().rel_type.is_some());
}

#[test]
fn group_map_to_proto_plain_and_stateful() {
    // Plain applyInPandas shape: every optional/stateful field empty (the `None` branches).
    let plain = LogicalPlan::try_new(
        LogicalPlanKind::GroupMap {
            grouping_expressions: vec![col_expr("g")],
            func: udf(),
            sorting_expressions: vec![],
            initial_grouping_expressions: vec![],
            is_map_groups_with_state: None,
            output_mode: None,
            timeout_conf: None,
            state_schema: None,
            transform_with_state_info: None,
        },
        vec![Arc::new(base())],
    )
    .unwrap();
    assert!(plain.to_proto().rel_type.is_some());

    // Include the initial state as a second child and exercise the `Some(...)`
    // branches for state_schema and transform_with_state_info.
    let stateful = LogicalPlan::try_new(
        LogicalPlanKind::GroupMap {
            grouping_expressions: vec![col_expr("g")],
            func: udf(),
            sorting_expressions: vec![col_expr("s")],
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
        },
        vec![Arc::new(base()), Arc::new(base())],
    )
    .unwrap();
    assert!(stateful.to_proto().rel_type.is_some());
}

#[test]
fn co_group_map_to_proto() {
    let p = LogicalPlan::try_new(
        LogicalPlanKind::CoGroupMap {
            input_grouping_expressions: vec![col_expr("a")],
            other_grouping_expressions: vec![col_expr("b")],
            func: udf(),
        },
        vec![Arc::new(base()), Arc::new(base())],
    )
    .unwrap();
    assert!(p.to_proto().rel_type.is_some());
}

#[test]
fn nearest_by_join_to_proto() {
    let p = LogicalPlan::try_new(
        LogicalPlanKind::NearestByJoin {
            ranking_expression: col_expr("dist"),
            num_results: 5,
            join_type: "inner".to_string(),
            mode: "brute_force".to_string(),
            direction: "ascending".to_string(),
        },
        vec![Arc::new(base()), Arc::new(base())],
    )
    .unwrap();
    assert!(p.to_proto().rel_type.is_some());
}

#[test]
fn common_inline_udtf_to_proto_with_and_without_return_type() {
    let with_rt = LogicalPlan::try_new(
        LogicalPlanKind::CommonInlineUdtf {
            function_name: "my_udtf".to_string(),
            deterministic: true,
            arguments: vec![col_expr("x")],
            return_type: Some(DataType::Struct { fields: vec![] }),
            eval_type: 200,
            command: vec![1, 2, 3],
            python_ver: "3.11".to_string(),
        },
        vec![],
    )
    .unwrap();
    assert!(with_rt.to_proto().rel_type.is_some());

    let no_rt = LogicalPlan::try_new(
        LogicalPlanKind::CommonInlineUdtf {
            function_name: "u2".to_string(),
            deterministic: false,
            arguments: vec![],
            return_type: None,
            eval_type: 201,
            command: vec![],
            python_ver: "3.12".to_string(),
        },
        vec![],
    )
    .unwrap();
    assert!(no_rt.to_proto().rel_type.is_some());
}

#[test]
fn common_inline_udtf_preserves_return_schema() {
    use spark_connect_proto::{common_inline_user_defined_table_function, relation};

    // DDL must survive unchanged, including malformed/scalar DDL: Connect
    // delegates parsing and validation to the server. Struct schemas stay typed.
    for return_type in [
        Some(DataType::Unparsed {
            data_type_string: "value ARRAY<STRUCT<x: INT>>".to_string(),
        }),
        Some(DataType::Unparsed {
            data_type_string: "invalid DDL !!!".to_string(),
        }),
        Some(DataType::Unparsed {
            data_type_string: "INT".to_string(),
        }),
        Some(DataType::from_ddl("value BIGINT").unwrap()),
        None,
    ] {
        let expected = return_type.as_ref().map(DataType::to_proto);
        let plan = LogicalPlan::try_new(
            LogicalPlanKind::CommonInlineUdtf {
                function_name: "echo".to_string(),
                deterministic: false,
                arguments: vec![col_expr("x")],
                return_type,
                eval_type: 300,
                command: vec![1, 2, 3],
                python_ver: "3.12".to_string(),
            },
            vec![],
        )
        .unwrap();
        let Some(relation::RelType::CommonInlineUserDefinedTableFunction(call)) =
            plan.to_proto().rel_type
        else {
            panic!("expected an inline UDTF relation");
        };
        let Some(common_inline_user_defined_table_function::Function::PythonUdtf(function)) =
            call.function
        else {
            panic!("expected a Python UDTF");
        };
        assert_eq!(function.return_type, expected);
        assert_eq!(function.eval_type, 300);
        assert_eq!(function.command.as_ref(), &[1, 2, 3]);
    }
}
