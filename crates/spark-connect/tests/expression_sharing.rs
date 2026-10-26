// Licensed to the Apache Software Foundation (ASF) under one or more
// contributor license agreements. See the NOTICE file distributed with
// this work for additional information regarding copyright ownership.
// The ASF licenses this file to You under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with
// the License. You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Server-free checks for immutable expression sharing and unchanged serialization.

use std::ptr;

use spark_connect::column::{col, lit, when, Column};
use spark_connect::expression::{
    Alias, CallFunctionWrapper, Expression, ExpressionNode, LiteralExpression, SortOrder,
};
use spark_connect::functions;
use spark_connect::types::DataType;
use spark_connect::udf::{eval_type, CommonInlineUserDefinedFunctionExpression, PythonUDFPayload};
use spark_connect::window::{FrameBound, WindowSpec};
use spark_connect_proto as proto;

fn assert_shared(left: &Expression, right: &Expression) {
    assert!(ptr::eq(left.node(), right.node()));
}

#[test]
fn clones_and_arithmetic_chains_share_existing_nodes() {
    let source = col("value");
    let cloned = source.clone();
    assert_shared(source.expression(), cloned.expression());
    assert_shared(source.expression(), &source.expression().clone());

    let mut columns = vec![source];
    for _ in 0..128 {
        columns.push(columns.last().unwrap().clone() + lit(1));
    }
    for pair in columns.windows(2) {
        let ExpressionNode::UnresolvedFunction(function) = pair[1].expression().node() else {
            panic!("expected addition");
        };
        assert_eq!(function.name, "+");
        assert_eq!(function.args.len(), 2);
        assert_shared(&function.args[0], pair[0].expression());
    }
}

#[test]
fn aliases_and_function_arguments_share_reused_subtrees() {
    let source = col("value") + lit(1);
    let before = source.to_proto();
    let left = source.clone().alias("left");
    let right = source.clone().alias("right");
    for alias in [&left, &right] {
        let ExpressionNode::Alias(alias) = alias.expression().node() else {
            panic!("expected alias");
        };
        assert_shared(&alias.child, source.expression());
    }
    let function = functions::call_function("greatest", vec![source.clone(), source.clone()]);
    let ExpressionNode::CallFunction(function) = function.expression().node() else {
        panic!("expected function call");
    };
    assert_eq!(function.arguments.len(), 2);
    for argument in &function.arguments {
        assert_shared(argument, source.expression());
    }
    assert_eq!(source.to_proto(), before);
    assert_eq!(left.expression().render(), "(value + 1) AS left");
    assert_eq!(right.expression().render(), "(value + 1) AS right");
}

#[test]
fn case_branches_share_children_without_mutating_the_original() {
    let condition = col("value").gt(lit(0));
    let value = col("value") + lit(1);
    let original = when(condition.clone(), value.clone());
    let before = original.to_proto();
    let extended = original.clone().when(col("value").lt(lit(0)), lit(-1));
    let completed = original.clone().otherwise(lit(0));
    let ExpressionNode::CaseWhen(original_case) = original.expression().node() else {
        panic!("expected CASE");
    };
    let ExpressionNode::CaseWhen(extended_case) = extended.expression().node() else {
        panic!("expected extended CASE");
    };
    let ExpressionNode::CaseWhen(completed_case) = completed.expression().node() else {
        panic!("expected completed CASE");
    };
    assert_eq!(original_case.branches.len(), 1);
    assert!(original_case.else_expr.is_none());
    assert_eq!(extended_case.branches.len(), 2);
    assert!(extended_case.else_expr.is_none());
    assert_eq!(completed_case.branches.len(), 1);
    assert!(completed_case.else_expr.is_some());
    for case in [original_case, extended_case, completed_case] {
        assert_shared(&case.branches[0].0, condition.expression());
        assert_shared(&case.branches[0].1, value.expression());
    }
    assert_eq!(original.to_proto(), before);
    assert_eq!(
        completed.expression().render(),
        "CASE WHEN (value > 0) THEN (value + 1) ELSE 0 END"
    );
}

#[test]
fn window_builders_share_expressions_without_mutating_the_specification() {
    let source = col("value") + lit(1);
    let original = WindowSpec::new()
        .partition_by(vec![source.expression().clone()])
        .order_by(vec![SortOrder::desc_nulls_last(
            source.expression().clone(),
        )]);
    let extended = original
        .clone()
        .partition_by(vec![col("other").expression().clone()])
        .rows_between(FrameBound::UnboundedPreceding, FrameBound::CurrentRow);
    assert_eq!(original.partition_spec.len(), 1);
    assert!(original.frame_spec.is_none());
    assert_eq!(extended.partition_spec.len(), 2);
    assert!(extended.frame_spec.is_some());
    assert_shared(&original.partition_spec[0], &extended.partition_spec[0]);
    assert_shared(&original.order_spec[0].child, &extended.order_spec[0].child);

    let windowed = source.clone().over(extended);
    let ExpressionNode::WindowExpression(window) = windowed.expression().node() else {
        panic!("expected window expression");
    };
    assert_shared(&window.window_function, source.expression());
    assert_shared(&window.partition_spec[0], source.expression());
    assert_shared(&window.order_spec[0].child, source.expression());
}

#[test]
fn binary_literal_payload_survives_without_being_cloned() {
    let original = Expression::new(ExpressionNode::Literal(LiteralExpression::Binary(vec![
        7;
        1024
    ])));
    let ExpressionNode::Literal(LiteralExpression::Binary(bytes)) = original.node() else {
        panic!("expected binary literal");
    };
    let address = bytes.as_ptr();
    let cloned = original.clone();
    assert_shared(&original, &cloned);
    drop(original);

    let ExpressionNode::Literal(LiteralExpression::Binary(bytes)) = cloned.node() else {
        panic!("expected binary literal");
    };
    assert_eq!(bytes.as_ptr(), address);
    assert_eq!(bytes.as_slice(), &[7; 1024]);
    assert_eq!(
        cloned.to_proto(),
        LiteralExpression::Binary(vec![7; 1024]).to_proto()
    );
}

#[test]
fn udf_clones_share_command_bytes_and_argument_lifetimes() {
    let argument = col("value") + lit(1);
    let argument_node = argument.expression().node() as *const ExpressionNode;
    let payload = PythonUDFPayload::new(
        DataType::Long,
        eval_type::SQL_BATCHED_UDF,
        vec![9; 1024],
        "3.11".to_string(),
    );
    let command_address = payload.command.as_ptr();
    let udf = CommonInlineUserDefinedFunctionExpression::new(
        "test_udf".to_string(),
        true,
        vec![argument.expression().clone()],
        payload,
    );
    let original = Expression::new(ExpressionNode::CommonInlineUserDefinedFunction(Box::new(
        udf,
    )));
    let cloned = original.clone();
    assert_shared(&original, &cloned);
    drop(original);
    drop(argument);

    let ExpressionNode::CommonInlineUserDefinedFunction(udf) = cloned.node() else {
        panic!("expected UDF");
    };
    assert_eq!(udf.python_udf.command.as_ptr(), command_address);
    assert_eq!(udf.python_udf.command.as_slice(), &[9; 1024]);
    assert!(ptr::eq(udf.arguments[0].node(), argument_node));
    assert_eq!(udf.arguments[0].render(), "(value + 1)");
    let Some(proto::expression::ExprType::CommonInlineUserDefinedFunction(encoded)) =
        cloned.to_proto().expr_type
    else {
        panic!("expected encoded UDF");
    };
    let Some(proto::common_inline_user_defined_function::Function::PythonUdf(payload)) =
        encoded.function
    else {
        panic!("expected encoded Python UDF");
    };
    assert_eq!(payload.command.as_ref(), &[9; 1024]);
    assert_eq!(payload.eval_type, eval_type::SQL_BATCHED_UDF);
}

#[test]
fn equality_remains_structural_and_does_not_shortcut_nan() {
    let left = (col("value") + lit(1)).expression().clone();
    let right = (col("value") + lit(1)).expression().clone();
    assert!(!ptr::eq(left.node(), right.node()));
    assert_eq!(left, right);
    assert_ne!(left, (col("value") + lit(2)).expression().clone());

    for literal in [
        LiteralExpression::Float(f32::NAN),
        LiteralExpression::Double(f64::NAN),
    ] {
        let expression: Expression = ExpressionNode::Literal(literal).into();
        let cloned = expression.clone();
        assert_shared(&expression, &cloned);
        assert_ne!(expression, cloned);
        let nested = Column::new(expression) + lit(1);
        assert_ne!(nested.expression(), nested.clone().expression());
    }
}

#[test]
fn debug_render_and_proto_keep_the_existing_representation() {
    let argument: Expression = ExpressionNode::Literal(LiteralExpression::Integer(7)).into();
    let wrapper = CallFunctionWrapper::new("f", vec![argument]);
    let expected_proto = wrapper.to_proto();
    let call = Expression::new(ExpressionNode::CallFunction(Box::new(wrapper)));
    let expected_debug = concat!(
        "CallFunction(CallFunctionWrapper { function_name: \"f\", ",
        "arguments: [Literal(Integer(7))] })"
    );
    assert_eq!(format!("{call:?}"), expected_debug);
    assert_eq!(call.render(), expected_debug);
    assert_eq!(call.to_proto(), expected_proto);

    let source = col("value") + lit(1);
    let alias = source.clone().alias("total");
    assert_eq!(alias.expression().render(), "(value + 1) AS total");
    assert_eq!(
        alias.to_proto(),
        Alias::new(source.expression().clone(), "total").to_proto()
    );
}

#[test]
fn shared_occurrences_expand_into_independent_proto_trees() {
    let source = col("value") + lit(1);
    let combined = source.clone() + source.clone();
    let ExpressionNode::UnresolvedFunction(function) = combined.expression().node() else {
        panic!("expected addition");
    };
    assert_shared(&function.args[0], &function.args[1]);
    let expected = ((col("value") + lit(1)) + (col("value") + lit(1))).to_proto();
    let mut encoded = combined.to_proto();
    assert_eq!(encoded, expected);
    let Some(proto::expression::ExprType::UnresolvedFunction(function)) = &mut encoded.expr_type
    else {
        panic!("expected encoded addition");
    };
    assert_eq!(function.arguments.len(), 2);
    assert_eq!(function.arguments[0], source.to_proto());
    assert_eq!(function.arguments[1], source.to_proto());
    function.arguments[0].expr_type = None;
    assert_eq!(function.arguments[1], source.to_proto());
    assert_eq!(combined.to_proto(), expected);
}
