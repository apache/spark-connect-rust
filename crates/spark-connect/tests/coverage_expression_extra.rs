//! Pure-logic coverage of `expression.rs`: exercises the `to_proto` arms and builder
//! methods that the behavioral (server-gated) column tests do not reach in a server-less
//! coverage run — every `LiteralExpression` variant, the inline-UDF expression arm,
//! `ColumnReference` plan-id/metadata, `Alias` metadata, `Cast` eval modes, `CaseWhen`
//! else, and window frame boundaries. No server required.

use spark_connect::expression::{
    Alias, CaseWhen, Cast, CastEvalMode, ColumnReference, Expression, FrameBoundary,
    LiteralExpression, SortOrder, UnresolvedFunction, WindowExpressionWrapper,
};
use spark_connect::types::DataType;
use spark_connect::udf::{eval_type, CommonInlineUserDefinedFunctionExpression, PythonUDFPayload};

fn col_expr(name: &str) -> Expression {
    Expression::ColumnReference(ColumnReference::new(name))
}

#[test]
fn every_literal_variant_to_proto() {
    let lits = vec![
        LiteralExpression::null(DataType::Integer),
        LiteralExpression::boolean(true),
        LiteralExpression::Byte(1),
        LiteralExpression::Short(2),
        LiteralExpression::int(3),
        LiteralExpression::long(4),
        LiteralExpression::Float(1.5),
        LiteralExpression::double(2.5),
        LiteralExpression::Decimal {
            value: "12.34".to_string(),
            precision: 4,
            scale: 2,
        },
        LiteralExpression::string("hi"),
        LiteralExpression::binary(vec![1, 2, 3]),
        LiteralExpression::Date(19000),
        LiteralExpression::Timestamp(1_693_526_400_000_000),
        LiteralExpression::TimestampNtz(1_693_526_400_000_000),
        LiteralExpression::Time {
            nano: 123,
            precision: 6,
        },
        LiteralExpression::Array {
            element_type: Box::new(DataType::Integer),
            elements: vec![LiteralExpression::int(1), LiteralExpression::int(2)],
        },
    ];
    for lit in lits {
        assert!(lit.to_proto().expr_type.is_some());
    }
}

#[test]
fn inline_udf_expression_to_proto() {
    let payload = PythonUDFPayload::new(
        DataType::Integer,
        eval_type::SQL_BATCHED_UDF,
        vec![1, 2, 3],
        "3.11".to_string(),
    );
    let udf = CommonInlineUserDefinedFunctionExpression::new(
        "my_udf".to_string(),
        true,
        vec![col_expr("x")],
        payload,
    );
    let expr = Expression::CommonInlineUserDefinedFunction(Box::new(udf));
    assert!(expr.to_proto().expr_type.is_some());
}

#[test]
fn column_reference_plan_id_and_metadata() {
    let bound = ColumnReference::new("a").with_plan_id(7);
    assert_eq!(bound.plan_id, Some(7));
    assert!(bound.to_proto().expr_type.is_some());

    let meta = ColumnReference::new("b").metadata();
    assert!(meta.is_metadata_column);
    assert!(meta.to_proto().expr_type.is_some());
}

#[test]
fn alias_with_metadata_to_proto() {
    let alias = Alias::new(col_expr("x"), "y").with_metadata("{\"comment\":\"c\"}".to_string());
    assert_eq!(alias.metadata, Some("{\"comment\":\"c\"}".to_string()));
    assert!(alias.to_proto().expr_type.is_some());
}

#[test]
fn cast_eval_modes_and_type_str() {
    for mode in [CastEvalMode::Legacy, CastEvalMode::Ansi, CastEvalMode::Try] {
        let cast = Cast::new(col_expr("x"), DataType::Integer).with_eval_mode(mode);
        assert!(cast.to_proto().expr_type.is_some());
    }
    let cast_str = Cast::new_str(col_expr("x"), "string");
    assert!(cast_str.to_proto().expr_type.is_some());
}

#[test]
fn case_when_with_else_to_proto() {
    let cw = CaseWhen::new(vec![(
        col_expr("cond"),
        Expression::Literal(LiteralExpression::int(1)),
    )])
    .with_else(Expression::Literal(LiteralExpression::int(0)));
    assert!(cw.else_expr.is_some());
    assert!(cw.to_proto().expr_type.is_some());
}

#[test]
fn window_frame_boundaries_to_proto() {
    let func = Expression::UnresolvedFunction(UnresolvedFunction::new("row_number", vec![]));
    let order = vec![SortOrder::asc_nulls_first(col_expr("x"))];

    // Preceding / Following value boundaries.
    let w1 = WindowExpressionWrapper::new(
        func.clone(),
        vec![col_expr("g")],
        order.clone(),
        Some((1, FrameBoundary::Preceding(1), FrameBoundary::Following(2))),
    );
    assert!(w1.to_proto().expr_type.is_some());

    // Unbounded boundaries.
    let w2 = WindowExpressionWrapper::new(
        func.clone(),
        vec![],
        order.clone(),
        Some((
            2,
            FrameBoundary::UnboundedPreceding,
            FrameBoundary::UnboundedFollowing,
        )),
    );
    assert!(w2.to_proto().expr_type.is_some());

    // Current-row boundary and no frame.
    let w3 = WindowExpressionWrapper::new(
        func.clone(),
        vec![],
        order,
        Some((1, FrameBoundary::CurrentRow, FrameBoundary::CurrentRow)),
    );
    assert!(w3.to_proto().expr_type.is_some());

    let w4 = WindowExpressionWrapper::new(func, vec![], vec![], None);
    assert!(w4.to_proto().expr_type.is_some());
}
