//! Normalization fixtures cover both expression enums exhaustively. Expected
//! inputs and wire results come from the typed builders, before normalization.

use super::*;
use crate::plan;
use crate::udf::PythonUDFPayload;
use std::ops::RangeInclusive;

// Check the 112/160-byte layout budgets when tests are built for 64-bit
// x86-64/AArch64 targets.
// Inline DataType payloads in LiteralExpression::Null and CastTarget::Type
// dominate the kind; children, declared type and origin bring the node to 160 B.
// These are upper bounds for these 64-bit targets, not a stable Rust ABI promise
// or an assumed budget for 32-bit targets.
#[cfg(all(
    target_pointer_width = "64",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
const _: () = {
    assert!(std::mem::size_of::<ExpressionKind>() <= 112);
    assert!(std::mem::size_of::<ExpressionNode>() <= 160);
};

#[test]
fn variadic_arguments_normalize_storage_by_length_and_preserve_handles() {
    let constructors: [fn(Vec<Expression>) -> ExpressionSpec; 3] = [
        |arguments| ExpressionSpec::UnresolvedFunction(UnresolvedFunction::new("f", arguments)),
        |arguments| {
            ExpressionSpec::CallFunction(Box::new(CallFunctionWrapper::new("f", arguments)))
        },
        |arguments| {
            ExpressionSpec::CommonInlineUserDefinedFunction(Box::new(
                CommonInlineUserDefinedFunctionExpression::new(
                    "f".into(),
                    false,
                    arguments,
                    PythonUDFPayload::new(DataType::Long, 100, vec![], "3.11".into()),
                ),
            ))
        },
    ];

    for (constructor_index, constructor) in constructors.into_iter().enumerate() {
        for count in 0..=4 {
            let expected: Vec<_> = (0..count)
                .map(|index| Expression::column(format!("argument_{index}")))
                .collect();
            let mut arguments = Vec::with_capacity(8);
            arguments.extend(expected.iter().cloned());
            let original_buffer = arguments.as_ptr();
            let expression = Expression::new(constructor(arguments));
            let children = &expression.node.children;

            assert_eq!(children.len(), count);
            assert_eq!(
                children.spilled(),
                count > children.inline_size(),
                "constructor {constructor_index}, argument count {count}"
            );
            for (actual, expected) in children.iter().zip(&expected) {
                assert!(actual.ptr_eq(expected));
            }
            if count > children.inline_size() {
                assert_eq!(children.as_ptr(), original_buffer);
            }
        }
    }
}

struct Fixture {
    spec: ExpressionSpec,
    inputs: Vec<Expression>,
    wire: proto::Expression,
    rendered: String,
    arity: RangeInclusive<usize>,
}

impl Fixture {
    fn new(spec: ExpressionSpec, inputs: Vec<Expression>, rendered: impl Into<String>) -> Self {
        let wire = builder_wire(&spec);
        let count = inputs.len();
        Self {
            spec,
            inputs,
            wire,
            rendered: rendered.into(),
            arity: count..=count,
        }
    }

    fn variadic(mut self) -> Self {
        self.arity = 0..=usize::MAX;
        self
    }
}

fn sql_wire(sql: &str) -> proto::Expression {
    proto::Expression {
        expr_type: Some(proto::expression::ExprType::ExpressionString(
            proto::expression::ExpressionString {
                expression: sql.into(),
            },
        )),
        ..Default::default()
    }
}

// These are the existing builder serializers, rather than the normalized node
// serializer. The direct variants have independent protocol expectations here.
fn builder_wire(spec: &ExpressionSpec) -> proto::Expression {
    match spec {
        ExpressionSpec::Literal(value) => value.to_proto(),
        ExpressionSpec::ColumnReference(value) => value.to_proto(),
        ExpressionSpec::UnresolvedFunction(value) => value.to_proto(),
        ExpressionSpec::UnresolvedStar(target, plan_id) => proto::Expression {
            expr_type: Some(proto::expression::ExprType::UnresolvedStar(
                proto::expression::UnresolvedStar {
                    unparsed_target: target.clone(),
                    plan_id: *plan_id,
                },
            )),
            ..Default::default()
        },
        ExpressionSpec::Alias(value) => value.to_proto(),
        ExpressionSpec::Cast(value) => value.to_proto(),
        ExpressionSpec::DirectShufflePartitionId(child) => proto::Expression {
            expr_type: Some(proto::expression::ExprType::DirectShufflePartitionId(
                Box::new(proto::expression::DirectShufflePartitionId {
                    child: Some(Box::new(child.to_proto())),
                }),
            )),
            ..Default::default()
        },
        ExpressionSpec::UnresolvedRegex(name, plan_id) => proto::Expression {
            expr_type: Some(proto::expression::ExprType::UnresolvedRegex(
                proto::expression::UnresolvedRegex {
                    col_name: name.clone(),
                    plan_id: *plan_id,
                },
            )),
            ..Default::default()
        },
        ExpressionSpec::SortOrder(value) => value.to_proto(),
        ExpressionSpec::CaseWhen(value) => value.to_proto(),
        ExpressionSpec::UnresolvedExtractValue(value) => value.to_proto(),
        ExpressionSpec::UpdateFields(value) => value.to_proto(),
        ExpressionSpec::SQLExpression(sql) => sql_wire(sql),
        ExpressionSpec::CallFunction(value) => value.to_proto(),
        ExpressionSpec::WindowExpression(value) => value.to_proto(),
        ExpressionSpec::LambdaFunction(value) => value.to_proto(),
        ExpressionSpec::UnresolvedNamedLambdaVariable(value) => value.to_proto(),
        ExpressionSpec::CommonInlineUserDefinedFunction(value) => proto::Expression {
            expr_type: Some(
                proto::expression::ExprType::CommonInlineUserDefinedFunction(value.to_proto()),
            ),
            ..Default::default()
        },
        ExpressionSpec::NamedArgument(value) => proto::Expression {
            expr_type: Some(proto::expression::ExprType::NamedArgumentExpression(
                Box::new(proto::NamedArgumentExpression {
                    key: value.key.clone(),
                    value: Some(Box::new(value.value.to_proto())),
                }),
            )),
            ..Default::default()
        },
        ExpressionSpec::Subquery(value) => value.to_proto(),
        ExpressionSpec::Sent(value) => value.expression.clone(),
    }
}

macro_rules! expression_inventory {
    ($(($spec:pat, $kind:pat, $fixture:expr)),+ $(,)?) => {
        fn fixtures() -> Vec<Fixture> {
            vec![$($fixture),+]
        }

        fn has_expected_kind(spec: &ExpressionSpec, kind: &ExpressionKind) -> bool {
            match spec {
                $($spec => matches!(kind, $kind)),+
            }
        }

        // This same-crate match intentionally has no wildcard. Adding a
        // canonical kind must add a fixture, even though the public enum is
        // non-exhaustive for downstream consumers.
        fn require_canonical_inventory(kind: &ExpressionKind) {
            match kind {
                $($kind => ()),+
            }
        }
    };
}

expression_inventory! {
    (ExpressionSpec::Literal(_), ExpressionKind::Literal(_), {
        Fixture::new(
            ExpressionSpec::Literal(LiteralExpression::Array {
                element_type: Box::new(DataType::Long),
                elements: vec![LiteralExpression::Long(4), LiteralExpression::Null(DataType::Long)],
            }),
            vec![],
            "[4, None]",
        )
    }),
    (ExpressionSpec::ColumnReference(_), ExpressionKind::ColumnReference(_), {
        Fixture::new(
            ExpressionSpec::ColumnReference(ColumnReference::new("bound").with_plan_id(17).metadata()),
            vec![],
            "bound",
        )
    }),
    (ExpressionSpec::UnresolvedFunction(_), ExpressionKind::UnresolvedFunction { .. }, {
        let left = Expression::column("left");
        let right = Expression::column("right");
        Fixture::new(
            ExpressionSpec::UnresolvedFunction(UnresolvedFunction::new_distinct("combine", vec![left.clone(), right.clone()])),
            vec![left, right],
            "combine(distinct left, right)",
        ).variadic()
    }),
    (ExpressionSpec::UnresolvedStar(..), ExpressionKind::UnresolvedStar(..), {
        Fixture::new(ExpressionSpec::UnresolvedStar(Some("source.*".into()), Some(18)), vec![], "unresolvedstar(source.*)")
    }),
    (ExpressionSpec::Alias(_), ExpressionKind::Alias { .. }, {
        let child = Expression::column("left");
        Fixture::new(
            ExpressionSpec::Alias(Box::new(Alias::with_names(child.clone(), ["first", "second"]).with_metadata("{\"tag\":1}".into()))),
            vec![child],
            "left AS first,second",
        )
    }),
    (ExpressionSpec::Cast(_), ExpressionKind::Cast { .. }, {
        let child = Expression::column("left");
        Fixture::new(ExpressionSpec::Cast(Box::new(Cast::new(child.clone(), DataType::Long).with_eval_mode(CastEvalMode::Try))), vec![child], "TRY_CAST(left AS BIGINT)")
    }),
    (ExpressionSpec::DirectShufflePartitionId(_), ExpressionKind::DirectShufflePartitionId, {
        let child = Expression::column("partition");
        Fixture::new(ExpressionSpec::DirectShufflePartitionId(child.clone()), vec![child], "DirectShufflePartitionID(child=partition)")
    }),
    (ExpressionSpec::UnresolvedRegex(..), ExpressionKind::UnresolvedRegex(..), {
        Fixture::new(ExpressionSpec::UnresolvedRegex("`a.*`".into(), Some(19)), vec![], "UnresolvedRegex(`a.*`)")
    }),
    (ExpressionSpec::SortOrder(_), ExpressionKind::SortOrder { .. }, {
        let child = Expression::column("left");
        Fixture::new(ExpressionSpec::SortOrder(Box::new(SortOrder::desc_nulls_first(child.clone()))), vec![child], "left DESC NULLS FIRST")
    }),
    (ExpressionSpec::CaseWhen(_), ExpressionKind::CaseWhen { .. }, {
        let condition = Expression::column("condition");
        let value = Expression::column("value");
        let otherwise = Expression::column("otherwise");
        Fixture::new(ExpressionSpec::CaseWhen(Box::new(CaseWhen::new(vec![(condition.clone(), value.clone())]).with_else(otherwise.clone()))), vec![condition, value, otherwise], "CASE WHEN condition THEN value ELSE otherwise END")
    }),
    (ExpressionSpec::UnresolvedExtractValue(_), ExpressionKind::UnresolvedExtractValue, {
        let child = Expression::column("record");
        let extraction = Expression::literal(LiteralExpression::String("field".into()));
        Fixture::new(ExpressionSpec::UnresolvedExtractValue(Box::new(ExtractValue::new(child.clone(), extraction.clone()))), vec![child, extraction], "record['field']")
    }),
    (ExpressionSpec::UpdateFields(_), ExpressionKind::UpdateFields { .. }, {
        let child = Expression::column("record");
        let value = Expression::column("replacement");
        Fixture::new(ExpressionSpec::UpdateFields(Box::new(UpdateFieldsExpr::new(child.clone(), "field", Some(value.clone())))), vec![child, value], "update_field(record, field, replacement)")
    }),
    (ExpressionSpec::SQLExpression(_), ExpressionKind::SQLExpression(_), {
        Fixture::new(ExpressionSpec::SQLExpression("1 + 2".into()), vec![], "1 + 2")
    }),
    (ExpressionSpec::CallFunction(_), ExpressionKind::CallFunction { .. }, {
        let left = Expression::column("left");
        let right = Expression::column("right");
        Fixture::new(ExpressionSpec::CallFunction(Box::new(CallFunctionWrapper::new("catalog.f", vec![left.clone(), right.clone()]))), vec![left, right], "CallFunction('catalog.f', left, right)").variadic()
    }),
    (ExpressionSpec::WindowExpression(_), ExpressionKind::WindowExpression { .. }, {
        let function = Expression::column("function");
        let partition = Expression::column("partition");
        let order = Expression::column("order");
        let window = WindowExpressionWrapper::new(function.clone(), vec![partition.clone()], vec![SortOrder::desc_nulls_last(order.clone())], Some((1, FrameBoundary::Preceding(3), FrameBoundary::CurrentRow)));
        let rendered = window.render();
        Fixture::new(ExpressionSpec::WindowExpression(Box::new(window)), vec![function, partition, order], rendered)
    }),
    (ExpressionSpec::LambdaFunction(_), ExpressionKind::LambdaFunction { .. }, {
        let body = Expression::column("body");
        Fixture::new(ExpressionSpec::LambdaFunction(Box::new(LambdaFunction::new(body.clone(), vec![UnresolvedNamedLambdaVariable::new("x"), UnresolvedNamedLambdaVariable::new("y")]))), vec![body], "LambdaFunction(body, x, y)")
    }),
    (ExpressionSpec::UnresolvedNamedLambdaVariable(_), ExpressionKind::UnresolvedNamedLambdaVariable(_), {
        Fixture::new(ExpressionSpec::UnresolvedNamedLambdaVariable(UnresolvedNamedLambdaVariable::new("x")), vec![], "x")
    }),
    (ExpressionSpec::CommonInlineUserDefinedFunction(_), ExpressionKind::CommonInlineUserDefinedFunction { .. }, {
        let left = Expression::column("left");
        let right = Expression::column("right");
        Fixture::new(ExpressionSpec::CommonInlineUserDefinedFunction(Box::new(CommonInlineUserDefinedFunctionExpression::new("udf".into(), false, vec![left.clone(), right.clone()], PythonUDFPayload::new(DataType::Long, 100, vec![1, 2], "3.11".into())))), vec![left, right], "udf(left, right)").variadic()
    }),
    (ExpressionSpec::NamedArgument(_), ExpressionKind::NamedArgument { .. }, {
        let child = Expression::column("value");
        Fixture::new(ExpressionSpec::NamedArgument(Box::new(NamedArgument::new("key", child.clone()))), vec![child], "key => value")
    }),
    (ExpressionSpec::Subquery(_), ExpressionKind::Subquery(_), {
        let partition = Expression::column("partition");
        let order = Expression::column("order");
        let inactive = Expression::column("inactive_value");
        let mut subquery = SubqueryExpression::new(Arc::new(plan::range(0, 3, 1)), 41, SubqueryType::TableArg);
        subquery.partition_spec = vec![partition.clone()];
        subquery.order_spec = vec![order.clone()];
        subquery.in_subquery_values = vec![inactive.clone()];
        subquery.with_single_partition = Some(false);
        Fixture::new(ExpressionSpec::Subquery(Box::new(subquery)), vec![partition, order, inactive], "SubqueryExpression(plan=<plan 41>, type=table_arg, partition_spec=[partition], order_spec=[order], with_single_partition=False)")
    }),
    (ExpressionSpec::Sent(_), ExpressionKind::Sent(_), {
        Fixture::new(ExpressionSpec::Sent(Box::new(SentExpression { expression: sql_wire("hidden_sql"), rendered: "opaque_display".into(), levels: 1 })), vec![], "opaque_display")
    }),
}

fn assert_input_order(expression: &Expression, expected: &[Expression]) {
    assert_eq!(expression.children().len(), expected.len());
    assert_eq!(expression.node().children().len(), expected.len());
    for ((child, node_child), expected) in expression
        .children()
        .iter()
        .zip(expression.node().children())
        .zip(expected)
    {
        assert!(child.ptr_eq(expected));
        assert!(node_child.ptr_eq(expected));
    }
}

#[test]
fn every_spec_normalizes_to_its_declared_kind_and_ordered_inputs() {
    let fixtures = fixtures();
    assert_eq!(fixtures.len(), 21);
    for fixture in fixtures {
        let expression = Expression::new(fixture.spec.clone());
        require_canonical_inventory(expression.kind());
        assert!(
            has_expected_kind(&fixture.spec, expression.kind()),
            "{:?}",
            fixture.spec
        );
        assert_input_order(&expression, &fixture.inputs);
        assert_eq!(expression.kind().arity().unwrap(), fixture.arity);
        assert_eq!(expression.to_spec(), fixture.spec);
        assert_eq!(expression.to_proto(), fixture.wire);
        assert_eq!(expression.render(), fixture.rendered);
        let constructed =
            Expression::try_new(expression.kind().clone(), fixture.inputs.iter().cloned()).unwrap();
        assert_input_order(&constructed, &fixture.inputs);
        assert_eq!(constructed.to_spec(), fixture.spec);
        assert_eq!(constructed.to_proto(), fixture.wire);
        let rebuilt = expression
            .with_children(fixture.inputs.iter().cloned())
            .unwrap();
        assert_input_order(&rebuilt, &fixture.inputs);
        assert_eq!(rebuilt.to_spec(), fixture.spec);
        assert_eq!(rebuilt.to_proto(), fixture.wire);
        assert_eq!(rebuilt.render(), fixture.rendered);
        if *fixture.arity.end() != usize::MAX {
            let mut invalid = fixture.inputs.clone();
            if invalid.is_empty() {
                invalid.push(Expression::column("unexpected"));
            } else {
                invalid.pop();
            }
            assert!(Expression::try_new(expression.kind().clone(), invalid.clone()).is_err());
            assert!(expression.with_children(invalid).is_err());
        }
    }
}

#[test]
fn optional_and_empty_shapes_round_trip_without_adding_inputs() {
    let child = Expression::column("record");
    let fixtures = [
        Fixture::new(
            ExpressionSpec::CaseWhen(Box::new(CaseWhen::new(vec![]))),
            vec![],
            "CASE END",
        ),
        Fixture::new(
            ExpressionSpec::CaseWhen(Box::new(CaseWhen::new(vec![]).with_else(child.clone()))),
            vec![child.clone()],
            "CASE ELSE record END",
        ),
        Fixture::new(
            ExpressionSpec::Alias(Box::new(Alias::new(child.clone(), "renamed"))),
            vec![child.clone()],
            "record AS renamed",
        ),
        Fixture::new(
            ExpressionSpec::Cast(Box::new(Cast::new_str(child.clone(), "double"))),
            vec![child.clone()],
            "CAST(record AS DOUBLE)",
        ),
        Fixture::new(
            ExpressionSpec::UpdateFields(Box::new(UpdateFieldsExpr::new(
                child.clone(),
                "field",
                None,
            ))),
            vec![child],
            "drop_field(record, field)",
        ),
        Fixture::new(
            ExpressionSpec::UnresolvedFunction(UnresolvedFunction::new("zero", vec![])),
            vec![],
            "zero()",
        )
        .variadic(),
        Fixture::new(
            ExpressionSpec::CallFunction(Box::new(CallFunctionWrapper::new("zero", vec![]))),
            vec![],
            "CallFunction('zero')",
        )
        .variadic(),
    ];
    for fixture in fixtures {
        let expression = Expression::new(fixture.spec.clone());
        assert_input_order(&expression, &fixture.inputs);
        assert_eq!(expression.to_spec(), fixture.spec);
        assert_eq!(expression.to_proto(), fixture.wire);
        assert_eq!(expression.render(), fixture.rendered);
    }
}

#[test]
fn subquery_rebuilds_preserve_plan_identity_ids_and_inactive_expression_roles() {
    let owned_plan_expression = crate::column::col("plan_only");
    let owned_plan = Arc::new(plan::project(
        Arc::new(plan::range(0, 3, 1)),
        vec![owned_plan_expression],
    ));
    for subquery_type in [
        SubqueryType::Scalar,
        SubqueryType::Exists,
        SubqueryType::In,
        SubqueryType::TableArg,
    ] {
        let partition = Expression::column("partition");
        let order = Expression::column("order");
        let value = Expression::column("value");
        let mut spec = SubqueryExpression::new(owned_plan.clone(), 57, subquery_type);
        spec.partition_spec = vec![partition.clone()];
        spec.order_spec = vec![order.clone()];
        spec.in_subquery_values = vec![value.clone()];
        spec.with_single_partition = Some(false);
        let expression = Expression::new(ExpressionSpec::Subquery(Box::new(spec.clone())));
        assert_input_order(
            &expression,
            &[partition.clone(), order.clone(), value.clone()],
        );
        assert_eq!(
            expression.to_spec(),
            ExpressionSpec::Subquery(Box::new(spec))
        );

        let mut visited = Vec::new();
        expression.visit(|node| visited.push(node.clone()));
        assert_eq!(
            visited.len(),
            4,
            "owned plan expressions must not be traversed"
        );
        assert!(visited[0].ptr_eq(&expression));
        assert_input_order(&expression, &visited[1..]);
        let rewritten = expression
            .try_transform_up(|node| Ok::<_, InvalidExpressionInputs>(node.clone()))
            .unwrap();
        assert!(rewritten.ptr_eq(&expression));

        let replacements = ["new_partition", "new_order", "new_value"].map(Expression::column);
        let rebuilt = expression
            .with_children(replacements.iter().cloned())
            .unwrap();
        assert_input_order(&rebuilt, &replacements);
        let ExpressionKind::Subquery(dependency) = rebuilt.kind() else {
            panic!("expected a subquery dependency");
        };
        assert!(Arc::ptr_eq(dependency.plan(), &owned_plan));
        assert_eq!(dependency.plan_id(), 57);
        assert_eq!(dependency.subquery_type, subquery_type);
        assert_eq!(dependency.with_single_partition, Some(false));
        let ExpressionSpec::Subquery(spec) = rebuilt.to_spec() else {
            panic!("expected a subquery specification");
        };
        assert!(Arc::ptr_eq(&spec.plan, &owned_plan));
        assert_eq!(spec.plan_id, 57);
        assert!(spec.partition_spec[0].ptr_eq(&replacements[0]));
        assert!(spec.order_spec[0].ptr_eq(&replacements[1]));
        assert!(spec.in_subquery_values[0].ptr_eq(&replacements[2]));
        assert_eq!(rebuilt.to_proto(), spec.to_proto());
        let Some(proto::expression::ExprType::SubqueryExpression(wire)) =
            rebuilt.to_proto().expr_type
        else {
            panic!("expected a subquery protocol expression");
        };
        assert_eq!(wire.plan_id, 57);
        assert_eq!(
            wire.table_arg_options.is_some(),
            subquery_type == SubqueryType::TableArg
        );
        assert_eq!(
            wire.in_subquery_values.len(),
            usize::from(subquery_type == SubqueryType::In)
        );
    }
}

#[test]
fn subquery_equality_keeps_plan_allocation_identity() {
    let plan = Arc::new(plan::range(0, 3, 1));
    let same = Expression::new(ExpressionSpec::Subquery(Box::new(SubqueryExpression::new(
        plan.clone(),
        9,
        SubqueryType::Scalar,
    ))));
    let shared = Expression::new(ExpressionSpec::Subquery(Box::new(SubqueryExpression::new(
        plan,
        9,
        SubqueryType::Scalar,
    ))));
    let distinct = Expression::new(ExpressionSpec::Subquery(Box::new(SubqueryExpression::new(
        Arc::new(plan::range(0, 3, 1)),
        9,
        SubqueryType::Scalar,
    ))));
    assert_eq!(same, shared);
    assert_ne!(same, distinct);
}
