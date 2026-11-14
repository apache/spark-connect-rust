//! Normalization fixtures cover both expression enums exhaustively. Expected
//! inputs and wire results come from the typed builders, before normalization.

use super::*;
use crate::plan;
use crate::udf::PythonUDFPayload;
use std::ops::RangeInclusive;

// Shared uncommon type/frame payloads and union-backed inline children keep
// scalar expression allocations compact. These 64-bit upper bounds are not a
// stable Rust ABI promise or an assumed budget for 32-bit targets.
#[cfg(all(
    target_pointer_width = "64",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
const _: () = {
    assert!(std::mem::size_of::<ExpressionKind>() <= 56);
    assert!(std::mem::size_of::<ExpressionNode>() <= 96);
};

#[test]
fn variadic_arguments_normalize_storage_by_length_and_preserve_handles() {
    let constructors: [fn(Vec<Expression>) -> Expression; 4] = [
        |arguments| {
            Expression::new(ExpressionSpec::UnresolvedFunction(UnresolvedFunction::new(
                "f", arguments,
            )))
        },
        |arguments| {
            Expression::new(ExpressionSpec::CallFunction(Box::new(
                CallFunctionWrapper::new("f", arguments),
            )))
        },
        |arguments| {
            Expression::new(ExpressionSpec::CommonInlineUserDefinedFunction(Box::new(
                CommonInlineUserDefinedFunctionExpression::new(
                    "f".into(),
                    false,
                    arguments,
                    PythonUDFPayload::new(DataType::Long, 100, vec![], "3.11".into()),
                ),
            )))
        },
        |arguments| Expression::function("f", arguments),
    ];

    for (constructor_index, constructor) in constructors.into_iter().enumerate() {
        for count in 0..=4 {
            let expected: Vec<_> = (0..count)
                .map(|index| Expression::column(format!("argument_{index}")))
                .collect();
            let mut arguments = Vec::with_capacity(8);
            arguments.extend(expected.iter().cloned());
            let original_buffer = arguments.as_ptr();
            let expression = constructor(arguments);
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

#[test]
fn direct_function_constructors_preserve_builder_behavior_and_shared_inputs() {
    for count in [0, 1, 2, 3, 4] {
        let inputs: Vec<_> = (0..count)
            .map(|index| Expression::column(format!("argument_{index}")))
            .collect();
        let origin = crate::origin::Origin {
            fragment: "direct function".into(),
            call_site: "construct.py:12".into(),
        };
        let _origin = crate::origin::set_current(origin.clone());
        let expected = Expression::new(ExpressionSpec::UnresolvedFunction(
            UnresolvedFunction::new("f", inputs.clone()),
        ));
        let direct = Expression::function("f", inputs.clone());
        let iterated = Expression::function_with_inputs("f", inputs.iter().cloned());

        for actual in [direct, iterated] {
            assert_input_order(&actual, &inputs);
            assert_eq!(actual.to_proto(), expected.to_proto());
            assert_eq!(actual.render(), expected.render());
            assert_eq!(actual.ty(), &ExpressionType::Unresolved);
            assert_eq!(actual.origin(), Some(&origin));
        }
    }

    let first = Expression::column("first");
    let second = Expression::column("second");
    let array = Expression::function_with_inputs("f", [first.clone(), second.clone()]);
    assert_input_order(&array, &[first, second]);
}

#[test]
fn direct_function_constructors_capture_origin_after_converting_inputs() {
    let mut iterator_origin = None;
    let inputs = std::iter::once(Expression::column("input")).inspect(|_| {
        iterator_origin = Some(crate::origin::set_current(crate::origin::Origin {
            fragment: "iterator".into(),
            call_site: "construct.py:20".into(),
        }));
    });
    let iterated = Expression::function_with_inputs("f", inputs);
    assert_eq!(iterated.origin().unwrap().fragment, "iterator");
    drop(iterator_origin);

    let mut generic_origin = None;
    let inputs = std::iter::once(Expression::column("input")).inspect(|_| {
        generic_origin = Some(crate::origin::set_current(crate::origin::Origin {
            fragment: "generic iterator".into(),
            call_site: "construct.py:22".into(),
        }));
    });
    let generic = Expression::try_new(
        ExpressionKind::UnresolvedFunction {
            name: "f".into(),
            is_distinct: false,
        },
        inputs,
    )
    .unwrap();
    assert_eq!(generic.origin().unwrap().fragment, "generic iterator");
    drop(generic_origin);

    struct NameWithOrigin<'a>(&'a mut Option<crate::origin::OriginGuard>);
    impl From<NameWithOrigin<'_>> for String {
        fn from(name: NameWithOrigin<'_>) -> Self {
            *name.0 = Some(crate::origin::set_current(crate::origin::Origin {
                fragment: "name conversion".into(),
                call_site: "construct.py:21".into(),
            }));
            "f".into()
        }
    }

    let mut name_origin = None;
    let named = Expression::function(NameWithOrigin(&mut name_origin), vec![]);
    assert_eq!(named.origin().unwrap().fragment, "name conversion");
}

#[test]
fn iterator_function_inputs_use_inline_storage_at_the_boundary() {
    let inputs = [
        Expression::column("first"),
        Expression::column("second"),
        Expression::column("third"),
    ];
    let inline = Expression::function_with_inputs("f", [inputs[0].clone(), inputs[1].clone()]);
    assert!(!inline.node.children.spilled());
    assert_input_order(&inline, &inputs[..2]);

    let spilled = Expression::function_with_inputs("f", inputs.clone());
    assert!(spilled.node.children.spilled());
    assert_eq!(spilled.node.children.capacity(), inputs.len());
    assert_input_order(&spilled, &inputs);
}

#[test]
fn wide_input_collection_reserves_exact_capacity() {
    let shared = Expression::column("shared");
    let expression =
        Expression::function_with_inputs("f", std::iter::repeat_n(shared.clone(), 10_001));
    let generic = Expression::try_new(
        expression.kind().clone(),
        std::iter::repeat_n(shared.clone(), 10_001),
    )
    .unwrap();
    let rebuilt = expression
        .with_children(std::iter::repeat_n(shared.clone(), 10_001))
        .unwrap();
    for node in [&expression, &generic, &rebuilt] {
        assert_eq!(node.children().len(), 10_001);
        assert_eq!(node.node.children.capacity(), 10_001);
        assert!(node.children().iter().all(|input| input.ptr_eq(&shared)));
    }

    let replacement = Expression::column("replacement");
    let transformed = expression
        .try_transform_up::<InvalidExpressionInputs>(|node| {
            Ok(if node.ptr_eq(&shared) {
                replacement.clone()
            } else {
                node.clone()
            })
        })
        .unwrap();
    assert_eq!(transformed.node.children.capacity(), 10_001);
    assert!(transformed
        .children()
        .iter()
        .all(|input| input.ptr_eq(&replacement)));
}

#[test]
fn composite_typed_builders_reserve_exact_input_capacity() {
    let shared = Expression::column("shared");
    let case = CaseWhen::new(vec![(shared.clone(), shared.clone()); 5]).with_else(shared.clone());
    let window = WindowExpressionWrapper::new(
        shared.clone(),
        vec![shared.clone(); 2],
        vec![SortOrder::desc_nulls_last(shared.clone()); 2],
        None,
    );
    let mut subquery = SubqueryExpression::new(Arc::new(plan::range(0, 1, 1)), 5, SubqueryType::In);
    subquery.in_subquery_values = vec![shared.clone(); 3];
    for (spec, count) in [
        (ExpressionSpec::CaseWhen(Box::new(case)), 11),
        (ExpressionSpec::WindowExpression(Box::new(window)), 5),
        (ExpressionSpec::Subquery(Box::new(subquery)), 3),
    ] {
        let expression = Expression::new(spec.clone());
        assert_eq!(expression.children().len(), count);
        assert_eq!(expression.node.children.capacity(), count);
        assert!(expression
            .children()
            .iter()
            .all(|input| input.ptr_eq(&shared)));
        assert_eq!(expression.to_spec(), spec);
    }
}

#[test]
fn deep_shared_destruction_preserves_external_leaves_and_releases_unique_nodes() {
    std::thread::Builder::new()
        .stack_size(128 << 10)
        .spawn(|| {
            let (root, survivor, unique_nodes, survivor_node, origin) = {
                let _origin = crate::origin::set_current(crate::origin::Origin {
                    fragment: "shared destruction".into(),
                    call_site: "drop.py:10".into(),
                });
                let survivor = Expression::literal(LiteralExpression::Integer(7));
                let survivor_node = Arc::downgrade(&survivor.node);
                let origin = Arc::downgrade(survivor.node.origin.as_ref().unwrap());
                let mut root = survivor.clone();
                let mut unique_nodes = Vec::new();
                for value in 0..10_000 {
                    let unique = Expression::literal(LiteralExpression::Integer(value));
                    unique_nodes.push(Arc::downgrade(&unique.node));
                    root = Expression::function("branch", vec![root, survivor.clone(), unique]);
                }
                (root, survivor, unique_nodes, survivor_node, origin)
            };

            drop(root);
            assert!(unique_nodes.iter().all(|node| node.upgrade().is_none()));
            assert!(survivor_node.upgrade().is_some());
            assert_eq!(survivor.render(), "7");
            assert_eq!(survivor.origin().unwrap().fragment, "shared destruction");
            drop(survivor);
            assert!(survivor_node.upgrade().is_none());
            assert!(origin.upgrade().is_none());
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn concurrent_wide_destruction_releases_last_owned_and_recursive_leaves() {
    const WORKERS: usize = 4;
    let declared = Arc::new(DataType::Long);
    let declared_weak = Arc::downgrade(&declared);
    let mut nested = LiteralExpression::Null(declared);
    for _ in 0..1000 {
        nested = LiteralExpression::Array {
            element_type: Box::new(DataType::Long),
            elements: vec![nested],
        };
    }
    let leaves = [
        Expression::literal(LiteralExpression::Integer(7)),
        Expression::column("survivor"),
        Expression::literal(nested),
    ];
    let leaf_nodes: Vec<_> = leaves
        .iter()
        .map(|leaf| Arc::downgrade(&leaf.node))
        .collect();
    let survivor = leaves[1].clone();
    let start = Arc::new(std::sync::Barrier::new(WORKERS + 1));
    let mut workers = Vec::new();
    for _ in 0..WORKERS {
        let arguments = (0..1024).flat_map(|_| leaves.iter().cloned()).collect();
        let root = Expression::function("wide", arguments);
        let root_node = Arc::downgrade(&root.node);
        let start = Arc::clone(&start);
        let worker = std::thread::Builder::new()
            .stack_size(64 << 10)
            .spawn(move || {
                start.wait();
                drop(root);
            })
            .unwrap();
        workers.push((worker, root_node));
    }
    drop(leaves);
    start.wait();
    for (worker, root_node) in workers {
        worker.join().unwrap();
        assert!(root_node.upgrade().is_none());
    }

    assert!(leaf_nodes[0].upgrade().is_none());
    assert!(leaf_nodes[2].upgrade().is_none());
    assert!(declared_weak.upgrade().is_none());
    assert_eq!(survivor.render(), "survivor");
    assert!(leaf_nodes[1].upgrade().is_some());
    drop(survivor);
    assert!(leaf_nodes[1].upgrade().is_none());
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
                elements: vec![LiteralExpression::Long(4), LiteralExpression::Null(Arc::new(DataType::Long))],
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
