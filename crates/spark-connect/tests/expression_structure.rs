//! Public structural expression contracts, independent of a running Connect server.

use spark_connect::expression::{
    Alias, CaseWhen, Cast, CastTarget, ColumnReference, Expression, ExpressionKind, ExpressionSpec,
    ExpressionType, FrameBoundary, InvalidExpressionInputs, LambdaFunction, LiteralExpression,
    NullOrdering, SortOrder, SortOrderProperties, UnresolvedNamedLambdaVariable,
    WindowExpressionWrapper, MAX_EXPRESSION_DEPTH,
};
use spark_connect::origin::{self, Origin};
use spark_connect::types::DataType;
use spark_connect::udf::{CommonInlineUserDefinedFunctionExpression, PythonUDFPayload};
use spark_connect_proto as proto;
use std::sync::Arc;

fn alias(child: Expression, name: &str) -> Expression {
    Expression::new(ExpressionSpec::Alias(Box::new(Alias::new(child, name))))
}

fn assert_inputs(expression: &Expression, expected: &[Expression]) {
    assert_eq!(expression.children().len(), expected.len());
    assert_eq!(expression.node().children().len(), expected.len());
    for ((handle_child, node_child), expected) in expression
        .children()
        .iter()
        .zip(expression.node().children())
        .zip(expected)
    {
        assert!(handle_child.ptr_eq(expected));
        assert!(node_child.ptr_eq(expected));
    }
}

fn names(root: &Expression, all_paths: bool) -> Vec<String> {
    let mut result = Vec::new();
    if all_paths {
        root.visit_all_paths(|node| result.push(node.render()));
    } else {
        root.visit(|node| result.push(node.render()));
    }
    result
}

#[test]
fn handles_share_nodes_and_rebuilds_share_inputs() {
    let input = Expression::column("before");
    let root = alias(input.clone(), "answer");
    assert!(root.ptr_eq(&root.clone()));
    assert_inputs(&root, std::slice::from_ref(&input));

    let replacement = Expression::column("after");
    let rebuilt = root.with_children([replacement.clone()]).unwrap();
    assert_inputs(&rebuilt, std::slice::from_ref(&replacement));
    assert_eq!(root.render(), "before AS answer");
    assert_eq!(rebuilt.render(), "after AS answer");
    assert_eq!(rebuilt.to_proto(), alias(replacement, "answer").to_proto());
}

#[test]
fn constructors_and_rebuilds_accept_iterators_and_check_input_counts() {
    let input = Expression::column("a");
    let original = alias(input.clone(), "named");
    let constructed =
        Expression::try_new(original.kind().clone(), std::iter::once(input.clone())).unwrap();
    assert_eq!(constructed.to_proto(), original.to_proto());
    assert_inputs(&constructed, std::slice::from_ref(&input));
    assert!(Expression::try_new(original.kind().clone(), []).is_err());
    assert!(original.with_children([]).is_err());
    assert!(original
        .with_children([input.clone(), input.clone()])
        .is_err());
    assert!(Expression::try_new(input.kind().clone(), [input.clone()]).is_err());
    assert_inputs(&original, &[input]);
}

#[test]
fn convenience_leaves_match_typed_specs_including_type_wire_and_origin() {
    let _origin = origin::set_current(Origin {
        fragment: "leaf construction".into(),
        call_site: "builder.py:10".into(),
    });
    let compare = |fast: Expression, spec: Expression| {
        assert_eq!(fast.kind(), spec.kind());
        assert_eq!(fast.ty(), spec.ty());
        assert_eq!(fast.to_proto(), spec.to_proto());
        assert_eq!(fast.origin(), spec.origin());
        assert_eq!(fast.origin().unwrap().fragment, "leaf construction");
        assert!(fast.children().is_empty());
        assert!(spec.children().is_empty());
    };
    for literal in [
        LiteralExpression::Long(7),
        LiteralExpression::Null(Arc::new(DataType::Array {
            element_type: Box::new(DataType::Long),
            contains_null: true,
        })),
    ] {
        compare(
            Expression::literal(literal.clone()),
            Expression::new(ExpressionSpec::Literal(literal)),
        );
    }
    compare(
        Expression::column("value"),
        Expression::new(ExpressionSpec::ColumnReference(ColumnReference::new(
            "value",
        ))),
    );
}

#[test]
fn visit_is_unique_preorder_and_all_paths_repeats_shared_inputs() {
    let leaf = Expression::column("x");
    let shared = Expression::function("abs", vec![leaf]);
    let root = Expression::function("+", vec![shared.clone(), shared.clone()]);
    assert_inputs(&root, &[shared.clone(), shared]);
    assert_eq!(names(&root, false), ["+(abs(x), abs(x))", "abs(x)", "x"]);
    assert_eq!(
        names(&root, true),
        ["+(abs(x), abs(x))", "abs(x)", "x", "abs(x)", "x"]
    );
}

#[test]
fn unique_visits_distinguish_equal_but_separately_allocated_nodes() {
    let left = Expression::column("same");
    let right = Expression::column("same");
    assert_eq!(left, right);
    assert!(!left.ptr_eq(&right));
    let root = Expression::function("+", vec![left, right]);
    assert_eq!(names(&root, false), ["+(same, same)", "same", "same"]);
}

#[test]
fn unique_visits_do_not_expand_a_shared_fanout_dag() {
    let mut root = Expression::column("leaf");
    for _ in 0..80 {
        root = Expression::function("+", vec![root.clone(), root]);
    }
    let mut visited = 0;
    root.visit(|_| visited += 1);
    assert_eq!(visited, 81);

    let mut small = Expression::column("leaf");
    for _ in 0..10 {
        small = Expression::function("+", vec![small.clone(), small]);
    }
    let mut occurrences = 0;
    small.visit_all_paths(|_| occurrences += 1);
    assert_eq!(occurrences, 2047);
}

#[test]
fn case_children_follow_condition_value_pairs_then_else() {
    let expected =
        ["condition0", "value0", "condition1", "value1", "otherwise"].map(Expression::column);
    let case = Expression::new(ExpressionSpec::CaseWhen(Box::new(
        CaseWhen::new(vec![
            (expected[0].clone(), expected[1].clone()),
            (expected[2].clone(), expected[3].clone()),
        ])
        .with_else(expected[4].clone()),
    )));
    assert_inputs(&case, &expected);
    let proto = case.to_proto();
    let Some(proto::expression::ExprType::UnresolvedFunction(function)) = proto.expr_type else {
        panic!("CASE must serialize as a when function");
    };
    assert_eq!(function.function_name, "when");
    assert_eq!(
        function.arguments,
        expected
            .iter()
            .map(Expression::to_proto)
            .collect::<Vec<_>>()
    );
    assert!(case.with_children(expected[..4].iter().cloned()).is_err());
    assert!(case
        .with_children(
            expected
                .iter()
                .cloned()
                .chain([Expression::column("extra")])
        )
        .is_err());
}

#[test]
fn case_shape_checks_reject_missing_inputs_and_overflowing_metadata() {
    let incomplete = Expression::try_new(
        ExpressionKind::CaseWhen {
            branch_count: 2,
            has_else: true,
        },
        ["condition0", "value0", "condition1", "value1"].map(Expression::column),
    );
    assert!(matches!(
        incomplete,
        Err(InvalidExpressionInputs::InvalidArity {
            operator: "CaseWhen",
            expected_min: 5,
            expected_max: Some(5),
            actual: 4,
            ..
        })
    ));
    let overflowing = ExpressionKind::CaseWhen {
        branch_count: usize::MAX / 2 + 1,
        has_else: false,
    };
    assert!(matches!(
        overflowing.arity(),
        Err(InvalidExpressionInputs::InputCountOverflow {
            operator: "CaseWhen",
            ..
        })
    ));
    assert!(matches!(
        Expression::try_new(overflowing, []),
        Err(InvalidExpressionInputs::InputCountOverflow {
            operator: "CaseWhen",
            ..
        })
    ));
}

#[test]
fn exact_maximum_input_counts_keep_a_finite_error_bound() {
    let kind = ExpressionKind::CaseWhen {
        branch_count: usize::MAX / 2,
        has_else: true,
    };
    assert_eq!(kind.arity().unwrap(), usize::MAX..=usize::MAX);
    let error = Expression::try_new(kind, []).unwrap_err();
    assert!(matches!(
        error,
        InvalidExpressionInputs::InvalidArity {
            operator: "CaseWhen",
            expected_min: usize::MAX,
            expected_max: Some(usize::MAX),
            actual: 0,
            ..
        }
    ));
    assert_eq!(
        error.to_string(),
        format!("CaseWhen expects {} expression inputs, got 0", usize::MAX)
    );
    let variadic = Expression::function("variadic", vec![]);
    assert_eq!(variadic.kind().arity().unwrap(), 0..=usize::MAX);
    assert!(variadic.with_children([]).is_ok());
    assert!(variadic
        .with_children([Expression::column("input")])
        .is_ok());
}

#[test]
fn window_children_preserve_order_flags_and_frame_payload() {
    let function = Expression::function("row_number", vec![]);
    let partition = Expression::column("key");
    let first = Expression::column("first");
    let second = Expression::column("second");
    let root = Expression::new(ExpressionSpec::WindowExpression(Box::new(
        WindowExpressionWrapper::new(
            function.clone(),
            vec![partition.clone()],
            vec![
                SortOrder::desc_nulls_last(first.clone()),
                SortOrder::asc_nulls_first(second.clone()),
            ],
            Some((1, FrameBoundary::Preceding(2), FrameBoundary::CurrentRow)),
        ),
    )));
    let expected = [function, partition, first, second];
    assert_inputs(&root, &expected);

    let replacement = Expression::column("replacement");
    let rebuilt = root
        .with_children(expected[..3].iter().cloned().chain([replacement.clone()]))
        .unwrap();
    let transformed = root
        .try_transform_up(|node| {
            Ok::<_, InvalidExpressionInputs>(if node.ptr_eq(&expected[3]) {
                replacement.clone()
            } else {
                node.clone()
            })
        })
        .unwrap();
    assert!(!transformed.ptr_eq(&root));
    assert_inputs(&transformed, rebuilt.children());
    assert_eq!(transformed.to_proto(), rebuilt.to_proto());
    let ExpressionKind::WindowExpression {
        frame_spec: Some(frame),
        ..
    } = root.kind()
    else {
        panic!("expected a canonical window frame");
    };
    for expression in [&rebuilt, &transformed] {
        let ExpressionKind::WindowExpression {
            frame_spec: Some(rebuilt_frame),
            ..
        } = expression.kind()
        else {
            panic!("rebuilt windows must retain their frame");
        };
        assert!(Arc::ptr_eq(frame, rebuilt_frame));
    }
    let Some(proto::expression::ExprType::Window(window)) = rebuilt.to_proto().expr_type else {
        panic!("expected a window expression");
    };
    assert_eq!(window.order_spec.len(), 2);
    assert_eq!(window.order_spec[0].direction, 2);
    assert_eq!(window.order_spec[0].null_ordering, 2);
    assert_eq!(window.order_spec[1].direction, 1);
    assert_eq!(window.order_spec[1].null_ordering, 1);
    assert_eq!(
        window.order_spec[1].child.as_deref(),
        Some(&replacement.to_proto())
    );
    let Some(proto::expression::ExprType::Window(original)) = root.to_proto().expr_type else {
        unreachable!();
    };
    assert_eq!(window.frame_spec, original.frame_spec);
    assert!(root.with_children(expected[..3].iter().cloned()).is_err());
}

#[test]
fn window_shape_checks_reject_both_segment_overflow_paths() {
    let cases = [
        ExpressionKind::WindowExpression {
            partition_count: usize::MAX,
            order_spec: vec![],
            frame_spec: None,
        },
        ExpressionKind::WindowExpression {
            partition_count: usize::MAX - 1,
            order_spec: vec![SortOrderProperties {
                ascending: false,
                null_ordering: NullOrdering::Last,
            }],
            frame_spec: None,
        },
    ];
    for kind in cases {
        assert!(matches!(
            kind.arity(),
            Err(InvalidExpressionInputs::InputCountOverflow {
                operator: "WindowExpression",
                ..
            })
        ));
        assert!(matches!(
            Expression::try_new(kind, [Expression::column("function")]),
            Err(InvalidExpressionInputs::InputCountOverflow {
                operator: "WindowExpression",
                ..
            })
        ));
    }
}

#[test]
fn lambda_children_expose_the_body_and_preserve_binder_names() {
    let body = Expression::function("+", vec![Expression::column("x"), Expression::column("y")]);
    let lambda = Expression::new(ExpressionSpec::LambdaFunction(Box::new(
        LambdaFunction::new(
            body.clone(),
            vec![
                UnresolvedNamedLambdaVariable::new("x"),
                UnresolvedNamedLambdaVariable::new("y"),
            ],
        ),
    )));
    assert_inputs(&lambda, &[body]);
    let replacement = Expression::column("x");
    let rebuilt = lambda.with_children([replacement.clone()]).unwrap();
    assert_inputs(&rebuilt, std::slice::from_ref(&replacement));
    let Some(proto::expression::ExprType::LambdaFunction(wire)) = rebuilt.to_proto().expr_type
    else {
        panic!("expected a lambda expression");
    };
    assert_eq!(wire.function.as_deref(), Some(&replacement.to_proto()));
    assert_eq!(wire.arguments.len(), 2);
    assert_eq!(wire.arguments[0].name_parts, ["x"]);
    assert_eq!(wire.arguments[1].name_parts, ["y"]);
}

#[test]
fn rebuild_keeps_captured_origin_and_does_not_recapture_the_current_one() {
    let root = {
        let _origin = origin::set_current(Origin {
            fragment: "original".into(),
            call_site: "original.py:10".into(),
        });
        alias(Expression::column("a"), "named")
    };
    let rebuilt = {
        let _origin = origin::set_current(Origin {
            fragment: "replacement".into(),
            call_site: "replacement.py:20".into(),
        });
        root.with_children([Expression::column("b")]).unwrap()
    };
    assert_eq!(rebuilt.origin(), root.origin());
    assert_eq!(rebuilt.origin().unwrap().fragment, "original");
    assert_eq!(
        rebuilt.children()[0].origin().unwrap().fragment,
        "replacement"
    );
    assert_eq!(rebuilt, alias(Expression::column("b"), "named"));
    let common = rebuilt.to_proto().common.unwrap();
    let Some(proto::origin::Function::PythonOrigin(origin)) =
        common.origin.and_then(|origin| origin.function)
    else {
        panic!("expected the captured Python origin");
    };
    assert_eq!(origin.fragment, "original");
    assert_eq!(origin.call_site, "original.py:10");
}

#[test]
fn declared_types_survive_rebuild_while_columns_and_functions_stay_unresolved() {
    let input = Expression::column("input");
    let cast = Expression::new(ExpressionSpec::Cast(Box::new(Cast::new(
        input.clone(),
        DataType::Integer,
    ))));
    assert_inputs(&cast, std::slice::from_ref(&input));
    let rebuilt = cast.with_children([Expression::column("other")]).unwrap();
    assert!(matches!(
        rebuilt.ty(),
        ExpressionType::Known(ty) if ty.as_ref() == &DataType::Integer
    ));
    assert!(matches!(input.ty(), ExpressionType::Unresolved));
    assert!(matches!(
        Expression::function("abs", vec![input]).ty(),
        ExpressionType::Unresolved
    ));
    let typed_null = Expression::literal(LiteralExpression::Null(Arc::new(DataType::Long)));
    assert_inputs(&typed_null, &[]);
    assert!(matches!(
        typed_null.ty(),
        ExpressionType::Known(ty) if ty.as_ref() == &DataType::Long
    ));
    let rebuilt_null = typed_null.with_children([]).unwrap();
    assert_eq!(rebuilt_null.to_proto(), typed_null.to_proto());
    for (original, rebuilt) in [(&cast, &rebuilt), (&typed_null, &rebuilt_null)] {
        let (ExpressionType::Known(original_type), ExpressionType::Known(rebuilt_type)) =
            (original.ty(), rebuilt.ty())
        else {
            panic!("typed expressions must retain their declared types");
        };
        assert!(Arc::ptr_eq(original_type, rebuilt_type));
        for expression in [original, rebuilt] {
            let payload = match expression.kind() {
                ExpressionKind::Literal(LiteralExpression::Null(payload))
                | ExpressionKind::Cast {
                    target: CastTarget::Type(payload),
                    ..
                } => payload,
                _ => panic!("expected a typed null or structured cast"),
            };
            assert!(Arc::ptr_eq(payload, original_type));
        }
    }
}

#[test]
fn udf_inputs_are_ordered_and_the_declared_return_type_is_retained() {
    let first = Expression::column("first");
    let second = Expression::column("second");
    let udf = Expression::new(ExpressionSpec::CommonInlineUserDefinedFunction(Box::new(
        CommonInlineUserDefinedFunctionExpression::new(
            "combine".into(),
            false,
            vec![first.clone(), second.clone()],
            PythonUDFPayload::new(DataType::Long, 100, vec![1, 2, 3], "3.11".into()),
        ),
    )));
    assert_inputs(&udf, &[first.clone(), second.clone()]);
    let rebuilt = udf.with_children([second, first]).unwrap();
    let ExpressionKind::CommonInlineUserDefinedFunction { python_udf, .. } = udf.kind() else {
        unreachable!();
    };
    let ExpressionKind::CommonInlineUserDefinedFunction {
        python_udf: rebuilt_payload,
        ..
    } = rebuilt.kind()
    else {
        unreachable!();
    };
    assert!(Arc::ptr_eq(python_udf, rebuilt_payload));
    let (ExpressionType::Known(original_type), ExpressionType::Known(rebuilt_type)) =
        (udf.ty(), rebuilt.ty())
    else {
        unreachable!();
    };
    assert!(Arc::ptr_eq(original_type, rebuilt_type));
    assert!(matches!(
        rebuilt.ty(),
        ExpressionType::Known(ty) if ty.as_ref() == &DataType::Long
    ));
    let Some(proto::expression::ExprType::CommonInlineUserDefinedFunction(wire)) =
        rebuilt.to_proto().expr_type
    else {
        panic!("expected an inline UDF");
    };
    assert_eq!(wire.function_name, "combine");
    assert!(!wire.deterministic);
    assert_eq!(
        wire.arguments,
        rebuilt
            .children()
            .iter()
            .map(Expression::to_proto)
            .collect::<Vec<_>>()
    );
    let Some(proto::common_inline_user_defined_function::Function::PythonUdf(payload)) =
        wire.function
    else {
        panic!("expected the Python UDF payload");
    };
    assert_eq!(payload.output_type, Some(DataType::Long.to_proto()));
    assert_eq!(payload.command.as_ref(), &[1, 2, 3]);
}

#[test]
fn opaque_sent_expressions_are_leaves_and_keep_their_serialized_depth() {
    let original = Expression::function("abs", vec![Expression::column("hidden")]);
    let (wire, levels) = original.try_to_proto_with_levels().unwrap();
    assert_eq!(levels, 2);
    let sent = Expression::sent(wire.clone(), original.render(), levels).unwrap();
    assert_inputs(&sent, &[]);
    assert_eq!(names(&sent, false), ["abs(hidden)"]);
    assert_eq!(names(&sent, true), ["abs(hidden)"]);
    assert!(matches!(
        sent.with_children([Expression::column("replacement")]),
        Err(InvalidExpressionInputs::InvalidArity {
            operator: "Sent",
            expected_min: 0,
            expected_max: Some(0),
            actual: 1,
            ..
        })
    ));
    let rebuilt = sent.with_children([]).unwrap();
    let (ExpressionKind::Sent(payload), ExpressionKind::Sent(rebuilt_payload)) =
        (sent.kind(), rebuilt.kind())
    else {
        unreachable!();
    };
    assert!(Arc::ptr_eq(payload, rebuilt_payload));
    assert_inputs(&rebuilt, &[]);
    assert_eq!(rebuilt.to_proto(), wire);
    assert_eq!(rebuilt.render(), original.render());
    assert_eq!(rebuilt.try_to_proto_with_levels().unwrap().1, 2);
    let wrapped = Expression::function("sqrt", vec![rebuilt]);
    assert_eq!(wrapped.try_to_proto_with_levels().unwrap().1, 3);
}

#[test]
fn sharing_an_opaque_payload_does_not_make_nan_equal_to_itself() {
    let wire = Expression::literal(LiteralExpression::Double(f64::NAN)).to_proto();
    let sent = Expression::sent(wire, "NaN".into(), 1).unwrap();
    assert_ne!(sent, sent.clone());
    assert_ne!(sent, sent.with_children([]).unwrap());
}

#[test]
fn no_op_bottom_up_transforms_keep_root_identity() {
    let shared = Expression::function("abs", vec![Expression::column("x")]);
    let root = Expression::function("+", vec![shared.clone(), shared]);
    let mut visited = Vec::new();
    let rewritten = root
        .try_transform_up(|node| {
            visited.push(node.render());
            Ok::<_, InvalidExpressionInputs>(node.clone())
        })
        .unwrap();
    assert!(rewritten.ptr_eq(&root));
    assert_eq!(visited, ["x", "abs(x)", "+(abs(x), abs(x))"]);
}

#[test]
fn bottom_up_transforms_rebuild_shared_ancestors_once() {
    let old = Expression::column("old");
    let replacement = Expression::column("new");
    let shared = Expression::function("abs", vec![old.clone()]);
    let root = Expression::function("+", vec![shared.clone(), shared]);
    let mut callbacks = 0;
    let rewritten = root
        .try_transform_up(|node| {
            callbacks += 1;
            Ok::<_, InvalidExpressionInputs>(if node.ptr_eq(&old) {
                replacement.clone()
            } else {
                node.clone()
            })
        })
        .unwrap();
    assert_eq!(callbacks, 3);
    assert_eq!(rewritten.render(), "+(abs(new), abs(new))");
    assert_eq!(root.render(), "+(abs(old), abs(old))");
    assert!(rewritten.children()[0].ptr_eq(&rewritten.children()[1]));
    assert!(rewritten.children()[0].children()[0].ptr_eq(&replacement));
}

#[derive(Debug)]
struct RewriteError;

impl From<InvalidExpressionInputs> for RewriteError {
    fn from(_: InvalidExpressionInputs) -> Self {
        Self
    }
}

#[test]
fn a_failing_transform_leaves_the_original_dag_unchanged() {
    let leaf = Expression::column("stop");
    let root = alias(leaf.clone(), "answer");
    let mut callbacks = 0;
    let result = root.try_transform_up(|_| {
        callbacks += 1;
        Err::<Expression, _>(RewriteError)
    });
    assert!(matches!(result, Err(RewriteError)));
    assert_eq!(callbacks, 1);
    assert_eq!(root.render(), "stop AS answer");
    assert!(root.children()[0].ptr_eq(&leaf));
}

#[test]
fn shared_nan_nodes_keep_value_equality_semantics() {
    let nan = Expression::literal(LiteralExpression::Double(f64::NAN));
    assert!(nan.ptr_eq(&nan.clone()));
    assert_ne!(nan, nan.clone());
    let root = alias(nan, "nan");
    assert_ne!(root, root.clone());
}

#[test]
fn deep_structural_walks_and_no_op_rewrites_work_on_a_small_stack() {
    std::thread::Builder::new()
        .stack_size(256 << 10)
        .spawn(|| {
            let mut root = Expression::column("leaf");
            for _ in 0..30_000 {
                root = Expression::function("abs", vec![root]);
            }
            let mut count = 0;
            root.visit(|_| count += 1);
            assert_eq!(count, 30_001);
            let rewritten = root
                .try_transform_up(|node| Ok::<_, InvalidExpressionInputs>(node.clone()))
                .unwrap();
            assert!(rewritten.ptr_eq(&root));
            drop(rewritten);
            drop(root);
        })
        .unwrap()
        .join()
        .unwrap();
}

fn nested_alias_wire(levels: usize) -> proto::Expression {
    let mut wire = proto::Expression {
        expr_type: Some(proto::expression::ExprType::ExpressionString(
            proto::expression::ExpressionString {
                expression: "hidden".into(),
            },
        )),
        ..Default::default()
    };
    for level in 0..levels {
        wire = proto::Expression {
            expr_type: Some(proto::expression::ExprType::Alias(Box::new(
                proto::expression::Alias {
                    expr: Some(Box::new(wire)),
                    name: vec![format!("level_{level}")],
                    ..Default::default()
                },
            ))),
            ..Default::default()
        };
    }
    wire
}

#[test]
fn deep_sent_payloads_clone_and_rebuild_on_a_small_stack() {
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(|| {
            let wire = nested_alias_wire(1000);
            let expected = wire.clone();
            // Logical depth remains a caller-supplied Sent contract. The physical
            // protobuf nesting must independently size the clone's stack.
            let sent = Expression::sent(wire, "opaque".into(), MAX_EXPRESSION_DEPTH).unwrap();
            let (sent, kind, rebuilt, spec, direct_wire, bounded_wire) =
                std::thread::Builder::new()
                    .stack_size(512 << 10)
                    .spawn(move || {
                        let kind = sent.kind().clone();
                        let rebuilt = sent.with_children([]).unwrap();
                        let spec = sent.to_spec();
                        let direct_wire = sent.to_proto();
                        let (bounded_wire, levels) = sent.try_to_proto_with_levels().unwrap();
                        assert_eq!(levels, MAX_EXPRESSION_DEPTH);
                        // Return every deep value: comparison and destruction run
                        // on the enclosing large stack, outside the guarantee here.
                        (sent, kind, rebuilt, spec, direct_wire, bounded_wire)
                    })
                    .unwrap()
                    .join()
                    .unwrap();

            let ExpressionKind::Sent(cloned) = kind else {
                panic!("expected an opaque Sent kind");
            };
            let ExpressionSpec::Sent(spec) = spec else {
                panic!("expected an opaque Sent specification");
            };
            assert_eq!(cloned.expression(), &expected);
            assert_eq!(cloned.levels(), MAX_EXPRESSION_DEPTH);
            assert_eq!(spec.expression, expected);
            assert_eq!(spec.levels, MAX_EXPRESSION_DEPTH);
            assert_eq!(direct_wire, expected);
            assert_eq!(bounded_wire, expected);
            assert_eq!(rebuilt.to_proto(), expected);
            assert_eq!(sent.to_proto(), expected);
            assert_eq!(rebuilt.render(), "opaque");
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn deep_sent_child_drops_on_a_small_stack() {
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(|| {
            let sent = Expression::sent(
                nested_alias_wire(1000),
                "opaque".into(),
                MAX_EXPRESSION_DEPTH,
            )
            .unwrap();
            let ExpressionKind::Sent(payload) = sent.kind() else {
                panic!("expected an opaque Sent payload");
            };
            let payload = Arc::downgrade(payload);
            std::thread::Builder::new()
                .stack_size(64 << 10)
                .spawn(move || drop(Expression::function("opaque_leaf", vec![sent])))
                .unwrap()
                .join()
                .unwrap();
            assert!(payload.upgrade().is_none());
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn deep_declared_types_rebuild_on_a_small_stack() {
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(|| {
            let mut declared = DataType::Long;
            for _ in 0..1000 {
                declared = DataType::Array {
                    element_type: Box::new(declared),
                    contains_null: true,
                };
            }
            let expected = declared.clone();
            let cast = Expression::new(ExpressionSpec::Cast(Box::new(Cast::new(
                Expression::column("before"),
                expected.clone(),
            ))));
            let (expression, rebuilt, cast, rebuilt_cast) = std::thread::Builder::new()
                .stack_size(512 << 10)
                .spawn(move || {
                    let expression =
                        Expression::literal(LiteralExpression::Null(Arc::new(declared)));
                    assert!(expression
                        .with_children([Expression::column("invalid")])
                        .is_err());
                    assert!(cast.with_children([]).is_err());
                    let rebuilt = expression.with_children([]).unwrap();
                    let rebuilt_cast = cast.with_children([Expression::column("after")]).unwrap();
                    (expression, rebuilt, cast, rebuilt_cast)
                })
                .unwrap()
                .join()
                .unwrap();
            // Keep nested type equality and destruction on the large stack.
            let ExpressionType::Known(cached) = expression.ty() else {
                panic!("a typed null must retain its declared type");
            };
            assert_eq!(cached.as_ref(), &expected);
            let ExpressionKind::Literal(LiteralExpression::Null(payload)) = expression.kind()
            else {
                panic!("expected a typed null payload");
            };
            assert_eq!(payload.as_ref(), &expected);
            assert!(Arc::ptr_eq(payload, cached));
            assert!(expression.children().is_empty());
            for root in [&rebuilt, &cast, &rebuilt_cast] {
                let ExpressionType::Known(cached) = root.ty() else {
                    panic!("rebuilt expressions must retain their declared types");
                };
                assert_eq!(cached.as_ref(), &expected);
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn expression_owned_declared_type_cache_drops_inside_the_stack_guard() {
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(|| {
            let mut declared = DataType::Long;
            for _ in 0..1000 {
                declared = DataType::Array {
                    element_type: Box::new(declared),
                    contains_null: true,
                };
            }
            std::thread::Builder::new()
                .stack_size(64 << 10)
                .spawn(move || {
                    let expression =
                        Expression::literal(LiteralExpression::Null(Arc::new(declared)));
                    assert!(matches!(expression.ty(), ExpressionType::Known(_)));
                    // The leaf payload and cached type handles must both be
                    // released inside its guard when the parent's drain drops it.
                    // No standalone DataType or opaque protobuf is dropped here.
                    drop(Expression::function("typed_leaf", vec![expression]));
                })
                .unwrap()
                .join()
                .unwrap();
        })
        .unwrap()
        .join()
        .unwrap();
}

// The caller builds, compares and drops protobufs on a large stack. This
// helper exercises just the opaque clone entry points on the small stack.
fn assert_sent_clone_entry_points(wire: proto::Expression, label: &'static str) {
    let expected = wire.clone();
    let sent = Expression::sent(wire, label.into(), 1).unwrap();
    let (sent, kind, rebuilt, spec, direct_wire, bounded_wire) = std::thread::Builder::new()
        .stack_size(512 << 10)
        .spawn(move || {
            let kind = sent.kind().clone();
            let rebuilt = sent.with_children([]).unwrap();
            let spec = sent.to_spec();
            let direct_wire = sent.to_proto();
            let (bounded_wire, levels) = sent.try_to_proto_with_levels().unwrap();
            assert_eq!(levels, 1);
            (sent, kind, rebuilt, spec, direct_wire, bounded_wire)
        })
        .unwrap()
        .join()
        .unwrap();
    let ExpressionKind::Sent(kind) = kind else {
        panic!("expected a Sent kind");
    };
    let ExpressionSpec::Sent(spec) = spec else {
        panic!("expected a Sent specification");
    };
    assert_eq!(kind.expression(), &expected, "{label}");
    assert_eq!(spec.expression, expected, "{label}");
    assert_eq!(direct_wire, expected, "{label}");
    assert_eq!(bounded_wire, expected, "{label}");
    assert_eq!(rebuilt.to_proto(), expected, "{label}");
    assert_eq!(sent.render(), label);
}

#[test]
fn sent_clone_walks_nested_array_map_and_struct_literal_values() {
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(|| {
            use proto::expression::literal::LiteralType;
            let long = || proto::expression::Literal {
                literal_type: Some(LiteralType::Long(7)),
                ..Default::default()
            };
            let mut literal = long();
            for level in 0..1000 {
                let literal_type = match level % 4 {
                    0 => LiteralType::Array(proto::expression::literal::Array {
                        elements: vec![literal],
                        ..Default::default()
                    }),
                    1 => LiteralType::Map(proto::expression::literal::Map {
                        keys: vec![long()],
                        values: vec![literal],
                        ..Default::default()
                    }),
                    2 => LiteralType::Struct(proto::expression::literal::Struct {
                        elements: vec![literal],
                        ..Default::default()
                    }),
                    _ => LiteralType::Map(proto::expression::literal::Map {
                        keys: vec![literal],
                        values: vec![long()],
                        ..Default::default()
                    }),
                };
                literal = proto::expression::Literal {
                    literal_type: Some(literal_type),
                    ..Default::default()
                };
            }
            assert_sent_clone_entry_points(
                proto::Expression {
                    expr_type: Some(proto::expression::ExprType::Literal(literal)),
                    ..Default::default()
                },
                "literal containers",
            );
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn sent_clone_walks_literal_type_annotations_and_null_types() {
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(|| {
            use proto::data_type::Kind;
            use proto::expression::literal::LiteralType;
            let long = || proto::DataType {
                kind: Some(Kind::Long(proto::data_type::Long::default())),
            };
            let mut data_type = long();
            for level in 0..1000 {
                let kind = match level % 5 {
                    0 => Kind::Array(Box::new(proto::data_type::Array {
                        element_type: Some(Box::new(data_type)),
                        contains_null: true,
                        ..Default::default()
                    })),
                    1 => Kind::Map(Box::new(proto::data_type::Map {
                        key_type: Some(Box::new(data_type)),
                        value_type: Some(Box::new(long())),
                        ..Default::default()
                    })),
                    2 => Kind::Struct(proto::data_type::Struct {
                        fields: vec![proto::data_type::StructField {
                            name: "nested".into(),
                            data_type: Some(data_type),
                            nullable: true,
                            ..Default::default()
                        }],
                        ..Default::default()
                    }),
                    3 => Kind::Map(Box::new(proto::data_type::Map {
                        key_type: Some(Box::new(long())),
                        value_type: Some(Box::new(data_type)),
                        ..Default::default()
                    })),
                    _ => Kind::Udt(Box::new(proto::data_type::Udt {
                        sql_type: Some(Box::new(data_type)),
                        ..Default::default()
                    })),
                };
                data_type = proto::DataType { kind: Some(kind) };
            }
            assert_sent_clone_entry_points(
                proto::Expression {
                    expr_type: Some(proto::expression::ExprType::Literal(
                        proto::expression::Literal {
                            data_type: Some(data_type.clone()),
                            ..Default::default()
                        },
                    )),
                    ..Default::default()
                },
                "literal data_type annotation",
            );
            assert_sent_clone_entry_points(
                proto::Expression {
                    expr_type: Some(proto::expression::ExprType::Literal(
                        proto::expression::Literal {
                            literal_type: Some(LiteralType::Null(data_type)),
                            ..Default::default()
                        },
                    )),
                    ..Default::default()
                },
                "typed null payload",
            );
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn sent_clone_walks_merge_assignments_and_window_frame_expressions() {
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(|| {
            use proto::expression::window::window_frame::frame_boundary::Boundary;
            let leaf = || proto::Expression {
                expr_type: Some(proto::expression::ExprType::ExpressionString(
                    proto::expression::ExpressionString {
                        expression: "leaf".into(),
                    },
                )),
                ..Default::default()
            };
            let mut expression = leaf();
            for level in 0..600 {
                let expr_type = match level % 5 {
                    0 => proto::expression::ExprType::MergeAction(Box::new(proto::MergeAction {
                        condition: Some(Box::new(expression)),
                        ..Default::default()
                    })),
                    1 => proto::expression::ExprType::MergeAction(Box::new(proto::MergeAction {
                        assignments: vec![proto::merge_action::Assignment {
                            key: Some(expression),
                            value: Some(leaf()),
                        }],
                        ..Default::default()
                    })),
                    2 => proto::expression::ExprType::MergeAction(Box::new(proto::MergeAction {
                        assignments: vec![proto::merge_action::Assignment {
                            key: Some(leaf()),
                            value: Some(expression),
                        }],
                        ..Default::default()
                    })),
                    position => {
                        let boundary = Some(Box::new(
                            proto::expression::window::window_frame::FrameBoundary {
                                boundary: Some(Boundary::Value(Box::new(expression))),
                            },
                        ));
                        let mut frame = proto::expression::window::WindowFrame::default();
                        if position == 3 {
                            frame.lower = boundary;
                        } else {
                            frame.upper = boundary;
                        }
                        proto::expression::ExprType::Window(Box::new(proto::expression::Window {
                            frame_spec: Some(Box::new(frame)),
                            ..Default::default()
                        }))
                    }
                };
                expression = proto::Expression {
                    expr_type: Some(expr_type),
                    ..Default::default()
                };
            }
            assert_sent_clone_entry_points(expression, "merge and frame expressions");
        })
        .unwrap()
        .join()
        .unwrap();
}
