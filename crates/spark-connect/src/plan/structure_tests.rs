//! Structural plan invariants and wire metadata that normalized goldens cannot check.

use std::sync::Arc;

use crate::column::col;
use crate::expression::{Expression, LiteralExpression};
use crate::plan::{
    self, AsOfJoinDirection, AsOfJoinType, JoinType, LogicalPlan, LogicalPlanKind, OpaqueRelation,
    PlanRebuildError, RelationOwner,
};
use crate::types::DataType;
use crate::udf::{eval_type, CommonInlineUserDefinedFunctionExpression, PythonUDFPayload};
use spark_connect_proto as proto;

fn leaf(end: i64) -> Arc<LogicalPlan> {
    Arc::new(plan::range(0, end, 1))
}

fn build(kind: LogicalPlanKind, children: Vec<Arc<LogicalPlan>>) -> LogicalPlan {
    LogicalPlan::try_new(kind, children).expect("valid plan structure")
}

fn join_kind() -> LogicalPlanKind {
    LogicalPlanKind::Join {
        join_type: JoinType::Inner,
        on: None,
        using_columns: vec![],
    }
}

fn udf() -> CommonInlineUserDefinedFunctionExpression {
    CommonInlineUserDefinedFunctionExpression::new(
        "grouped".to_string(),
        true,
        vec![],
        PythonUDFPayload::new(
            DataType::Integer,
            eval_type::SQL_GROUPED_MAP_PANDAS_UDF,
            vec![1, 2, 3],
            "3.11".to_string(),
        ),
    )
}

fn group_map_kind() -> LogicalPlanKind {
    LogicalPlanKind::GroupMap {
        grouping_expressions: vec![],
        func: udf(),
        sorting_expressions: vec![],
        initial_grouping_expressions: vec![],
        is_map_groups_with_state: None,
        output_mode: None,
        timeout_conf: None,
        state_schema: None,
        transform_with_state_info: None,
    }
}

#[allow(deprecated)]
fn common(plan_id: i64) -> proto::RelationCommon {
    proto::RelationCommon {
        source_info: format!("node-{plan_id}"),
        plan_id: Some(plan_id),
        ..Default::default()
    }
}

fn range_end(relation: &proto::Relation) -> i64 {
    match relation.rel_type.as_ref() {
        Some(proto::relation::RelType::Range(range)) => range.end,
        other => panic!("expected a sentinel Range, got {other:?}"),
    }
}

fn wire_input_range_ends(relation: &proto::Relation) -> Vec<i64> {
    use proto::relation::RelType::*;

    // Inspect the protobuf's input fields, independently of the AST children.
    // The root's own Range payload is not a child sentinel.
    let inputs = match relation
        .rel_type
        .as_ref()
        .expect("serialized relation type")
    {
        Range(_)
        | Sql(_)
        | LocalRelation(_)
        | CachedRemoteRelation(_)
        | ChunkedCachedLocalRelation(_)
        | Read(_)
        | RelationChanges(_)
        | UnresolvedTableValuedFunction(_)
        | CommonInlineUserDefinedTableFunction(_)
        | Catalog(_) => vec![],
        Project(op) => vec![op.input.as_deref()],
        Filter(op) => vec![op.input.as_deref()],
        Aggregate(op) => vec![op.input.as_deref()],
        Join(op) => vec![op.left.as_deref(), op.right.as_deref()],
        LateralJoin(op) => vec![op.left.as_deref(), op.right.as_deref()],
        SetOp(op) => vec![op.left_input.as_deref(), op.right_input.as_deref()],
        Limit(op) => vec![op.input.as_deref()],
        Offset(op) => vec![op.input.as_deref()],
        Tail(op) => vec![op.input.as_deref()],
        ShowString(op) => vec![op.input.as_deref()],
        HtmlString(op) => vec![op.input.as_deref()],
        Parse(op) => vec![op.input.as_deref()],
        ApplyInPandasWithState(op) => vec![op.input.as_deref()],
        AsOfJoin(op) => vec![op.left.as_deref(), op.right.as_deref()],
        WithRelations(op) => {
            let mut inputs = vec![op.root.as_deref()];
            inputs.extend(op.references.iter().map(Some));
            inputs
        }
        Deduplicate(op) => vec![op.input.as_deref()],
        Sort(op) => vec![op.input.as_deref()],
        Sample(op) => vec![op.input.as_deref()],
        Repartition(op) => vec![op.input.as_deref()],
        RepartitionByExpression(op) => vec![op.input.as_deref()],
        WithColumns(op) => vec![op.input.as_deref()],
        WithColumnsRenamed(op) => vec![op.input.as_deref()],
        Drop(op) => vec![op.input.as_deref()],
        ToDf(op) => vec![op.input.as_deref()],
        ToSchema(op) => vec![op.input.as_deref()],
        Hint(op) => vec![op.input.as_deref()],
        Unpivot(op) => vec![op.input.as_deref()],
        FillNa(op) => vec![op.input.as_deref()],
        DropNa(op) => vec![op.input.as_deref()],
        Replace(op) => vec![op.input.as_deref()],
        Describe(op) => vec![op.input.as_deref()],
        Summary(op) => vec![op.input.as_deref()],
        SubqueryAlias(op) => vec![op.input.as_deref()],
        WithWatermark(op) => vec![op.input.as_deref()],
        Crosstab(op) => vec![op.input.as_deref()],
        FreqItems(op) => vec![op.input.as_deref()],
        ApproxQuantile(op) => vec![op.input.as_deref()],
        Corr(op) => vec![op.input.as_deref()],
        Cov(op) => vec![op.input.as_deref()],
        SampleBy(op) => vec![op.input.as_deref()],
        CollectMetrics(op) => vec![op.input.as_deref()],
        Zip(op) => vec![op.left.as_deref(), op.right.as_deref()],
        Transpose(op) => vec![op.input.as_deref()],
        MapPartitions(op) => vec![op.input.as_deref()],
        GroupMap(op) => vec![op.input.as_deref(), op.initial_input.as_deref()],
        CoGroupMap(op) => vec![op.input.as_deref(), op.other.as_deref()],
        NearestByJoin(op) => vec![op.left.as_deref(), op.right.as_deref()],
        other => panic!("unexpected wire operator in structural fixtures: {other:?}"),
    };
    inputs.into_iter().flatten().map(range_end).collect()
}

#[test]
fn construction_rejects_invalid_leaf_unary_binary_and_optional_child_counts() {
    let cases = [
        (plan::range(0, 1, 1).kind().clone(), vec![0], vec![1, 2]),
        (
            LogicalPlanKind::Project {
                columns: vec![col("id")],
            },
            vec![1],
            vec![0, 2],
        ),
        (join_kind(), vec![2], vec![0, 1, 3]),
        (group_map_kind(), vec![1, 2], vec![0, 3]),
    ];
    for (kind, valid_counts, invalid_counts) in cases {
        for count in valid_counts {
            let children = (0..count).map(|i| leaf(i + 1));
            assert!(
                LogicalPlan::try_new(kind.clone(), children).is_ok(),
                "{kind:?} must accept {count} inputs"
            );
        }
        for count in invalid_counts {
            let children = (0..count).map(|i| leaf(i + 1));
            assert!(
                LogicalPlan::try_new(kind.clone(), children).is_err(),
                "{kind:?} must reject {count} inputs"
            );
        }
    }
}

// The match has no wildcard, so adding a LogicalPlanKind variant requires a
// fixture and its independently stated input contract in the same declaration.
macro_rules! operator_fixtures {
    ($($variant:ident => ($constructor:expr, $expected:expr $(,)?)),+ $(,)?) => {{
        #[deny(unreachable_patterns)]
        fn expected_arity(kind: &LogicalPlanKind) -> std::ops::RangeInclusive<usize> {
            match kind {
                $(LogicalPlanKind::$variant { .. } => $expected,)+
            }
        }
        (vec![$(($constructor, $expected, stringify!($variant))),+], expected_arity)
    }};
}

#[test]
fn construction_and_rebuilding_accept_arrays_and_iterators_in_input_order() {
    let left = leaf(11);
    let right = leaf(22);
    let from_array =
        LogicalPlan::try_new(join_kind(), [Arc::clone(&left), Arc::clone(&right)]).unwrap();
    let from_iterator = LogicalPlan::try_new(
        join_kind(),
        [Arc::clone(&left), Arc::clone(&right)].into_iter().rev(),
    )
    .unwrap();
    assert_eq!(wire_input_range_ends(&from_array.to_proto()), [11, 22]);
    assert_eq!(wire_input_range_ends(&from_iterator.to_proto()), [22, 11]);

    let rebuilt_from_iterator = from_array
        .with_children(from_iterator.children().iter().cloned())
        .unwrap();
    let rebuilt_from_array = from_iterator
        .with_children([Arc::clone(&left), Arc::clone(&right)])
        .unwrap();
    assert_eq!(
        wire_input_range_ends(&rebuilt_from_iterator.to_proto()),
        [22, 11]
    );
    assert_eq!(
        wire_input_range_ends(&rebuilt_from_array.to_proto()),
        [11, 22]
    );
    assert!(Arc::ptr_eq(&rebuilt_from_iterator.children()[0], &right));
    assert!(Arc::ptr_eq(&rebuilt_from_array.children()[0], &left));
    assert!(LogicalPlan::try_new(join_kind(), [leaf(1)]).is_err());
    assert!(from_array.with_children(std::iter::empty()).is_err());
}

#[test]
fn every_operator_arity_agrees_with_construction_and_serialization() {
    use crate::plan::{AggregateGroupType, LogicalPlanKind::*, ParseFormat, SetOpType};
    use crate::readwriter::ReadType;
    use crate::row::Value;

    // These expected ranges describe the protocol inputs independently of the
    // implementation's arity table, so both over- and under-counting are caught.
    let (cases, expected_arity) = operator_fixtures! {
        Range => (
            Range {
                start: 0,
                end: 1,
                step: 1,
                num_partitions: None,
            },
            0..=0,
        ),
        Sql => (
            Sql {
                query: "SELECT 1".into(),
                pos_args: vec![],
                named_args: Default::default(),
            },
            0..=0,
        ),
        Project => (
            Project {
                columns: vec![col("id")],
            },
            1..=1,
        ),
        Filter => (
            Filter {
                condition: col("id"),
            },
            1..=1,
        ),
        Aggregate => (
            Aggregate {
                group_type: AggregateGroupType::GroupBy,
                grouping_expressions: vec![],
                aggregate_expressions: vec![],
                pivot_col: None,
                pivot_values: vec![],
                grouping_sets: vec![],
            },
            1..=1,
        ),
        Join => (join_kind(), 2..=2),
        LateralJoin => (
            LateralJoin {
                join_type: JoinType::Inner,
                on: None,
            },
            2..=2,
        ),
        SetOperation => (
            SetOperation {
                set_op_type: SetOpType::Union,
                is_all: true,
                by_name: false,
                allow_missing_columns: false,
            },
            2..=2,
        ),
        Limit => (Limit { limit: 1 }, 1..=1),
        Offset => (Offset { offset: 1 }, 1..=1),
        Tail => (Tail { limit: 1 }, 1..=1),
        ShowString => (
            ShowString {
                num_rows: 1,
                truncate: 20,
                vertical: false,
            },
            1..=1,
        ),
        HtmlString => (
            HtmlString {
                num_rows: 1,
                truncate: 20,
            },
            1..=1,
        ),
        Raw => (
            Raw {
                relation: OpaqueRelation::new(plan::range(0, 99, 1).to_proto()),
                owner: None,
                depth: 1,
                observations: Vec::new(),
            },
            0..=0,
        ),
        Deduplicate => (
            Deduplicate {
                all_columns_as_keys: true,
                column_names: vec![],
                within_watermark: false,
            },
            1..=1,
        ),
        Sort => (
            Sort {
                order: vec![],
                is_global: true,
            },
            1..=1,
        ),
        Sample => (
            Sample {
                lower_bound: 0.0,
                upper_bound: 0.5,
                with_replacement: false,
                seed: None,
                deterministic_order: false,
            },
            1..=1,
        ),
        Repartition => (
            Repartition {
                num_partitions: 1,
                shuffle: true,
            },
            1..=1,
        ),
        RepartitionByExpression => (
            RepartitionByExpression {
                num_partitions: 1,
                expressions: vec![],
            },
            1..=1,
        ),
        WithColumns => (
            WithColumns {
                column_names: vec![],
                columns: vec![],
            },
            1..=1,
        ),
        WithColumnMetadata => (
            WithColumnMetadata {
                column_name: "id".into(),
                column: Expression::column("id"),
                metadata_json: "{}".into(),
            },
            1..=1,
        ),
        WithColumnsRenamed => (
            WithColumnsRenamed {
                renames: Default::default(),
            },
            1..=1,
        ),
        Drop => (
            Drop {
                columns: vec![],
                expressions: vec![],
            },
            1..=1,
        ),
        ToDF => (
            ToDF {
                column_names: vec![],
            },
            1..=1,
        ),
        ToSchema => (
            ToSchema {
                schema: DataType::Struct { fields: vec![] },
            },
            1..=1,
        ),
        Hint => (
            Hint {
                name: "COALESCE".into(),
                parameters: vec![Expression::literal(LiteralExpression::Integer(1))],
            },
            1..=1,
        ),
        Parse => (
            Parse {
                format: ParseFormat::Json,
                schema: None,
                options: Default::default(),
            },
            1..=1,
        ),
        Unpivot => (
            Unpivot {
                ids: vec![],
                values: None,
                variable_column_name: "variable".into(),
                value_column_name: "value".into(),
            },
            1..=1,
        ),
        NAFill => (
            NAFill {
                fill_value: Value::Integer(0),
                columns: vec![],
            },
            1..=1,
        ),
        NAFillColumns => (
            NAFillColumns {
                cols: vec![],
                values: vec![],
            },
            1..=1,
        ),
        NADrop => (
            NADrop {
                how: "any".into(),
                min_non_null: None,
                columns: vec![],
            },
            1..=1,
        ),
        NAReplace => (
            NAReplace {
                replacements: vec![],
                columns: vec![],
            },
            1..=1,
        ),
        Describe => (Describe { columns: vec![] }, 1..=1),
        Summary => (
            Summary {
                percentiles: vec![],
            },
            1..=1,
        ),
        ColRegex => (
            ColRegex {
                col_name: "id".into(),
            },
            1..=1,
        ),
        SubqueryAlias => (SubqueryAlias { alias: "t".into() }, 1..=1),
        WithRelations => (WithRelations, 1..=usize::MAX),
        LocalRelation => (
            LocalRelation {
                schema: Some(DataType::Struct { fields: vec![] }),
                data: None,
            },
            0..=0,
        ),
        ChunkedCachedLocalRelation => (
            ChunkedCachedLocalRelation {
                data_hashes: vec!["data".into()],
                schema_hash: None,
            },
            0..=0,
        ),
        CachedRelation => (
            CachedRelation {
                relation: OpaqueRelation::new(plan::range(0, 99, 1).to_proto()),
            },
            0..=0,
        ),
        CachedRemoteRelation => (
            CachedRemoteRelation {
                relation_id: "cached".into(),
                owner: None,
            },
            0..=0,
        ),
        Read => (
            Read {
                read_type: ReadType::NamedTable {
                    table_name: "t".into(),
                    options: Default::default(),
                },
                is_streaming: false,
            },
            0..=0,
        ),
        RelationChanges => (
            RelationChanges {
                table_name: "t".into(),
                options: Default::default(),
                is_streaming: None,
            },
            0..=0,
        ),
        WithWatermark => (
            WithWatermark {
                time_column: "ts".into(),
                delay_threshold: "1 minute".into(),
            },
            1..=1,
        ),
        RepartitionByRange => (
            RepartitionByRange {
                num_partitions: None,
                partition_exprs: vec![],
            },
            1..=1,
        ),
        StatCrosstab => (
            StatCrosstab {
                col1: "a".into(),
                col2: "b".into(),
            },
            1..=1,
        ),
        StatFreqItems => (
            StatFreqItems {
                columns: vec![],
                support: 0.1,
            },
            1..=1,
        ),
        StatApproxQuantile => (
            StatApproxQuantile {
                columns: vec![],
                probabilities: vec![],
                relative_error: 0.1,
            },
            1..=1,
        ),
        StatCorr => (
            StatCorr {
                col1: "a".into(),
                col2: "b".into(),
            },
            1..=1,
        ),
        StatCov => (
            StatCov {
                col1: "a".into(),
                col2: "b".into(),
            },
            1..=1,
        ),
        StatSampleBy => (
            StatSampleBy {
                col: Expression::column("id"),
                fractions: vec![],
                seed: None,
            },
            1..=1,
        ),
        Observe => (
            Observe {
                name: "metrics".into(),
                exprs: vec![],
                observation: None,
            },
            1..=1,
        ),
        UnresolvedTableValuedFunction => (
            UnresolvedTableValuedFunction {
                name: "range".into(),
                arguments: vec![],
            },
            0..=0,
        ),
        Zip => (Zip, 2..=2),
        MlTransform => (
            MlTransform {
                ml_relation: OpaqueRelation::new(plan::range(0, 1, 1).to_proto()),
                owner: None,
            },
            1..=1,
        ),
        Catalog => (
            Catalog {
                catalog: proto::Catalog {
                    cat_type: Some(proto::catalog::CatType::CurrentDatabase(
                        proto::CurrentDatabase::default(),
                    )),
                },
            },
            0..=0,
        ),
        Transpose => (
            Transpose {
                index_columns: vec![],
            },
            1..=1,
        ),
        MapPartitions => (
            MapPartitions {
                func: udf(),
                is_barrier: false,
                profile_id: None,
            },
            1..=1,
        ),
        GroupMap => (group_map_kind(), 1..=2),
        ApplyInPandasWithState => (
            ApplyInPandasWithState {
                grouping_expressions: vec![],
                func: udf(),
                output_schema: DataType::Struct { fields: vec![] },
                state_schema: DataType::Struct { fields: vec![] },
                output_mode: "append".into(),
                timeout_conf: "NoTimeout".into(),
            },
            1..=1,
        ),
        AsOfJoin => (
            AsOfJoin {
                left_as_of: col("id"),
                right_as_of: col("id"),
                on: None,
                using_columns: vec![],
                join_type: AsOfJoinType::Inner,
                tolerance: None,
                allow_exact_matches: true,
                direction: AsOfJoinDirection::Backward,
            },
            2..=2,
        ),
        CoGroupMap => (
            CoGroupMap {
                input_grouping_expressions: vec![],
                other_grouping_expressions: vec![],
                func: udf(),
            },
            2..=2,
        ),
        NearestByJoin => (
            NearestByJoin {
                ranking_expression: col("id").expression().clone(),
                num_results: 1,
                join_type: "inner".into(),
                mode: "brute_force".into(),
                direction: "ascending".into(),
            },
            2..=2,
        ),
        CommonInlineUdtf => (
            CommonInlineUdtf {
                function_name: "udtf".into(),
                deterministic: true,
                arguments: vec![],
                return_type: Some(DataType::Struct { fields: vec![] }),
                eval_type: eval_type::SQL_TABLE_UDF,
                command: vec![],
                python_ver: "3.11".into(),
            },
            0..=0,
        ),
    };
    let names: std::collections::BTreeSet<_> =
        cases.iter().map(|(kind, _, _)| kind.name()).collect();
    assert_eq!(names.len(), cases.len(), "operator cases must be unique");

    for (kind, expected, variant) in cases {
        assert_eq!(
            kind.name(),
            variant,
            "fixture must construct its declared variant"
        );
        assert_eq!(
            expected_arity(&kind),
            expected,
            "fixture variant must match its declaration"
        );
        assert_eq!(kind.arity(), expected, "{} input contract", kind.name());
        // WithRelations is variadic; five inputs also exercise spilled child storage.
        for count in expected.clone().take(5) {
            let children = (0..count).map(|i| leaf(i as i64 + 1));
            let node = LogicalPlan::try_new(kind.clone(), children).unwrap_or_else(|error| {
                panic!("{} must accept {count} inputs: {error}", kind.name())
            });
            let relation = node.to_proto();
            assert!(
                relation.rel_type.is_some(),
                "{} must serialize",
                kind.name()
            );
            // ML wire payloads are opaque to the AST input contract, even when
            // this fixture happens to contain a leaf relation.
            if !matches!(&kind, MlTransform { .. }) {
                let expected_inputs: Vec<_> = (1..=count).map(|i| i as i64).collect();
                assert_eq!(
                    wire_input_range_ends(&relation),
                    expected_inputs,
                    "{} must preserve every wire input in order",
                    kind.name()
                );
            }
        }
        let invalid_counts = expected
            .end()
            .checked_add(1)
            .into_iter()
            .chain(expected.start().checked_sub(1));
        for count in invalid_counts {
            let children = (0..count).map(|i| leaf(i as i64 + 1));
            let error = LogicalPlan::try_new(kind.clone(), children).unwrap_err();
            assert_eq!(error.operator, kind.name());
            assert_eq!(error.expected_min, *expected.start());
            assert_eq!(
                error.expected_max,
                (*expected.end() != usize::MAX).then_some(*expected.end())
            );
            assert_eq!(error.actual, count);
        }
    }
}

#[test]
fn binary_children_keep_left_and_right_wire_positions() {
    let relation = build(join_kind(), vec![leaf(11), leaf(22)]).to_proto();
    let Some(proto::relation::RelType::Join(join)) = relation.rel_type else {
        panic!("expected Join");
    };
    assert_eq!(range_end(join.left.as_deref().unwrap()), 11);
    assert_eq!(range_end(join.right.as_deref().unwrap()), 22);
}

#[test]
fn group_map_optional_child_is_the_initial_input() {
    for initial in [false, true] {
        let mut children = vec![leaf(11)];
        if initial {
            children.push(leaf(22));
        }
        let relation = build(group_map_kind(), children).to_proto();
        let Some(proto::relation::RelType::GroupMap(group)) = relation.rel_type else {
            panic!("expected GroupMap");
        };
        assert_eq!(range_end(group.input.as_deref().unwrap()), 11);
        assert_eq!(
            group.initial_input.as_deref().map(range_end),
            initial.then_some(22)
        );
    }
}

#[test]
fn cogroup_children_keep_input_and_other_wire_positions() {
    let relation = build(
        LogicalPlanKind::CoGroupMap {
            input_grouping_expressions: vec![],
            other_grouping_expressions: vec![],
            func: udf(),
        },
        vec![leaf(11), leaf(22)],
    )
    .to_proto();
    let Some(proto::relation::RelType::CoGroupMap(group)) = relation.rel_type else {
        panic!("expected CoGroupMap");
    };
    assert_eq!(range_end(group.input.as_deref().unwrap()), 11);
    assert_eq!(range_end(group.other.as_deref().unwrap()), 22);
}

#[test]
fn clone_and_child_replacement_share_subtrees_and_keep_operator_payloads() {
    let child = leaf(11);
    let original = build(
        LogicalPlanKind::Project {
            columns: vec![col("id")],
        },
        vec![Arc::clone(&child)],
    );
    let cloned = original.clone();
    assert!(Arc::ptr_eq(&original.children()[0], &cloned.children()[0]));
    assert_eq!(cloned.to_proto(), original.to_proto());

    let replacement = leaf(22);
    let rebuilt = original.with_children([Arc::clone(&replacement)]).unwrap();
    assert!(Arc::ptr_eq(&rebuilt.children()[0], &replacement));
    assert!(Arc::ptr_eq(&original.children()[0], &child));
    let Some(proto::relation::RelType::Project(project)) = rebuilt.to_proto().rel_type else {
        panic!("expected Project");
    };
    assert_eq!(project.expressions, [col("id").to_proto()]);
    assert_eq!(range_end(project.input.as_deref().unwrap()), 22);
    assert!(original.with_children([]).is_err());
    assert!(original.with_children([leaf(1), leaf(2)]).is_err());
}

#[test]
fn into_parts_moves_payload_and_shared_inputs_without_copying() {
    let input = leaf(11);
    let columns = vec![col("id")];
    let columns_allocation = columns.as_ptr();
    let node = build(
        LogicalPlanKind::Project { columns },
        vec![Arc::clone(&input)],
    );
    let wire = node.to_proto();

    let (kind, children) = node.into_parts();
    let LogicalPlanKind::Project { columns } = &kind else {
        panic!("expected Project payload");
    };
    assert_eq!(columns.as_ptr(), columns_allocation);
    assert_eq!(children.len(), 1);
    assert!(Arc::ptr_eq(&children[0], &input));
    assert_eq!(Arc::strong_count(&input), 2);

    let rebuilt = LogicalPlan::try_new(kind, children).unwrap();
    assert_eq!(rebuilt.to_proto(), wire);
    assert!(Arc::ptr_eq(&rebuilt.children()[0], &input));
}

#[test]
fn into_parts_preserves_opaque_allocation_owner_and_wire_payload() {
    let input = leaf(11);
    let owner = Arc::new(String::from("cached model"));
    let weak_owner = Arc::downgrade(&owner);
    let wire = plan::limit(plan::range(0, 99, 1), 2).to_proto();
    let opaque = OpaqueRelation::new(wire.clone());
    let relation_allocation = opaque.as_proto() as *const proto::Relation;
    let node = build(
        LogicalPlanKind::MlTransform {
            ml_relation: opaque,
            owner: Some(RelationOwner(owner.clone())),
        },
        vec![Arc::clone(&input)],
    );
    drop(owner);

    let (kind, children) = node.into_parts();
    let LogicalPlanKind::MlTransform { ml_relation, owner } = &kind else {
        panic!("expected MlTransform payload");
    };
    assert_eq!(
        ml_relation.as_proto() as *const proto::Relation,
        relation_allocation
    );
    assert!(owner.is_some());
    assert!(weak_owner.upgrade().is_some());
    assert_eq!(children.len(), 1);
    assert!(Arc::ptr_eq(&children[0], &input));

    let rebuilt = LogicalPlan::try_new(kind, children).unwrap();
    assert_eq!(rebuilt.to_proto(), wire);
    assert!(Arc::ptr_eq(&rebuilt.children()[0], &input));
    drop(rebuilt);
    assert!(weak_owner.upgrade().is_none());
}

#[test]
fn into_parts_keeps_deep_input_destruction_iterative() {
    std::thread::Builder::new()
        .stack_size(512 << 10)
        .spawn(|| {
            const LAYERS: usize = 20_000;
            let input = leaf(1);
            let weak_input = Arc::downgrade(&input);
            let mut root = plan::limit(input, 1);
            for _ in 1..LAYERS {
                root = plan::limit(Arc::new(root), 1);
            }
            assert_eq!(root.depth(), LAYERS + 1);

            let (kind, children) = root.into_parts();
            assert!(matches!(&kind, LogicalPlanKind::Limit { .. }));
            assert_eq!(children.len(), 1);
            assert!(weak_input.upgrade().is_some());
            drop(kind);
            drop(children);
            assert!(weak_input.upgrade().is_none());
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn ml_rebuilding_accepts_only_its_existing_opaque_input() {
    let owner = Arc::new(String::from("cached model"));
    let weak_owner = Arc::downgrade(&owner);
    let child = Arc::new(plan::limit(leaf(11), 2));
    let wire = plan::limit(plan::range(0, 99, 1), 1).to_proto();
    let original = Arc::new(build(
        LogicalPlanKind::MlTransform {
            ml_relation: OpaqueRelation::new(wire.clone()),
            owner: Some(RelationOwner(owner.clone())),
        },
        vec![Arc::clone(&child)],
    ));

    let rebuilt = Arc::new(original.with_children(vec![Arc::clone(&child)]).unwrap());
    assert!(!Arc::ptr_eq(&original, &rebuilt));
    assert!(Arc::ptr_eq(&rebuilt.children()[0], &child));
    assert_eq!(rebuilt.depth(), 3);
    assert_eq!(rebuilt.to_proto(), wire);

    // An equivalent owned node has another allocation; replacing it would not
    // update the opaque wire relation.
    let equivalent_child = Arc::new(child.as_ref().clone());
    for replacement in [equivalent_child, leaf(22)] {
        assert!(matches!(
            original.with_children(vec![replacement]),
            Err(PlanRebuildError::OpaqueInputReplacement {
                operator: "MlTransform",
                ..
            })
        ));
    }
    let Err(PlanRebuildError::InvalidArity(error)) = original.with_children(vec![]) else {
        panic!("ML rebuilding must still validate input count");
    };
    assert_eq!(error.operator, "MlTransform");
    assert_eq!(error.actual, 0);

    drop(owner);
    drop(original);
    assert!(weak_owner.upgrade().is_some());
    drop(rebuilt);
    assert!(weak_owner.upgrade().is_none());
}

#[test]
fn visit_is_preorder_and_follows_structural_child_order() {
    let left = Arc::new(build(
        LogicalPlanKind::Project {
            columns: vec![col("id")],
        },
        vec![leaf(11)],
    ));
    let root = build(join_kind(), vec![left, leaf(22)]);
    let mut visited = vec![];
    root.visit(|node| {
        visited.push(match node.kind() {
            LogicalPlanKind::Join { .. } => "join".to_string(),
            LogicalPlanKind::Project { .. } => "project".to_string(),
            LogicalPlanKind::Range { end, .. } => format!("range-{end}"),
            other => panic!("unexpected node {other:?}"),
        });
    });
    assert_eq!(visited, ["join", "project", "range-11", "range-22"]);
}

#[test]
fn visit_uses_first_encounter_preorder_and_shared_node_identity() {
    let shared = leaf(22);
    let other = leaf(33);
    let equivalent = leaf(22);
    let left = Arc::new(build(
        join_kind(),
        vec![Arc::clone(&shared), Arc::clone(&other)],
    ));
    let right = Arc::new(build(
        LogicalPlanKind::Project {
            columns: vec![col("id")],
        },
        vec![Arc::clone(&equivalent)],
    ));
    let root = build(
        LogicalPlanKind::WithRelations,
        vec![Arc::clone(&left), Arc::clone(&shared), Arc::clone(&right)],
    );

    let mut visited = vec![];
    root.visit(|node| visited.push(node as *const LogicalPlan));
    assert_eq!(
        visited,
        [
            &root as *const LogicalPlan,
            Arc::as_ptr(&left),
            Arc::as_ptr(&shared),
            Arc::as_ptr(&other),
            Arc::as_ptr(&right),
            Arc::as_ptr(&equivalent),
        ]
    );
}

#[test]
fn visit_handles_a_deep_shared_dag_once_per_node_on_a_small_stack() {
    std::thread::Builder::new()
        .stack_size(512 << 10)
        .spawn(|| {
            const LAYERS: usize = 20_000;
            let mut root = leaf(1);
            for _ in 0..LAYERS {
                root = Arc::new(
                    LogicalPlan::try_new(join_kind(), [Arc::clone(&root), Arc::clone(&root)])
                        .unwrap(),
                );
            }
            let mut visited = std::collections::HashSet::new();
            root.visit(|node| {
                assert!(
                    visited.insert(node as *const LogicalPlan),
                    "each shared node must be visited only once"
                );
            });
            assert_eq!(visited.len(), LAYERS + 1);
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn visit_all_paths_revisits_shared_subtrees_in_preorder() {
    let input = leaf(11);
    let shared = Arc::new(plan::limit(Arc::clone(&input), 2));
    let root = build(join_kind(), vec![Arc::clone(&shared), Arc::clone(&shared)]);
    let mut visited = vec![];
    root.visit_all_paths(|node| visited.push(node as *const LogicalPlan));
    assert_eq!(
        visited,
        [
            &root as *const LogicalPlan,
            Arc::as_ptr(&shared),
            Arc::as_ptr(&input),
            Arc::as_ptr(&shared),
            Arc::as_ptr(&input),
        ]
    );
}

#[test]
fn transform_up_visits_a_diamond_once_in_child_order_and_keeps_noop_identity() {
    let shared = leaf(11);
    let equivalent = leaf(11);
    let sibling = leaf(22);
    let left = Arc::new(plan::limit(Arc::clone(&shared), 2));
    let right = Arc::new(build(
        join_kind(),
        vec![Arc::clone(&shared), Arc::clone(&equivalent)],
    ));
    let root = Arc::new(build(
        LogicalPlanKind::WithRelations,
        vec![Arc::clone(&left), Arc::clone(&right), Arc::clone(&sibling)],
    ));

    let mut visited = vec![];
    let transformed = root
        .try_transform_up(|node| {
            visited.push(Arc::as_ptr(node));
            Ok::<_, PlanRebuildError>(Arc::clone(node))
        })
        .unwrap();
    assert_eq!(
        visited,
        [
            Arc::as_ptr(&shared),
            Arc::as_ptr(&left),
            Arc::as_ptr(&equivalent),
            Arc::as_ptr(&right),
            Arc::as_ptr(&sibling),
            Arc::as_ptr(&root),
        ]
    );
    assert!(Arc::ptr_eq(&transformed, &root));
    assert!(Arc::ptr_eq(&transformed.children()[0], &left));
    assert!(Arc::ptr_eq(&transformed.children()[1], &right));
    assert!(Arc::ptr_eq(&transformed.children()[2], &sibling));
}

#[test]
fn transform_up_resolves_cross_edge_dependencies_in_both_input_orders() {
    for leaf_first in [true, false] {
        let shared = leaf(11);
        let parent = Arc::new(plan::limit(Arc::clone(&shared), 2));
        let children = if leaf_first {
            vec![Arc::clone(&shared), Arc::clone(&parent)]
        } else {
            vec![Arc::clone(&parent), Arc::clone(&shared)]
        };
        let root = Arc::new(build(join_kind(), children));
        let replacement = leaf(33);
        let mut visited = vec![];
        let transformed = root
            .try_transform_up(|node| {
                match node.kind() {
                    LogicalPlanKind::Range { end, .. } => {
                        assert_eq!(*end, 11);
                        assert!(Arc::ptr_eq(node, &shared));
                        visited.push("range");
                        return Ok::<_, PlanRebuildError>(Arc::clone(&replacement));
                    }
                    LogicalPlanKind::Limit { limit } => {
                        assert_eq!(*limit, 2);
                        assert!(Arc::ptr_eq(&node.children()[0], &replacement));
                        visited.push("limit");
                    }
                    LogicalPlanKind::Join { .. } => visited.push("join"),
                    other => panic!("unexpected rewritten operator: {other:?}"),
                }
                Ok(Arc::clone(node))
            })
            .unwrap();
        assert_eq!(visited, ["range", "limit", "join"]);
        let (leaf_index, parent_index) = if leaf_first { (0, 1) } else { (1, 0) };
        assert!(Arc::ptr_eq(
            &transformed.children()[leaf_index],
            &replacement
        ));
        let transformed_parent = &transformed.children()[parent_index];
        assert!(!Arc::ptr_eq(transformed_parent, &parent));
        assert!(Arc::ptr_eq(&transformed_parent.children()[0], &replacement));
        assert!(Arc::ptr_eq(&parent.children()[0], &shared));
    }
}

#[test]
fn transform_up_shares_replacements_and_keeps_unchanged_inputs() {
    let shared = leaf(11);
    let sibling = leaf(22);
    let left = Arc::new(build(
        join_kind(),
        vec![Arc::clone(&shared), Arc::clone(&sibling)],
    ));
    let right = Arc::new(plan::limit(Arc::clone(&shared), 2));
    let root = Arc::new(build(
        join_kind(),
        vec![Arc::clone(&left), Arc::clone(&right)],
    ));
    let replacement = Arc::new(plan::offset(leaf(99), 7));
    let mut visited = vec![];
    let transformed = root
        .try_transform_up(|node| {
            match node.kind() {
                LogicalPlanKind::Range { end: 11, .. } => {
                    assert!(Arc::ptr_eq(node, &shared));
                    visited.push("shared");
                    return Ok::<_, PlanRebuildError>(Arc::clone(&replacement));
                }
                LogicalPlanKind::Range { end: 22, .. } => {
                    assert!(Arc::ptr_eq(node, &sibling));
                    visited.push("sibling");
                }
                LogicalPlanKind::Join { .. } => visited.push("join"),
                LogicalPlanKind::Limit { .. } => {
                    assert!(Arc::ptr_eq(&node.children()[0], &replacement));
                    visited.push("limit");
                }
                other => panic!("rule replacements must not be revisited: {other:?}"),
            }
            Ok(Arc::clone(node))
        })
        .unwrap();
    assert_eq!(visited, ["shared", "sibling", "join", "limit", "join"]);
    assert!(!Arc::ptr_eq(&transformed, &root));
    let transformed_left = &transformed.children()[0];
    let transformed_right = &transformed.children()[1];
    assert!(!Arc::ptr_eq(transformed_left, &left));
    assert!(!Arc::ptr_eq(transformed_right, &right));
    assert!(Arc::ptr_eq(&transformed_left.children()[0], &replacement));
    assert!(Arc::ptr_eq(&transformed_right.children()[0], &replacement));
    assert!(Arc::ptr_eq(&transformed_left.children()[1], &sibling));
    assert!(Arc::ptr_eq(&left.children()[0], &shared));
    assert!(Arc::ptr_eq(&right.children()[0], &shared));
}

#[test]
fn transform_up_returns_callback_errors_without_visiting_later_nodes() {
    let first = leaf(11);
    let later = leaf(22);
    let root = Arc::new(build(
        join_kind(),
        vec![Arc::clone(&first), Arc::clone(&later)],
    ));
    let mut visited = vec![];
    let result: Result<_, Box<dyn std::error::Error>> = root.try_transform_up(|node| {
        visited.push(Arc::as_ptr(node));
        Err(std::io::Error::other("stop rewriting").into())
    });
    let error = result.unwrap_err();
    assert_eq!(
        error.downcast_ref::<std::io::Error>().unwrap().to_string(),
        "stop rewriting"
    );
    assert_eq!(visited, [Arc::as_ptr(&first)]);
    assert!(Arc::ptr_eq(&root.children()[0], &first));
    assert!(Arc::ptr_eq(&root.children()[1], &later));
}

#[test]
fn transform_up_preserves_opaque_ml_inputs_or_reports_replacement() {
    let input = leaf(11);
    let ml = Arc::new(build(
        LogicalPlanKind::MlTransform {
            ml_relation: OpaqueRelation::new(input.to_proto()),
            owner: None,
        },
        vec![Arc::clone(&input)],
    ));
    let root = Arc::new(plan::limit(Arc::clone(&ml), 2));
    let mut visited = vec![];
    let unchanged = root
        .try_transform_up(|node| {
            visited.push(Arc::as_ptr(node));
            Ok::<_, PlanRebuildError>(Arc::clone(node))
        })
        .unwrap();
    assert!(Arc::ptr_eq(&unchanged, &root));
    assert_eq!(
        visited,
        [Arc::as_ptr(&input), Arc::as_ptr(&ml), Arc::as_ptr(&root)]
    );

    visited.clear();
    let replacement = leaf(22);
    let result: Result<_, Box<dyn std::error::Error>> = root.try_transform_up(|node| {
        visited.push(Arc::as_ptr(node));
        Ok(if Arc::ptr_eq(node, &input) {
            Arc::clone(&replacement)
        } else {
            Arc::clone(node)
        })
    });
    let error = result.unwrap_err();
    assert_eq!(
        error.downcast_ref::<PlanRebuildError>(),
        Some(&PlanRebuildError::OpaqueInputReplacement {
            operator: "MlTransform",
        })
    );
    assert_eq!(visited, [Arc::as_ptr(&input)]);
    assert!(Arc::ptr_eq(&ml.children()[0], &input));
    assert!(Arc::ptr_eq(&root.children()[0], &ml));
}

#[test]
fn transform_up_can_replace_a_whole_ml_node_when_its_input_is_unchanged() {
    let input = leaf(11);
    let ml = Arc::new(build(
        LogicalPlanKind::MlTransform {
            ml_relation: OpaqueRelation::new(input.to_proto()),
            owner: None,
        },
        vec![Arc::clone(&input)],
    ));
    let root = Arc::new(plan::limit(Arc::clone(&ml), 2));
    let replacement = leaf(22);
    let mut visited = vec![];
    let rewritten = root
        .try_transform_up::<PlanRebuildError>(|node| {
            visited.push(node.kind().name());
            Ok(if Arc::ptr_eq(node, &ml) {
                Arc::clone(&replacement)
            } else {
                Arc::clone(node)
            })
        })
        .unwrap();

    assert_eq!(visited, ["Range", "MlTransform", "Limit"]);
    assert!(!Arc::ptr_eq(&rewritten, &root));
    assert!(Arc::ptr_eq(&rewritten.children()[0], &replacement));
    assert!(Arc::ptr_eq(&ml.children()[0], &input));
    assert!(Arc::ptr_eq(&root.children()[0], &ml));
    let Some(proto::relation::RelType::Limit(limit)) = rewritten.to_proto().rel_type else {
        panic!("expected Limit");
    };
    assert_eq!(range_end(limit.input.as_deref().unwrap()), 22);
}

#[test]
fn transform_up_handles_a_deep_shared_dag_on_a_small_stack() {
    std::thread::Builder::new()
        .stack_size(512 << 10)
        .spawn(|| {
            const LAYERS: usize = 20_000;
            let input = leaf(1);
            let mut root = Arc::clone(&input);
            for _ in 0..LAYERS {
                root = Arc::new(
                    LogicalPlan::try_new(join_kind(), [Arc::clone(&root), Arc::clone(&root)])
                        .unwrap(),
                );
            }
            let replacement = leaf(2);
            for replace_input in [false, true] {
                let mut visits = 0;
                let transformed = root
                    .try_transform_up(|node| {
                        visits += 1;
                        Ok::<_, PlanRebuildError>(if replace_input && Arc::ptr_eq(node, &input) {
                            Arc::clone(&replacement)
                        } else {
                            Arc::clone(node)
                        })
                    })
                    .unwrap();
                assert_eq!(visits, LAYERS + 1);
                assert_eq!(Arc::ptr_eq(&transformed, &root), !replace_input);
                assert_eq!(transformed.depth(), LAYERS + 1);
                let mut transformed_visits = 0;
                transformed.visit(|node| {
                    transformed_visits += 1;
                    match node.kind() {
                        LogicalPlanKind::Join { .. } => {
                            assert!(Arc::ptr_eq(&node.children()[0], &node.children()[1]));
                        }
                        LogicalPlanKind::Range { end, .. } => {
                            assert_eq!(*end, if replace_input { 2 } else { 1 });
                        }
                        other => panic!("unexpected rewritten operator: {other:?}"),
                    }
                });
                assert_eq!(transformed_visits, LAYERS + 1);
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn opaque_relations_preserve_operator_wire_metadata_defaults() {
    for original_common in [
        None,
        Some(proto::RelationCommon::default()),
        Some(common(7)),
    ] {
        let mut raw = plan::range(0, 99, 1).to_proto();
        raw.common = original_common;
        let nodes = [
            (
                build(
                    LogicalPlanKind::Raw {
                        relation: OpaqueRelation::new(raw.clone()),
                        owner: None,
                        depth: 1,
                        observations: Vec::new(),
                    },
                    vec![],
                ),
                Some(raw.common.clone().unwrap_or_default()),
            ),
            (
                build(
                    LogicalPlanKind::CachedRelation {
                        relation: OpaqueRelation::new(raw.clone()),
                    },
                    vec![],
                ),
                Some(proto::RelationCommon::default()),
            ),
            (
                build(
                    LogicalPlanKind::MlTransform {
                        ml_relation: OpaqueRelation::new(raw.clone()),
                        owner: None,
                    },
                    vec![leaf(11)],
                ),
                raw.common.clone(),
            ),
        ];
        for (node, expected_common) in nodes {
            let mut baseline = raw.clone();
            baseline.common = expected_common;
            assert_eq!(node.to_proto(), baseline);
        }
    }
}

#[test]
fn variadic_inputs_keep_root_and_references_in_order_and_visit_each_edge() {
    let shared = leaf(22);
    let node = build(
        LogicalPlanKind::WithRelations,
        vec![
            leaf(11),
            Arc::clone(&shared),
            leaf(33),
            Arc::clone(&shared),
            leaf(44),
        ],
    );
    assert_eq!(
        wire_input_range_ends(&node.to_proto()),
        [11, 22, 33, 22, 44]
    );
    let mut ranges = vec![];
    node.visit_all_paths(|node| {
        if let LogicalPlanKind::Range { end, .. } = node.kind() {
            ranges.push(*end);
        }
    });
    assert_eq!(ranges, [11, 22, 33, 22, 44]);
    let mut unique_ranges = vec![];
    node.visit(|node| {
        if let LogicalPlanKind::Range { end, .. } = node.kind() {
            unique_ranges.push(*end);
        }
    });
    assert_eq!(unique_ranges, [11, 22, 33, 44]);
    assert!(Arc::ptr_eq(&node.children()[1], &node.children()[3]));
    assert!(node.with_children(vec![]).is_err());
}

#[test]
fn rebuilding_an_opaque_plan_keeps_its_owner_and_depth() {
    let owner = Arc::new(String::from("cached model"));
    let weak = Arc::downgrade(&owner);
    let raw = plan::limit(plan::limit(plan::range(0, 99, 1), 2), 1).to_proto();
    let node = build(
        LogicalPlanKind::Raw {
            relation: OpaqueRelation::new(raw.clone()),
            owner: Some(RelationOwner(owner.clone())),
            depth: 3,
            observations: Vec::new(),
        },
        vec![],
    );
    let rebuilt = node.with_children(vec![]).unwrap();
    drop(owner);
    drop(node);
    assert!(weak.upgrade().is_some());
    assert_eq!(rebuilt.depth(), 3);
    assert_eq!(rebuilt.to_proto(), raw);
    let mut visited = 0;
    rebuilt.visit(|_| visited += 1);
    assert_eq!(visited, 1);
    drop(rebuilt);
    assert!(weak.upgrade().is_none());
}

#[test]
fn dropping_deep_shared_fanout_plans_does_not_overflow_a_small_stack() {
    std::thread::Builder::new()
        .stack_size(512 << 10)
        .spawn(|| {
            const LAYERS: usize = 20_000;
            for (kind, child_count) in [(group_map_kind(), 2), (LogicalPlanKind::WithRelations, 4)]
            {
                let mut root = leaf(1);
                let weak_leaf = Arc::downgrade(&root);
                for _ in 0..LAYERS {
                    let children = (0..child_count).map(|_| Arc::clone(&root));
                    root = Arc::new(LogicalPlan::try_new(kind.clone(), children).unwrap());
                }
                assert_eq!(root.depth(), LAYERS + 1);
                // Each layer repeats the same input. Occurrence visiting or serializing
                // these compact DAGs would unfold exponentially many paths.
                drop(root);
                assert!(weak_leaf.upgrade().is_none());
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn public_plan_errors_preserve_messages_and_causes() {
    use crate::plan::InvalidPlanArity;
    use std::error::Error as _;

    for (kind, children, expected_max, message) in [
        (
            LogicalPlanKind::Project { columns: vec![] },
            vec![],
            Some(1),
            "Project expects 1 input, got 0",
        ),
        (
            join_kind(),
            vec![leaf(1), leaf(2), leaf(3)],
            Some(2),
            "Join expects 2 inputs, got 3",
        ),
        (
            group_map_kind(),
            vec![leaf(1), leaf(2), leaf(3)],
            Some(2),
            "GroupMap expects 1..=2 inputs, got 3",
        ),
        (
            LogicalPlanKind::WithRelations,
            vec![],
            None,
            "WithRelations expects at least 1 input, got 0",
        ),
    ] {
        let error = LogicalPlan::try_new(kind, children).unwrap_err();
        assert_eq!(error.expected_max, expected_max);
        assert_eq!(error.to_string(), message);
        assert!(error.source().is_none());
    }

    let project = build(LogicalPlanKind::Project { columns: vec![] }, vec![leaf(1)]);
    let error = project.with_children([]).unwrap_err();
    assert_eq!(error.to_string(), "Project expects 1 input, got 0");
    assert_eq!(
        error.source().unwrap().downcast_ref::<InvalidPlanArity>(),
        Some(&InvalidPlanArity {
            operator: "Project",
            expected_min: 1,
            expected_max: Some(1),
            actual: 0,
        })
    );

    let input = leaf(1);
    let ml = build(
        LogicalPlanKind::MlTransform {
            ml_relation: OpaqueRelation::new(input.to_proto()),
            owner: None,
        },
        vec![input],
    );
    let error = ml.with_children([leaf(2)]).unwrap_err();
    assert_eq!(
        error.to_string(),
        "MlTransform has an opaque wire input; rebuilding requires the original shared input"
    );
    assert!(error.source().is_none());
}
