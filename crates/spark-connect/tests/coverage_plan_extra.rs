//! Golden/unit coverage for `plan.rs` paths not exercised by `plans_golden.rs`:
//! the `plan::*` builder helpers that no golden case constructs, the
//! `value_to_proto_literal` match arms (via `NAFill` with each `Value`), and the
//! `to_proto()` arms for `LateralJoin`, `RepartitionByRange`, `Read` (DataSource),
//! and `StatSampleBy`. No server: every test just builds a `LogicalPlan` and calls
//! `to_proto()`, so the proto-building lines run. Assertions are light on purpose.

use std::collections::HashMap;

use spark_connect::column::lit;
use spark_connect::plan::{self, JoinType, LogicalPlan};
use spark_connect::readwriter::ReadType;
use spark_connect::row::Value;
use spark_connect::types::DataType;

fn base() -> LogicalPlan {
    plan::range(0, 10, 1)
}

/// Exercise the `plan::*` constructor helpers that `plans_golden.rs` does not build.
#[test]
fn plan_builder_helpers_to_proto() {
    let plans = vec![
        plan::range_with_partitions(0, 10, 1, 4),
        plan::tail(base(), 5),
        plan::to_schema(base(), DataType::Struct { fields: vec![] }),
        plan::col_regex(base(), "a.*"),
        plan::sample(base(), 0.0, 0.5, true, Some(42)),
        // num_partitions <= 0 -> the `None` branch of RepartitionByExpression.
        plan::repartition_by_expression(base(), 0, vec![]),
        // num_partitions > 0 -> the `Some` branch.
        plan::repartition_by_expression(base(), 2, vec![]),
        plan::summary(base(), vec!["mean".to_string(), "stddev".to_string()]),
        plan::describe(base(), vec!["id".to_string()]),
        plan::subquery_alias(base(), "t"),
        plan::na_drop(base(), "any", Some(2), vec!["id".to_string()]),
        plan::na_replace(
            base(),
            vec![("a".to_string(), "b".to_string())],
            vec!["id".to_string()],
        ),
        plan::cached_remote_relation("rid-123"),
        plan::local_relation(DataType::Struct { fields: vec![] }, None),
        plan::hint(base(), "COALESCE", vec!["1".to_string()]),
    ];
    for p in plans {
        // Building the proto must not panic; a rel_type must be produced.
        let proto = p.to_proto();
        assert!(proto.rel_type.is_some());
    }
}

/// `value_to_proto_literal` (used by `NAFill`) has one match arm per `Value`
/// variant; drive them all through `na_fill(...).to_proto()`.
#[test]
fn nafill_covers_all_value_literal_arms() {
    let values = vec![
        Value::Bool(true),
        Value::Byte(1),
        Value::Short(2),
        Value::Integer(3),
        Value::Long(4),
        Value::Float(1.5),
        Value::Double(2.5),
        Value::String("s".to_string()),
        Value::Date(5),
        Value::Timestamp(6),
        Value::Decimal {
            value: "1.23".to_string(),
            precision: Some(5),
            scale: Some(2),
        },
        // Binary/Null fall through to the catch-all `other =>` arm.
        Value::Binary(vec![1, 2, 3]),
        Value::Null,
    ];
    for v in values {
        let proto = plan::na_fill(base(), v, vec![]).to_proto();
        assert!(proto.rel_type.is_some());
    }
}

/// `LateralJoin::to_proto` (join-type match + optional condition), for every join type.
#[test]
fn lateral_join_to_proto_all_join_types() {
    for jt in [
        JoinType::Inner,
        JoinType::LeftOuter,
        JoinType::RightOuter,
        JoinType::FullOuter,
        JoinType::LeftSemi,
        JoinType::LeftAnti,
        JoinType::Cross,
    ] {
        let p = LogicalPlan::LateralJoin {
            left: Box::new(base()),
            right: Box::new(base()),
            join_type: jt,
            on: None,
        };
        assert!(p.to_proto().rel_type.is_some());
    }
}

/// `RepartitionByRange::to_proto` with an explicit partition count (the `Some` branch).
#[test]
fn repartition_by_range_to_proto() {
    let p = LogicalPlan::RepartitionByRange {
        input: Box::new(base()),
        num_partitions: Some(4),
        partition_exprs: vec![],
    };
    assert!(p.to_proto().rel_type.is_some());
}

/// `Read::to_proto` for the DataSource read type with format/schema/source_name set.
#[test]
fn read_datasource_to_proto() {
    let p = LogicalPlan::Read {
        read_type: ReadType::DataSource {
            format: Some("parquet".to_string()),
            schema: Some("a INT, b STRING".to_string()),
            options: HashMap::new(),
            paths: vec!["/tmp/data".to_string()],
            predicates: vec![],
            source_name: Some("src".to_string()),
        },
        is_streaming: false,
    };
    assert!(p.to_proto().rel_type.is_some());
}

/// `StatSampleBy::to_proto`: builds the col expression, fraction literals, and seed.
#[test]
fn stat_sample_by_to_proto() {
    let p = LogicalPlan::StatSampleBy {
        input: Box::new(base()),
        col: "k".to_string(),
        fractions: vec![
            (lit(0).expression().clone(), 0.1),
            (lit(1).expression().clone(), 0.9),
        ],
        seed: Some(7),
    };
    assert!(p.to_proto().rel_type.is_some());
}
