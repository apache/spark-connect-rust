//! Server-free serialization checks for plan variants whose behavioral coverage would
//! otherwise only come from a live server: every JoinType maps to the correct proto
//! enum, and local/cached relations serialize to the right rel_type. These assert the
//! actual proto mapping (not a recorded snapshot) and run in CI without a server.

use spark_connect::column::col;
use spark_connect::plan::{self, JoinType, LogicalPlan};
use spark_connect::types::DataType;
use spark_connect_proto as proto;

fn base() -> LogicalPlan {
    plan::range(0, 5, 1)
}

#[test]
fn join_types_map_to_correct_proto() {
    let cases = [
        (JoinType::Inner, proto::join::JoinType::Inner),
        (JoinType::LeftOuter, proto::join::JoinType::LeftOuter),
        (JoinType::RightOuter, proto::join::JoinType::RightOuter),
        (JoinType::FullOuter, proto::join::JoinType::FullOuter),
        (JoinType::LeftSemi, proto::join::JoinType::LeftSemi),
        (JoinType::LeftAnti, proto::join::JoinType::LeftAnti),
        (JoinType::Cross, proto::join::JoinType::Cross),
    ];
    for (jt, expected) in cases {
        let rel = plan::join(base(), base(), jt, Some(col("id")), vec![]).to_proto();
        match rel.rel_type {
            Some(proto::relation::RelType::Join(j)) => {
                assert_eq!(
                    j.join_type, expected as i32,
                    "JoinType {jt:?} must serialize to proto {expected:?}"
                );
                assert!(
                    j.join_condition.is_some(),
                    "the `on` condition must be carried"
                );
            }
            other => panic!("expected Join relation, got {other:?}"),
        }
    }
    // using-columns form (no `on` condition)
    let rel = plan::join(
        base(),
        base(),
        JoinType::Inner,
        None,
        vec!["id".to_string()],
    )
    .to_proto();
    match rel.rel_type {
        Some(proto::relation::RelType::Join(j)) => {
            assert_eq!(j.using_columns, vec!["id".to_string()]);
        }
        other => panic!("expected Join, got {other:?}"),
    }
}

#[test]
fn local_and_cached_relations_serialize() {
    // LocalRelation with and without inline data.
    match plan::local_relation(DataType::Integer, None)
        .to_proto()
        .rel_type
    {
        Some(proto::relation::RelType::LocalRelation(_)) => {}
        other => panic!("expected LocalRelation, got {other:?}"),
    }
    match plan::local_relation(DataType::Integer, Some(vec![1, 2, 3]))
        .to_proto()
        .rel_type
    {
        Some(proto::relation::RelType::LocalRelation(lr)) => {
            assert_eq!(lr.data.as_deref(), Some(&[1u8, 2, 3][..]));
        }
        other => panic!("expected LocalRelation, got {other:?}"),
    }
    // CachedRemoteRelation carries the relation id.
    match plan::cached_remote_relation("rel-42").to_proto().rel_type {
        Some(proto::relation::RelType::CachedRemoteRelation(cr)) => {
            assert_eq!(cr.relation_id, "rel-42");
        }
        other => panic!("expected CachedRemoteRelation, got {other:?}"),
    }
}
