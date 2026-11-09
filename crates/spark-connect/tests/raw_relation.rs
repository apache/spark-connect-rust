// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! DataFrames of relations built outside the core, as the plans of `pyspark.ml` build theirs,
//! and the commands of messages built outside it. The tests of a server run when SPARK_REMOTE
//! is set.

use std::any::Any;
use std::sync::Arc;

use spark_connect::column::{col, lit};
use spark_connect::{DataFrame, SparkSession};
use spark_connect_proto as proto;

fn offline() -> SparkSession {
    // The channel connects lazily: building plans needs no server.
    SparkSession::builder()
        .remote("sc://127.0.0.1:1")
        .get_or_create()
        .unwrap()
}

fn live() -> Option<SparkSession> {
    let remote = std::env::var("SPARK_REMOTE").ok()?;
    Some(
        SparkSession::builder()
            .remote(&remote)
            .get_or_create()
            .unwrap(),
    )
}

/// Runs `f` on a thread with a stack of 64 MiB. A plan is converted recursively, so a deep plan
/// takes more than the 2 MiB stack of a test thread in a debug build, whose frames are much
/// larger than those of a release build.
fn on_a_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(f)
        .unwrap()
        .join()
        .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
}

fn sql_relation(query: &str) -> proto::Relation {
    proto::Relation {
        common: Some(proto::RelationCommon {
            plan_id: Some(1 << 48),
            ..Default::default()
        }),
        rel_type: Some(proto::relation::RelType::Sql(proto::Sql {
            query: query.to_string(),
            ..Default::default()
        })),
    }
}

#[test]
fn raw_relations_are_sent_as_they_are_with_a_plan_id_of_the_session() {
    let df = DataFrame::from_relation(offline(), sql_relation("SELECT 1 AS a"), None).unwrap();
    let relation = df.to_relation();
    let Some(proto::relation::RelType::Sql(sql)) = &relation.rel_type else {
        panic!("expected the SQL relation, got {relation:?}");
    };
    assert_eq!(sql.query, "SELECT 1 AS a");
    let plan_id = relation.common.as_ref().and_then(|common| common.plan_id);
    assert!(plan_id.is_some());
    // The plan ID is that of the node, as the session gives it, the same each time.
    assert_eq!(df.to_relation(), relation);
    assert_eq!(df.clone().to_relation(), relation);
}

#[test]
fn transformations_of_raw_relations_take_them_as_input() {
    let df = DataFrame::from_relation(offline(), sql_relation("SELECT 1 AS a"), None).unwrap();
    let relation = df.to_relation();
    let filtered = df.filter(col("a").gt(lit(0)));
    let Some(proto::relation::RelType::Filter(filter)) = filtered.to_relation().rel_type else {
        panic!("expected a filter");
    };
    assert_eq!(filter.input.as_deref(), Some(&relation));
}

#[test]
fn raw_relations_keep_their_owners_alive_while_a_dataframe_uses_them() {
    let owner: Arc<dyn Any + Send + Sync> = Arc::new(String::from("plan"));
    let alive = Arc::downgrade(&owner);
    let df =
        DataFrame::from_relation(offline(), sql_relation("SELECT 1 AS a"), Some(owner)).unwrap();
    let derived = df.select([col("a")]);
    drop(df);
    assert!(alive.upgrade().is_some());
    assert!(format!("{:?}", derived.to_relation()).contains("SELECT 1 AS a"));
    drop(derived);
    assert!(alive.upgrade().is_none());
}

#[test]
fn sent_relations_keep_the_plan_ids_of_the_dataframes_they_were_of() {
    // As a pickled DataFrame of PySpark Connect, whose plan keeps its plan IDs: the DataFrame of
    // a relation that its session sent is sent as it is, and the columns of the DataFrame it
    // was of refer to it.
    let session = offline();
    let df = session.range(3).unwrap().select([col("id").alias("x")]);
    let relation = df.to_relation();
    let plan_id = relation.common.as_ref().and_then(|common| common.plan_id);
    let copy = DataFrame::from_sent_relation(session.clone(), relation.clone());
    assert_eq!(copy.to_relation(), relation);
    assert_eq!(copy.clone().to_relation(), relation);
    let filtered = copy.filter(df.col("x").gt(lit(0)));
    let Some(proto::relation::RelType::Filter(filter)) = filtered.to_relation().rel_type else {
        panic!("expected a filter");
    };
    assert_eq!(filter.input.as_deref(), Some(&relation));
    assert!(format!("{:?}", filter.condition).contains(&format!("plan_id: {:?}", plan_id)));
    // The other plans of the session take other plan IDs.
    let other = session.range(1).unwrap().to_relation();
    assert_ne!(other.common.and_then(|common| common.plan_id), plan_id);
}

#[test]
fn placeholders_of_a_dataframe_are_replaced_by_its_relation() {
    on_a_large_stack(|| {
        // As the plans of pyspark.ml over the DataFrames that they take as input: a relation over
        // the placeholder of a DataFrame, which is replaced by that DataFrame's relation.
        let session = offline();
        let mut df = session.range(3).unwrap();
        for i in 0..120 {
            df = df.with_column(&format!("c{i}"), col("id").add(lit(i)));
        }
        let placeholder = df.placeholder_relation();
        // Shallow: the placeholder of a DataFrame of 120 levels is a few bytes.
        assert!(prost::Message::encoded_len(&placeholder) < 128);
        let input = DataFrame::from_relation(
            session.clone(),
            proto::Relation {
                common: None,
                rel_type: Some(proto::relation::RelType::WithRelations(Box::new(
                    proto::WithRelations {
                        root: Some(Box::new(placeholder.clone())),
                        references: vec![placeholder.clone()],
                    },
                ))),
            },
            None,
        )
        .unwrap();
        let Some(proto::relation::RelType::WithRelations(with)) = input.to_relation().rel_type
        else {
            panic!("expected WithRelations");
        };
        assert_eq!(with.root.as_deref(), Some(&df.to_relation()));
        assert_eq!(with.references, vec![df.to_relation()]);
        // The same placeholder gives the same relation.
        assert_eq!(df.placeholder_relation(), placeholder);
        // The DataFrame of the placeholder is a node of its own, whose relation is that of `df`.
        let single = DataFrame::from_relation(session.clone(), placeholder.clone(), None).unwrap();
        assert_eq!(single.to_relation().rel_type, df.to_relation().rel_type);
        // A placeholder of a DataFrame of another session is the relation of that DataFrame too.
        let other = offline();
        let copy = DataFrame::from_relation(other, placeholder, None).unwrap();
        assert_eq!(copy.to_relation().rel_type, df.to_relation().rel_type);
    });
}

#[test]
fn placeholders_of_dropped_dataframes_are_errors() {
    let session = offline();
    let df = session.range(3).unwrap();
    let placeholder = df.placeholder_relation();
    drop(df);
    let Err(error) = DataFrame::from_relation(session, placeholder, None) else {
        panic!("a placeholder of a dropped DataFrame resolves");
    };
    assert!(error.to_string().contains("no longer exists"), "{error}");
}

#[test]
fn a_deep_placeholder_runs_on_the_server() {
    on_a_large_stack(|| {
        let Some(session) = live() else {
            return;
        };
        let mut df = session.range(3).unwrap();
        for i in 0..120 {
            df = df.with_column(&format!("c{i}"), col("id").add(lit(i)));
        }
        let wrapped = DataFrame::from_relation(
            session.clone(),
            proto::Relation {
                common: None,
                rel_type: Some(proto::relation::RelType::WithRelations(Box::new(
                    proto::WithRelations {
                        root: Some(Box::new(df.placeholder_relation())),
                        references: vec![],
                    },
                ))),
            },
            None,
        )
        .unwrap();
        assert_eq!(wrapped.count().unwrap(), 3);
        assert_eq!(wrapped.columns().unwrap().len(), 121);
    });
}

#[test]
fn commands_of_no_command_type_are_errors() {
    let error = offline()
        .execute_command_proto(proto::Command::default())
        .unwrap_err();
    assert!(error.to_string().contains("no command type"), "{error}");
}

#[test]
fn raw_relations_run_on_the_server() {
    let Some(session) = live() else {
        return;
    };
    let df = DataFrame::from_relation(
        session.clone(),
        sql_relation("SELECT id AS a FROM range(3)"),
        None,
    )
    .unwrap();
    assert_eq!(df.columns().unwrap(), vec!["a".to_string()]);
    assert_eq!(df.clone().filter(col("a").gt(lit(0))).count().unwrap(), 2);
    let other = session.range(2).unwrap();
    let joined = df.join(
        &other,
        Some(df.col("a").eq(other.col("id"))),
        spark_connect::plan::JoinType::Inner,
    );
    assert_eq!(joined.count().unwrap(), 2);
}

#[test]
fn commands_of_messages_return_the_responses_of_the_server() {
    let Some(session) = live() else {
        return;
    };
    let command = proto::Command {
        command_type: Some(proto::command::CommandType::SqlCommand(proto::SqlCommand {
            input: Some(sql_relation("SELECT 1 AS a")),
            ..Default::default()
        })),
    };
    let responses = session.execute_command_proto(command).unwrap();
    let relation = responses
        .iter()
        .find_map(|response| match &response.response_type {
            Some(proto::execute_plan_response::ResponseType::SqlCommandResult(result)) => {
                result.relation.clone()
            }
            _ => None,
        })
        .expect("the result of the SQL command");
    assert!(relation.rel_type.is_some());
}
