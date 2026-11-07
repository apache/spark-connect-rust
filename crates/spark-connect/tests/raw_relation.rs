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
    let df = DataFrame::from_relation(offline(), sql_relation("SELECT 1 AS a"), None);
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
    let df = DataFrame::from_relation(offline(), sql_relation("SELECT 1 AS a"), None);
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
    let df = DataFrame::from_relation(offline(), sql_relation("SELECT 1 AS a"), Some(owner));
    let derived = df.select([col("a")]);
    drop(df);
    assert!(alive.upgrade().is_some());
    assert!(format!("{:?}", derived.to_relation()).contains("SELECT 1 AS a"));
    drop(derived);
    assert!(alive.upgrade().is_none());
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
    );
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
