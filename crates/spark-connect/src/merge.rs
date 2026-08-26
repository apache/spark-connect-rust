//! `MergeIntoWriter` mirroring `pyspark.sql.connect.merge.MergeIntoWriter`.
//!
//! Builds a `MergeIntoTableCommand` from a fluent set of `whenMatched` /
//! `whenNotMatched` / `whenNotMatchedBySource` clauses and executes it.

use std::collections::HashMap;

use spark_connect_core::error::Result;
use spark_connect_proto as proto;

use crate::column::Column;
use crate::dataframe::{build_input_relation, execute_command};
use crate::plan::LogicalPlan;
use crate::session::SparkSession;

fn merge_action(
    action_type: proto::merge_action::ActionType,
    condition: Option<&Column>,
    assignments: Option<HashMap<String, Column>>,
) -> proto::Expression {
    let mut action = proto::MergeAction::default();
    action.action_type = action_type as i32;
    action.condition = condition.map(|c| Box::new(c.to_proto()));
    if let Some(assignments) = assignments {
        action.assignments = assignments
            .into_iter()
            .map(|(k, v)| proto::merge_action::Assignment {
                // Mirrors the reference `expr(k)` for the assignment target.
                key: Some(crate::functions::expr(&k).to_proto()),
                value: Some(v.to_proto()),
            })
            .collect();
    }
    let mut expr = proto::Expression::default();
    expr.expr_type = Some(proto::expression::ExprType::MergeAction(Box::new(action)));
    expr
}

/// Fluent builder for a MERGE INTO command. Create it via [`crate::dataframe::DataFrame::merge_into`].
pub struct MergeIntoWriter {
    session: SparkSession,
    source_plan: LogicalPlan,
    target_table: String,
    condition: Column,
    schema_evolution: bool,
    matched_actions: Vec<proto::Expression>,
    not_matched_actions: Vec<proto::Expression>,
    not_matched_by_source_actions: Vec<proto::Expression>,
}

impl MergeIntoWriter {
    pub(crate) fn new(
        session: SparkSession,
        source_plan: LogicalPlan,
        target_table: String,
        condition: Column,
    ) -> Self {
        MergeIntoWriter {
            session,
            source_plan,
            target_table,
            condition,
            schema_evolution: false,
            matched_actions: Vec::new(),
            not_matched_actions: Vec::new(),
            not_matched_by_source_actions: Vec::new(),
        }
    }

    /// Begin a `WHEN MATCHED [AND condition]` clause.
    pub fn when_matched(self, condition: Option<Column>) -> WhenMatched {
        WhenMatched {
            writer: self,
            condition,
        }
    }

    /// Begin a `WHEN NOT MATCHED [AND condition]` clause.
    pub fn when_not_matched(self, condition: Option<Column>) -> WhenNotMatched {
        WhenNotMatched {
            writer: self,
            condition,
        }
    }

    /// Begin a `WHEN NOT MATCHED BY SOURCE [AND condition]` clause.
    pub fn when_not_matched_by_source(self, condition: Option<Column>) -> WhenNotMatchedBySource {
        WhenNotMatchedBySource {
            writer: self,
            condition,
        }
    }

    /// Enable schema evolution for this merge.
    pub fn with_schema_evolution(mut self) -> Self {
        self.schema_evolution = true;
        self
    }

    /// Execute the merge.
    pub fn merge(self) -> Result<()> {
        let mut cmd = proto::MergeIntoTableCommand::default();
        cmd.target_table_name = self.target_table;
        cmd.source_table_plan = Some(build_input_relation(&self.source_plan, &self.session)?);
        cmd.merge_condition = Some(self.condition.to_proto());
        cmd.match_actions = self.matched_actions;
        cmd.not_matched_actions = self.not_matched_actions;
        cmd.not_matched_by_source_actions = self.not_matched_by_source_actions;
        cmd.with_schema_evolution = self.schema_evolution;
        execute_command(
            &self.session,
            proto::command::CommandType::MergeIntoTableCommand(cmd),
        )
    }
}

/// `WHEN MATCHED` clause builder.
pub struct WhenMatched {
    writer: MergeIntoWriter,
    condition: Option<Column>,
}

impl WhenMatched {
    pub fn update_all(mut self) -> MergeIntoWriter {
        let action = merge_action(
            proto::merge_action::ActionType::UpdateStar,
            self.condition.as_ref(),
            None,
        );
        self.writer.matched_actions.push(action);
        self.writer
    }

    pub fn update(mut self, assignments: HashMap<String, Column>) -> MergeIntoWriter {
        let action = merge_action(
            proto::merge_action::ActionType::Update,
            self.condition.as_ref(),
            Some(assignments),
        );
        self.writer.matched_actions.push(action);
        self.writer
    }

    pub fn delete(mut self) -> MergeIntoWriter {
        let action = merge_action(
            proto::merge_action::ActionType::Delete,
            self.condition.as_ref(),
            None,
        );
        self.writer.matched_actions.push(action);
        self.writer
    }
}

/// `WHEN NOT MATCHED` clause builder.
pub struct WhenNotMatched {
    writer: MergeIntoWriter,
    condition: Option<Column>,
}

impl WhenNotMatched {
    pub fn insert_all(mut self) -> MergeIntoWriter {
        let action = merge_action(
            proto::merge_action::ActionType::InsertStar,
            self.condition.as_ref(),
            None,
        );
        self.writer.not_matched_actions.push(action);
        self.writer
    }

    pub fn insert(mut self, assignments: HashMap<String, Column>) -> MergeIntoWriter {
        let action = merge_action(
            proto::merge_action::ActionType::Insert,
            self.condition.as_ref(),
            Some(assignments),
        );
        self.writer.not_matched_actions.push(action);
        self.writer
    }
}

/// `WHEN NOT MATCHED BY SOURCE` clause builder.
pub struct WhenNotMatchedBySource {
    writer: MergeIntoWriter,
    condition: Option<Column>,
}

impl WhenNotMatchedBySource {
    pub fn update_all(mut self) -> MergeIntoWriter {
        let action = merge_action(
            proto::merge_action::ActionType::UpdateStar,
            self.condition.as_ref(),
            None,
        );
        self.writer.not_matched_by_source_actions.push(action);
        self.writer
    }

    pub fn update(mut self, assignments: HashMap<String, Column>) -> MergeIntoWriter {
        let action = merge_action(
            proto::merge_action::ActionType::Update,
            self.condition.as_ref(),
            Some(assignments),
        );
        self.writer.not_matched_by_source_actions.push(action);
        self.writer
    }

    pub fn delete(mut self) -> MergeIntoWriter {
        let action = merge_action(
            proto::merge_action::ActionType::Delete,
            self.condition.as_ref(),
            None,
        );
        self.writer.not_matched_by_source_actions.push(action);
        self.writer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::SparkSession;

    fn session() -> SparkSession {
        SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .expect("failed to build session")
    }

    #[test]
    fn merge_into_when_matched_update() {
        let spark = session();
        let df = spark.range(3).unwrap();
        let mut assignments = std::collections::HashMap::new();
        assignments.insert("col1".to_string(), crate::column::col("new_val"));

        let writer = df
            .merge_into("target_table", crate::column::col("id"))
            .when_matched(None)
            .update(assignments);

        assert_eq!(writer.matched_actions.len(), 1);
        assert_eq!(writer.not_matched_actions.len(), 0);
    }

    #[test]
    fn merge_into_when_matched_delete() {
        let spark = session();
        let df = spark.range(3).unwrap();

        let writer = df
            .merge_into("target_table", crate::column::col("id"))
            .when_matched(None)
            .delete();

        assert_eq!(writer.matched_actions.len(), 1);
    }

    #[test]
    fn merge_into_when_matched_update_all() {
        let spark = session();
        let df = spark.range(3).unwrap();

        let writer = df
            .merge_into("target_table", crate::column::col("id"))
            .when_matched(None)
            .update_all();

        assert_eq!(writer.matched_actions.len(), 1);
    }

    #[test]
    fn merge_into_when_not_matched_insert() {
        let spark = session();
        let df = spark.range(3).unwrap();
        let mut assignments = std::collections::HashMap::new();
        assignments.insert("col1".to_string(), crate::column::col("val"));

        let writer = df
            .merge_into("target_table", crate::column::col("id"))
            .when_not_matched(None)
            .insert(assignments);

        assert_eq!(writer.not_matched_actions.len(), 1);
        assert_eq!(writer.matched_actions.len(), 0);
    }

    #[test]
    fn merge_into_when_not_matched_insert_all() {
        let spark = session();
        let df = spark.range(3).unwrap();

        let writer = df
            .merge_into("target_table", crate::column::col("id"))
            .when_not_matched(None)
            .insert_all();

        assert_eq!(writer.not_matched_actions.len(), 1);
    }

    #[test]
    fn merge_into_when_not_matched_by_source() {
        let spark = session();
        let df = spark.range(3).unwrap();

        let writer = df
            .merge_into("target_table", crate::column::col("id"))
            .when_not_matched_by_source(None)
            .delete();

        assert_eq!(writer.not_matched_by_source_actions.len(), 1);
    }

    #[test]
    fn merge_into_multiple_clauses() {
        let spark = session();
        let df = spark.range(3).unwrap();
        let mut assign1 = std::collections::HashMap::new();
        assign1.insert("col1".to_string(), crate::column::col("val1"));
        let mut assign2 = std::collections::HashMap::new();
        assign2.insert("col2".to_string(), crate::column::col("val2"));

        let writer = df
            .merge_into("target_table", crate::column::col("id"))
            .when_matched(None)
            .update(assign1)
            .when_not_matched(None)
            .insert(assign2);

        assert_eq!(writer.matched_actions.len(), 1);
        assert_eq!(writer.not_matched_actions.len(), 1);
    }

    #[test]
    fn merge_into_with_schema_evolution() {
        let spark = session();
        let df = spark.range(3).unwrap();

        let writer = df
            .merge_into("target_table", crate::column::col("id"))
            .with_schema_evolution()
            .when_matched(None)
            .update_all();

        assert!(writer.schema_evolution);
    }
}
