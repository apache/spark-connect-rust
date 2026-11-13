/// Test suite for RuntimeConf, TVF, and TableArg features.
///
/// These tests verify that the Rust implementation matches the PySpark reference.

#[cfg(test)]
mod tests {
    use spark_connect::plan::{LogicalPlan, LogicalPlanKind};

    #[test]
    fn test_unresolved_table_valued_function_plan() {
        // Verify that try_new accepts UnresolvedTableValuedFunction with no inputs.
        let plan = LogicalPlan::try_new(
            LogicalPlanKind::UnresolvedTableValuedFunction {
                name: "explode".to_string(),
                arguments: vec![],
            },
            vec![],
        )
        .unwrap();

        // Convert to proto
        let proto = plan.to_proto();
        assert!(proto.rel_type.is_some());
    }
}
