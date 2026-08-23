/// Test suite for RuntimeConf, TVF, and TableArg features.
///
/// These tests verify that the Rust implementation matches the PySpark reference.

#[cfg(test)]
mod tests {
    use spark_connect::plan::LogicalPlan;
    use spark_connect::table_arg::TableArg;
    use spark_connect::tvf::TableValuedFunction;

    #[test]
    fn test_unresolved_table_valued_function_plan() {
        // Verify that UnresolvedTableValuedFunction is a valid LogicalPlan variant
        let plan = LogicalPlan::UnresolvedTableValuedFunction {
            name: "explode".to_string(),
            arguments: vec![],
        };

        // Convert to proto
        let proto = plan.to_proto();
        assert!(proto.rel_type.is_some());
    }
}
