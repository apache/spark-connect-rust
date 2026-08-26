//! Golden tests that verify plan-builder arguments are correctly serialized to proto.
//!
//! These tests ensure that the following bugs, when reintroduced, would cause test failures:
//! 1. dropna(how="all") dropping the `how` argument (min_non_nulls not set to 1)
//! 2. hint() dropping hint parameters entirely
//! 3. replace() dropping string-valued replacements
//! 4. pivot() with explicit values dropping those values
//! 5. fillna_double() dropping the double fill value
//!
//! Each test builds a DataFrame operation, converts to proto, and asserts the
//! exact proto field that must be present to prevent argument loss.

use spark_connect::column::col;
use spark_connect::expression::{Expression, LiteralExpression};
use spark_connect::plan::{self, AggregateGroupType};
use spark_connect_proto as proto;

/// Helper to extract a proto::Relation from a LogicalPlan.
fn plan_to_relation(lp: spark_connect::plan::LogicalPlan) -> proto::Relation {
    lp.to_proto()
}

#[test]
fn test_dropna_how_all_min_non_nulls_assertion() {
    // Regression: dropna(how="all") must assert min_non_nulls == 1
    // If the `how` argument is dropped, this test fails.
    let base = plan::range(0, 10, 1);
    let df_plan = plan::na_drop(base, "all", None, vec![]);
    let relation = plan_to_relation(df_plan);

    // Extract the DropNa proto message
    let drop_na = match relation.rel_type {
        Some(proto::relation::RelType::DropNa(ref dn)) => dn,
        _ => panic!("expected DropNa relation"),
    };

    // **ASSERTION**: min_non_nulls must be 1 for how="all"
    // If the bug is reintroduced (how argument dropped), this will be None/unset and fail.
    assert_eq!(
        drop_na.min_non_nulls,
        Some(1),
        "dropna(how='all') must set min_non_nulls=1 in proto"
    );
}

#[test]
fn test_dropna_how_any_min_non_nulls_unset() {
    // Regression: dropna(how="any") must leave min_non_nulls unset (None)
    // This ensures `how` values are distinct in the proto.
    let base = plan::range(0, 10, 1);
    let df_plan = plan::na_drop(base, "any", None, vec![]);
    let relation = plan_to_relation(df_plan);

    let drop_na = match relation.rel_type {
        Some(proto::relation::RelType::DropNa(ref dn)) => dn,
        _ => panic!("expected DropNa relation"),
    };

    // **ASSERTION**: min_non_nulls must be None for how="any"
    assert_eq!(
        drop_na.min_non_nulls, None,
        "dropna(how='any') must leave min_non_nulls unset in proto"
    );
}

#[test]
fn test_dropna_explicit_thresh_overrides() {
    // Regression: explicit thresh (min_non_null) must override `how` in proto.
    let base = plan::range(0, 10, 1);
    let df_plan = plan::na_drop(base, "any", Some(3), vec![]);
    let relation = plan_to_relation(df_plan);

    let drop_na = match relation.rel_type {
        Some(proto::relation::RelType::DropNa(ref dn)) => dn,
        _ => panic!("expected DropNa relation"),
    };

    // **ASSERTION**: min_non_nulls must be the explicit threshold value
    assert_eq!(
        drop_na.min_non_nulls,
        Some(3),
        "explicit thresh must override how in proto"
    );
}

#[test]
fn test_hint_with_parameters_assertion() {
    // Regression: hint parameters must be serialized (previously dropped entirely).
    // If the hint parameters are dropped, the parameters list is empty and test fails.
    let base = plan::range(0, 10, 1);
    let df_plan = plan::hint(
        base,
        "REPARTITION",
        vec!["10".to_string(), "name".to_string()],
    );
    let relation = plan_to_relation(df_plan);

    let hint = match relation.rel_type {
        Some(proto::relation::RelType::Hint(ref h)) => h,
        _ => panic!("expected Hint relation"),
    };

    // **ASSERTION**: parameters list must not be empty
    // If the bug is reintroduced, parameters will be an empty vec and this fails.
    assert_eq!(
        hint.parameters.len(),
        2,
        "hint parameters must be serialized to proto (previously dropped)"
    );

    // Further validate that the parameters are correctly typed:
    // "10" should become an Integer(10) literal (int32, matching reference lit(int)),
    // "name" should become a String literal.
    match hint.parameters[0].expr_type.as_ref() {
        Some(proto::expression::ExprType::Literal(lit)) => {
            let lit_type = lit.literal_type.clone().unwrap();
            assert!(
                matches!(
                    lit_type,
                    proto::expression::literal::LiteralType::Integer(10)
                ),
                "first param '10' should be Integer(10) literal"
            );
        }
        _ => panic!("expected literal for first parameter"),
    }

    match hint.parameters[1].expr_type.as_ref() {
        Some(proto::expression::ExprType::Literal(lit)) => {
            let lit_type = lit.literal_type.clone().unwrap();
            assert!(
                matches!(
                    lit_type,
                    proto::expression::literal::LiteralType::String(ref s) if s == "name"
                ),
                "second param 'name' should be String literal"
            );
        }
        _ => panic!("expected literal for second parameter"),
    }
}

#[test]
fn test_hint_with_empty_parameters() {
    // Regression: even empty hints should serialize correctly (e.g., broadcast()).
    let base = plan::range(0, 10, 1);
    let df_plan = plan::hint(base, "broadcast", vec![]);
    let relation = plan_to_relation(df_plan);

    let hint = match relation.rel_type {
        Some(proto::relation::RelType::Hint(ref h)) => h,
        _ => panic!("expected Hint relation"),
    };

    // **ASSERTION**: hint name must be set even with empty parameters
    assert_eq!(
        hint.name, "broadcast",
        "hint name must be preserved in proto"
    );
    assert_eq!(hint.parameters.len(), 0, "broadcast hint has no parameters");
}

#[test]
fn test_replace_string_values_assertion() {
    // Regression: replace() with string values must serialize both old and new as string literals
    // (previously, non-numeric replacements were silently dropped).
    let base = plan::range(0, 10, 1);
    let df_plan = plan::na_replace(base, vec![("foo".to_string(), "bar".to_string())], vec![]);
    let relation = plan_to_relation(df_plan);

    let replace = match relation.rel_type {
        Some(proto::relation::RelType::Replace(ref r)) => r,
        _ => panic!("expected Replace relation"),
    };

    // **ASSERTION**: must have at least one replacement
    assert_eq!(
        replace.replacements.len(),
        1,
        "replace must serialize the replacements"
    );

    let repl = &replace.replacements[0];

    // **ASSERTION**: both old and new values must be set as strings
    // If the bug is reintroduced, they will be None or numeric and test fails.
    let old_lit = repl
        .old_value
        .as_ref()
        .expect("old_value must be set")
        .literal_type
        .clone()
        .expect("old_value literal_type must be set");
    assert!(
        matches!(
            old_lit,
            proto::expression::literal::LiteralType::String(ref s) if s == "foo"
        ),
        "old value must be string 'foo' in proto"
    );

    let new_lit = repl
        .new_value
        .as_ref()
        .expect("new_value must be set")
        .literal_type
        .clone()
        .expect("new_value literal_type must be set");
    assert!(
        matches!(
            new_lit,
            proto::expression::literal::LiteralType::String(ref s) if s == "bar"
        ),
        "new value must be string 'bar' in proto"
    );
}

#[test]
fn test_replace_numeric_string_values() {
    // Regression: numeric-looking strings ("1.0", "2.0") should parse as doubles
    // (per the plan.rs str_to_proto_literal behavior).
    let base = plan::range(0, 10, 1);
    let df_plan = plan::na_replace(base, vec![("1.0".to_string(), "2.0".to_string())], vec![]);
    let relation = plan_to_relation(df_plan);

    let replace = match relation.rel_type {
        Some(proto::relation::RelType::Replace(ref r)) => r,
        _ => panic!("expected Replace relation"),
    };

    let repl = &replace.replacements[0];

    // Both should be Double(1.0) and Double(2.0)
    let old_lit = repl
        .old_value
        .as_ref()
        .unwrap()
        .literal_type
        .clone()
        .unwrap();
    assert!(
        matches!(
            old_lit,
            proto::expression::literal::LiteralType::Double(v) if (v - 1.0).abs() < 1e-9
        ),
        "numeric string '1.0' should parse as Double"
    );

    let new_lit = repl
        .new_value
        .as_ref()
        .unwrap()
        .literal_type
        .clone()
        .unwrap();
    assert!(
        matches!(
            new_lit,
            proto::expression::literal::LiteralType::Double(v) if (v - 2.0).abs() < 1e-9
        ),
        "numeric string '2.0' should parse as Double"
    );
}

#[test]
fn test_pivot_with_explicit_values_assertion() {
    // Regression: pivot() with explicit values must serialize them (previously dropped).
    // The pivot values are serialized as Literal expressions in the Aggregate proto.
    let base = plan::range(0, 10, 1);

    // Create an aggregation with pivot and explicit values.
    let pivot_values = vec![
        Expression::Literal(LiteralExpression::string("a")),
        Expression::Literal(LiteralExpression::string("b")),
    ];
    let df_plan = plan::aggregate_with_pivot(
        base,
        AggregateGroupType::Pivot,
        vec![],
        vec![],
        col("id").expression().clone(),
        pivot_values,
    );
    let relation = plan_to_relation(df_plan);

    let agg = match relation.rel_type {
        Some(proto::relation::RelType::Aggregate(ref a)) => a,
        _ => panic!("expected Aggregate relation for pivot"),
    };

    // **ASSERTION**: pivot must be set with values
    let pivot = agg
        .pivot
        .as_ref()
        .expect("pivot must be set for Pivot group type");

    // **ASSERTION**: values list must not be empty
    // If the bug is reintroduced, values will be an empty vec and test fails.
    assert_eq!(
        pivot.values.len(),
        2,
        "pivot explicit values must be serialized to proto (previously dropped)"
    );

    // Validate value types
    assert!(
        matches!(
            pivot.values[0].literal_type,
            Some(proto::expression::literal::LiteralType::String(ref s)) if s == "a"
        ),
        "first pivot value should be string 'a'"
    );

    assert!(
        matches!(
            pivot.values[1].literal_type,
            Some(proto::expression::literal::LiteralType::String(ref s)) if s == "b"
        ),
        "second pivot value should be string 'b'"
    );
}

#[test]
fn test_pivot_without_explicit_values() {
    // Regression: pivot without explicit values should have empty values list.
    // The server will compute distinct values server-side.
    let base = plan::range(0, 10, 1);

    let df_plan = plan::aggregate_with_pivot(
        base,
        AggregateGroupType::Pivot,
        vec![],
        vec![],
        col("id").expression().clone(),
        vec![], // No explicit values
    );
    let relation = plan_to_relation(df_plan);

    let agg = match relation.rel_type {
        Some(proto::relation::RelType::Aggregate(ref a)) => a,
        _ => panic!("expected Aggregate relation for pivot"),
    };

    let pivot = agg.pivot.as_ref().expect("pivot must be set");

    // **ASSERTION**: values list should be empty (server computes values)
    assert_eq!(
        pivot.values.len(),
        0,
        "pivot without explicit values should have empty values list"
    );
}

#[test]
fn test_fillna_double_value_assertion() {
    // Regression: fillna_double() must serialize the double fill value (previously dropped).
    // If the double value is dropped, the values list is empty and test fails.
    let base = plan::range(0, 10, 1);
    let fill_val = spark_connect::row::Value::Double(1.5);
    let df_plan = plan::na_fill(base, fill_val.clone(), vec![]);
    let relation = plan_to_relation(df_plan);

    let fill_na = match relation.rel_type {
        Some(proto::relation::RelType::FillNa(ref f)) => f,
        _ => panic!("expected FillNa relation"),
    };

    // **ASSERTION**: values list must not be empty
    // If the bug is reintroduced, values will be empty and test fails.
    assert_eq!(
        fill_na.values.len(),
        1,
        "fillna_double must serialize the fill value to proto (previously dropped)"
    );

    // **ASSERTION**: the fill value must be a double
    let lit = fill_na.values[0]
        .literal_type
        .clone()
        .expect("literal_type must be set");
    assert!(
        matches!(
            lit,
            proto::expression::literal::LiteralType::Double(v) if (v - 1.5).abs() < 1e-9
        ),
        "fill value must be Double(1.5) in proto"
    );
}

#[test]
fn test_fillna_long_value_assertion() {
    // Regression: fillna_long() must serialize the long fill value.
    let base = plan::range(0, 10, 1);
    let fill_val = spark_connect::row::Value::Long(42);
    let df_plan = plan::na_fill(base, fill_val.clone(), vec![]);
    let relation = plan_to_relation(df_plan);

    let fill_na = match relation.rel_type {
        Some(proto::relation::RelType::FillNa(ref f)) => f,
        _ => panic!("expected FillNa relation"),
    };

    // **ASSERTION**: must serialize the long value
    assert_eq!(fill_na.values.len(), 1);

    let lit = fill_na.values[0]
        .literal_type
        .clone()
        .expect("literal_type must be set");
    assert!(
        matches!(lit, proto::expression::literal::LiteralType::Long(42)),
        "fill value must be Long(42) in proto"
    );
}

#[test]
fn test_fillna_string_value_assertion() {
    // Regression: fillna_string() must serialize the string fill value.
    let base = plan::range(0, 10, 1);
    let fill_val = spark_connect::row::Value::String("N/A".to_string());
    let df_plan = plan::na_fill(base, fill_val.clone(), vec![]);
    let relation = plan_to_relation(df_plan);

    let fill_na = match relation.rel_type {
        Some(proto::relation::RelType::FillNa(ref f)) => f,
        _ => panic!("expected FillNa relation"),
    };

    // **ASSERTION**: must serialize the string value
    assert_eq!(fill_na.values.len(), 1);

    let lit = fill_na.values[0]
        .literal_type
        .clone()
        .expect("literal_type must be set");
    assert!(
        matches!(
            lit,
            proto::expression::literal::LiteralType::String(ref s) if s == "N/A"
        ),
        "fill value must be String('N/A') in proto"
    );
}

#[test]
fn test_grouping_sets_preserved_assertion() {
    // Regression: grouping_sets([[a,b],[a]]) must serialize as GROUP_TYPE_GROUPING_SETS
    // with the explicit sets preserved — not flattened into a plain group-by (the old
    // bug combined all columns and used GROUPBY, silently losing the set structure).
    let base = plan::range(0, 10, 1);
    let sets = vec![
        vec![col("a").expression().clone(), col("b").expression().clone()],
        vec![col("a").expression().clone()],
    ];
    let grouping_expressions = vec![col("a").expression().clone(), col("b").expression().clone()];
    let df_plan =
        plan::aggregate_with_grouping_sets(base, grouping_expressions, vec![], sets.clone());
    let relation = plan_to_relation(df_plan);

    let agg = match relation.rel_type {
        Some(proto::relation::RelType::Aggregate(ref a)) => a,
        _ => panic!("expected Aggregate relation for grouping sets"),
    };

    // **ASSERTION**: group_type must be GROUPING_SETS, not GROUPBY.
    assert_eq!(
        agg.group_type,
        proto::aggregate::GroupType::GroupingSets as i32,
        "group_type must be GROUPING_SETS; if flattened to GROUPBY the sets are lost"
    );

    // **ASSERTION**: the explicit sets are preserved (2 sets, of sizes 2 and 1).
    assert_eq!(
        agg.grouping_sets.len(),
        2,
        "both grouping sets must be serialized; if dropped this is 0"
    );
    assert_eq!(agg.grouping_sets[0].grouping_set.len(), 2);
    assert_eq!(agg.grouping_sets[1].grouping_set.len(), 1);
}
