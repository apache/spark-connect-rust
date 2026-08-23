//! Golden parity test: all 46 logical plan types must serialize to the exact
//! same protobuf the reference PySpark client produces.
//!
//! Goldens live in `tests/golden/plans.jsonl` (captured by `scripts/capture_golden.py`,
//! base64-encoded `spark.connect.Plan`). We normalize out non-deterministic noise
//! (`common`/origin, attribute `plan_id`) on BOTH sides, then require byte-equality.
//! A required case that is missing or mismatches FAILS the test - no silent skips.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

use base64::{engine::general_purpose::STANDARD, Engine};
use prost::Message;

use spark_connect::column::col;
use spark_connect::expression::{ColumnReference, Expression};
use spark_connect::plan::{self, AggregateGroupType, JoinType, LogicalPlan, SetOpType};
use spark_connect_proto as proto;

/// Recursively clear fields that vary run-to-run and are not client-authored:
/// `common` (holds Python origin) everywhere, and relation/attribute `plan_id`.
fn normalize_relation(r: &mut proto::Relation) {
    r.common = None;
    if let Some(rel_type) = &mut r.rel_type {
        use proto::relation::RelType;
        match rel_type {
            RelType::Range(range) => {}
            RelType::Sql(_) => {}
            RelType::Project(proj) => {
                if let Some(input) = &mut proj.input {
                    normalize_relation(input);
                }
                for expr in &mut proj.expressions {
                    normalize_expression(expr);
                }
            }
            RelType::Filter(filter) => {
                if let Some(input) = &mut filter.input {
                    normalize_relation(input);
                }
                if let Some(cond) = &mut filter.condition {
                    normalize_expression(cond);
                }
            }
            RelType::Join(join) => {
                if let Some(left) = &mut join.left {
                    normalize_relation(left);
                }
                if let Some(right) = &mut join.right {
                    normalize_relation(right);
                }
                if let Some(on) = &mut join.join_condition {
                    normalize_expression(on);
                }
            }
            RelType::SetOp(set_op) => {
                if let Some(left) = &mut set_op.left_input {
                    normalize_relation(left);
                }
                if let Some(right) = &mut set_op.right_input {
                    normalize_relation(right);
                }
            }
            RelType::Sort(sort) => {
                if let Some(input) = &mut sort.input {
                    normalize_relation(input);
                }
                for order in &mut sort.order {
                    if let Some(child) = &mut order.child {
                        normalize_expression(child);
                    }
                }
            }
            RelType::Limit(limit) => {
                if let Some(input) = &mut limit.input {
                    normalize_relation(input);
                }
            }
            RelType::Aggregate(agg) => {
                if let Some(input) = &mut agg.input {
                    normalize_relation(input);
                }
                for expr in &mut agg.grouping_expressions {
                    normalize_expression(expr);
                }
                for expr in &mut agg.aggregate_expressions {
                    normalize_expression(expr);
                }
                if let Some(pivot) = &mut agg.pivot {
                    if let Some(col) = &mut pivot.col {
                        normalize_expression(col);
                    }
                }
            }
            RelType::Offset(offset) => {
                if let Some(input) = &mut offset.input {
                    normalize_relation(input);
                }
            }
            RelType::Tail(tail) => {
                if let Some(input) = &mut tail.input {
                    normalize_relation(input);
                }
            }
            RelType::Deduplicate(dedup) => {
                if let Some(input) = &mut dedup.input {
                    normalize_relation(input);
                }
            }
            RelType::Sample(sample) => {
                if let Some(input) = &mut sample.input {
                    normalize_relation(input);
                }
            }
            RelType::Repartition(repart) => {
                if let Some(input) = &mut repart.input {
                    normalize_relation(input);
                }
            }
            RelType::RepartitionByExpression(repart_expr) => {
                if let Some(input) = &mut repart_expr.input {
                    normalize_relation(input);
                }
                for expr in &mut repart_expr.partition_exprs {
                    normalize_expression(expr);
                }
            }
            RelType::WithColumns(wc) => {
                if let Some(input) = &mut wc.input {
                    normalize_relation(input);
                }
                // The aliases field contains Alias objects, which need to be normalized recursively
                for alias in &mut wc.aliases {
                    if let Some(expr) = &mut alias.expr {
                        normalize_expression(expr);
                    }
                }
            }
            RelType::WithColumnsRenamed(wcr) => {
                if let Some(input) = &mut wcr.input {
                    normalize_relation(input);
                }
            }
            RelType::Drop(drop) => {
                if let Some(input) = &mut drop.input {
                    normalize_relation(input);
                }
            }
            RelType::SubqueryAlias(sq_alias) => {
                if let Some(input) = &mut sq_alias.input {
                    normalize_relation(input);
                }
            }
            RelType::Hint(hint) => {
                if let Some(input) = &mut hint.input {
                    normalize_relation(input);
                }
            }
            RelType::Unpivot(unpivot) => {
                if let Some(input) = &mut unpivot.input {
                    normalize_relation(input);
                }
                for id in &mut unpivot.ids {
                    normalize_expression(id);
                }
                if let Some(values) = &mut unpivot.values {
                    for val in &mut values.values {
                        normalize_expression(val);
                    }
                }
            }
            RelType::FillNa(fill_na) => {
                if let Some(input) = &mut fill_na.input {
                    normalize_relation(input);
                }
            }
            RelType::DropNa(drop_na) => {
                if let Some(input) = &mut drop_na.input {
                    normalize_relation(input);
                }
            }
            RelType::Replace(replace) => {
                if let Some(input) = &mut replace.input {
                    normalize_relation(input);
                }
            }
            RelType::Describe(describe) => {
                if let Some(input) = &mut describe.input {
                    normalize_relation(input);
                }
            }
            RelType::Summary(summary) => {
                if let Some(input) = &mut summary.input {
                    normalize_relation(input);
                }
            }
            RelType::ToSchema(to_schema) => {
                if let Some(input) = &mut to_schema.input {
                    normalize_relation(input);
                }
            }
            RelType::LocalRelation(_) => {}
            RelType::CachedRemoteRelation(_) => {}
            RelType::ToDf(to_df) => {
                if let Some(input) = &mut to_df.input {
                    normalize_relation(input);
                }
            }
            _ => {}
        }
    }
}

fn normalize_expression(e: &mut proto::Expression) {
    use proto::expression::ExprType as T;
    e.common = None;
    if let Some(t) = e.expr_type.as_mut() {
        match t {
            T::UnresolvedAttribute(a) => a.plan_id = None,
            T::UnresolvedRegex(r) => r.plan_id = None,
            T::UnresolvedFunction(f) => {
                for a in f.arguments.iter_mut() {
                    normalize_expression(a);
                }
            }
            T::Alias(a) => {
                if let Some(x) = a.expr.as_deref_mut() {
                    normalize_expression(x);
                }
            }
            T::Cast(c) => {
                if let Some(x) = c.expr.as_deref_mut() {
                    normalize_expression(x);
                }
            }
            T::SortOrder(s) => {
                if let Some(x) = s.child.as_deref_mut() {
                    normalize_expression(x);
                }
            }
            T::UnresolvedExtractValue(v) => {
                if let Some(x) = v.child.as_deref_mut() {
                    normalize_expression(x);
                }
                if let Some(x) = v.extraction.as_deref_mut() {
                    normalize_expression(x);
                }
            }
            T::CallFunction(cf) => {
                for a in cf.arguments.iter_mut() {
                    normalize_expression(a);
                }
            }
            _ => {}
        }
    }
}

fn load_goldens() -> HashMap<String, proto::Plan> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/golden/plans.jsonl"
    );
    let file = File::open(path).expect("golden file plans.jsonl missing");
    let mut out = HashMap::new();
    for line in BufReader::new(file).lines() {
        let line = line.unwrap();
        if line.trim().is_empty() {
            continue;
        }
        let obj: serde_json::Value = serde_json::from_str(&line).unwrap();
        let name = obj["name"].as_str().unwrap().to_string();
        let b64 = obj["b64"].as_str().unwrap();
        let bytes = STANDARD.decode(b64).unwrap();
        let mut plan = proto::Plan::decode(&bytes[..]).unwrap();
        if let Some(proto::plan::OpType::Root(root)) = &mut plan.op_type {
            normalize_relation(root);
        }
        out.insert(name, plan);
    }
    out
}

fn build(name: &str) -> Option<LogicalPlan> {
    let plan = match name {
        "range" => plan::range(0, 10, 1),
        "range_start_end_step" => plan::range(2, 20, 3),
        "sql_select" => plan::sql("SELECT 1 AS a, 'x' AS b"),
        "filter_gt" => {
            let range = plan::range(0, 10, 1);
            plan::filter(range, col("id").gt(spark_connect::column::lit(3)))
        }
        "where_gt" => {
            let range = plan::range(0, 10, 1);
            plan::filter(range, col("id").gt(spark_connect::column::lit(3)))
        }
        "select_star" => {
            let range = plan::range(0, 10, 1);
            let star_col = spark_connect::column::Column::new(Expression::UnresolvedStar(None));
            plan::project(range, vec![star_col])
        }
        "select_alias" => {
            let range = plan::range(0, 10, 1);
            let expr = col("id") * spark_connect::column::lit(2);
            plan::project(range, vec![expr.alias("x")])
        }
        "select_expr" => {
            let range = plan::range(0, 10, 1);
            let expr1 = spark_connect::column::Column::new(Expression::SQLExpression(
                "id * 2 as x".to_string(),
            ));
            let expr2 = spark_connect::column::Column::new(Expression::SQLExpression(
                "id + 1 as y".to_string(),
            ));
            plan::project(range, vec![expr1, expr2])
        }
        "with_column" => {
            let range = plan::range(0, 10, 1);
            let expr = col("id") + spark_connect::column::lit(1);
            plan::with_columns(range, vec!["y".to_string()], vec![expr])
        }
        "with_column_renamed" => {
            let range = plan::range(0, 10, 1);
            let mut renames = HashMap::new();
            renames.insert("id".to_string(), "n".to_string());
            plan::with_columns_renamed(range, renames)
        }
        "with_columns_renamed" => {
            let range = plan::range(0, 10, 1);
            let mut renames = HashMap::new();
            renames.insert("id".to_string(), "n".to_string());
            plan::with_columns_renamed(range, renames)
        }
        "drop" => {
            let range = plan::range(0, 10, 1);
            let wc = plan::with_columns(
                range,
                vec!["y".to_string()],
                vec![spark_connect::column::lit(1)],
            );
            plan::drop(wc, vec!["y".to_string()])
        }
        "to_df" => {
            let range = plan::range(0, 10, 1);
            plan::to_df(range, vec!["renamed".to_string()])
        }
        "limit" => {
            let range = plan::range(0, 10, 1);
            plan::limit(range, 5)
        }
        "offset" => {
            let range = plan::range(0, 10, 1);
            plan::offset(range, 3)
        }
        "tail_via_limit" => {
            let range = plan::range(0, 10, 1);
            let lim = plan::limit(range, 5);
            plan::offset(lim, 2)
        }
        "distinct" => {
            let range = plan::range(0, 10, 1);
            let expr = (col("id") % spark_connect::column::lit(2)).alias("m");
            let proj = plan::project(range, vec![expr]);
            plan::deduplicate(proj, true, vec![], false)
        }
        "drop_duplicates" => {
            let range = plan::range(0, 10, 1);
            plan::deduplicate(range, false, vec!["id".to_string()], false)
        }
        "sort" => {
            let range = plan::range(0, 10, 1);
            let order_expr = Expression::SortOrder(Box::new(
                spark_connect::expression::SortOrder::desc_nulls_last(Expression::ColumnReference(
                    ColumnReference::new("id"),
                )),
            ));
            plan::sort(range, vec![order_expr], true)
        }
        "order_by_multi" => {
            let range = plan::range(0, 10, 1);
            let order_expr = Expression::SortOrder(Box::new(
                spark_connect::expression::SortOrder::asc_nulls_last(Expression::ColumnReference(
                    ColumnReference::new("id"),
                )),
            ));
            plan::sort(range, vec![order_expr], true)
        }
        "group_agg" => {
            let range = plan::range(0, 10, 1);
            let group_expr = (col("id") % spark_connect::column::lit(2))
                .alias("k")
                .expression()
                .clone();
            let agg_count = spark_connect::column::Column::new(Expression::UnresolvedFunction(
                spark_connect::expression::UnresolvedFunction::new(
                    "count",
                    vec![Expression::UnresolvedStar(None)],
                ),
            ))
            .alias("c")
            .expression()
            .clone();
            let agg_sum = spark_connect::column::Column::new(Expression::UnresolvedFunction(
                spark_connect::expression::UnresolvedFunction::new(
                    "sum",
                    vec![Expression::ColumnReference(ColumnReference::new("id"))],
                ),
            ))
            .alias("s")
            .expression()
            .clone();
            plan::aggregate(
                range,
                AggregateGroupType::GroupBy,
                vec![group_expr],
                vec![agg_count, agg_sum],
            )
        }
        "group_count" => {
            let range = plan::range(0, 10, 1);
            let group_expr = Expression::ColumnReference(ColumnReference::new("id"));
            let agg_count = spark_connect::column::Column::new(Expression::UnresolvedFunction(
                spark_connect::expression::UnresolvedFunction::new(
                    "count",
                    vec![Expression::Literal(
                        spark_connect::expression::LiteralExpression::int(1),
                    )],
                ),
            ))
            .alias("count")
            .expression()
            .clone();
            plan::aggregate(
                range,
                AggregateGroupType::GroupBy,
                vec![group_expr],
                vec![agg_count],
            )
        }
        "cube" => {
            let range = plan::range(0, 10, 1);
            let group_expr = Expression::ColumnReference(ColumnReference::new("id"));
            let agg_count = spark_connect::column::Column::new(Expression::UnresolvedFunction(
                spark_connect::expression::UnresolvedFunction::new(
                    "count",
                    vec![Expression::Literal(
                        spark_connect::expression::LiteralExpression::int(1),
                    )],
                ),
            ))
            .alias("count")
            .expression()
            .clone();
            plan::aggregate(
                range,
                AggregateGroupType::Cube,
                vec![group_expr],
                vec![agg_count],
            )
        }
        "rollup" => {
            let range = plan::range(0, 10, 1);
            let group_expr = Expression::ColumnReference(ColumnReference::new("id"));
            let agg_count = spark_connect::column::Column::new(Expression::UnresolvedFunction(
                spark_connect::expression::UnresolvedFunction::new(
                    "count",
                    vec![Expression::Literal(
                        spark_connect::expression::LiteralExpression::int(1),
                    )],
                ),
            ))
            .alias("count")
            .expression()
            .clone();
            plan::aggregate(
                range,
                AggregateGroupType::Rollup,
                vec![group_expr],
                vec![agg_count],
            )
        }
        "pivot" => {
            let range = plan::range(0, 10, 1);
            let group_expr = Expression::ColumnReference(ColumnReference::new("id"));
            let agg_count = spark_connect::column::Column::new(Expression::UnresolvedFunction(
                spark_connect::expression::UnresolvedFunction::new(
                    "count",
                    vec![Expression::Literal(
                        spark_connect::expression::LiteralExpression::int(1),
                    )],
                ),
            ))
            .alias("count")
            .expression()
            .clone();
            let pivot_col = Expression::ColumnReference(ColumnReference::new("id"));
            plan::aggregate_with_pivot(
                range,
                AggregateGroupType::Pivot,
                vec![group_expr],
                vec![agg_count],
                pivot_col,
                vec![],
            )
        }
        "union" => {
            let left = plan::range(0, 3, 1);
            let right = plan::range(0, 3, 1);
            plan::set_operation(left, right, SetOpType::Union, true, false, false)
        }
        "union_by_name" => {
            let left = plan::range(0, 3, 1);
            let right = plan::range(0, 3, 1);
            plan::set_operation(left, right, SetOpType::Union, true, true, false)
        }
        "intersect" => {
            let left = plan::range(0, 5, 1);
            let right = plan::range(0, 3, 1);
            plan::set_operation(left, right, SetOpType::Intersect, false, false, false)
        }
        "intersect_all" => {
            let left = plan::range(0, 5, 1);
            let right = plan::range(0, 3, 1);
            plan::set_operation(left, right, SetOpType::Intersect, true, false, false)
        }
        "subtract" => {
            let left = plan::range(0, 5, 1);
            let right = plan::range(0, 3, 1);
            plan::set_operation(left, right, SetOpType::Except, false, false, false)
        }
        "except_all" => {
            let left = plan::range(0, 5, 1);
            let right = plan::range(0, 3, 1);
            plan::set_operation(left, right, SetOpType::Except, true, false, false)
        }
        "join_inner" => {
            let left = plan::subquery_alias(plan::range(0, 5, 1), "a");
            let right = plan::subquery_alias(plan::range(0, 5, 1), "b");
            let on = spark_connect::column::Column::new(Expression::UnresolvedFunction(
                spark_connect::expression::UnresolvedFunction::new(
                    "==",
                    vec![
                        Expression::ColumnReference(ColumnReference::new("a.id")),
                        Expression::ColumnReference(ColumnReference::new("b.id")),
                    ],
                ),
            ));
            plan::join(left, right, JoinType::Inner, Some(on), vec![])
        }
        "join_left" => {
            let left = plan::subquery_alias(plan::range(0, 5, 1), "a");
            let right = plan::subquery_alias(plan::range(0, 5, 1), "b");
            let on = spark_connect::column::Column::new(Expression::UnresolvedFunction(
                spark_connect::expression::UnresolvedFunction::new(
                    "==",
                    vec![
                        Expression::ColumnReference(ColumnReference::new("a.id")),
                        Expression::ColumnReference(ColumnReference::new("b.id")),
                    ],
                ),
            ));
            plan::join(left, right, JoinType::LeftOuter, Some(on), vec![])
        }
        "cross_join" => {
            let left = plan::range(0, 5, 1);
            let right = plan::range(0, 3, 1);
            plan::join(left, right, JoinType::Cross, None, vec![])
        }
        "sample" => {
            let range = plan::range(0, 10, 1);
            plan::sample(range, 0.0, 0.5, false, Some(6749189490610701821))
        }
        "repartition" => {
            let range = plan::range(0, 10, 1);
            plan::repartition(range, 4, true)
        }
        "repartition_by" => {
            let range = plan::range(0, 10, 1);
            plan::repartition_by_expression(
                range,
                4,
                vec![Expression::ColumnReference(ColumnReference::new("id"))],
            )
        }
        "coalesce" => {
            let range = plan::range(0, 10, 1);
            plan::repartition(range, 1, false)
        }
        "hint" => {
            let range = plan::range(0, 10, 1);
            plan::hint(range, "broadcast", vec![])
        }
        "na_drop" => {
            let range = plan::range(0, 10, 1);
            let wc = plan::with_columns(
                range,
                vec!["y".to_string()],
                vec![spark_connect::column::lit(1)],
            );
            plan::na_drop(wc, "any", None, vec![])
        }
        "na_fill" => {
            let range = plan::range(0, 10, 1);
            let wc = plan::with_columns(
                range,
                vec!["y".to_string()],
                vec![spark_connect::column::lit(1)],
            );
            plan::na_fill(wc, 0, vec![])
        }
        "replace" => {
            let range = plan::range(0, 10, 1);
            plan::na_replace(
                range,
                vec![(
                    serde_json::json!(0.0).to_string(),
                    serde_json::json!(100.0).to_string(),
                )],
                vec![],
            )
        }
        "describe" => {
            let range = plan::range(0, 10, 1);
            plan::describe(range, vec!["id".to_string()])
        }
        "summary" => {
            let range = plan::range(0, 10, 1);
            plan::summary(range, vec!["count".to_string(), "min".to_string()])
        }
        "col_regex" => {
            let range = plan::range(0, 10, 1);
            let regex_col =
                spark_connect::column::Column::new(Expression::UnresolvedRegex("`id`".to_string()));
            plan::project(range, vec![regex_col])
        }
        "unpivot" => {
            let range = plan::range(0, 10, 1);
            let wc = plan::with_columns(
                range,
                vec!["y".to_string()],
                vec![spark_connect::column::lit(1)],
            );
            let ids = vec![col("id")];
            let values = vec![col("y")];
            plan::unpivot(wc, ids, Some(values), "var", "val")
        }
        _ => return None,
    };
    Some(plan)
}

const REQUIRED: &[&str] = &[
    "range",
    "range_start_end_step",
    "sql_select",
    "filter_gt",
    "where_gt",
    "select_star",
    "select_alias",
    "select_expr",
    "with_column",
    "with_column_renamed",
    "with_columns_renamed",
    "drop",
    "to_df",
    "limit",
    "offset",
    "tail_via_limit",
    "distinct",
    "drop_duplicates",
    "sort",
    "order_by_multi",
    "group_agg",
    "group_count",
    "cube",
    "rollup",
    "pivot",
    "union",
    "union_by_name",
    "intersect",
    "intersect_all",
    "subtract",
    "except_all",
    "join_inner",
    "join_left",
    "cross_join",
    "sample",
    "repartition",
    "repartition_by",
    "coalesce",
    "hint",
    "na_drop",
    "na_fill",
    "replace",
    "describe",
    "summary",
    "col_regex",
    "unpivot",
];

#[test]
fn all_46_golden_plan_cases_pass() {
    let goldens = load_goldens();
    let mut failures: Vec<String> = Vec::new();

    for &name in REQUIRED {
        let expected = match goldens.get(name) {
            Some(p) => p.clone(),
            None => {
                failures.push(format!("{name}: MISSING from golden file"));
                continue;
            }
        };

        let plan_obj = build(name).unwrap_or_else(|| panic!("no builder for required case {name}"));
        let plan_proto = plan_obj.to_proto();
        let mut actual_plan = proto::Plan::default();
        actual_plan.op_type = Some(proto::plan::OpType::Root(plan_proto));

        if let Some(proto::plan::OpType::Root(root)) = &mut actual_plan.op_type {
            normalize_relation(root);
        }

        if actual_plan != expected {
            failures.push(format!(
                "{name}: MISMATCH\n  expected: {expected:?}\n  actual:   {actual_plan:?}"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} golden plan cases failed:\n{}",
        failures.len(),
        REQUIRED.len(),
        failures.join("\n")
    );
    assert_eq!(
        REQUIRED.len(),
        46,
        "expected exactly 46 cases, got {}",
        REQUIRED.len()
    );
    println!("all {} golden plan cases passed", REQUIRED.len());
}
