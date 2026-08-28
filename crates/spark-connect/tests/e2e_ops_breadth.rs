//! Server-gated breadth coverage: build DataFrames with (nearly) every operation and
//! execute them, so the per-operation execution methods and the recursive
//! `assign_plan_ids` match over `RelType` variants in `dataframe.rs` actually run.
//! Correctness is asserted elsewhere; here the point is to exercise the reachable
//! execution surface, so most calls are resilient (`let _ = ...`) and a few safe ones
//! are asserted. Run with SPARK_REMOTE set.

use spark_connect::column::{col, lit};
use spark_connect::dataframe::DataFrame;
use spark_connect::functions as f;
use spark_connect::plan::JoinType;
use spark_connect::session::SparkSession;
use spark_connect::types::DataType;

fn should_run() -> bool {
    std::env::var("SPARK_REMOTE").is_ok()
}

fn session() -> SparkSession {
    let url = std::env::var("SPARK_REMOTE").unwrap_or_else(|_| "sc://localhost:15002".to_string());
    SparkSession::builder()
        .remote(&url)
        .get_or_create()
        .expect("session")
}

// (id, name, v) with a duplicate row for dedup/set-op paths.
fn base(s: &SparkSession) -> DataFrame {
    s.sql("SELECT * FROM VALUES (1,'a',1.0),(2,'b',2.0),(2,'b',2.0),(3,'c',3.0) AS t(id, name, v)")
        .expect("base df")
}

fn expr(c: spark_connect::column::Column) -> spark_connect::expression::Expression {
    c.expression().clone()
}

/// Relational transforms: project, filter, joins, set ops, aggregates, sort, slicing,
/// dedup, repartition, sample, reshape. Each is collected so `assign_plan_ids` walks
/// the corresponding `RelType` arm.
#[test]
fn transforms_execute() {
    if !should_run() {
        return;
    }
    let s = session();
    let df = base(&s);

    // Project / filter / column ops.
    assert_eq!(df.select(vec![col("id")]).count().unwrap(), 4);
    let _ = df.select_expr(vec!["id + 1 AS x"]).collect().unwrap();
    let _ = df.filter(col("id").gt(lit(1))).collect().unwrap();
    let _ = df.where_(col("id").gt(lit(1))).collect().unwrap();
    let _ = df.with_column("y", lit(1)).collect().unwrap();
    let _ = df
        .with_columns(vec![("p".to_string(), lit(1)), ("q".to_string(), lit(2))])
        .collect()
        .unwrap();
    let _ = df.with_column_renamed("name", "nm").collect().unwrap();
    let _ = df
        .with_columns_renamed(vec![("v".to_string(), "val".to_string())])
        .collect()
        .unwrap();
    let _ = df.drop(vec!["name"]).collect().unwrap();

    // Slicing / dedup / distinct.
    let _ = df.limit(2).collect().unwrap();
    let _ = df.offset(1).collect().unwrap();
    let _ = df.tail(2).collect().unwrap();
    let _ = df.distinct().collect().unwrap();
    let _ = df.drop_duplicates(None).collect().unwrap();
    let _ = df.drop_duplicates(Some(vec!["id"])).collect().unwrap();

    // Sort variants.
    let _ = df.sort(vec![expr(col("id"))]).collect().unwrap();
    let _ = df.order_by(vec![expr(col("id"))]).collect().unwrap();
    let _ = df
        .sort_within_partitions(vec![expr(col("id"))])
        .collect()
        .unwrap();

    // Repartition variants / hint / broadcast / to_df / alias.
    let _ = df.repartition(3).collect().unwrap();
    let _ = df.coalesce(1).collect().unwrap();
    let _ = df
        .repartition_by_range(2, vec![expr(col("id"))])
        .collect()
        .unwrap();
    let _ = df.repartition_by_id(2).collect().unwrap();
    let _ = df.hint("broadcast", vec![]).collect().unwrap();
    let _ = df.broadcast().collect().unwrap();
    let _ = df.to_df(vec!["a", "b", "c"]).collect().unwrap();
    let _ = df.alias("t2").collect().unwrap();

    // Sample.
    let _ = df.sample(0.9, Some(7)).collect().unwrap();
}

/// Joins (multiple kinds) and set operations.
#[test]
fn joins_and_setops_execute() {
    if !should_run() {
        return;
    }
    let s = session();
    let df = base(&s);
    let other = s
        .sql("SELECT * FROM VALUES (2,'b',2.0),(9,'z',9.0) AS t(id, name, v)")
        .unwrap();
    let right = s
        .sql("SELECT * FROM VALUES (1,10),(2,20) AS t(rid, w)")
        .unwrap();

    // Join kinds.
    for jt in [
        JoinType::Inner,
        JoinType::LeftOuter,
        JoinType::RightOuter,
        JoinType::FullOuter,
        JoinType::LeftSemi,
        JoinType::LeftAnti,
    ] {
        let _ = df
            .join(&right, Some(col("id").eq(col("rid"))), jt)
            .collect();
    }
    let _ = df
        .join_using(&other, vec!["id".to_string()], JoinType::Inner)
        .collect()
        .unwrap();
    let _ = df.cross_join(&right).collect().unwrap();

    // Set ops (distinct + all variants).
    let _ = df.union(&other).collect().unwrap();
    let _ = df.union_all(&other).collect().unwrap();
    let _ = df.union_by_name(&other).collect().unwrap();
    let _ = df.union_by_name_opt(&other, true).collect().unwrap();
    let _ = df.intersect(&other).collect().unwrap();
    let _ = df.intersect_all(&other).collect().unwrap();
    let _ = df.subtract(&other).collect().unwrap();
    let _ = df.except_all(&other).collect().unwrap();
}

/// Grouping / aggregation surfaces (group_by/agg/rollup/cube/pivot + shortcuts).
#[test]
fn aggregates_execute() {
    if !should_run() {
        return;
    }
    let s = session();
    let df = base(&s);

    let _ = df
        .group_by(vec![col("id")])
        .agg(vec![expr(f::sum(col("v")))])
        .collect()
        .unwrap();
    let _ = df.group_by(vec![col("id")]).count().collect().unwrap();
    let _ = df
        .group_by(vec![col("id")])
        .sum(vec!["v"])
        .collect()
        .unwrap();
    let _ = df
        .group_by(vec![col("id")])
        .avg(vec!["v"])
        .collect()
        .unwrap();
    let _ = df
        .group_by(vec![col("id")])
        .min(vec!["v"])
        .collect()
        .unwrap();
    let _ = df
        .group_by(vec![col("id")])
        .max(vec!["v"])
        .collect()
        .unwrap();
    let _ = df
        .group_by(vec![col("id")])
        .mean(vec!["v"])
        .collect()
        .unwrap();
    let _ = df.rollup(vec![col("id")]).count().collect().unwrap();
    let _ = df.cube(vec![col("id")]).count().collect().unwrap();
    let _ = df
        .group_by(vec![col("id")])
        .pivot(col("name"), None)
        .agg(vec![expr(f::sum(col("v")))])
        .collect();
    let _ = df.agg(vec![expr(f::sum(col("v")))]).collect().unwrap();
    let _ = df.observe("obs", vec![expr(f::sum(col("v")))]).collect();
}

/// NA / stat / reshape / schema-shaping operations.
#[test]
fn na_reshape_execute() {
    if !should_run() {
        return;
    }
    let s = session();
    let df = base(&s);

    // NA functions.
    let _ = df.fillna(0, Some(vec!["id"])).collect().unwrap();
    let _ = df.fillna_double(0.0, Some(vec!["v"])).collect().unwrap();
    let _ = df.fillna_string("x", Some(vec!["name"])).collect().unwrap();
    let _ = df.dropna(Some("any"), None, None).collect().unwrap();
    let _ = df
        .replace(vec![("a".to_string(), "A".to_string())], Some(vec!["name"]))
        .collect()
        .unwrap();

    // Describe / summary.
    let _ = df.describe(vec!["id", "v"]).collect().unwrap();
    let _ = df.summary(vec!["count", "mean"]).collect().unwrap();

    // Reshape: melt / unpivot / transpose / to(schema).
    let _ = df
        .melt(vec!["id"], Some(vec!["v"]), "var", "val")
        .collect()
        .unwrap();
    let _ = df
        .unpivot(vec![col("id")], Some(vec![col("v")]), "var", "val")
        .collect()
        .unwrap();
    let _ = df.select(vec![col("id"), col("v")]).transpose();
    let _ = df
        .to(DataType::from_ddl("id INT, name STRING, v DOUBLE").unwrap())
        .collect();
}

/// Range/SQL sources and metadata/scalar accessors that trigger execution.
#[test]
fn sources_and_accessors_execute() {
    if !should_run() {
        return;
    }
    let s = session();
    let df = base(&s);

    assert!(s.range(5).unwrap().count().unwrap() >= 0);
    assert!(s.range_full(0, 6, 2, None).unwrap().count().unwrap() >= 0);

    let _ = df.schema().unwrap();
    let _ = df.columns().unwrap();
    let _ = df.dtypes().unwrap();
    let _ = df.is_empty().unwrap();
    let _ = df.first().unwrap();
    let _ = df.head().unwrap();
    let _ = df.take(2).unwrap();
    let _ = df.to_json().unwrap();
    let _ = df.select(vec![col("id")]).limit(1).scalar();
    let _ = df.semantic_hash();
    let _ = df.same_semantics(&df);
    let _ = df.to_local_iterator(false);
    let _ = df.collect_record_batches();
}
