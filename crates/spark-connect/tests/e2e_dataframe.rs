//! Behavioral e2e coverage of DataFrame methods not exercised elsewhere: results are
//! asserted against reference pyspark semantics. Covers the untested reachable
//! surface of dataframe.rs (set ops, dedup, joins, reshape, metadata, JSON, temp
//! views, semantics). Run with SPARK_REMOTE set.

use spark_connect::column::{col, lit};
use spark_connect::dataframe::DataFrame;
use spark_connect::functions as f;
use spark_connect::plan::JoinType;
use spark_connect::session::SparkSession;

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
// 4 rows, with a duplicate (2,'b') for dedup/set-op assertions.
fn df4(s: &SparkSession) -> DataFrame {
    s.sql("SELECT * FROM VALUES (1,'a'),(2,'b'),(2,'b'),(3,'c') AS t(id, name)")
        .expect("df4")
}

#[test]
fn dataframe_transforms_and_setops() {
    if !should_run() {
        return;
    }
    let s = session();
    let df = df4(&s);

    // where_ (alias of filter)
    assert_eq!(df.where_(col("id").gt(lit(1))).count().unwrap(), 3);
    // selectExpr
    let se = df.select_expr(vec!["id + 1 AS x"]);
    let xs: Vec<i64> = se
        .collect()
        .unwrap()
        .iter()
        .map(|r| r.get(0).unwrap().as_i64().unwrap())
        .collect();
    assert_eq!(xs, vec![2, 3, 3, 4]);
    // dedup
    assert_eq!(df.drop_duplicates(None).count().unwrap(), 3);
    assert_eq!(df.drop_duplicates(Some(vec!["id"])).count().unwrap(), 3);
    // set ops (all vs distinct)
    let other = s
        .sql("SELECT * FROM VALUES (2,'b'),(9,'z') AS t(id, name)")
        .unwrap();
    assert_eq!(df.union_all(&other).count().unwrap(), 6);
    assert_eq!(df.intersect_all(&other).count().unwrap(), 1); // one (2,'b')
    assert_eq!(df.except_all(&other).count().unwrap(), 3); // 4 - 1 matched
    assert_eq!(df.union_by_name(&other).count().unwrap(), 6);
    // join_using
    let right = s
        .sql("SELECT * FROM VALUES (1,100),(2,200) AS t(id, v)")
        .unwrap();
    assert_eq!(
        df.join_using(&right, vec!["id".to_string()], JoinType::Inner)
            .count()
            .unwrap(),
        3
    );
    // reshape / ordering
    assert_eq!(
        df.sort_within_partitions(vec![col("id").expression().clone()])
            .count()
            .unwrap(),
        4
    );
    assert_eq!(
        df.repartition_by_range(2, vec![col("id").expression().clone()])
            .count()
            .unwrap(),
        4
    );
    assert_eq!(df.to_schema(vec!["id", "name"]).count().unwrap(), 4);
    let m = df.melt(vec!["id"], Some(vec!["name"]), "var", "val");
    assert_eq!(m.count().unwrap(), 4);
}

#[test]
fn dataframe_metadata_json_semantics() {
    if !should_run() {
        return;
    }
    let s = session();
    let df = df4(&s);

    // take(n)
    assert_eq!(df.take(2).unwrap().len(), 2);
    // toJSON: one JSON string per row
    let json = df.to_json().unwrap();
    assert_eq!(json.len(), 4);
    // Each row is a JSON object carrying both columns.
    assert!(json
        .iter()
        .all(|j| j.starts_with('{') && j.contains("id") && j.contains("name")));
    // exists / scalar
    assert!(df.exists().unwrap());
    let one = s.sql("SELECT 42 AS v").unwrap();
    assert_eq!(
        one.scalar().unwrap(),
        Some(spark_connect::row::Value::Integer(42))
    );
    // semantics (server-backed AnalyzePlan): two identical plans are equal
    assert!(df.same_semantics(&df4(&s)).unwrap());
    let _ = df.semantic_hash().unwrap();
    // flags
    assert!(!df.is_streaming());
    let _ = df.is_local();
    // input_files (a sql df reads no files -> empty, but the call path runs)
    let _ = df.input_files().unwrap();
    // observe: attach an observed metric then trigger it
    let observed = df.observe("obs", vec![f::count(lit(1)).expression().clone()]);
    let _ = observed.collect().unwrap();
    // storage level / is_cached around cache
    let cached = df.cache().unwrap();
    let _ = cached.count().unwrap();
    let _ = cached.is_cached().unwrap();
    let _ = cached.storage_level().unwrap();
    // temp views (session + global)
    df.create_or_replace_temp_view("cov_df_view").unwrap();
    assert_eq!(s.table("cov_df_view").unwrap().count().unwrap(), 4);
    let _ = df.create_or_replace_global_temp_view("cov_df_gview");
    // random_split returns the requested number of frames
    let parts = df.random_split(vec![0.5, 0.5], Some(1));
    assert_eq!(parts.len(), 2);
}
