//! Server-gated end-to-end coverage of the high-level exec paths that unit/golden
//! tests can't reach: table-valued functions, catalog, session config/tags, grouped
//! aggregation, na/stat, describe/summary/unpivot, and mergeInto. These execute
//! against a live server (so they cover `tvf.rs`, `catalog.rs`, `session.rs`,
//! `group.rs`, and the na/stat/reshape parts of `dataframe.rs`, which read 0% in a
//! server-less run) and run natively, so unlike the Python-driven coverage they are
//! free of the cross-arch extension artifact.
//!
//! Run with: `SPARK_REMOTE=sc://localhost:15002 cargo test --test e2e_coverage`

use std::collections::HashMap;

use spark_connect::column::{col, lit, lit_string};
use spark_connect::functions as f;
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

/// A small typed DataFrame with a null, for na/stat/agg coverage.
fn sample(s: &SparkSession) -> spark_connect::dataframe::DataFrame {
    s.sql(
        "SELECT * FROM VALUES (1,'a',10.0),(2,'b',20.0),(2,'c',30.0),(3,NULL,40.0) \
         AS t(id, name, val)",
    )
    .expect("sample df")
}

#[test]
fn tvf_surface() {
    if !should_run() {
        return;
    }
    let s = session();
    let tvf = s.tvf();
    assert_eq!(tvf.range(0, Some(5), 1, None).unwrap().count().unwrap(), 5);
    assert_eq!(tvf.range(5, None, 1, Some(2)).unwrap().count().unwrap(), 5);

    let arr = f::array(vec![lit(1), lit(2), lit(3)]);
    assert_eq!(tvf.explode(&arr).unwrap().count().unwrap(), 3);
    assert_eq!(tvf.explode_outer(&arr).unwrap().count().unwrap(), 3);
    assert_eq!(tvf.posexplode(&arr).unwrap().count().unwrap(), 3);
    assert_eq!(tvf.posexplode_outer(&arr).unwrap().count().unwrap(), 3);

    // stack(2, 1, 2, 3, 4) -> 2 rows
    assert_eq!(
        tvf.stack(&lit(2), vec![lit(1), lit(2), lit(3), lit(4)])
            .unwrap()
            .count()
            .unwrap(),
        2
    );
    // json_tuple over a single JSON object -> 1 row.
    let _ = tvf
        .json_tuple(
            &lit_string(r#"{"a":1,"b":2}"#),
            vec![lit_string("a"), lit_string("b")],
        )
        .unwrap()
        .collect()
        .unwrap();
    // inline(array of structs) -> one row per struct.
    let structs = f::array(vec![
        f::r#struct(vec![lit(1), lit_string("x")]),
        f::r#struct(vec![lit(2), lit_string("y")]),
    ]);
    assert_eq!(tvf.inline(&structs).unwrap().count().unwrap(), 2);
    let _ = tvf.inline_outer(&structs).unwrap().collect().unwrap();
    // Server-catalog TVFs (no args).
    assert!(tvf.collations().unwrap().count().unwrap() > 0);
    assert!(tvf.sql_keywords().unwrap().count().unwrap() > 0);

    // variant_explode / variant_explode_outer over a 2-key JSON object -> 2 rows.
    let variant = f::parse_json(lit_string(r#"{"a":1,"b":2}"#));
    assert_eq!(tvf.variant_explode(&variant).unwrap().count().unwrap(), 2);
    assert_eq!(
        tvf.variant_explode_outer(&variant)
            .unwrap()
            .count()
            .unwrap(),
        2
    );

    // python_worker_logs: the client builds + submits the TVF; the server either
    // returns rows or reports the feature is disabled (a server-side config
    // toggle). Either outcome exercises the client path, so accept both.
    match tvf.python_worker_logs().and_then(|d| d.collect()) {
        Ok(_) => {}
        Err(e) => assert!(
            format!("{e:?}").contains("FEATURE_NOT_ENABLED"),
            "unexpected python_worker_logs error: {e:?}"
        ),
    }
}

#[test]
fn catalog_surface() {
    if !should_run() {
        return;
    }
    let s = session();
    let c = s.catalog();
    let _ = c.current_catalog().unwrap();
    let _ = c.list_catalogs().unwrap().collect().unwrap();
    let _ = c
        .list_catalogs_with_pattern(Some("*"))
        .unwrap()
        .collect()
        .unwrap();
    let _ = c.current_database().unwrap();
    let _ = c.list_databases().unwrap().collect().unwrap();
    let _ = c
        .list_databases_with_pattern(Some("*"))
        .unwrap()
        .collect()
        .unwrap();
    assert!(c.database_exists("default").unwrap());
    assert!(!c.database_exists("no_such_db_xyz").unwrap());
    let _ = c.get_database("default").unwrap().collect().unwrap();
    let _ = c.list_tables().unwrap().collect().unwrap();
    let _ = c.list_functions().unwrap().collect().unwrap();
    assert!(c.function_exists("abs").unwrap());

    // Table-scoped methods against a temp view.
    s.range(3)
        .unwrap()
        .create_or_replace_temp_view("cov_view")
        .unwrap();
    assert!(c.table_exists("cov_view").unwrap());
    let _ = c.get_table("cov_view").unwrap().collect().unwrap();
    let _ = c.list_columns("cov_view").unwrap().collect().unwrap();
    let _ = c.cache_table("cov_view");
    let _ = c.is_cached("cov_view");
    let _ = c.uncache_table("cov_view");
    let _ = c.set_current_database("default");
}

/// The database/pattern-qualified and metadata catalog variants that
/// catalog_surface doesn't reach (100% catalog API coverage).
#[test]
fn catalog_extras_surface() {
    if !should_run() {
        return;
    }
    let s = session();
    let c = s.catalog();

    // set_current_catalog: round-trip through the current value so we don't
    // disturb other tests sharing this session.
    let cur_cat = c.current_catalog().unwrap();
    c.set_current_catalog(&cur_cat).unwrap();

    s.range(3)
        .unwrap()
        .create_or_replace_temp_view("cov_view2")
        .unwrap();

    // Database-qualified table variants.
    let _ = c
        .list_tables_in_database("default")
        .unwrap()
        .collect()
        .unwrap();
    let _ = c
        .list_tables_with_pattern(Some("default"), Some("*"))
        .unwrap()
        .collect()
        .unwrap();
    let _ = c
        .get_table_with_database("cov_view2", None)
        .unwrap()
        .collect()
        .unwrap();
    assert!(c.table_exists_with_database("cov_view2", None).unwrap());
    let _ = c
        .list_columns_with_database("cov_view2", None)
        .unwrap()
        .collect()
        .unwrap();

    // Function variants.
    let _ = c.get_function("abs").unwrap().collect().unwrap();
    let _ = c
        .get_function_with_database("abs", None)
        .unwrap()
        .collect()
        .unwrap();
    assert!(c.function_exists_with_database("abs", None).unwrap());
    let _ = c
        .list_functions_in_database("default")
        .unwrap()
        .collect()
        .unwrap();
    let _ = c
        .list_functions_with_pattern(Some("default"), Some("*"))
        .unwrap()
        .collect()
        .unwrap();

    // dropGlobalTempView for a non-existent view returns false (no error).
    assert!(!c.drop_global_temp_view("no_such_global_view").unwrap());

    // Cache-maintenance ops (no-op-safe against a temp view / whole catalog).
    let _ = c.refresh_table("cov_view2");
    let _ = c.clear_cache();

    // create_table / create_external_table: the client builds and submits the
    // catalog command; with a bogus source the server rejects it, but the client
    // serialization path runs either way. Accept success or a server error.
    let _ = c.create_table("cov_created_tbl", None, Some("parquet"), Some("desc"));
    let _ = c.create_external_table("cov_ext_tbl", None, Some("parquet"));
}

/// DDL catalog operations added for v4.2.0 parity: create/drop database, drop/analyze/
/// truncate table, drop view, get create-table string, get table properties, list
/// partitions, list views. Exercises the real client->server round-trip and cleans up.
#[test]
fn catalog_ddl_v420_surface() {
    if !should_run() {
        return;
    }
    let s = session();
    let c = s.catalog();

    let db = "cov_ddl_db";
    // createDatabase (idempotent) + verify it exists.
    let mut props = HashMap::new();
    props.insert("purpose".to_string(), "coverage".to_string());
    c.create_database(db, true, props).unwrap();
    assert!(c.database_exists(db).unwrap());

    // listViews: with a pattern (no db -> current database is used) and db-qualified.
    let _ = c.list_views(None, None).unwrap().collect().unwrap();
    let _ = c.list_views(None, Some("*")).unwrap().collect().unwrap();
    let _ = c.list_views(Some(db), Some("*")).unwrap().collect().unwrap();

    // A managed, partitioned table in that database for the table-scoped ops.
    let tbl = format!("{db}.cov_ddl_tbl");
    s.sql(&format!(
        "CREATE TABLE IF NOT EXISTS {tbl} (a INT) USING parquet PARTITIONED BY (p INT)"
    ))
    .unwrap()
    .collect()
    .unwrap();
    s.sql(&format!("INSERT INTO {tbl} PARTITION (p=1) VALUES (10)"))
        .unwrap()
        .collect()
        .unwrap();

    // listPartitions, getTableProperties, getCreateTableString, analyzeTable.
    let _ = c.list_partitions(&tbl).unwrap().collect().unwrap();
    let _ = c.get_table_properties(&tbl).unwrap();
    let create_str = c.get_create_table_string(&tbl, false).unwrap();
    assert!(create_str.to_uppercase().contains("CREATE"));
    c.analyze_table(&tbl, true).unwrap();

    // truncateTable then dropTable.
    c.truncate_table(&tbl).unwrap();
    c.drop_table(&tbl, true, false).unwrap();
    assert!(!c.table_exists(&tbl).unwrap());

    // A persistent view for dropView.
    let view = format!("{db}.cov_ddl_view");
    s.sql(&format!("CREATE OR REPLACE VIEW {view} AS SELECT 1 AS a"))
        .unwrap()
        .collect()
        .unwrap();
    c.drop_view(&view, true).unwrap();

    // dropDatabase(cascade) cleans up.
    c.drop_database(db, true, true).unwrap();
    assert!(!c.database_exists(db).unwrap());
}

#[test]
fn session_surface() {
    if !should_run() {
        return;
    }
    let s = session();
    assert!(s.version().unwrap().starts_with('4'));
    assert_eq!(s.range(5).unwrap().count().unwrap(), 5);
    assert_eq!(s.range_full(0, 10, 2, Some(2)).unwrap().count().unwrap(), 5);
    let _ = s.sql("SELECT 1 AS a").unwrap().collect().unwrap();

    let conf = s.conf();
    conf.set("spark.sql.shuffle.partitions", "7").unwrap();
    assert_eq!(
        conf.get("spark.sql.shuffle.partitions").unwrap(),
        Some("7".to_string())
    );
    let _ = conf.get_all().unwrap();
    let _ = conf.is_modifiable("spark.sql.shuffle.partitions").unwrap();
    conf.unset("spark.sql.shuffle.partitions").unwrap();

    s.add_tag("cov-tag").unwrap();
    assert!(s.get_tags().contains(&"cov-tag".to_string()));
    s.remove_tag("cov-tag");
    s.clear_tags();
}

#[test]
fn agg_na_stat_reshape_surface() {
    if !should_run() {
        return;
    }
    let s = session();
    let df = sample(&s);

    // grouped aggregation + rollup/cube/pivot
    assert_eq!(df.group_by(vec![col("id")]).count().count().unwrap(), 3);
    let _ = df
        .group_by(vec![col("id")])
        .agg(vec![f::sum(col("val")).expression().clone()])
        .collect()
        .unwrap();
    let _ = df
        .group_by(vec![col("id")])
        .sum(vec!["val"])
        .collect()
        .unwrap();
    let _ = df
        .group_by(vec![col("id")])
        .avg(vec!["val"])
        .collect()
        .unwrap();
    let _ = df
        .group_by(vec![col("id")])
        .min(vec!["val"])
        .collect()
        .unwrap();
    let _ = df
        .group_by(vec![col("id")])
        .max(vec!["val"])
        .collect()
        .unwrap();
    let _ = df
        .group_by(vec![col("id")])
        .mean(vec!["val"])
        .collect()
        .unwrap();
    let _ = df.rollup(vec![col("id")]).count().collect().unwrap();
    let _ = df.cube(vec![col("id")]).count().collect().unwrap();
    let _ = df
        .group_by(vec![col("id")])
        .pivot(col("name"), None)
        .count()
        .collect();

    // na
    let _ = df.na().drop(Some("any"), None, None).collect().unwrap();
    let _ = df.na().fill(0, None).collect().unwrap();
    let _ = df
        .na()
        .replace(vec![("a".to_string(), "A".to_string())], Some(vec!["name"]))
        .collect()
        .unwrap();
    let _ = df.fillna(0, Some(vec!["val"])).collect().unwrap();
    let _ = df
        .dropna(Some("all"), Some(1), Some(vec!["name"]))
        .collect()
        .unwrap();
    let _ = df
        .replace(vec![("a".to_string(), "A".to_string())], None)
        .collect()
        .unwrap();

    // stat
    let st = df.stat();
    let _ = st.crosstab("id", "name").collect().unwrap();
    let _ = st.freq_items(vec!["name"], 0.5).collect().unwrap();
    let _ = st
        .approx_quantile(vec!["val"], vec![0.5], 0.1)
        .collect()
        .unwrap();
    let _ = st.corr("id", "val").unwrap();
    let _ = st.cov("id", "val").unwrap();

    // describe / summary / unpivot
    let _ = df.describe(vec!["val"]).collect().unwrap();
    let _ = df.summary(vec!["25%", "50%", "75%"]).collect().unwrap();
    let _ = df
        .unpivot(vec![col("id")], Some(vec![col("val")]), "var", "value")
        .collect()
        .unwrap();
}

#[test]
fn merge_into_builder_surface() {
    if !should_run() {
        return;
    }
    let s = session();
    let source = sample(&s);
    // Exercise every when_* branch and the plan builder. The server may reject MERGE
    // on a non-row-level table, but the client-side builder + to_proto (the bulk of
    // merge.rs) runs before the RPC, which is what we're covering here.
    let mut up = HashMap::new();
    up.insert("val".to_string(), col("source.val"));
    let mut ins = HashMap::new();
    ins.insert("id".to_string(), col("source.id"));

    let _ = source
        .merge_into("cov_target", col("target.id").eq(col("source.id")))
        .when_matched(None)
        .update(up.clone())
        .when_matched(Some(col("source.val").gt(lit(0))))
        .delete()
        .when_not_matched(None)
        .insert(ins.clone())
        .when_not_matched(Some(col("source.id").gt(lit(0))))
        .insert_all()
        .when_not_matched_by_source(None)
        .update_all()
        .when_not_matched_by_source(Some(col("target.id").gt(lit(0))))
        .delete()
        .with_schema_evolution()
        .merge();

    let _ = source
        .merge_into("cov_target2", col("target.id").eq(col("source.id")))
        .when_matched(None)
        .update_all()
        .when_not_matched(None)
        .insert_all()
        .merge();
}

#[test]
fn streaming_surface() {
    if !should_run() {
        return;
    }
    use spark_connect::streaming::Trigger;
    use std::collections::HashMap as Map;

    let s = session();

    // DataStreamReader surface (format/option/options/schema/load), all lazy.
    let sdf = s
        .read_stream()
        .format("rate")
        .option("rowsPerSecond", "1")
        .load(None);
    assert!(sdf.is_streaming());
    let mut opts = Map::new();
    opts.insert("rowsPerSecond".to_string(), "1".to_string());
    assert!(s
        .read_stream()
        .format("rate")
        .options(opts)
        .load(None)
        .is_streaming());
    assert!(s
        .read_stream()
        .format("rate")
        .schema("timestamp timestamp, value long")
        .load(None)
        .is_streaming());

    // Start a query and exercise the full StreamingQuery read surface.
    let q = sdf
        .write_stream()
        .format("memory")
        .query_name("cov_rs_stream")
        .output_mode("append")
        .trigger(Trigger::ProcessingTime("1 second".to_string()))
        .start("")
        .expect("start streaming query");
    std::thread::sleep(std::time::Duration::from_secs(2));
    let _ = (q.id(), q.run_id(), q.name());
    assert!(q.is_active().unwrap());
    let _ = q.status().unwrap();
    let _ = q.recent_progress().unwrap();
    let _ = q.last_progress().unwrap();
    let _ = q.explain(false).unwrap();
    let _ = q.exception().unwrap();
    let _ = q.await_termination(Some(0.2)).unwrap();

    // Manager surface.
    let mgr = s.streams();
    let _ = mgr.active().unwrap();
    let _ = mgr.get(q.id()).unwrap();
    let _ = mgr.await_any_termination(Some(0.2)).unwrap();
    q.stop().unwrap();
    mgr.reset_terminated().unwrap();

    // Trigger variants + other writer options (built, not started).
    let base = s
        .read_stream()
        .format("rate")
        .option("rowsPerSecond", "1")
        .load(None);
    let _ = base
        .write_stream()
        .trigger(Trigger::Once)
        .trigger(Trigger::AvailableNow)
        .trigger(Trigger::Continuous("1 second".to_string()))
        .partition_by(vec!["value"])
        .cluster_by(vec!["value"])
        .output_mode("complete");

    // Streaming file-source readers + name(): all build lazy streaming plans
    // (no execution until start), so is_streaming() is a client-side check.
    let schema = "value long";
    assert!(s
        .read_stream()
        .name("rate")
        .format("rate")
        .load(None)
        .is_streaming());
    for df in [
        s.read_stream().schema(schema).json("/tmp/cov_s_json"),
        s.read_stream().schema(schema).parquet("/tmp/cov_s_pq"),
        s.read_stream().schema(schema).orc("/tmp/cov_s_orc"),
        s.read_stream()
            .schema(schema)
            .option("header", "true")
            .csv("/tmp/cov_s_csv"),
        s.read_stream()
            .schema("value string")
            .text("/tmp/cov_s_txt"),
    ] {
        assert!(df.is_streaming());
    }

    // process_all_available drains a rate stream feeding a memory sink, then stop.
    let q2 = s
        .read_stream()
        .format("rate")
        .option("rowsPerSecond", "5")
        .load(None)
        .write_stream()
        .format("memory")
        .query_name("cov_rs_drain")
        .output_mode("append")
        .start("")
        .expect("start drain query");
    q2.process_all_available().unwrap();
    q2.stop().unwrap();
}

#[test]
fn dataframe_methods_surface() {
    if !should_run() {
        return;
    }
    let s = session();
    let df = sample(&s);

    let _ = df.select(vec![col("id"), col("name")]).collect().unwrap();
    let _ = df.filter(col("id")).collect(); // non-boolean filter: exercises the path
    let _ = df.with_column("v2", col("val")).collect().unwrap();
    assert_eq!(df.limit(2).count().unwrap(), 2);
    let _ = df.tail(2);
    assert_eq!(df.distinct().count().unwrap(), 4);
    let _ = df.drop_duplicates(Some(vec!["id"])).collect().unwrap();
    let _ = df
        .order_by(vec![col("id").expression().clone()])
        .collect()
        .unwrap();
    let _ = df.cross_join(&s.range(2).unwrap()).count().unwrap();
    let _ = df.union(&df).count().unwrap();
    let _ = df.repartition(2).count().unwrap();
    let _ = df.coalesce(1).count().unwrap();
    let _ = df.first().unwrap();
    let _ = df.head().unwrap();
    let _ = df.is_empty().unwrap();
    let _ = df.columns().unwrap();
    let _ = df.dtypes().unwrap();
    let _ = df.print_schema();
    let _ = df.explain();
    let _ = df.show(2);

    // Local iterator (both prefetch modes).
    let it = df.to_local_iterator(false).unwrap();
    assert_eq!(it.count(), 4);
    let it2 = df.to_local_iterator(true).unwrap();
    assert_eq!(it2.count(), 4);
}

#[test]
fn session_extras_surface() {
    if !should_run() {
        return;
    }
    let s = session();
    let _ = s.session_id();
    let _ = s.profile();
    let _ = s.profiler();
    let _ = s.read(); // batch DataFrameReader
    assert_eq!(s.empty_data_frame().unwrap().count().unwrap(), 0);
    let ns = s.new_session();
    assert_eq!(ns.range(2).unwrap().count().unwrap(), 2);
    let cs = s.clone_session();
    assert_eq!(cs.range(2).unwrap().count().unwrap(), 2);
    // Run something so there is execution info to read.
    let _ = s.range(1).unwrap().collect().unwrap();
    let _ = s.last_execution_info();
    let _ = s.interrupt_all();
    let _ = s.interrupt_tag("nope");
    let _ = s.interrupt_operation("nope");
}

#[test]
fn dataframe_more_surface() {
    if !should_run() {
        return;
    }
    let s = session();
    let df = sample(&s);
    let other = df.select(vec![col("id"), col("name"), col("val")]);

    let _ = df
        .with_columns(vec![("v2".to_string(), col("val"))])
        .collect()
        .unwrap();
    let _ = df
        .with_columns_renamed(vec![("val".to_string(), "value".to_string())])
        .collect()
        .unwrap();
    let _ = df.drop(vec!["val"]).collect().unwrap();
    let _ = df.offset(1).collect().unwrap();
    let _ = df.union_by_name(&other).count().unwrap();
    let _ = df.intersect(&other).count().unwrap();
    let _ = df.intersect_all(&other).count().unwrap();
    let _ = df.subtract(&other).count().unwrap();
    let _ = df.hint("broadcast", vec![]).collect().unwrap();
    let _ = df.to_df(vec!["a", "b", "c"]).collect().unwrap();
    let _ = df.sample(0.5, Some(1)).count().unwrap();
    let _ = df.col_regex("`.*`").collect().unwrap();

    // cache / persist / unpersist round-trip.
    let cached = df.cache().unwrap();
    let _ = cached.count().unwrap();
    let persisted = df
        .persist(spark_connect_proto::StorageLevel::default())
        .unwrap();
    let _ = persisted.unpersist(false).unwrap();
}

/// Further DataFrame methods not reached by the two surfaces above, including the
/// `to` (schema reconcile) and `with_metadata` paths that were previously stubs.
#[test]
fn dataframe_extra_surface() {
    use spark_connect::plan::JoinType;
    use spark_connect::row::Value;
    use spark_connect::types::{DataType, StructField};
    use std::collections::BTreeMap;

    if !should_run() {
        return;
    }
    let s = session();
    let df = sample(&s); // (id long, name string, val double)

    // broadcast hint, sort (distinct from order_by), single-column rename.
    assert_eq!(df.broadcast().count().unwrap(), 4);
    let _ = df
        .sort(vec![col("id").expression().clone()])
        .collect()
        .unwrap();
    let renamed = df.with_column_renamed("val", "value");
    assert!(renamed.columns().unwrap().contains(&"value".to_string()));

    // grouping_sets aggregation.
    let _ = df
        .grouping_sets(vec![vec![col("id")], vec![]])
        .agg(vec![f::sum(col("val")).expression().clone()])
        .collect()
        .unwrap();

    // repartition variants + spark_session round-trip.
    assert_eq!(
        df.repartition_by_expressions(2, vec![col("id").expression().clone()])
            .count()
            .unwrap(),
        4
    );
    assert_eq!(df.repartition_by_id(2).count().unwrap(), 4);
    assert_eq!(df.spark_session().range(2).unwrap().count().unwrap(), 2);

    // metadata_column just builds a Column (client-side); temp-view registration.
    let _ = df.metadata_column("_metadata");
    df.create_temp_view("cov_ctv").unwrap();
    df.create_global_temp_view("cov_cgtv").unwrap();
    df.register_temp_table("cov_rtt").unwrap();

    // to(schema): reconcile by name to a reordered subset schema.
    let target = DataType::Struct {
        fields: vec![
            StructField {
                name: "val".into(),
                data_type: DataType::Double,
                nullable: true,
                metadata: BTreeMap::new(),
            },
            StructField {
                name: "id".into(),
                data_type: DataType::Long,
                nullable: true,
                metadata: BTreeMap::new(),
            },
        ],
    };
    let reconciled = df.to(target);
    assert_eq!(
        reconciled.columns().unwrap(),
        vec!["val".to_string(), "id".to_string()]
    );

    // with_metadata: attach metadata to a column; data/columns are preserved.
    let mut meta = HashMap::new();
    meta.insert("comment".to_string(), "the id column".to_string());
    let with_meta = df.with_metadata("id", meta);
    assert_eq!(with_meta.count().unwrap(), 4);
    assert_eq!(with_meta.columns().unwrap(), df.columns().unwrap());

    // na-fill variants over a typed, all-null-second-row frame.
    let na_df = s
        .sql("SELECT * FROM VALUES (1.0,'a',true),(NULL,NULL,NULL) AS t(d, s, b)")
        .unwrap();
    assert_eq!(
        na_df.fillna_double(0.0, Some(vec!["d"])).count().unwrap(),
        2
    );
    let _ = na_df.fillna_string("z", Some(vec!["s"])).collect().unwrap();
    let _ = na_df.fillna_bool(false, Some(vec!["b"])).collect().unwrap();
    let _ = na_df
        .fillna_value(Value::Double(1.0), Some(vec!["d"]))
        .collect()
        .unwrap();
    let _ = na_df
        .fillna_map(vec![("d".to_string(), Value::Double(9.0))])
        .collect()
        .unwrap();

    // transpose actually swaps rows/columns (server-side Transpose relation): a
    // (k, x, y) frame with rows a/b transposes so the x/y columns become the 2 rows
    // and the first column's values (a, b) become the headers.
    let tdf = s
        .sql("SELECT * FROM VALUES ('a',1,2),('b',3,4) AS t(k, x, y)")
        .unwrap();
    let transposed = tdf.transpose().unwrap();
    assert_eq!(transposed.count().unwrap(), 2);
    let tcols = transposed.columns().unwrap();
    assert!(tcols.contains(&"a".to_string()) && tcols.contains(&"b".to_string()));
    // transpose_with_index picks an explicit header column.
    assert_eq!(
        tdf.transpose_with_index(col("k")).unwrap().count().unwrap(),
        2
    );

    // explain in every mode (simple/extended/codegen/cost/formatted).
    for m in ["simple", "extended", "codegen", "cost", "formatted"] {
        df.explain_mode(m).unwrap();
    }

    let _ = df.lateral_join(&s.range(2).unwrap(), None, JoinType::Inner);

    // Cleanup the views/tables.
    for stmt in [
        "DROP VIEW IF EXISTS cov_ctv",
        "DROP VIEW IF EXISTS cov_rtt",
        "DROP VIEW IF EXISTS global_temp.cov_cgtv",
    ] {
        let _ = s.sql(stmt).unwrap().collect();
    }
}

#[test]
fn catalog_ddl_surface() {
    if !should_run() {
        return;
    }
    let s = session();
    let c = s.catalog();
    s.range(2)
        .unwrap()
        .create_or_replace_temp_view("cov_ddl_view")
        .unwrap();
    assert!(c.drop_temp_view("cov_ddl_view").unwrap());
    // These can be no-ops or errors depending on the catalog; exercise the paths.
    let _ = c.clear_cache();
    let _ = c.refresh_by_path("/tmp/nonexistent");

    // create_table (from a written Parquet dir) + refresh + recover_partitions. Tolerate
    // server-side limitations (recover on a non-partitioned table, etc.); the coverage
    // target is the client-side catalog request path.
    let dir = std::env::temp_dir().join(format!("cov_ct_{}", std::process::id()));
    let path = dir.join("t").to_string_lossy().into_owned();
    s.range(4)
        .unwrap()
        .write()
        .mode("overwrite")
        .parquet(&path)
        .unwrap();
    let _ = s
        .sql("DROP TABLE IF EXISTS cov_created_tbl")
        .unwrap()
        .collect();
    if let Ok(created) = c.create_table("cov_created_tbl", Some(&path), Some("parquet"), None) {
        let _ = created.count();
        let _ = c.refresh_table("cov_created_tbl");
        let _ = c.recover_partitions("cov_created_tbl");
        let _ = c.is_cached("cov_created_tbl");
    }
    let _ = s
        .sql("DROP TABLE IF EXISTS cov_created_tbl")
        .unwrap()
        .collect();
}

#[test]
fn read_write_roundtrip_surface() {
    if !should_run() {
        return;
    }
    let s = session();
    let dir = std::env::temp_dir().join(format!("cov_rw_{}", std::process::id()));
    let p = |name: &str| dir.join(name).to_string_lossy().into_owned();
    let df = s.range(5).unwrap();

    // Writer: format/mode/option + each concrete sink, then read back and verify rows.
    df.write().mode("overwrite").parquet(&p("pq")).unwrap();
    assert_eq!(s.read().parquet(&p("pq")).count().unwrap(), 5);
    df.write().mode("overwrite").json(&p("js")).unwrap();
    assert_eq!(s.read().json(&p("js")).count().unwrap(), 5);
    df.write()
        .mode("overwrite")
        .option("header", "true")
        .csv(&p("csv"))
        .unwrap();
    assert_eq!(
        s.read()
            .option("header", "true")
            .csv(&p("csv"))
            .count()
            .unwrap(),
        5
    );
    df.write().mode("overwrite").orc(&p("orc")).unwrap();
    assert_eq!(s.read().orc(&p("orc")).count().unwrap(), 5);

    // Text sink needs a single string column.
    let text_df = s
        .sql("SELECT CAST(id AS STRING) AS value FROM range(3)")
        .unwrap();
    text_df.write().mode("overwrite").text(&p("txt")).unwrap();
    assert_eq!(s.read().text(&p("txt")).count().unwrap(), 3);

    // format(...).save(...) + load(...), and partitioning/bucketing builders.
    df.write()
        .format("parquet")
        .mode("overwrite")
        .save(Some(&p("saved")))
        .unwrap();
    assert_eq!(
        s.read()
            .format("parquet")
            .load(Some(&p("saved")))
            .count()
            .unwrap(),
        5
    );
    // partition_by needs a non-partition column too (can't partition by every column).
    let two_col = s.sql("SELECT id, id * 2 AS v FROM range(5)").unwrap();
    two_col
        .write()
        .mode("overwrite")
        .partition_by(vec!["id".to_string()])
        .parquet(&p("partitioned"))
        .unwrap();
    assert_eq!(s.read().parquet(&p("partitioned")).count().unwrap(), 5);

    // saveAsTable + insertInto (managed-table sinks), then read back via table().
    let _ = s.sql("DROP TABLE IF EXISTS cov_rw_tbl").unwrap().collect();
    df.write()
        .mode("overwrite")
        .save_as_table("cov_rw_tbl")
        .unwrap();
    assert_eq!(s.table("cov_rw_tbl").unwrap().count().unwrap(), 5);
    df.write().insert_into("cov_rw_tbl").unwrap();
    assert_eq!(s.table("cov_rw_tbl").unwrap().count().unwrap(), 10);

    // bucket_by + sort_by are only valid with saveAsTable.
    let _ = s
        .sql("DROP TABLE IF EXISTS cov_rw_bucketed")
        .unwrap()
        .collect();
    df.write()
        .mode("overwrite")
        .bucket_by(2, vec!["id".to_string()])
        .sort_by(vec!["id".to_string()])
        .save_as_table("cov_rw_bucketed")
        .unwrap();
    assert_eq!(s.table("cov_rw_bucketed").unwrap().count().unwrap(), 5);

    // options(map) plural form on the reader.
    let mut opts = HashMap::new();
    opts.insert("header".to_string(), "true".to_string());
    assert_eq!(s.read().options(opts).csv(&p("csv")).count().unwrap(), 5);

    // DataFrameWriterV2: create works on the session catalog; append / replace /
    // create_or_replace / overwrite / overwrite_partitions require a v2 catalog,
    // which the default (v1) catalog isn't - the server rejects them. Either way
    // the client build+submit path runs, which is what we're covering here.
    let _ = s.sql("DROP TABLE IF EXISTS cov_rw_v2").unwrap().collect();
    df.write_to("cov_rw_v2").using("parquet").create().unwrap();
    assert_eq!(s.table("cov_rw_v2").unwrap().count().unwrap(), 5);
    fn tolerate<E: std::fmt::Debug>(r: Result<(), E>) {
        if let Err(e) = r {
            let m = format!("{e:?}");
            assert!(
                m.contains("v1 table") || m.contains("UNSUPPORTED_FEATURE"),
                "unexpected v2 write error: {m}"
            );
        }
    }
    tolerate(df.write_to("cov_rw_v2").append());
    tolerate(
        df.write_to("cov_rw_v2")
            .using("parquet")
            .create_or_replace(),
    );
    tolerate(df.write_to("cov_rw_v2").using("parquet").replace());
    tolerate(df.write_to("cov_rw_v2").overwrite(col("id").gt(lit(0))));
    tolerate(df.write_to("cov_rw_v2").overwrite_partitions());

    // XML round-trip (Spark 4.2 has native XML support); jdbc builds a lazy plan
    // (executing it needs a real DB, so just cover the builder).
    df.write()
        .mode("overwrite")
        .option("rowTag", "row")
        .xml(&p("xml"))
        .unwrap();
    assert_eq!(
        s.read()
            .option("rowTag", "row")
            .xml(&p("xml"))
            .count()
            .unwrap(),
        5
    );
    let _ = s
        .read()
        .jdbc("jdbc:h2:mem:covtest", "t", Some(vec!["1=1".to_string()]));

    // Cleanup managed tables.
    for t in ["cov_rw_tbl", "cov_rw_bucketed", "cov_rw_v2"] {
        let _ = s
            .sql(&format!("DROP TABLE IF EXISTS {t}"))
            .unwrap()
            .collect();
    }
}

#[test]
fn artifact_and_tag_surface() {
    if !should_run() {
        return;
    }
    let s = session();

    // --- Artifact API: uploads real files, exercising the client-side chunking /
    // CRC32 / batched-request path in spark-connect-core::artifact + the session
    // wrappers. ---
    let dir = std::env::temp_dir().join(format!("cov_artifacts_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f1 = dir.join("a.txt");
    let f2 = dir.join("b.txt");
    std::fs::write(&f1, b"hello artifact one").unwrap();
    // A larger file to exercise the multi-chunk path (well over one gRPC chunk).
    std::fs::write(&f2, vec![b'x'; 512 * 1024]).unwrap();

    s.add_artifact(f1.to_str().unwrap()).expect("add_artifact");
    s.add_artifacts(&[f1.to_str().unwrap(), f2.to_str().unwrap()])
        .expect("add_artifacts");
    // copyFromLocalToFs: exercises the client-side named-artifact upload
    // (forward_to_fs/...). A single-node server rejects a *local* destination, but the
    // client upload code we're covering runs before that, so tolerate the server error.
    let dest = dir.join("copied.txt");
    let _ = s.copy_from_local_to_fs(f1.to_str().unwrap(), dest.to_str().unwrap());

    // --- Tag API: full lifecycle (add several, read, interrupt-by-tag, remove, clear). ---
    s.clear_tags();
    s.add_tag("tag-alpha").unwrap();
    s.add_tag("tag-beta").unwrap();
    let tags = s.get_tags();
    assert!(tags.contains(&"tag-alpha".to_string()));
    assert!(tags.contains(&"tag-beta".to_string()));
    // interrupt operations carrying a tag (none running -> empty list, but exercises it).
    let _ = s.interrupt_tag("tag-alpha").unwrap();
    s.remove_tag("tag-alpha");
    assert!(!s.get_tags().contains(&"tag-alpha".to_string()));
    s.clear_tags();
    assert!(s.get_tags().is_empty());
}

#[test]
fn ml_estimators_and_transformers() {
    if !should_run() {
        return;
    }
    use spark_connect::ml::{
        Estimator, LogisticRegression, MaxAbsScaler, StandardScaler, StringIndexer, Transformer,
        VectorAssembler,
    };

    let s = session();
    // connect-ML operators expect an array<double> features column, so build it
    // directly in SQL (also gives a plain-double `a`/`b` for VectorAssembler input).
    let df = s
        .sql(
            "SELECT CAST(id AS DOUBLE) AS a, CAST(id * 2 AS DOUBLE) AS b, \
             array(CAST(id AS DOUBLE), CAST(id * 2 AS DOUBLE)) AS features, \
             CAST(id % 2 AS DOUBLE) AS label, \
             CASE WHEN id % 2 = 0 THEN 'x' ELSE 'y' END AS s \
             FROM range(20)",
        )
        .expect("ml sample");

    // Each operator below exercises the CLIENT-side ML code in ml.rs - building the
    // MlOperator/MlRelation params and the fit/transform requests - which runs before
    // the RPC. The actual ML computation is server-side (connect-ML support varies by
    // build), so we tolerate a server rejection rather than asserting the result: the
    // coverage target is our request-construction code, not Spark's ML engine.

    // Transformer.transform builds a fetch of the transform relation.
    let mut va = VectorAssembler::new()
        .set_input_cols(vec!["a", "b"])
        .set_output_col("assembled");
    let _ = va.transform(&df).and_then(|d| d.count());

    // Estimator.fit builds+sends the fit request; Model.transform builds a transform.
    let mut ss = StandardScaler::new()
        .set_input_col("features")
        .set_output_col("scaled");
    if let Ok(mut m) = ss.fit(&df) {
        let _ = m.transform(&df).and_then(|d| d.count());
    }
    let mut mas = MaxAbsScaler::new()
        .set_input_col("features")
        .set_output_col("maxabs");
    if let Ok(mut m) = mas.fit(&df) {
        let _ = m.transform(&df).and_then(|d| d.count());
    }
    let mut si = StringIndexer::new()
        .set_input_col("s")
        .set_output_col("s_idx");
    if let Ok(mut m) = si.fit(&df) {
        let _ = m.transform(&df).and_then(|d| d.count());
    }
    let mut lr = LogisticRegression::new()
        .set_feature_col("features")
        .set_label_col("label")
        .set_prediction_col("prediction")
        .set_max_iter(5);
    if let Ok(mut m) = lr.fit(&df) {
        let _ = m.transform(&df).and_then(|d| d.count());
    }

    // RegressionEvaluator.evaluate now sends a real MlCommand::Evaluate and reads
    // the metric back (previously a stub returning 0.0). With prediction == label
    // the RMSE is exactly 0.
    use spark_connect::ml::{
        BinaryClassificationEvaluator, Evaluator, Pipeline, RegressionEvaluator,
    };
    let eval_df = s
        .sql("SELECT CAST(id AS DOUBLE) AS label, CAST(id AS DOUBLE) AS prediction FROM range(10)")
        .expect("eval df");
    let reg_eval = RegressionEvaluator::new()
        .set_label_col("label")
        .set_prediction_col("prediction")
        .set_metric_name("rmse");
    assert_eq!(reg_eval.evaluate(&eval_df).unwrap(), 0.0);

    // BinaryClassificationEvaluator: the client build+submit path runs; the server
    // needs a vector score column for a real AUC, so tolerate a shape rejection.
    let bin_eval = BinaryClassificationEvaluator::new()
        .set_label_col("label")
        .set_score_col("prediction")
        .set_metric_name("areaUnderROC");
    let _ = bin_eval.evaluate(&eval_df);

    // Pipeline estimator builds+fits (stages carried as params); tolerate server ML.
    let mut pipeline = Pipeline::new().set_stages(vec!["scaler", "lr"]);
    if let Ok(mut m) = pipeline.fit(&df) {
        let _ = m.transform(&df).and_then(|d| d.count());
    }
}

#[test]
fn create_dataframe_typed_schemas() {
    if !should_run() {
        return;
    }
    use spark_connect::row::{Row, Value};
    use spark_connect::types::{DataType, StructField};
    use std::collections::BTreeMap;

    let s = session();
    let field = |name: &str, dt: DataType| StructField {
        name: name.to_string(),
        data_type: dt,
        nullable: true,
        metadata: BTreeMap::new(),
    };

    // Regression: a TIMESTAMP (LTZ) column must build an Arrow array whose timezone
    // matches the schema (UTC), or RecordBatch::try_new rejects it.
    let ts_schema = DataType::Struct {
        fields: vec![field("a", DataType::Timestamp)],
    };
    let ts_rows = vec![
        Row::new(
            vec!["a".into()],
            vec![Value::Timestamp(1_577_836_800_000_000)],
        ),
        Row::new(
            vec!["a".into()],
            vec![Value::Timestamp(1_577_923_200_000_000)],
        ),
    ];
    let df = s
        .create_dataframe(ts_rows, ts_schema)
        .expect("createDataFrame TIMESTAMP");
    assert_eq!(df.count().unwrap(), 2);

    // TIMESTAMP_NTZ (no zone) must still work.
    let ntz_schema = DataType::Struct {
        fields: vec![field("a", DataType::TimestampNtz)],
    };
    let ntz_rows = vec![Row::new(
        vec!["a".into()],
        vec![Value::Timestamp(1_577_836_800_000_000)],
    )];
    assert_eq!(
        s.create_dataframe(ntz_rows, ntz_schema)
            .unwrap()
            .count()
            .unwrap(),
        1
    );

    // Float target from an int value (coerce_value must widen Long/Integer -> Float).
    let f_schema = DataType::Struct {
        fields: vec![field("a", DataType::Float)],
    };
    let f_rows = vec![
        Row::new(vec!["a".into()], vec![Value::Long(1)]),
        Row::new(vec!["a".into()], vec![Value::Long(2)]),
    ];
    assert_eq!(
        s.create_dataframe(f_rows, f_schema)
            .unwrap()
            .count()
            .unwrap(),
        2
    );

    // Exercise every primitive Arrow-array builder arm in one round-trip: bool,
    // byte, short, int, long, double, string, binary, date, decimal - and a null
    // in each. createDataFrame builds a local Arrow relation for these, so a
    // successful collect proves each arm produces a server-acceptable array.
    let all_schema = DataType::Struct {
        fields: vec![
            field("b", DataType::Boolean),
            field("bt", DataType::Byte),
            field("sh", DataType::Short),
            field("i", DataType::Integer),
            field("l", DataType::Long),
            field("d", DataType::Double),
            field(
                "s",
                DataType::String {
                    collation: "UTF8_BINARY".into(),
                },
            ),
            field("bin", DataType::Binary),
            field("dt", DataType::Date),
            field(
                "dec",
                DataType::Decimal {
                    precision: 38,
                    scale: 2,
                },
            ),
        ],
    };
    let names: Vec<String> = vec![
        "b".into(),
        "bt".into(),
        "sh".into(),
        "i".into(),
        "l".into(),
        "d".into(),
        "s".into(),
        "bin".into(),
        "dt".into(),
        "dec".into(),
    ];
    let row0 = Row::new(
        names.clone(),
        vec![
            Value::Bool(true),
            Value::Byte(7),
            Value::Short(300),
            Value::Integer(70_000),
            Value::Long(5_000_000_000),
            Value::Double(2.5),
            Value::String("hello".into()),
            Value::Binary(vec![1u8, 2, 3]),
            Value::Date(19_000), // days since epoch
            Value::Decimal {
                value: "-1.50".into(),
                precision: Some(38),
                scale: Some(2),
            },
        ],
    );
    // A second row with a null in every column exercises the null-handling path.
    let row1 = Row::new(names.clone(), vec![Value::Null; 10]);
    let df = s
        .create_dataframe(vec![row0, row1], all_schema)
        .expect("createDataFrame all primitive types");
    let rows = df.collect().unwrap();
    assert_eq!(rows.len(), 2);
    // Spot-check the non-null row round-tripped correctly.
    let r = &rows[0];
    assert_eq!(r.get_by_name("b").and_then(|v| v.as_bool()), Some(true));
    assert_eq!(r.get_by_name("bt").and_then(|v| v.as_i64()), Some(7));
    assert_eq!(r.get_by_name("sh").and_then(|v| v.as_i64()), Some(300));
    assert_eq!(r.get_by_name("i").and_then(|v| v.as_i64()), Some(70_000));
    assert_eq!(
        r.get_by_name("l").and_then(|v| v.as_i64()),
        Some(5_000_000_000)
    );
    assert_eq!(r.get_by_name("d").and_then(|v| v.as_f64()), Some(2.5));
    assert_eq!(r.get_by_name("s").and_then(|v| v.as_str()), Some("hello"));
    assert_eq!(
        r.get_by_name("bin").and_then(|v| v.as_bytes()),
        Some(&[1u8, 2, 3][..])
    );
}

#[test]
fn udf_builders_client_side() {
    if !should_run() {
        return;
    }
    use spark_connect::types::DataType;
    use spark_connect::udf::{CommonInlineUserDefinedFunctionExpression, PythonUDFPayload};

    let s = session();
    let df = s.range(4).unwrap();
    let empty_struct = || DataType::Struct { fields: vec![] };
    // A dummy cloudpickle-less payload: we build the plans (client-side) but never
    // collect(), so the server/worker never runs the UDF - the coverage target is our
    // request-building in group.rs / dataframe.rs, which the reference client owns too.
    let udf = || {
        CommonInlineUserDefinedFunctionExpression::new(
            "f".to_string(),
            true,
            vec![],
            PythonUDFPayload::new(empty_struct(), 200, vec![1, 2, 3], "3.11".to_string()),
        )
    };

    // GroupedData UDF applies (all lazy plan builders).
    let gd = df.group_by(vec![col("id")]);
    let _ = gd.apply_in_pandas(udf());
    let _ = gd.apply_in_arrow(udf());
    let _ = gd.apply_in_pandas_with_state(udf(), empty_struct(), "append", "");
    let _ = gd.transform_with_state(udf(), "append", "NoTime", None, None);
    let _ =
        gd.transform_with_state_in_pandas(udf(), empty_struct(), "append", "NoTime", None, None);

    // Cogroup + apply.
    let gd2 = s.range(4).unwrap().group_by(vec![col("id")]);
    let cg = gd.cogroup(&gd2);
    let _ = cg.apply_in_pandas(udf());
    let _ = cg.apply_in_arrow(udf());

    // DataFrame mapInPandas / mapInArrow (barrier off/on).
    let _ = df.map_in_pandas(udf(), false);
    let _ = df.map_in_arrow(udf(), true);
}
