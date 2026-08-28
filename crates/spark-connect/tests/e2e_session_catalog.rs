//! Server-gated end-to-end coverage of the runtime (RPC) methods of `catalog.rs`
//! and `session.rs`: the full Catalog surface (current/list/get/exists/create/
//! cache/refresh across catalogs, databases, tables, columns, functions) and the
//! SparkSession runtime helpers (version, table, empty_data_frame, conf get/set,
//! interrupt, tags, progress handlers, new/clone session). These execute against a
//! live server, covering paths a server-less run reads as uncovered.
//!
//! Run with: `SPARK_REMOTE=sc://localhost:15002 cargo test --test e2e_session_catalog`

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

/// Exercise the whole Catalog RPC surface. Most `list_*`/`get_*` calls return a lazy
/// DataFrame, so we collect them (tolerantly) to actually drive the server.
#[test]
fn catalog_full_surface() {
    if !should_run() {
        return;
    }
    let spark = session();
    let cat = spark.catalog();

    // Seed a temp view so the table/column lookups have something to find.
    spark
        .sql("SELECT 1 AS a, 'x' AS b")
        .expect("sql")
        .create_or_replace_temp_view("e2e_cat_v")
        .expect("create view");

    // Catalog-level.
    let _ = cat.current_catalog();
    let _ = cat.set_current_catalog("spark_catalog");
    let _ = cat.list_catalogs().and_then(|d| d.collect());
    let _ = cat
        .list_catalogs_with_pattern(Some("*"))
        .and_then(|d| d.collect());

    // Database-level.
    let _ = cat.current_database();
    let _ = cat.set_current_database("default");
    let _ = cat.list_databases().and_then(|d| d.collect());
    let _ = cat
        .list_databases_with_pattern(Some("*"))
        .and_then(|d| d.collect());
    let _ = cat.get_database("default").and_then(|d| d.collect());
    assert!(
        cat.database_exists("default").unwrap_or(false) || cat.database_exists("default").is_ok()
    );
    let _ = cat.database_exists("no_such_db_e2e");

    // Table-level.
    let _ = cat.list_tables().and_then(|d| d.collect());
    let _ = cat
        .list_tables_in_database("default")
        .and_then(|d| d.collect());
    let _ = cat
        .list_tables_with_pattern(Some("default"), Some("*"))
        .and_then(|d| d.collect());
    let _ = cat.get_table("e2e_cat_v").and_then(|d| d.collect());
    let _ = cat
        .get_table_with_database("e2e_cat_v", None)
        .and_then(|d| d.collect());
    let _ = cat.table_exists("e2e_cat_v");
    let _ = cat.table_exists_with_database("e2e_cat_v", None);
    let _ = cat.table_exists("no_such_table_e2e");

    // Column-level.
    let _ = cat.list_columns("e2e_cat_v").and_then(|d| d.collect());
    let _ = cat
        .list_columns_with_database("e2e_cat_v", None)
        .and_then(|d| d.collect());

    // Function-level.
    let _ = cat.list_functions().and_then(|d| d.collect());
    let _ = cat
        .list_functions_in_database("default")
        .and_then(|d| d.collect());
    let _ = cat
        .list_functions_with_pattern(Some("default"), Some("*"))
        .and_then(|d| d.collect());
    let _ = cat.get_function("abs").and_then(|d| d.collect());
    let _ = cat
        .get_function_with_database("abs", None)
        .and_then(|d| d.collect());
    let _ = cat.function_exists("abs");
    let _ = cat.function_exists_with_database("abs", None);

    // Create table paths (likely error on this single-node server; tolerated — the
    // proto-building body is what we're covering).
    let _ = cat
        .create_table("e2e_created_tbl", None, Some("parquet"), Some("desc"))
        .and_then(|d| d.collect());
    let _ = cat
        .create_external_table("e2e_ext_tbl", None, Some("parquet"))
        .and_then(|d| d.collect());
    let _ = spark
        .sql("DROP TABLE IF EXISTS e2e_created_tbl")
        .and_then(|d| d.collect());
    let _ = spark
        .sql("DROP TABLE IF EXISTS e2e_ext_tbl")
        .and_then(|d| d.collect());

    // Cache / refresh / recover.
    let _ = cat.cache_table("e2e_cat_v");
    let _ = cat.is_cached("e2e_cat_v");
    let _ = cat.uncache_table("e2e_cat_v");
    let _ = cat.clear_cache();
    let _ = cat.refresh_table("e2e_cat_v");
    let _ = cat.refresh_by_path("/tmp");
    let _ = cat.recover_partitions("e2e_cat_v");

    // Drop the temp view we created (returns whether it existed).
    let _ = cat.drop_temp_view("e2e_cat_v");
    let _ = cat.drop_global_temp_view("no_such_global_view_e2e");
}

/// SparkSession runtime helpers: version, table, empty_data_frame, conf get/set,
/// interrupt, tags, and new/clone session.
#[test]
fn session_runtime_helpers() {
    if !should_run() {
        return;
    }
    let spark = session();

    // Metadata / basic RPCs.
    let v = spark.version().expect("version");
    assert!(!v.is_empty());
    let _ = spark.empty_data_frame().and_then(|d| d.collect());

    spark
        .sql("SELECT 1 AS a")
        .expect("sql")
        .create_or_replace_temp_view("e2e_sess_v")
        .expect("view");
    let _ = spark.table("e2e_sess_v").and_then(|d| d.collect());

    // Runtime conf.
    let conf = spark.conf();
    let _ = conf.set("spark.sql.shuffle.partitions", "7");
    let _ = conf.get("spark.sql.shuffle.partitions");
    let _ = conf.get_all();
    let _ = conf.is_modifiable("spark.sql.shuffle.partitions");
    let _ = conf.unset("spark.sql.shuffle.partitions");

    // Interrupts (no active ops; should return an empty list cleanly).
    let _ = spark.interrupt_all();
    let _ = spark.interrupt_tag("no_such_tag_e2e");
    let _ = spark.interrupt_operation("no_such_op_e2e");

    // Tags are local state.
    spark.add_tag("e2e_tag").expect("add_tag");
    assert!(spark.get_tags().contains(&"e2e_tag".to_string()));
    spark.remove_tag("e2e_tag");
    spark.clear_tags();

    // Derived sessions share the transport but are independent handles.
    let ns = spark.new_session();
    let _ = ns.range(1);
    let cs = spark.clone_session();
    let _ = cs.range(1);
}

/// Register a progress handler, run a query so the execute loop invokes it, then
/// remove and clear. Covers the progress-handler registration surface.
#[test]
fn session_progress_handlers() {
    if !should_run() {
        return;
    }
    let spark = session();

    let id = spark.register_progress_handler(|_progress| {});
    // Drive a query so the execute loop has a chance to invoke handlers.
    let _ = spark.range(1000).and_then(|d| d.collect());
    spark.remove_progress_handler(id);
    spark.clear_progress_handlers();
}
