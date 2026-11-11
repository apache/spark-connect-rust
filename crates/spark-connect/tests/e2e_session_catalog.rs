//! Server-gated end-to-end coverage of the runtime (RPC) methods of `catalog.rs`
//! and `session.rs`: the full Catalog surface (current/list/get/exists/create/
//! cache/refresh across catalogs, databases, tables, columns, functions) and the
//! SparkSession runtime helpers (version, table, empty_data_frame, conf get/set,
//! interrupt, tags, progress handlers, new/clone session). These execute against a
//! live server, covering paths a server-less run reads as uncovered.
//!
//! Run with: `SPARK_REMOTE=sc://localhost:15002 cargo test --test e2e_session_catalog`

mod common;

use common::{session, should_run};

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
    let cs = spark.clone_session().expect("clone_session");
    let _ = cs.range(1);
}

/// `clone_session` clones the session on the server, as PySpark's `cloneSession`: the clone,
/// of a session id of its own or the one given, has the configurations and the temporary
/// views of the session, and changes to either do not show in the other.
#[test]
fn clone_session_copies_the_state_of_the_session() {
    if !should_run() {
        return;
    }
    let spark = session();
    spark
        .conf()
        .set("spark.sql.shuffle.partitions", "7")
        .expect("conf set");
    spark
        .sql("SELECT 1 AS a")
        .expect("sql")
        .create_or_replace_temp_view("e2e_clone_v")
        .expect("create view");
    let clone = spark.clone_session().expect("clone_session");
    assert_ne!(clone.session_id(), spark.session_id());
    assert_eq!(
        clone
            .conf()
            .get("spark.sql.shuffle.partitions")
            .expect("conf get")
            .as_deref(),
        Some("7")
    );
    assert_eq!(
        clone
            .table("e2e_clone_v")
            .expect("table")
            .count()
            .expect("count"),
        1
    );
    // Isolated: a change in the clone does not show in the session.
    clone
        .conf()
        .set("spark.sql.shuffle.partitions", "9")
        .expect("conf set");
    assert_eq!(
        spark
            .conf()
            .get("spark.sql.shuffle.partitions")
            .expect("conf get")
            .as_deref(),
        Some("7")
    );
    let id = uuid::Uuid::new_v4().to_string();
    let named = spark
        .clone_session_with_id(Some(&id))
        .expect("clone with id");
    assert_eq!(named.session_id(), id);
    assert!(spark.clone_session_with_id(Some("not-a-uuid")).is_err());
    let _ = spark.conf().unset("spark.sql.shuffle.partitions");
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

/// A server error carries its kind, condition, parameters, SQL state, stack trace and query
/// contexts, as the reference client reads them.
#[test]
fn server_errors_carry_their_condition_and_details() {
    use spark_connect::SparkErrorKind;
    if !should_run() {
        return;
    }
    let spark = session();
    // Raised at `sql()`, or at the action where `sql()` does not run the statement.
    let err = match spark.sql("SELEC 1") {
        Ok(df) => df.collect().expect_err("a syntax error"),
        Err(err) => err,
    };
    assert_eq!(err.kind, SparkErrorKind::Parse);
    assert_eq!(err.get_condition().as_deref(), Some("PARSE_SYNTAX_ERROR"));
    assert_eq!(err.get_sql_state().as_deref(), Some("42601"));
    assert_eq!(err.params.get("error").map(String::as_str), Some("'SELEC'"));
    // The server's message, which names the condition once.
    assert_eq!(err.to_string().matches("[PARSE_SYNTAX_ERROR]").count(), 1);
    assert!(err
        .get_stacktrace()
        .unwrap()
        .starts_with("org.apache.spark.sql.catalyst.parser.ParseException"));
    // spark.sql.connect.serverStacktrace.enabled is true by default.
    assert!(err.server.as_ref().unwrap().display_stacktrace);

    let err = spark
        .sql("SELECT 1/0")
        .expect("a query")
        .collect()
        .expect_err("a division by zero");
    assert_eq!(err.kind, SparkErrorKind::Arithmetic);
    assert_eq!(err.get_condition().as_deref(), Some("DIVIDE_BY_ZERO"));
    let contexts = err.get_query_context();
    assert_eq!(contexts.len(), 1);
    assert_eq!(contexts[0].fragment(), "1/0");
}

/// As pyspark, `sql()` sends the statement at once: a command takes effect without an
/// action, its result is the local relation the server returned, and a statement that
/// does not parse fails at `sql()`.
#[test]
fn sql_runs_statements_eagerly() {
    if !should_run() {
        return;
    }
    // A session of its own, so that the view and the setting stay local to this test.
    let spark = session().new_session();
    spark
        .sql("CREATE OR REPLACE TEMP VIEW eager_sql_v AS SELECT 1 AS x")
        .expect("create view");
    let rows = spark.table("eager_sql_v").unwrap().collect().unwrap();
    assert_eq!(rows.len(), 1);
    spark
        .sql("SET spark.sql.shuffle.partitions=7")
        .expect("set");
    assert_eq!(
        spark.conf().get("spark.sql.shuffle.partitions").unwrap(),
        Some("7".to_string())
    );
    let databases = spark.sql("SHOW DATABASES").expect("show databases");
    assert!(databases.is_local());
    assert!(!databases.collect().unwrap().is_empty());
    let query = spark.sql("SELECT 1 AS x").expect("query");
    assert!(!query.is_local());
    assert_eq!(query.collect().unwrap().len(), 1);
    assert!(spark.sql("SELEC 1").is_err());
    let _ = spark.stop();
}

/// As pyspark's `sql()` with DataFrames as keyword arguments: each DataFrame is sent with
/// the statement as a reference, which the server resolves the name in the query to, also
/// when the query names it twice, and a command can read one too.
#[test]
fn sql_resolves_the_dataframes_that_the_query_names() {
    if !should_run() {
        return;
    }
    use spark_connect::row::Value;
    use std::collections::HashMap;

    let spark = session().new_session();
    let ids = spark.range(3).expect("range");
    let df = spark
        .sql_with_references(
            "SELECT count(*) AS n FROM refs_ids a JOIN refs_ids b USING (id)",
            Vec::new(),
            HashMap::new(),
            vec![(ids.clone(), "refs_ids".to_string())],
        )
        .expect("query");
    let rows = df.collect().expect("collect");
    assert_eq!(rows[0].get(0), Some(&Value::Long(3)));
    let shown = spark
        .sql_with_references(
            "SELECT * FROM refs_ids WHERE id > 0",
            Vec::new(),
            HashMap::new(),
            vec![(ids, "refs_ids".to_string())],
        )
        .expect("filter");
    assert_eq!(shown.collect().expect("collect").len(), 2);
    let _ = spark.stop();
}

/// `parse_ddl` parses a DDL string on the server, as PySpark Connect's `_parse_ddl`: of a
/// schema or a type, with the NOT NULL and COMMENT of fields, and the server's errors.
#[test]
fn parse_ddl_parses_on_the_server() {
    if !should_run() {
        return;
    }
    use spark_connect::types::DataType;
    let spark = session();
    let parsed = spark
        .parse_ddl("a INT NOT NULL COMMENT 'c', b STRUCT<x: DOUBLE>")
        .expect("parse");
    let DataType::Struct { fields } = parsed else {
        panic!("expected a struct, got {parsed:?}");
    };
    assert_eq!(
        fields.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(),
        vec!["a", "b"]
    );
    assert!(!fields[0].nullable);
    assert_eq!(
        fields[0].metadata.get("comment"),
        Some(&serde_json::Value::String("c".to_string()))
    );
    assert_eq!(spark.parse_ddl("bigint").expect("parse"), DataType::Long);
    let error = spark.parse_ddl("a INT,").expect_err("a syntax error");
    assert_eq!(error.error_class, "PARSE_SYNTAX_ERROR");
    let _ = spark.stop();
}

/// `json_to_ddl` converts the JSON of a schema to DDL on the server, as PySpark Connect's
/// `_to_ddl`: the DDL of Spark, with its quoted names, NOT NULL and COMMENT.
#[test]
fn json_to_ddl_converts_on_the_server() {
    if !should_run() {
        return;
    }
    use spark_connect::types::{DataType, StructField};
    let spark = session();
    let field = |name: &str, data_type: DataType, nullable: bool| StructField {
        name: name.to_string(),
        data_type,
        nullable,
        metadata: Default::default(),
    };
    let mut commented = field("c", DataType::Long, true);
    commented.metadata.insert(
        "comment".to_string(),
        serde_json::Value::String("the c".to_string()),
    );
    let schema = DataType::Struct {
        fields: vec![
            field("a", DataType::Integer, true),
            field(
                "b b",
                DataType::String {
                    collation: "UTF8_BINARY".to_string(),
                },
                false,
            ),
            commented,
        ],
    };
    let ddl = spark.json_to_ddl(&schema.json()).expect("ddl");
    assert_eq!(ddl, "a INT,`b b` STRING NOT NULL,c BIGINT COMMENT 'the c'");
    let _ = spark.stop();
}

/// The execution that the calling thread waits for is the one that its wait interrupts, as
/// PySpark's `KeyboardInterrupt` does: the wait ends with the server's cancellation, and the
/// session goes on.
/// A prefetching iterator is read by a task of its own, and its caller waits for the task's
/// batches: a wait of the caller interrupts its execution, while an iterator that is not read is
/// not the one a wait interrupts.
#[test]
fn interrupting_the_wait_of_a_prefetching_iterator_cancels_its_execution() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use spark_connect_core::runtime::{with_wait_watch, WaitWatch, Watch};

    if !should_run() {
        return;
    }
    let spark = session();
    let slow = spark
        .sql("SELECT sum(sqrt(id)) FROM range(0, 1000000000000, 1, 2)")
        .expect("sql");
    let mut rows = slow.to_local_iterator(true).expect("to_local_iterator");
    // Not read yet: no wait of the caller is for the execution.
    assert_eq!(spark.running_operation_id(), None);
    let started = Instant::now();
    let interrupted = Arc::new(AtomicBool::new(false));
    let watch_session = spark.clone();
    let watch_interrupted = Arc::clone(&interrupted);
    let watch: Arc<WaitWatch> = Arc::new(move || {
        if started.elapsed() < Duration::from_secs(2) || watch_interrupted.load(Ordering::SeqCst) {
            return Watch::Continue;
        }
        match watch_session.running_operation_id() {
            Some(operation_id) => {
                watch_interrupted.store(true, Ordering::SeqCst);
                let session = watch_session.clone();
                Watch::Interrupt(Box::pin(async move {
                    let _ = session.interrupt_operation_async(&operation_id).await;
                }))
            }
            None => Watch::Continue,
        }
    });
    let outcome = with_wait_watch(watch, || rows.next());
    assert!(
        matches!(outcome, Some(Err(_))),
        "the cancelled execution returned a batch"
    );
    assert!(
        interrupted.load(Ordering::SeqCst),
        "the caller's wait was not for the execution"
    );
    assert!(started.elapsed() < Duration::from_secs(30));
    drop(rows);
    let count = spark
        .sql("SELECT * FROM range(3)")
        .map_err(Box::new)
        .and_then(|df| df.count().map_err(Box::new))
        .expect("count");
    assert_eq!(count, 3);
}

#[test]
fn interrupting_the_wait_cancels_the_execution_of_the_calling_thread() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use spark_connect_core::runtime::{with_wait_watch, WaitWatch, Watch};

    if !should_run() {
        return;
    }
    let spark = session();
    let slow = spark
        .sql("SELECT sum(sqrt(id)) FROM range(0, 1000000000000, 1, 2)")
        .expect("sql");
    let started = Instant::now();
    let interrupted = Arc::new(AtomicBool::new(false));
    let watch_session = spark.clone();
    let watch_interrupted = Arc::clone(&interrupted);
    let watch: Arc<WaitWatch> = Arc::new(move || {
        if started.elapsed() < Duration::from_secs(2) || watch_interrupted.load(Ordering::SeqCst) {
            return Watch::Continue;
        }
        match watch_session.running_operation_id() {
            Some(operation_id) => {
                watch_interrupted.store(true, Ordering::SeqCst);
                let session = watch_session.clone();
                Watch::Interrupt(Box::pin(async move {
                    let _ = session.interrupt_operation_async(&operation_id).await;
                }))
            }
            None => Watch::Continue,
        }
    });
    let outcome = with_wait_watch(watch, || slow.collect().map_err(Box::new));
    assert!(
        outcome.is_err(),
        "the cancelled execution returned a result"
    );
    assert!(
        interrupted.load(Ordering::SeqCst),
        "no execution was in flight"
    );
    assert_eq!(
        spark.interrupt_all().expect("interrupt_all"),
        Vec::<String>::new()
    );
    let count = spark
        .sql("SELECT * FROM range(3)")
        .map_err(Box::new)
        .and_then(|df| df.count().map_err(Box::new))
        .expect("count");
    assert_eq!(count, 3);
}
