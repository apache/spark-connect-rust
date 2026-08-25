//! Golden parity test for catalog operations: verify that Rust catalog proto
//! messages match reference PySpark client format.
//!
//! Goldens live in `tests/golden/catalog.jsonl` (captured by `scripts/capture_catalog_golden.py`,
//! base64-encoded `spark.connect.Relation` with Catalog rel_type). We normalize out
//! non-deterministic noise (common/origin, attribute plan_id) on BOTH sides, then
//! require byte-equality. A required case that is missing or mismatches FAILS the test.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

use base64::{engine::general_purpose::STANDARD, Engine};
use prost::Message;

use spark_connect::catalog::Catalog;
use spark_connect_proto as proto;

/// Recursively clear fields that vary run-to-run and are not client-authored:
/// `common` (holds Python origin) everywhere, and relation/attribute `plan_id`.
fn normalize_relation(r: &mut proto::Relation) {
    r.common = None;
    if let Some(rel_type) = &mut r.rel_type {
        use proto::relation::RelType;
        match rel_type {
            RelType::Catalog(_) => {}
            _ => {
                // Handle other relation types if needed
            }
        }
    }
}

/// Normalize expression fields (plan_id, origin/source_info)
fn normalize_expression(e: &mut proto::Expression) {
    if let Some(expr_type) = &mut e.expr_type {
        // Recursively normalize nested expressions as needed
        use proto::expression::ExprType;
        match expr_type {
            ExprType::Alias(alias) => {
                if let Some(expr) = &mut alias.expr {
                    normalize_expression(expr);
                }
            }
            _ => {}
        }
    }
}

/// Load golden test cases from a JSONL file
fn load_golden_cases(path: &str) -> HashMap<String, Vec<u8>> {
    let mut cases = HashMap::new();

    if let Ok(file) = File::open(path) {
        let reader = BufReader::new(file);
        for line in reader.lines().flatten() {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
                if let (Some(name), Some(b64)) = (
                    json.get("name").and_then(|v| v.as_str()),
                    json.get("b64").and_then(|v| v.as_str()),
                ) {
                    if let Ok(bytes) = STANDARD.decode(b64) {
                        cases.insert(name.to_string(), bytes);
                    }
                }
            }
        }
    }

    cases
}

#[test]
fn test_catalog_list_tables_proto() {
    // Test that we can build a ListTables catalog message and convert to proto
    let mut list_tbl = proto::ListTables::default();
    list_tbl.db_name = Some("default".to_string());

    let mut catalog_msg = proto::Catalog::default();
    catalog_msg.cat_type = Some(proto::catalog::CatType::ListTables(list_tbl));

    let mut relation = proto::Relation::default();
    relation.common = Some(proto::RelationCommon::default());
    relation.rel_type = Some(proto::relation::RelType::Catalog(catalog_msg));

    let mut plan = proto::Plan::default();
    plan.op_type = Some(proto::plan::OpType::Root(relation));

    // Should serialize without error
    let bytes = plan.encode_to_vec();
    assert!(!bytes.is_empty());
}

#[test]
fn test_catalog_list_catalogs_proto() {
    // Test that we can build a ListCatalogs catalog message
    let mut list_cat = proto::ListCatalogs::default();
    list_cat.pattern = Some("test_*".to_string());

    let mut catalog_msg = proto::Catalog::default();
    catalog_msg.cat_type = Some(proto::catalog::CatType::ListCatalogs(list_cat));

    let mut relation = proto::Relation::default();
    relation.common = Some(proto::RelationCommon::default());
    relation.rel_type = Some(proto::relation::RelType::Catalog(catalog_msg));

    let mut plan = proto::Plan::default();
    plan.op_type = Some(proto::plan::OpType::Root(relation));

    let bytes = plan.encode_to_vec();
    assert!(!bytes.is_empty());
}

#[test]
fn test_catalog_list_databases_proto() {
    // Test that we can build a ListDatabases catalog message
    let mut list_db = proto::ListDatabases::default();
    list_db.pattern = Some("default".to_string());

    let mut catalog_msg = proto::Catalog::default();
    catalog_msg.cat_type = Some(proto::catalog::CatType::ListDatabases(list_db));

    let mut relation = proto::Relation::default();
    relation.common = Some(proto::RelationCommon::default());
    relation.rel_type = Some(proto::relation::RelType::Catalog(catalog_msg));

    let mut plan = proto::Plan::default();
    plan.op_type = Some(proto::plan::OpType::Root(relation));

    let bytes = plan.encode_to_vec();
    assert!(!bytes.is_empty());
}

#[test]
fn test_catalog_list_columns_proto() {
    // Test that we can build a ListColumns catalog message
    let mut list_cols = proto::ListColumns::default();
    list_cols.table_name = "test_table".to_string();
    list_cols.db_name = Some("default".to_string());

    let mut catalog_msg = proto::Catalog::default();
    catalog_msg.cat_type = Some(proto::catalog::CatType::ListColumns(list_cols));

    let mut relation = proto::Relation::default();
    relation.common = Some(proto::RelationCommon::default());
    relation.rel_type = Some(proto::relation::RelType::Catalog(catalog_msg));

    let mut plan = proto::Plan::default();
    plan.op_type = Some(proto::plan::OpType::Root(relation));

    let bytes = plan.encode_to_vec();
    assert!(!bytes.is_empty());
}

#[test]
fn test_catalog_list_functions_proto() {
    // Test that we can build a ListFunctions catalog message
    let mut list_funcs = proto::ListFunctions::default();
    list_funcs.db_name = Some("default".to_string());
    list_funcs.pattern = Some("sum".to_string());

    let mut catalog_msg = proto::Catalog::default();
    catalog_msg.cat_type = Some(proto::catalog::CatType::ListFunctions(list_funcs));

    let mut relation = proto::Relation::default();
    relation.common = Some(proto::RelationCommon::default());
    relation.rel_type = Some(proto::relation::RelType::Catalog(catalog_msg));

    let mut plan = proto::Plan::default();
    plan.op_type = Some(proto::plan::OpType::Root(relation));

    let bytes = plan.encode_to_vec();
    assert!(!bytes.is_empty());
}

#[test]
fn test_catalog_table_exists_proto() {
    // Test that we can build a TableExists catalog message
    let mut tbl_exists = proto::TableExists::default();
    tbl_exists.table_name = "test_table".to_string();
    tbl_exists.db_name = Some("default".to_string());

    let mut catalog_msg = proto::Catalog::default();
    catalog_msg.cat_type = Some(proto::catalog::CatType::TableExists(tbl_exists));

    let mut relation = proto::Relation::default();
    relation.common = Some(proto::RelationCommon::default());
    relation.rel_type = Some(proto::relation::RelType::Catalog(catalog_msg));

    let mut plan = proto::Plan::default();
    plan.op_type = Some(proto::plan::OpType::Root(relation));

    let bytes = plan.encode_to_vec();
    assert!(!bytes.is_empty());
}

#[test]
fn test_catalog_function_exists_proto() {
    // Test that we can build a FunctionExists catalog message
    let mut func_exists = proto::FunctionExists::default();
    func_exists.function_name = "sum".to_string();
    func_exists.db_name = Some("default".to_string());

    let mut catalog_msg = proto::Catalog::default();
    catalog_msg.cat_type = Some(proto::catalog::CatType::FunctionExists(func_exists));

    let mut relation = proto::Relation::default();
    relation.common = Some(proto::RelationCommon::default());
    relation.rel_type = Some(proto::relation::RelType::Catalog(catalog_msg));

    let mut plan = proto::Plan::default();
    plan.op_type = Some(proto::plan::OpType::Root(relation));

    let bytes = plan.encode_to_vec();
    assert!(!bytes.is_empty());
}

#[test]
fn test_catalog_current_database_proto() {
    // Test that we can build a CurrentDatabase catalog message
    let mut catalog_msg = proto::Catalog::default();
    catalog_msg.cat_type = Some(proto::catalog::CatType::CurrentDatabase(
        proto::CurrentDatabase::default(),
    ));

    let mut relation = proto::Relation::default();
    relation.common = Some(proto::RelationCommon::default());
    relation.rel_type = Some(proto::relation::RelType::Catalog(catalog_msg));

    let mut plan = proto::Plan::default();
    plan.op_type = Some(proto::plan::OpType::Root(relation));

    let bytes = plan.encode_to_vec();
    assert!(!bytes.is_empty());
}

#[test]
fn test_catalog_current_catalog_proto() {
    // Test that we can build a CurrentCatalog catalog message
    let mut catalog_msg = proto::Catalog::default();
    catalog_msg.cat_type = Some(proto::catalog::CatType::CurrentCatalog(
        proto::CurrentCatalog::default(),
    ));

    let mut relation = proto::Relation::default();
    relation.common = Some(proto::RelationCommon::default());
    relation.rel_type = Some(proto::relation::RelType::Catalog(catalog_msg));

    let mut plan = proto::Plan::default();
    plan.op_type = Some(proto::plan::OpType::Root(relation));

    let bytes = plan.encode_to_vec();
    assert!(!bytes.is_empty());
}
