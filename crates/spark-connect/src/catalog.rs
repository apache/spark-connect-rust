//! Catalog API mirroring `pyspark.sql.connect.catalog.Catalog`.
//!
//! Provides access to database and table metadata, and catalog operations.

use spark_connect_core::error::{Result, SparkError};
use spark_connect_core::runtime::block_on;
use spark_connect_proto as proto;

use crate::dataframe::DataFrame;
use crate::row::{Row, Value};
use crate::session::SparkSession;

/// Metadata result classes returned by the typed catalog methods, mirroring
/// `pyspark.sql.catalog.{CatalogMetadata,Database,Table,Column,Function,TablePartition}`.
#[derive(Debug, Clone, PartialEq)]
pub struct CatalogMetadata {
    pub name: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Database {
    pub name: String,
    pub catalog: Option<String>,
    pub description: Option<String>,
    pub location_uri: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Table {
    pub name: String,
    pub catalog: Option<String>,
    pub namespace: Option<Vec<String>>,
    pub description: Option<String>,
    pub table_type: String,
    pub is_temporary: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Column {
    pub name: String,
    pub description: Option<String>,
    pub data_type: String,
    pub nullable: bool,
    pub is_partition: bool,
    pub is_bucket: bool,
    pub is_cluster: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: String,
    pub catalog: Option<String>,
    pub namespace: Option<Vec<String>>,
    pub description: Option<String>,
    pub class_name: String,
    pub is_temporary: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TablePartition {
    pub partition: String,
}

// Row-parsing helpers for the typed catalog results.
fn row_str(row: &Row, i: usize) -> String {
    row.get(i)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}
fn row_opt_str(row: &Row, i: usize) -> Option<String> {
    match row.get(i) {
        Some(v) if !v.is_null() => v.as_str().map(|s| s.to_string()),
        _ => None,
    }
}
fn row_bool(row: &Row, i: usize) -> bool {
    row.get(i).and_then(|v| v.as_bool()).unwrap_or(false)
}
fn row_opt_namespace(row: &Row, i: usize) -> Option<Vec<String>> {
    match row.get(i) {
        Some(Value::List(items)) => Some(
            items
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect(),
        ),
        _ => None,
    }
}
fn parse_database(r: &Row) -> Database {
    Database {
        name: row_str(r, 0),
        catalog: row_opt_str(r, 1),
        description: row_opt_str(r, 2),
        location_uri: row_str(r, 3),
    }
}
fn parse_table(r: &Row) -> Table {
    Table {
        name: row_str(r, 0),
        catalog: row_opt_str(r, 1),
        namespace: row_opt_namespace(r, 2),
        description: row_opt_str(r, 3),
        table_type: row_str(r, 4),
        is_temporary: row_bool(r, 5),
    }
}
fn parse_function(r: &Row) -> Function {
    Function {
        name: row_str(r, 0),
        catalog: row_opt_str(r, 1),
        namespace: row_opt_namespace(r, 2),
        description: row_opt_str(r, 3),
        class_name: row_str(r, 4),
        is_temporary: row_bool(r, 5),
    }
}

/// Catalog provides access to database and table metadata.
///
/// Mirrors `pyspark.sql.connect.catalog.Catalog`.
#[derive(Clone)]
pub struct Catalog {
    session: SparkSession,
}

impl Catalog {
    /// Create a new Catalog.
    pub(crate) fn new(session: SparkSession) -> Self {
        Catalog { session }
    }

    /// Get the current catalog.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.currentCatalog`.
    pub fn current_catalog(&self) -> Result<String> {
        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::CurrentCatalog(
            proto::CurrentCatalog::default(),
        ));

        let result = self.execute_catalog(&catalog_msg)?;
        if !result.is_empty() {
            if let Some(row) = result.first() {
                if let Some(value) = row.get(0) {
                    match value.as_str() {
                        Some(s) => return Ok(s.to_string()),
                        None => {
                            return Err(SparkError::connect_msg(
                                "currentCatalog returned non-string",
                            ))
                        }
                    }
                }
            }
        }
        Err(SparkError::connect_msg("currentCatalog returned no result"))
    }

    /// Set the current catalog.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.setCurrentCatalog`.
    pub fn set_current_catalog(&self, catalog_name: &str) -> Result<()> {
        let mut set_cat = proto::SetCurrentCatalog::default();
        set_cat.catalog_name = catalog_name.to_string();

        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::SetCurrentCatalog(set_cat));

        self.execute_catalog(&catalog_msg)?;
        Ok(())
    }

    /// List all catalogs.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.listCatalogs`.
    pub fn list_catalogs(&self) -> Result<DataFrame> {
        self.list_catalogs_with_pattern(None)
    }

    /// List all catalogs matching a pattern.
    pub fn list_catalogs_with_pattern(&self, pattern: Option<&str>) -> Result<DataFrame> {
        let mut list_cat = proto::ListCatalogs::default();
        if let Some(p) = pattern {
            list_cat.pattern = Some(p.to_string());
        }

        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::ListCatalogs(list_cat));

        self.execute_catalog_as_dataframe(&catalog_msg)
    }

    /// Get the current database.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.currentDatabase`.
    pub fn current_database(&self) -> Result<String> {
        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::CurrentDatabase(
            proto::CurrentDatabase::default(),
        ));

        let result = self.execute_catalog(&catalog_msg)?;
        if !result.is_empty() {
            if let Some(row) = result.first() {
                if let Some(value) = row.get(0) {
                    match value.as_str() {
                        Some(s) => return Ok(s.to_string()),
                        None => {
                            return Err(SparkError::connect_msg(
                                "currentDatabase returned non-string",
                            ))
                        }
                    }
                }
            }
        }
        Err(SparkError::connect_msg(
            "currentDatabase returned no result",
        ))
    }

    /// Set the current database.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.setCurrentDatabase`.
    pub fn set_current_database(&self, db_name: &str) -> Result<()> {
        let mut set_db = proto::SetCurrentDatabase::default();
        set_db.db_name = db_name.to_string();

        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::SetCurrentDatabase(set_db));

        self.execute_catalog(&catalog_msg)?;
        Ok(())
    }

    /// List all databases.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.listDatabases`.
    pub fn list_databases(&self) -> Result<DataFrame> {
        self.list_databases_with_pattern(None)
    }

    /// List all databases matching a pattern.
    pub fn list_databases_with_pattern(&self, pattern: Option<&str>) -> Result<DataFrame> {
        let mut list_db = proto::ListDatabases::default();
        if let Some(p) = pattern {
            list_db.pattern = Some(p.to_string());
        }

        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::ListDatabases(list_db));

        self.execute_catalog_as_dataframe(&catalog_msg)
    }

    /// Get database metadata.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.getDatabase`.
    pub fn get_database(&self, db_name: &str) -> Result<DataFrame> {
        let mut get_db = proto::GetDatabase::default();
        get_db.db_name = db_name.to_string();

        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::GetDatabase(get_db));

        self.execute_catalog_as_dataframe(&catalog_msg)
    }

    /// Check if a database exists.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.databaseExists`.
    pub fn database_exists(&self, db_name: &str) -> Result<bool> {
        let mut db_exists = proto::DatabaseExists::default();
        db_exists.db_name = db_name.to_string();

        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::DatabaseExists(db_exists));

        let result = self.execute_catalog(&catalog_msg)?;
        if !result.is_empty() {
            if let Some(row) = result.first() {
                if let Some(value) = row.get(0) {
                    match value.as_bool() {
                        Some(b) => return Ok(b),
                        None => {
                            return Err(SparkError::connect_msg(
                                "databaseExists returned non-boolean",
                            ))
                        }
                    }
                }
            }
        }
        Ok(false)
    }

    /// List all tables in a database.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.listTables`.
    pub fn list_tables(&self) -> Result<DataFrame> {
        self.list_tables_with_pattern(None, None)
    }

    /// List all tables in a specific database.
    pub fn list_tables_in_database(&self, db_name: &str) -> Result<DataFrame> {
        self.list_tables_with_pattern(Some(db_name), None)
    }

    /// List all tables in a database matching a pattern.
    pub fn list_tables_with_pattern(
        &self,
        db_name: Option<&str>,
        pattern: Option<&str>,
    ) -> Result<DataFrame> {
        let mut list_tbl = proto::ListTables::default();
        if let Some(db) = db_name {
            list_tbl.db_name = Some(db.to_string());
        }
        if let Some(p) = pattern {
            list_tbl.pattern = Some(p.to_string());
        }

        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::ListTables(list_tbl));

        self.execute_catalog_as_dataframe(&catalog_msg)
    }

    /// Get table metadata.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.getTable`.
    pub fn get_table(&self, table_name: &str) -> Result<DataFrame> {
        self.get_table_with_database(table_name, None)
    }

    /// Get table metadata from a specific database.
    pub fn get_table_with_database(
        &self,
        table_name: &str,
        db_name: Option<&str>,
    ) -> Result<DataFrame> {
        let mut get_tbl = proto::GetTable::default();
        get_tbl.table_name = table_name.to_string();
        if let Some(db) = db_name {
            get_tbl.db_name = Some(db.to_string());
        }

        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::GetTable(get_tbl));

        self.execute_catalog_as_dataframe(&catalog_msg)
    }

    /// Check if a table exists.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.tableExists`.
    pub fn table_exists(&self, table_name: &str) -> Result<bool> {
        self.table_exists_with_database(table_name, None)
    }

    /// Check if a table exists in a specific database.
    pub fn table_exists_with_database(
        &self,
        table_name: &str,
        db_name: Option<&str>,
    ) -> Result<bool> {
        let mut tbl_exists = proto::TableExists::default();
        tbl_exists.table_name = table_name.to_string();
        if let Some(db) = db_name {
            tbl_exists.db_name = Some(db.to_string());
        }

        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::TableExists(tbl_exists));

        let result = self.execute_catalog(&catalog_msg)?;
        if !result.is_empty() {
            if let Some(row) = result.first() {
                if let Some(value) = row.get(0) {
                    match value.as_bool() {
                        Some(b) => return Ok(b),
                        None => {
                            return Err(SparkError::connect_msg("tableExists returned non-boolean"))
                        }
                    }
                }
            }
        }
        Ok(false)
    }

    /// List columns of a table.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.listColumns`.
    pub fn list_columns(&self, table_name: &str) -> Result<DataFrame> {
        self.list_columns_with_database(table_name, None)
    }

    /// List columns of a table in a specific database.
    pub fn list_columns_with_database(
        &self,
        table_name: &str,
        db_name: Option<&str>,
    ) -> Result<DataFrame> {
        let mut list_cols = proto::ListColumns::default();
        list_cols.table_name = table_name.to_string();
        if let Some(db) = db_name {
            list_cols.db_name = Some(db.to_string());
        }

        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::ListColumns(list_cols));

        self.execute_catalog_as_dataframe(&catalog_msg)
    }

    /// List all functions in the current database.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.listFunctions`.
    pub fn list_functions(&self) -> Result<DataFrame> {
        self.list_functions_with_pattern(None, None)
    }

    /// List all functions in a specific database.
    pub fn list_functions_in_database(&self, db_name: &str) -> Result<DataFrame> {
        self.list_functions_with_pattern(Some(db_name), None)
    }

    /// List all functions matching a pattern.
    pub fn list_functions_with_pattern(
        &self,
        db_name: Option<&str>,
        pattern: Option<&str>,
    ) -> Result<DataFrame> {
        let mut list_funcs = proto::ListFunctions::default();
        if let Some(db) = db_name {
            list_funcs.db_name = Some(db.to_string());
        }
        if let Some(p) = pattern {
            list_funcs.pattern = Some(p.to_string());
        }

        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::ListFunctions(list_funcs));

        self.execute_catalog_as_dataframe(&catalog_msg)
    }

    /// Get function metadata.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.getFunction`.
    pub fn get_function(&self, function_name: &str) -> Result<DataFrame> {
        self.get_function_with_database(function_name, None)
    }

    /// Get function metadata from a specific database.
    pub fn get_function_with_database(
        &self,
        function_name: &str,
        db_name: Option<&str>,
    ) -> Result<DataFrame> {
        let mut get_func = proto::GetFunction::default();
        get_func.function_name = function_name.to_string();
        if let Some(db) = db_name {
            get_func.db_name = Some(db.to_string());
        }

        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::GetFunction(get_func));

        self.execute_catalog_as_dataframe(&catalog_msg)
    }

    /// Check if a function exists.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.functionExists`.
    pub fn function_exists(&self, function_name: &str) -> Result<bool> {
        self.function_exists_with_database(function_name, None)
    }

    /// Check if a function exists in a specific database.
    pub fn function_exists_with_database(
        &self,
        function_name: &str,
        db_name: Option<&str>,
    ) -> Result<bool> {
        let mut func_exists = proto::FunctionExists::default();
        func_exists.function_name = function_name.to_string();
        if let Some(db) = db_name {
            func_exists.db_name = Some(db.to_string());
        }

        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::FunctionExists(func_exists));

        let result = self.execute_catalog(&catalog_msg)?;
        if !result.is_empty() {
            if let Some(row) = result.first() {
                if let Some(value) = row.get(0) {
                    match value.as_bool() {
                        Some(b) => return Ok(b),
                        None => {
                            return Err(SparkError::connect_msg(
                                "functionExists returned non-boolean",
                            ))
                        }
                    }
                }
            }
        }
        Ok(false)
    }

    /// Create an external table.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.createExternalTable`.
    pub fn create_external_table(
        &self,
        table_name: &str,
        path: Option<&str>,
        source: Option<&str>,
    ) -> Result<DataFrame> {
        let mut create_ext = proto::CreateExternalTable::default();
        create_ext.table_name = table_name.to_string();
        if let Some(p) = path {
            create_ext.path = Some(p.to_string());
        }
        if let Some(s) = source {
            create_ext.source = Some(s.to_string());
        }

        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::CreateExternalTable(create_ext));

        self.execute_catalog_as_dataframe(&catalog_msg)
    }

    /// Create a table.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.createTable`.
    pub fn create_table(
        &self,
        table_name: &str,
        path: Option<&str>,
        source: Option<&str>,
        description: Option<&str>,
    ) -> Result<DataFrame> {
        let mut create_tbl = proto::CreateTable::default();
        create_tbl.table_name = table_name.to_string();
        if let Some(p) = path {
            create_tbl.path = Some(p.to_string());
        }
        if let Some(s) = source {
            create_tbl.source = Some(s.to_string());
        }
        if let Some(d) = description {
            create_tbl.description = Some(d.to_string());
        }

        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::CreateTable(create_tbl));

        self.execute_catalog_as_dataframe(&catalog_msg)
    }

    /// Drop a temporary view.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.dropTempView`.
    pub fn drop_temp_view(&self, view_name: &str) -> Result<bool> {
        let mut drop_view = proto::DropTempView::default();
        drop_view.view_name = view_name.to_string();

        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::DropTempView(drop_view));

        let result = self.execute_catalog(&catalog_msg)?;
        if !result.is_empty() {
            if let Some(row) = result.first() {
                if let Some(value) = row.get(0) {
                    match value.as_bool() {
                        Some(b) => return Ok(b),
                        None => {
                            return Err(SparkError::connect_msg(
                                "dropTempView returned non-boolean",
                            ))
                        }
                    }
                }
            }
        }
        Ok(false)
    }

    /// Drop a global temporary view.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.dropGlobalTempView`.
    pub fn drop_global_temp_view(&self, view_name: &str) -> Result<bool> {
        let mut drop_global = proto::DropGlobalTempView::default();
        drop_global.view_name = view_name.to_string();

        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::DropGlobalTempView(drop_global));

        let result = self.execute_catalog(&catalog_msg)?;
        if !result.is_empty() {
            if let Some(row) = result.first() {
                if let Some(value) = row.get(0) {
                    match value.as_bool() {
                        Some(b) => return Ok(b),
                        None => {
                            return Err(SparkError::connect_msg(
                                "dropGlobalTempView returned non-boolean",
                            ))
                        }
                    }
                }
            }
        }
        Ok(false)
    }

    /// Cache a table.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.cacheTable`.
    pub fn cache_table(&self, table_name: &str) -> Result<()> {
        self.cache_table_with_storage_level(table_name, None)
    }

    /// Cache a table at an optional storage level (`CacheTable.storage_level`, tag 2).
    /// Mirrors `Catalog.cacheTable(tableName, storageLevel)`.
    pub fn cache_table_with_storage_level(
        &self,
        table_name: &str,
        storage_level: Option<proto::StorageLevel>,
    ) -> Result<()> {
        let mut cache = proto::CacheTable::default();
        cache.table_name = table_name.to_string();
        cache.storage_level = storage_level;

        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::CacheTable(cache));

        self.execute_catalog(&catalog_msg)?;
        Ok(())
    }

    /// Uncache a table.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.uncacheTable`.
    pub fn uncache_table(&self, table_name: &str) -> Result<()> {
        let mut uncache = proto::UncacheTable::default();
        uncache.table_name = table_name.to_string();

        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::UncacheTable(uncache));

        self.execute_catalog(&catalog_msg)?;
        Ok(())
    }

    /// Check if a table is cached.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.isCached`.
    pub fn is_cached(&self, table_name: &str) -> Result<bool> {
        let mut is_cached = proto::IsCached::default();
        is_cached.table_name = table_name.to_string();

        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::IsCached(is_cached));

        let result = self.execute_catalog(&catalog_msg)?;
        if !result.is_empty() {
            if let Some(row) = result.first() {
                if let Some(value) = row.get(0) {
                    match value.as_bool() {
                        Some(b) => return Ok(b),
                        None => {
                            return Err(SparkError::connect_msg("isCached returned non-boolean"))
                        }
                    }
                }
            }
        }
        Ok(false)
    }

    /// Clear the cache.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.clearCache`.
    pub fn clear_cache(&self) -> Result<()> {
        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::ClearCache(
            proto::ClearCache::default(),
        ));

        self.execute_catalog(&catalog_msg)?;
        Ok(())
    }

    /// Refresh a table.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.refreshTable`.
    pub fn refresh_table(&self, table_name: &str) -> Result<()> {
        let mut refresh = proto::RefreshTable::default();
        refresh.table_name = table_name.to_string();

        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::RefreshTable(refresh));

        self.execute_catalog(&catalog_msg)?;
        Ok(())
    }

    /// Refresh by path.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.refreshByPath`.
    pub fn refresh_by_path(&self, path: &str) -> Result<()> {
        let mut refresh_path = proto::RefreshByPath::default();
        refresh_path.path = path.to_string();

        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::RefreshByPath(refresh_path));

        self.execute_catalog(&catalog_msg)?;
        Ok(())
    }

    /// Recover partitions.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.recoverPartitions`.
    pub fn recover_partitions(&self, table_name: &str) -> Result<()> {
        let mut recover = proto::RecoverPartitions::default();
        recover.table_name = table_name.to_string();

        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::RecoverPartitions(recover));

        self.execute_catalog(&catalog_msg)?;
        Ok(())
    }

    /// Create a database.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.createDatabase`.
    pub fn create_database(
        &self,
        db_name: &str,
        if_not_exists: bool,
        properties: std::collections::HashMap<String, String>,
    ) -> Result<()> {
        let create_db = proto::CreateDatabase {
            db_name: db_name.to_string(),
            if_not_exists,
            properties,
        };
        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::CreateDatabase(create_db));
        self.execute_catalog(&catalog_msg)?;
        Ok(())
    }

    /// Drop a database.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.dropDatabase`.
    pub fn drop_database(&self, db_name: &str, if_exists: bool, cascade: bool) -> Result<()> {
        let drop_db = proto::DropDatabase {
            db_name: db_name.to_string(),
            if_exists,
            cascade,
        };
        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::DropDatabase(drop_db));
        self.execute_catalog(&catalog_msg)?;
        Ok(())
    }

    /// Drop a table.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.dropTable`.
    pub fn drop_table(&self, table_name: &str, if_exists: bool, purge: bool) -> Result<()> {
        let drop_tbl = proto::DropTable {
            table_name: table_name.to_string(),
            if_exists,
            purge,
        };
        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::DropTable(drop_tbl));
        self.execute_catalog(&catalog_msg)?;
        Ok(())
    }

    /// Drop a view.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.dropView`.
    pub fn drop_view(&self, view_name: &str, if_exists: bool) -> Result<()> {
        let drop_view = proto::DropView {
            view_name: view_name.to_string(),
            if_exists,
        };
        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::DropView(drop_view));
        self.execute_catalog(&catalog_msg)?;
        Ok(())
    }

    /// Truncate a table.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.truncateTable`.
    pub fn truncate_table(&self, table_name: &str) -> Result<()> {
        let truncate = proto::TruncateTable {
            table_name: table_name.to_string(),
        };
        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::TruncateTable(truncate));
        self.execute_catalog(&catalog_msg)?;
        Ok(())
    }

    /// Recover the statistics of a table.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.analyzeTable`.
    pub fn analyze_table(&self, table_name: &str, no_scan: bool) -> Result<()> {
        let analyze = proto::AnalyzeTable {
            table_name: table_name.to_string(),
            no_scan,
        };
        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::AnalyzeTable(analyze));
        self.execute_catalog(&catalog_msg)?;
        Ok(())
    }

    /// Get the `CREATE TABLE` string of a table.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.getCreateTableString`. Returns the
    /// first row's first column, or an empty string when there are no rows.
    pub fn get_create_table_string(&self, table_name: &str, as_serde: bool) -> Result<String> {
        let get = proto::GetCreateTableString {
            table_name: table_name.to_string(),
            as_serde,
        };
        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::GetCreateTableString(get));
        let result = self.execute_catalog(&catalog_msg)?;
        match result.first().and_then(|row| row.get(0)) {
            Some(value) => match value.as_str() {
                Some(s) => Ok(s.to_string()),
                None => Err(SparkError::connect_msg(
                    "getCreateTableString returned non-string",
                )),
            },
            None => Ok(String::new()),
        }
    }

    /// Get the properties of a table as (key, value) pairs.
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.getTableProperties`: the result has
    /// two columns (key, value), one row per property.
    pub fn get_table_properties(&self, table_name: &str) -> Result<Vec<(String, String)>> {
        let get = proto::GetTableProperties {
            table_name: table_name.to_string(),
        };
        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::GetTableProperties(get));
        let result = self.execute_catalog(&catalog_msg)?;
        let mut props = Vec::with_capacity(result.len());
        for row in &result {
            let key = row
                .get(0)
                .and_then(|v| v.as_str())
                .ok_or_else(|| SparkError::connect_msg("getTableProperties key not a string"))?;
            let value = row
                .get(1)
                .and_then(|v| v.as_str())
                .ok_or_else(|| SparkError::connect_msg("getTableProperties value not a string"))?;
            props.push((key.to_string(), value.to_string()));
        }
        Ok(props)
    }

    /// List the partitions of a table (returns a DataFrame, matching the other
    /// `list_*` catalog methods).
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.listPartitions`.
    pub fn list_partitions(&self, table_name: &str) -> Result<DataFrame> {
        let list = proto::ListPartitions {
            table_name: table_name.to_string(),
        };
        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::ListPartitions(list));
        self.execute_catalog_as_dataframe(&catalog_msg)
    }

    /// List the views (returns a DataFrame, matching the other `list_*` methods).
    ///
    /// Mirrors `pyspark.sql.connect.catalog.Catalog.listViews`: when a pattern is given
    /// without a database, the current database is used.
    pub fn list_views(&self, db_name: Option<&str>, pattern: Option<&str>) -> Result<DataFrame> {
        let resolved_db = if pattern.is_some() && db_name.is_none() {
            Some(self.current_database()?)
        } else {
            db_name.map(|s| s.to_string())
        };
        let list = proto::ListViews {
            db_name: resolved_db,
            pattern: pattern.map(|s| s.to_string()),
        };
        let mut catalog_msg = proto::Catalog::default();
        catalog_msg.cat_type = Some(proto::catalog::CatType::ListViews(list));
        self.execute_catalog_as_dataframe(&catalog_msg)
    }

    /// Helper: execute a catalog operation and return results as Rows.
    // ---- Typed catalog results (mirror pyspark's List[Table]/List[Database]/... ) ----
    // Each reuses the DataFrame-returning method, collects the rows, and parses them
    // into the metadata structs above (column order matches the reference client).

    /// Typed `listCatalogs` -> `Vec<CatalogMetadata>`.
    pub fn list_catalogs_typed(&self, pattern: Option<&str>) -> Result<Vec<CatalogMetadata>> {
        let rows = self.list_catalogs_with_pattern(pattern)?.collect()?;
        Ok(rows
            .iter()
            .map(|r| CatalogMetadata {
                name: row_str(r, 0),
                description: row_opt_str(r, 1),
            })
            .collect())
    }

    /// Typed `listDatabases` -> `Vec<Database>`.
    pub fn list_databases_typed(&self, pattern: Option<&str>) -> Result<Vec<Database>> {
        let rows = self.list_databases_with_pattern(pattern)?.collect()?;
        Ok(rows.iter().map(parse_database).collect())
    }

    /// Typed `getDatabase` -> `Database`.
    pub fn get_database_typed(&self, db_name: &str) -> Result<Database> {
        let rows = self.get_database(db_name)?.collect()?;
        rows.first()
            .map(parse_database)
            .ok_or_else(|| SparkError::connect_msg("getDatabase returned no result"))
    }

    /// Typed `listTables` -> `Vec<Table>`.
    pub fn list_tables_typed(
        &self,
        db_name: Option<&str>,
        pattern: Option<&str>,
    ) -> Result<Vec<Table>> {
        let rows = self.list_tables_with_pattern(db_name, pattern)?.collect()?;
        Ok(rows.iter().map(parse_table).collect())
    }

    /// Typed `getTable` -> `Table`.
    pub fn get_table_typed(&self, table_name: &str) -> Result<Table> {
        let rows = self.get_table(table_name)?.collect()?;
        rows.first()
            .map(parse_table)
            .ok_or_else(|| SparkError::connect_msg("getTable returned no result"))
    }

    /// Typed `listFunctions` -> `Vec<Function>`.
    pub fn list_functions_typed(
        &self,
        db_name: Option<&str>,
        pattern: Option<&str>,
    ) -> Result<Vec<Function>> {
        let rows = self
            .list_functions_with_pattern(db_name, pattern)?
            .collect()?;
        Ok(rows.iter().map(parse_function).collect())
    }

    /// Typed `getFunction` -> `Function`.
    pub fn get_function_typed(&self, function_name: &str) -> Result<Function> {
        let rows = self.get_function(function_name)?.collect()?;
        rows.first()
            .map(parse_function)
            .ok_or_else(|| SparkError::connect_msg("getFunction returned no result"))
    }

    /// Typed `listColumns` -> `Vec<Column>`.
    pub fn list_columns_typed(
        &self,
        table_name: &str,
        db_name: Option<&str>,
    ) -> Result<Vec<Column>> {
        let rows = self
            .list_columns_with_database(table_name, db_name)?
            .collect()?;
        Ok(rows
            .iter()
            .map(|r| Column {
                name: row_str(r, 0),
                description: row_opt_str(r, 1),
                data_type: row_str(r, 2),
                nullable: row_bool(r, 3),
                is_partition: row_bool(r, 4),
                is_bucket: row_bool(r, 5),
                is_cluster: row_bool(r, 6),
            })
            .collect())
    }

    /// Typed `listPartitions` -> `Vec<TablePartition>`.
    pub fn list_partitions_typed(&self, table_name: &str) -> Result<Vec<TablePartition>> {
        let rows = self.list_partitions(table_name)?.collect()?;
        Ok(rows
            .iter()
            .map(|r| TablePartition {
                partition: row_str(r, 0),
            })
            .collect())
    }

    /// Typed `listViews` -> `Vec<Table>` (views share the Table result shape).
    pub fn list_views_typed(
        &self,
        db_name: Option<&str>,
        pattern: Option<&str>,
    ) -> Result<Vec<Table>> {
        let rows = self.list_views(db_name, pattern)?.collect()?;
        Ok(rows.iter().map(parse_table).collect())
    }

    fn execute_catalog(&self, catalog: &proto::Catalog) -> Result<Vec<Row>> {
        let request = self.build_execute_catalog_request(catalog)?;
        let mut stream = block_on(self.session.client().execute_plan(request))?;

        let mut rows = vec![];

        loop {
            let resp = block_on(stream.message()).map_err(SparkError::from_grpc_status)?;
            let Some(resp) = resp else {
                break;
            };
            if let Some(proto::execute_plan_response::ResponseType::ArrowBatch(batch)) =
                resp.response_type
            {
                let batch_rows = decode_arrow_batch(&batch)?;
                rows.extend(batch_rows);
            }
        }

        Ok(rows)
    }

    /// Helper: expose a catalog operation as a lazy DataFrame.
    ///
    /// The catalog op is a relation on the server, so we wrap it in a plan and let
    /// `.collect()` (or any downstream op) evaluate it. This preserves the real
    /// server-side schema and row data - and, crucially, returns an empty result
    /// for an empty database (e.g. `listTables` with no tables) rather than erroring.
    fn execute_catalog_as_dataframe(&self, catalog: &proto::Catalog) -> Result<DataFrame> {
        let plan = crate::plan::LogicalPlan::Catalog {
            catalog: catalog.clone(),
        };
        Ok(DataFrame::new(self.session.clone(), plan))
    }

    /// Build an ExecutePlanRequest for a catalog operation.
    fn build_execute_catalog_request(
        &self,
        catalog: &proto::Catalog,
    ) -> Result<proto::ExecutePlanRequest> {
        let mut relation = proto::Relation::default();
        relation.common = Some(proto::RelationCommon::default());
        relation.rel_type = Some(proto::relation::RelType::Catalog(catalog.clone()));

        let mut plan = proto::Plan::default();
        plan.op_type = Some(proto::plan::OpType::Root(relation));

        let mut request = proto::ExecutePlanRequest::default();
        request.session_id = self.session.client().session_id().to_string();
        request.user_context = Some(proto::UserContext::default());
        request.plan = Some(plan);

        Ok(request)
    }
}

/// Decode an Arrow batch into rows.
pub(crate) fn decode_arrow_batch(
    batch: &proto::execute_plan_response::ArrowBatch,
) -> Result<Vec<Row>> {
    use arrow::ipc::reader::StreamReader;
    use std::io::Cursor;

    if batch.data.is_empty() {
        return Ok(vec![]);
    }

    let cursor = Cursor::new(&batch.data);
    let mut reader = StreamReader::try_new(cursor, None).map_err(|e| {
        SparkError::connect_msg(format!("Failed to create Arrow stream reader: {}", e))
    })?;

    let mut rows = vec![];

    while let Some(record_batch) = reader
        .next()
        .transpose()
        .map_err(|e| SparkError::connect_msg(format!("Failed to decode Arrow batch: {}", e)))?
    {
        let schema = record_batch.schema();
        let num_rows = record_batch.num_rows();
        let num_cols = record_batch.num_columns();

        for row_idx in 0..num_rows {
            let mut field_names = vec![];
            let mut values = vec![];

            for col_idx in 0..num_cols {
                let field_name = schema.field(col_idx).name().clone();
                let column = record_batch.column(col_idx);

                let value = arrow_value_at(column.as_ref(), row_idx)?;
                field_names.push(field_name);
                values.push(value);
            }

            rows.push(Row::new(field_names, values));
        }
    }

    Ok(rows)
}

/// Extract a value at a row index from an Arrow array. Delegates to the single,
/// comprehensive decoder in `dataframe` so catalog results support every type too.
fn arrow_value_at(array: &dyn arrow::array::Array, index: usize) -> Result<Value> {
    crate::dataframe::arrow_value_at(array, index)
}
