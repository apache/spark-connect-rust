//! Catalog API mirroring `pyspark.sql.connect.catalog.Catalog`.
//!
//! Provides access to database and table metadata, and catalog operations.

use spark_connect_core::error::{Result, SparkError};
use spark_connect_core::runtime::block_on;
use spark_connect_proto as proto;

use crate::dataframe::DataFrame;
use crate::row::{Row, Value};
use crate::session::SparkSession;
use crate::types::DataType;

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
        let mut cache = proto::CacheTable::default();
        cache.table_name = table_name.to_string();

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

    /// Helper: execute a catalog operation and return results as Rows.
    fn execute_catalog(&self, catalog: &proto::Catalog) -> Result<Vec<Row>> {
        let request = self.build_execute_catalog_request(catalog)?;
        let mut stream = block_on(self.session.client().execute_plan(request))?;

        let mut rows = vec![];

        loop {
            let resp = block_on(stream.message()).map_err(|e| SparkError::from_grpc_status(e))?;
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

    /// Helper: execute a catalog operation and return as DataFrame.
    fn execute_catalog_as_dataframe(&self, catalog: &proto::Catalog) -> Result<DataFrame> {
        let rows = self.execute_catalog(catalog)?;

        if rows.is_empty() {
            return Err(SparkError::connect_msg(
                "Catalog operation returned no rows",
            ));
        }

        let fields: Vec<crate::types::StructField> = rows[0]
            .fields()
            .iter()
            .map(|name| crate::types::StructField {
                name: name.clone(),
                data_type: DataType::String {
                    collation: "UTF8_BINARY".to_string(),
                },
                nullable: true,
                metadata: std::collections::BTreeMap::new(),
            })
            .collect();

        let schema = DataType::Struct { fields };
        let plan = crate::plan::LogicalPlan::LocalRelation { schema, data: None };

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

/// Extract a value at a specific index from an Arrow array.
fn arrow_value_at(array: &dyn arrow::array::Array, index: usize) -> Result<Value> {
    use arrow::array::*;

    if array.is_null(index) {
        return Ok(Value::Null);
    }

    if let Some(arr) = array.as_any().downcast_ref::<BooleanArray>() {
        return Ok(Value::Bool(arr.value(index)));
    }
    if let Some(arr) = array.as_any().downcast_ref::<Int8Array>() {
        return Ok(Value::Byte(arr.value(index)));
    }
    if let Some(arr) = array.as_any().downcast_ref::<Int16Array>() {
        return Ok(Value::Short(arr.value(index)));
    }
    if let Some(arr) = array.as_any().downcast_ref::<Int32Array>() {
        return Ok(Value::Integer(arr.value(index)));
    }
    if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
        return Ok(Value::Long(arr.value(index)));
    }
    if let Some(arr) = array.as_any().downcast_ref::<Float32Array>() {
        return Ok(Value::Float(arr.value(index)));
    }
    if let Some(arr) = array.as_any().downcast_ref::<Float64Array>() {
        return Ok(Value::Double(arr.value(index)));
    }
    if let Some(arr) = array.as_any().downcast_ref::<StringArray>() {
        return Ok(Value::String(arr.value(index).to_string()));
    }
    if let Some(arr) = array.as_any().downcast_ref::<BinaryArray>() {
        return Ok(Value::Binary(arr.value(index).to_vec()));
    }
    if let Some(arr) = array.as_any().downcast_ref::<Date32Array>() {
        return Ok(Value::Date(arr.value(index)));
    }
    if let Some(arr) = array.as_any().downcast_ref::<TimestampMicrosecondArray>() {
        return Ok(Value::Timestamp(arr.value(index)));
    }

    Err(SparkError::connect_msg(
        "Unsupported Arrow type - cannot convert to Value",
    ))
}
