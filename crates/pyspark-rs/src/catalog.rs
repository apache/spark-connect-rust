//! PyO3 wrapper for spark_connect::catalog::Catalog.

use pyo3::prelude::*;
use spark_connect::catalog::{
    Catalog, CatalogMetadata, Column, Database, Function, Table, TablePartition,
};

use crate::dataframe::PyDataFrame;
use crate::errors::ResultExt;

/// `pyspark.sql.catalog.CatalogMetadata`.
#[pyclass(name = "CatalogMetadata")]
pub struct PyCatalogMetadata {
    inner: CatalogMetadata,
}
#[pymethods]
impl PyCatalogMetadata {
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }
    #[getter]
    fn description(&self) -> Option<String> {
        self.inner.description.clone()
    }
    fn __repr__(&self) -> String {
        format!(
            "CatalogMetadata(name='{}', description={:?})",
            self.inner.name, self.inner.description
        )
    }
}

/// `pyspark.sql.catalog.Database`.
#[pyclass(name = "Database")]
pub struct PyDatabase {
    inner: Database,
}
#[pymethods]
impl PyDatabase {
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }
    #[getter]
    fn catalog(&self) -> Option<String> {
        self.inner.catalog.clone()
    }
    #[getter]
    fn description(&self) -> Option<String> {
        self.inner.description.clone()
    }
    #[getter]
    fn locationUri(&self) -> String {
        self.inner.location_uri.clone()
    }
    fn __repr__(&self) -> String {
        format!("Database(name='{}')", self.inner.name)
    }
}

/// `pyspark.sql.catalog.Table`.
#[pyclass(name = "Table")]
pub struct PyTable {
    inner: Table,
}
#[pymethods]
impl PyTable {
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }
    #[getter]
    fn catalog(&self) -> Option<String> {
        self.inner.catalog.clone()
    }
    #[getter]
    fn namespace(&self) -> Option<Vec<String>> {
        self.inner.namespace.clone()
    }
    #[getter]
    fn description(&self) -> Option<String> {
        self.inner.description.clone()
    }
    #[getter]
    fn tableType(&self) -> String {
        self.inner.table_type.clone()
    }
    #[getter]
    fn isTemporary(&self) -> bool {
        self.inner.is_temporary
    }
    /// The single-element namespace as a database name, else None (mirrors `Table.database`).
    #[getter]
    fn database(&self) -> Option<String> {
        match &self.inner.namespace {
            Some(ns) if ns.len() == 1 => Some(ns[0].clone()),
            _ => None,
        }
    }
    fn __repr__(&self) -> String {
        format!("Table(name='{}', tableType='{}')", self.inner.name, self.inner.table_type)
    }
}

/// `pyspark.sql.catalog.Column` (exposed as `CatalogColumn` in the extension to avoid
/// colliding with the expression `Column`; re-exported as `Column` by pyspark.sql.catalog).
#[pyclass(name = "CatalogColumn")]
pub struct PyCatalogColumn {
    inner: Column,
}
#[pymethods]
impl PyCatalogColumn {
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }
    #[getter]
    fn description(&self) -> Option<String> {
        self.inner.description.clone()
    }
    #[getter]
    fn dataType(&self) -> String {
        self.inner.data_type.clone()
    }
    #[getter]
    fn nullable(&self) -> bool {
        self.inner.nullable
    }
    #[getter]
    fn isPartition(&self) -> bool {
        self.inner.is_partition
    }
    #[getter]
    fn isBucket(&self) -> bool {
        self.inner.is_bucket
    }
    #[getter]
    fn isCluster(&self) -> bool {
        self.inner.is_cluster
    }
    fn __repr__(&self) -> String {
        format!("Column(name='{}', dataType='{}')", self.inner.name, self.inner.data_type)
    }
}

/// `pyspark.sql.catalog.Function`.
#[pyclass(name = "Function")]
pub struct PyFunction {
    inner: Function,
}
#[pymethods]
impl PyFunction {
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }
    #[getter]
    fn catalog(&self) -> Option<String> {
        self.inner.catalog.clone()
    }
    #[getter]
    fn namespace(&self) -> Option<Vec<String>> {
        self.inner.namespace.clone()
    }
    #[getter]
    fn description(&self) -> Option<String> {
        self.inner.description.clone()
    }
    #[getter]
    fn className(&self) -> String {
        self.inner.class_name.clone()
    }
    #[getter]
    fn isTemporary(&self) -> bool {
        self.inner.is_temporary
    }
    fn __repr__(&self) -> String {
        format!("Function(name='{}', className='{}')", self.inner.name, self.inner.class_name)
    }
}

/// `pyspark.sql.catalog.TablePartition`.
#[pyclass(name = "TablePartition")]
pub struct PyTablePartition {
    inner: TablePartition,
}
#[pymethods]
impl PyTablePartition {
    #[getter]
    fn partition(&self) -> String {
        self.inner.partition.clone()
    }
    fn __repr__(&self) -> String {
        format!("TablePartition(partition='{}')", self.inner.partition)
    }
}

/// Python wrapper for Spark Catalog.
#[pyclass(name = "Catalog")]
pub struct PyCatalog {
    pub(crate) catalog: Catalog,
}

impl PyCatalog {
    pub fn new(catalog: Catalog) -> Self {
        PyCatalog { catalog }
    }
}

#[pymethods]
impl PyCatalog {
    /// Get the current catalog.
    fn currentCatalog(&self) -> PyResult<String> {
        self.catalog.current_catalog().to_pyerr()
    }

    /// Set the current catalog.
    fn setCurrentCatalog(&self, catalog_name: &str) -> PyResult<()> {
        self.catalog.set_current_catalog(catalog_name).to_pyerr()
    }

    /// List catalogs. Mirrors `Catalog.listCatalogs(pattern=None) -> List[CatalogMetadata]`.
    #[pyo3(signature = (pattern=None))]
    fn listCatalogs(&self, pattern: Option<&str>) -> PyResult<Vec<PyCatalogMetadata>> {
        Ok(self
            .catalog
            .list_catalogs_typed(pattern)
            .to_pyerr()?
            .into_iter()
            .map(|inner| PyCatalogMetadata { inner })
            .collect())
    }

    /// Get current database.
    fn currentDatabase(&self) -> PyResult<String> {
        self.catalog.current_database().to_pyerr()
    }

    /// Set current database.
    fn setCurrentDatabase(&self, db_name: &str) -> PyResult<()> {
        self.catalog.set_current_database(db_name).to_pyerr()
    }

    /// List databases. Mirrors `Catalog.listDatabases(pattern=None) -> List[Database]`.
    #[pyo3(signature = (pattern=None))]
    fn listDatabases(&self, pattern: Option<&str>) -> PyResult<Vec<PyDatabase>> {
        Ok(self
            .catalog
            .list_databases_typed(pattern)
            .to_pyerr()?
            .into_iter()
            .map(|inner| PyDatabase { inner })
            .collect())
    }

    /// Check if database exists.
    fn databaseExists(&self, db_name: &str) -> PyResult<bool> {
        self.catalog.database_exists(db_name).to_pyerr()
    }

    /// Get database info. Mirrors `Catalog.getDatabase(dbName) -> Database`.
    fn getDatabase(&self, db_name: &str) -> PyResult<PyDatabase> {
        Ok(PyDatabase {
            inner: self.catalog.get_database_typed(db_name).to_pyerr()?,
        })
    }

    /// List tables. Mirrors `Catalog.listTables(dbName=None, pattern=None) -> List[Table]`.
    #[pyo3(signature = (dbName=None, pattern=None))]
    #[allow(non_snake_case)]
    fn listTables(&self, dbName: Option<&str>, pattern: Option<&str>) -> PyResult<Vec<PyTable>> {
        Ok(self
            .catalog
            .list_tables_typed(dbName, pattern)
            .to_pyerr()?
            .into_iter()
            .map(|inner| PyTable { inner })
            .collect())
    }

    /// Check if table exists.
    fn tableExists(&self, table_name: &str) -> PyResult<bool> {
        self.catalog.table_exists(table_name).to_pyerr()
    }

    /// Get table info. Mirrors `Catalog.getTable(tableName) -> Table`.
    fn getTable(&self, table_name: &str) -> PyResult<PyTable> {
        Ok(PyTable {
            inner: self.catalog.get_table_typed(table_name).to_pyerr()?,
        })
    }

    /// List functions. Mirrors `Catalog.listFunctions(dbName=None, pattern=None) -> List[Function]`.
    #[pyo3(signature = (dbName=None, pattern=None))]
    #[allow(non_snake_case)]
    fn listFunctions(
        &self,
        dbName: Option<&str>,
        pattern: Option<&str>,
    ) -> PyResult<Vec<PyFunction>> {
        Ok(self
            .catalog
            .list_functions_typed(dbName, pattern)
            .to_pyerr()?
            .into_iter()
            .map(|inner| PyFunction { inner })
            .collect())
    }

    /// Check if function exists.
    fn functionExists(&self, function_name: &str) -> PyResult<bool> {
        self.catalog.function_exists(function_name).to_pyerr()
    }

    /// Get function info. Mirrors `Catalog.getFunction(functionName) -> Function`.
    fn getFunction(&self, function_name: &str) -> PyResult<PyFunction> {
        Ok(PyFunction {
            inner: self.catalog.get_function_typed(function_name).to_pyerr()?,
        })
    }

    /// List columns of a table. Mirrors `Catalog.listColumns(tableName, dbName=None) -> List[Column]`.
    #[pyo3(signature = (tableName, dbName=None))]
    #[allow(non_snake_case)]
    fn listColumns(&self, tableName: &str, dbName: Option<&str>) -> PyResult<Vec<PyCatalogColumn>> {
        Ok(self
            .catalog
            .list_columns_typed(tableName, dbName)
            .to_pyerr()?
            .into_iter()
            .map(|inner| PyCatalogColumn { inner })
            .collect())
    }

    /// Cache a table.
    fn cacheTable(&self, table_name: &str) -> PyResult<()> {
        self.catalog.cache_table(table_name).to_pyerr()
    }

    /// Uncache a table.
    fn uncacheTable(&self, table_name: &str) -> PyResult<()> {
        self.catalog.uncache_table(table_name).to_pyerr()
    }

    /// Drop temporary view.
    fn dropTempView(&self, view_name: &str) -> PyResult<bool> {
        self.catalog.drop_temp_view(view_name).to_pyerr()
    }

    /// Drop a global temporary view.
    fn dropGlobalTempView(&self, view_name: &str) -> PyResult<bool> {
        self.catalog.drop_global_temp_view(view_name).to_pyerr()
    }

    /// Whether a table/view is cached.
    fn isCached(&self, table_name: &str) -> PyResult<bool> {
        self.catalog.is_cached(table_name).to_pyerr()
    }

    /// Remove all cached tables from the in-memory cache.
    fn clearCache(&self) -> PyResult<()> {
        self.catalog.clear_cache().to_pyerr()
    }

    /// Invalidate and refresh cached metadata for a table.
    fn refreshTable(&self, table_name: &str) -> PyResult<()> {
        self.catalog.refresh_table(table_name).to_pyerr()
    }

    /// Invalidate and refresh cached data for any table at a path.
    fn refreshByPath(&self, path: &str) -> PyResult<()> {
        self.catalog.refresh_by_path(path).to_pyerr()
    }

    /// Recover all the partitions of a table.
    fn recoverPartitions(&self, table_name: &str) -> PyResult<()> {
        self.catalog.recover_partitions(table_name).to_pyerr()
    }

    /// Create a table from a data source.
    #[pyo3(signature = (tableName, path=None, source=None, schema=None, description=None, **options))]
    #[allow(non_snake_case, unused_variables)]
    fn createTable(
        &self,
        tableName: &str,
        path: Option<&str>,
        source: Option<&str>,
        schema: Option<&Bound<'_, PyAny>>,
        description: Option<&str>,
        options: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<PyDataFrame> {
        self.catalog
            .create_table(tableName, path, source, description)
            .map(PyDataFrame::new)
            .to_pyerr()
    }

    /// Create an external table from a data source.
    #[pyo3(signature = (tableName, path=None, source=None, schema=None, **options))]
    #[allow(non_snake_case, unused_variables)]
    fn createExternalTable(
        &self,
        tableName: &str,
        path: Option<&str>,
        source: Option<&str>,
        schema: Option<&Bound<'_, PyAny>>,
        options: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<PyDataFrame> {
        self.catalog
            .create_external_table(tableName, path, source)
            .map(PyDataFrame::new)
            .to_pyerr()
    }

    /// Register a Python function as a UDF, returning the registered UDF. Deprecated
    /// in pyspark in favour of `spark.udf.register`; mirrors `Catalog.registerFunction`.
    #[pyo3(signature = (name, f, returnType=None))]
    #[allow(non_snake_case)]
    fn registerFunction<'a>(
        &self,
        py: Python<'a>,
        name: &str,
        f: Bound<'a, PyAny>,
        returnType: Option<Bound<'a, PyAny>>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let udf_cls = py
            .import("pyspark.sql.udf")?
            .getattr("UserDefinedFunction")?;
        let rt: Bound<'a, PyAny> = match returnType {
            Some(t) => t,
            None => py
                .import("pyspark.sql.types")?
                .getattr("StringType")?
                .call0()?,
        };
        // UserDefinedFunction(func, returnType, evalType=100, name=name)
        udf_cls.call1((f, rt, 100, name))
    }

    /// Create a database.
    #[pyo3(signature = (dbName, ifNotExists=false, properties=None))]
    #[allow(non_snake_case)]
    fn createDatabase(
        &self,
        dbName: &str,
        ifNotExists: bool,
        properties: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<()> {
        let mut props = std::collections::HashMap::new();
        if let Some(dict) = properties {
            for (k, v) in dict.iter() {
                props.insert(k.extract::<String>()?, v.extract::<String>()?);
            }
        }
        self.catalog
            .create_database(dbName, ifNotExists, props)
            .to_pyerr()
    }

    /// Drop a database.
    #[pyo3(signature = (dbName, ifExists=false, cascade=false))]
    #[allow(non_snake_case)]
    fn dropDatabase(&self, dbName: &str, ifExists: bool, cascade: bool) -> PyResult<()> {
        self.catalog
            .drop_database(dbName, ifExists, cascade)
            .to_pyerr()
    }

    /// Drop a table.
    #[pyo3(signature = (tableName, ifExists=false, purge=false))]
    #[allow(non_snake_case)]
    fn dropTable(&self, tableName: &str, ifExists: bool, purge: bool) -> PyResult<()> {
        self.catalog.drop_table(tableName, ifExists, purge).to_pyerr()
    }

    /// Drop a view.
    #[pyo3(signature = (viewName, ifExists=false))]
    #[allow(non_snake_case)]
    fn dropView(&self, viewName: &str, ifExists: bool) -> PyResult<()> {
        self.catalog.drop_view(viewName, ifExists).to_pyerr()
    }

    /// Truncate a table.
    #[allow(non_snake_case)]
    fn truncateTable(&self, tableName: &str) -> PyResult<()> {
        self.catalog.truncate_table(tableName).to_pyerr()
    }

    /// Recover the statistics of a table.
    #[pyo3(signature = (tableName, noScan=false))]
    #[allow(non_snake_case)]
    fn analyzeTable(&self, tableName: &str, noScan: bool) -> PyResult<()> {
        self.catalog.analyze_table(tableName, noScan).to_pyerr()
    }

    /// Get the `CREATE TABLE` string of a table.
    #[pyo3(signature = (tableName, asSerde=false))]
    #[allow(non_snake_case)]
    fn getCreateTableString(&self, tableName: &str, asSerde: bool) -> PyResult<String> {
        self.catalog
            .get_create_table_string(tableName, asSerde)
            .to_pyerr()
    }

    /// Get the properties of a table as a dict.
    #[allow(non_snake_case)]
    fn getTableProperties<'py>(
        &self,
        py: Python<'py>,
        tableName: &str,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let props = self.catalog.get_table_properties(tableName).to_pyerr()?;
        let dict = pyo3::types::PyDict::new(py);
        for (k, v) in props {
            dict.set_item(k, v)?;
        }
        Ok(dict)
    }

    /// List the partitions of a table. Mirrors `Catalog.listPartitions -> List[TablePartition]`.
    #[allow(non_snake_case)]
    fn listPartitions(&self, tableName: &str) -> PyResult<Vec<PyTablePartition>> {
        Ok(self
            .catalog
            .list_partitions_typed(tableName)
            .to_pyerr()?
            .into_iter()
            .map(|inner| PyTablePartition { inner })
            .collect())
    }

    /// List the views in a database. Mirrors `Catalog.listViews -> List[Table]`.
    #[pyo3(signature = (dbName=None, pattern=None))]
    #[allow(non_snake_case)]
    fn listViews(&self, dbName: Option<&str>, pattern: Option<&str>) -> PyResult<Vec<PyTable>> {
        Ok(self
            .catalog
            .list_views_typed(dbName, pattern)
            .to_pyerr()?
            .into_iter()
            .map(|inner| PyTable { inner })
            .collect())
    }

    fn __repr__(&self) -> String {
        "Catalog()".to_string()
    }
}
