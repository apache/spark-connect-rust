//! PyO3 wrapper for spark_connect::catalog::Catalog.

use pyo3::prelude::*;
use spark_connect::catalog::Catalog;

use crate::dataframe::PyDataFrame;
use crate::errors::ResultExt;

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

    /// List catalogs.
    fn listCatalogs(&self) -> PyResult<PyDataFrame> {
        let df = self.catalog.list_catalogs().to_pyerr()?;
        Ok(PyDataFrame::new(df))
    }

    /// Get current database.
    fn currentDatabase(&self) -> PyResult<String> {
        self.catalog.current_database().to_pyerr()
    }

    /// Set current database.
    fn setCurrentDatabase(&self, db_name: &str) -> PyResult<()> {
        self.catalog.set_current_database(db_name).to_pyerr()
    }

    /// List databases.
    fn listDatabases(&self) -> PyResult<PyDataFrame> {
        let df = self.catalog.list_databases().to_pyerr()?;
        Ok(PyDataFrame::new(df))
    }

    /// Check if database exists.
    fn databaseExists(&self, db_name: &str) -> PyResult<bool> {
        self.catalog.database_exists(db_name).to_pyerr()
    }

    /// Get database info.
    fn getDatabase(&self, db_name: &str) -> PyResult<PyDataFrame> {
        let df = self.catalog.get_database(db_name).to_pyerr()?;
        Ok(PyDataFrame::new(df))
    }

    /// List tables.
    fn listTables(&self) -> PyResult<PyDataFrame> {
        let df = self.catalog.list_tables().to_pyerr()?;
        Ok(PyDataFrame::new(df))
    }

    /// List tables in database.
    fn listTablesWithDatabase(&self, db_name: &str) -> PyResult<PyDataFrame> {
        let df = self.catalog.list_tables_in_database(db_name).to_pyerr()?;
        Ok(PyDataFrame::new(df))
    }

    /// Check if table exists.
    fn tableExists(&self, table_name: &str) -> PyResult<bool> {
        self.catalog.table_exists(table_name).to_pyerr()
    }

    /// Get table info.
    fn getTable(&self, table_name: &str) -> PyResult<PyDataFrame> {
        let df = self.catalog.get_table(table_name).to_pyerr()?;
        Ok(PyDataFrame::new(df))
    }

    /// List functions.
    fn listFunctions(&self) -> PyResult<PyDataFrame> {
        let df = self.catalog.list_functions().to_pyerr()?;
        Ok(PyDataFrame::new(df))
    }

    /// Check if function exists.
    fn functionExists(&self, function_name: &str) -> PyResult<bool> {
        self.catalog.function_exists(function_name).to_pyerr()
    }

    /// Get function info.
    fn getFunction(&self, function_name: &str) -> PyResult<PyDataFrame> {
        let df = self.catalog.get_function(function_name).to_pyerr()?;
        Ok(PyDataFrame::new(df))
    }

    /// List columns of a table.
    fn listColumns(&self, table_name: &str) -> PyResult<PyDataFrame> {
        let df = self.catalog.list_columns(table_name).to_pyerr()?;
        Ok(PyDataFrame::new(df))
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

    /// List the partitions of a table.
    #[allow(non_snake_case)]
    fn listPartitions(&self, tableName: &str) -> PyResult<PyDataFrame> {
        let df = self.catalog.list_partitions(tableName).to_pyerr()?;
        Ok(PyDataFrame::new(df))
    }

    /// List the views in a database.
    #[pyo3(signature = (dbName=None, pattern=None))]
    #[allow(non_snake_case)]
    fn listViews(&self, dbName: Option<&str>, pattern: Option<&str>) -> PyResult<PyDataFrame> {
        let df = self.catalog.list_views(dbName, pattern).to_pyerr()?;
        Ok(PyDataFrame::new(df))
    }

    fn __repr__(&self) -> String {
        "Catalog()".to_string()
    }
}
