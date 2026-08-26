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

    fn __repr__(&self) -> String {
        "Catalog()".to_string()
    }
}
