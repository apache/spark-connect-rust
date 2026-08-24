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

    fn __repr__(&self) -> String {
        "Catalog()".to_string()
    }
}
