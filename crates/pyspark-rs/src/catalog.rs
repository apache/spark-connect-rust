//! PyO3 wrapper for spark_connect::catalog::Catalog.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use spark_connect::catalog::{
    Catalog, CatalogMetadata, Column, Database, Function, Table, TablePartition,
};

use spark_connect_core::error::SparkError;

use crate::dataframe::PyDataFrame;
use crate::errors::ResultExt;
use crate::interrupt::interruptible;

/// PySpark's FutureWarning of the deprecated `dbName` of the catalog method `method`.
fn warn_db_name(py: Python<'_>, method: &str, db_name: Option<&str>) -> PyResult<()> {
    if db_name.is_some() {
        py.import("warnings")?.call_method1(
            "warn",
            (
                format!(
                    "`dbName` has been deprecated since Spark 3.4 and might be removed in a \
                     future version. Use {method}(`dbName.tableName`) instead."
                ),
                py.get_type::<pyo3::exceptions::PyFutureWarning>(),
            ),
        )?;
    }
    Ok(())
}

/// The schema of `createTable`, a StructType, as PySpark's, else `NOT_EXPECTED_TYPE`.
fn table_schema(
    schema: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<spark_connect::types::DataType>> {
    let Some(schema) = schema.filter(|schema| !schema.is_none()) else {
        return Ok(None);
    };
    if let Ok(st) = schema.extract::<PyRef<'_, crate::types::PyStructType>>() {
        return Ok(Some(spark_connect::types::DataType::Struct {
            fields: st.current_fields(schema.py())?,
        }));
    }
    let type_name = schema.get_type().name()?.to_string();
    Err(crate::column::raise_pyspark(
        schema.py(),
        "PySparkTypeError",
        "NOT_EXPECTED_TYPE",
        &[
            ("arg_name", "schema"),
            ("expected_type", "struct type"),
            ("arg_type", &type_name),
        ],
    ))
}

/// The `**options` of `createTable`, as PySpark's: strings, where a `None` is left out.
fn table_options(
    options: Option<&Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<std::collections::HashMap<String, String>> {
    let mut map = std::collections::HashMap::new();
    for (key, value) in options.into_iter().flat_map(|options| options.iter()) {
        if value.is_none() {
            continue;
        }
        if !value.is_instance_of::<pyo3::types::PyString>() {
            // As the protobuf map of strings that PySpark sets the options in.
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "{} has type {}, but expected one of: (<class 'bytes'>, <class 'str'>)",
                value.repr()?,
                value.get_type().str()?
            )));
        }
        map.insert(key.extract()?, value.extract()?);
    }
    Ok(map)
}

/// The NamedTuple of `pyspark.sql.catalog` of a result of the catalog, as PySpark's `Catalog`
/// returns: `CatalogMetadata`, `Database`, `Table`, `Column`, `Function` or `TablePartition`.
trait NamedTuple {
    /// The name of the NamedTuple, and its fields.
    fn fields<'py>(&self, py: Python<'py>) -> PyResult<(&'static str, Bound<'py, PyDict>)>;

    fn named_tuple(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let (name, fields) = self.fields(py)?;
        Ok(py
            .import("pyspark.sql.catalog")?
            .getattr(name)?
            .call((), Some(&fields))?
            .unbind())
    }
}

impl NamedTuple for CatalogMetadata {
    fn fields<'py>(&self, py: Python<'py>) -> PyResult<(&'static str, Bound<'py, PyDict>)> {
        let fields = PyDict::new(py);
        fields.set_item("name", &self.name)?;
        fields.set_item("description", &self.description)?;
        Ok(("CatalogMetadata", fields))
    }
}

impl NamedTuple for Database {
    fn fields<'py>(&self, py: Python<'py>) -> PyResult<(&'static str, Bound<'py, PyDict>)> {
        let fields = PyDict::new(py);
        fields.set_item("name", &self.name)?;
        fields.set_item("catalog", &self.catalog)?;
        fields.set_item("description", &self.description)?;
        fields.set_item("locationUri", &self.location_uri)?;
        Ok(("Database", fields))
    }
}

impl NamedTuple for Table {
    fn fields<'py>(&self, py: Python<'py>) -> PyResult<(&'static str, Bound<'py, PyDict>)> {
        let fields = PyDict::new(py);
        fields.set_item("name", &self.name)?;
        fields.set_item("catalog", &self.catalog)?;
        fields.set_item("namespace", &self.namespace)?;
        fields.set_item("description", &self.description)?;
        fields.set_item("tableType", &self.table_type)?;
        fields.set_item("isTemporary", self.is_temporary)?;
        Ok(("Table", fields))
    }
}

impl NamedTuple for Column {
    fn fields<'py>(&self, py: Python<'py>) -> PyResult<(&'static str, Bound<'py, PyDict>)> {
        let fields = PyDict::new(py);
        fields.set_item("name", &self.name)?;
        fields.set_item("description", &self.description)?;
        fields.set_item("dataType", &self.data_type)?;
        fields.set_item("nullable", self.nullable)?;
        fields.set_item("isPartition", self.is_partition)?;
        fields.set_item("isBucket", self.is_bucket)?;
        fields.set_item("isCluster", self.is_cluster)?;
        Ok(("Column", fields))
    }
}

impl NamedTuple for Function {
    fn fields<'py>(&self, py: Python<'py>) -> PyResult<(&'static str, Bound<'py, PyDict>)> {
        let fields = PyDict::new(py);
        fields.set_item("name", &self.name)?;
        fields.set_item("catalog", &self.catalog)?;
        fields.set_item("namespace", &self.namespace)?;
        fields.set_item("description", &self.description)?;
        fields.set_item("className", &self.class_name)?;
        fields.set_item("isTemporary", self.is_temporary)?;
        Ok(("Function", fields))
    }
}

impl NamedTuple for TablePartition {
    fn fields<'py>(&self, py: Python<'py>) -> PyResult<(&'static str, Bound<'py, PyDict>)> {
        let fields = PyDict::new(py);
        fields.set_item("partition", &self.partition)?;
        Ok(("TablePartition", fields))
    }
}

/// The NamedTuples of `results`.
fn named_tuples<T: NamedTuple>(py: Python<'_>, results: Vec<T>) -> PyResult<Vec<Py<PyAny>>> {
    results
        .iter()
        .map(|result| result.named_tuple(py))
        .collect()
}

/// Python wrapper for Spark Catalog.
#[pyclass(name = "Catalog", module = "pyspark.sql.catalog", weakref, dict)]
pub struct PyCatalog {
    pub(crate) catalog: Catalog,
}

impl PyCatalog {
    pub fn new(catalog: Catalog) -> Self {
        PyCatalog { catalog }
    }

    /// `f` of the catalog without the GIL, as PySpark's requests to the server release it
    /// while they wait, so that other Python threads run and a signal interrupts the wait.
    fn detached<T: Send>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&Catalog) -> Result<T, Box<SparkError>> + Send,
    ) -> PyResult<T> {
        interruptible(py, self.catalog.spark_session(), || f(&self.catalog))
    }
}

#[pymethods]
impl PyCatalog {
    /// Get the current catalog.
    fn currentCatalog(&self, py: Python<'_>) -> PyResult<String> {
        self.detached(py, |catalog| catalog.current_catalog().map_err(Box::new))
    }

    /// Set the current catalog.
    fn setCurrentCatalog(&self, py: Python<'_>, catalogName: &str) -> PyResult<()> {
        self.detached(py, |catalog| {
            catalog.set_current_catalog(catalogName).map_err(Box::new)
        })
    }

    /// List catalogs. Mirrors `Catalog.listCatalogs(pattern=None) -> List[CatalogMetadata]`.
    #[pyo3(signature = (pattern=None))]
    fn listCatalogs(&self, py: Python<'_>, pattern: Option<&str>) -> PyResult<Vec<Py<PyAny>>> {
        named_tuples(
            py,
            self.detached(py, |catalog| {
                catalog.list_catalogs_typed(pattern).map_err(Box::new)
            })?,
        )
    }

    /// Get current database.
    fn currentDatabase(&self, py: Python<'_>) -> PyResult<String> {
        self.detached(py, |catalog| catalog.current_database().map_err(Box::new))
    }

    /// Set current database.
    fn setCurrentDatabase(&self, py: Python<'_>, dbName: &str) -> PyResult<()> {
        self.detached(py, |catalog| {
            catalog.set_current_database(dbName).map_err(Box::new)
        })
    }

    /// List databases. Mirrors `Catalog.listDatabases(pattern=None) -> List[Database]`.
    #[pyo3(signature = (pattern=None))]
    fn listDatabases(&self, py: Python<'_>, pattern: Option<&str>) -> PyResult<Vec<Py<PyAny>>> {
        named_tuples(
            py,
            self.detached(py, |catalog| {
                catalog.list_databases_typed(pattern).map_err(Box::new)
            })?,
        )
    }

    /// Check if database exists.
    fn databaseExists(&self, py: Python<'_>, dbName: &str) -> PyResult<bool> {
        self.detached(py, |catalog| {
            catalog.database_exists(dbName).map_err(Box::new)
        })
    }

    /// Get database info. Mirrors `Catalog.getDatabase(dbName) -> Database`.
    fn getDatabase(&self, py: Python<'_>, dbName: &str) -> PyResult<Py<PyAny>> {
        self.detached(py, |catalog| {
            catalog.get_database_typed(dbName).map_err(Box::new)
        })?
        .named_tuple(py)
    }

    /// List tables. Mirrors `Catalog.listTables(dbName=None, pattern=None) -> List[Table]`.
    #[pyo3(signature = (dbName=None, pattern=None))]
    #[allow(non_snake_case)]
    fn listTables(
        &self,
        py: Python<'_>,
        dbName: Option<&str>,
        pattern: Option<&str>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        named_tuples(
            py,
            self.detached(py, |catalog| {
                catalog.list_tables_typed(dbName, pattern).map_err(Box::new)
            })?,
        )
    }

    /// Check if table exists. Mirrors `Catalog.tableExists(tableName, dbName=None) -> bool`.
    #[pyo3(signature = (tableName, dbName=None))]
    fn tableExists(&self, py: Python<'_>, tableName: &str, dbName: Option<&str>) -> PyResult<bool> {
        warn_db_name(py, "tableExists", dbName)?;
        self.detached(py, |catalog| {
            { catalog.table_exists_with_database(tableName, dbName) }.map_err(Box::new)
        })
    }

    /// Get table info. Mirrors `Catalog.getTable(tableName) -> Table`.
    fn getTable(&self, py: Python<'_>, tableName: &str) -> PyResult<Py<PyAny>> {
        self.detached(py, |catalog| {
            catalog.get_table_typed(tableName).map_err(Box::new)
        })?
        .named_tuple(py)
    }

    /// List functions. Mirrors `Catalog.listFunctions(dbName=None, pattern=None) -> List[Function]`.
    #[pyo3(signature = (dbName=None, pattern=None))]
    #[allow(non_snake_case)]
    fn listFunctions(
        &self,
        py: Python<'_>,
        dbName: Option<&str>,
        pattern: Option<&str>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        named_tuples(
            py,
            self.detached(py, |catalog| {
                catalog
                    .list_functions_typed(dbName, pattern)
                    .map_err(Box::new)
            })?,
        )
    }

    /// Check if function exists. Mirrors `Catalog.functionExists(functionName, dbName=None) -> bool`.
    #[pyo3(signature = (functionName, dbName=None))]
    fn functionExists(
        &self,
        py: Python<'_>,
        functionName: &str,
        dbName: Option<&str>,
    ) -> PyResult<bool> {
        warn_db_name(py, "functionExists", dbName)?;
        self.detached(py, |catalog| {
            { catalog.function_exists_with_database(functionName, dbName) }.map_err(Box::new)
        })
    }

    /// Get function info. Mirrors `Catalog.getFunction(functionName) -> Function`.
    fn getFunction(&self, py: Python<'_>, functionName: &str) -> PyResult<Py<PyAny>> {
        self.detached(py, |catalog| {
            catalog.get_function_typed(functionName).map_err(Box::new)
        })?
        .named_tuple(py)
    }

    /// List columns of a table. Mirrors `Catalog.listColumns(tableName, dbName=None) -> List[Column]`.
    #[pyo3(signature = (tableName, dbName=None))]
    #[allow(non_snake_case)]
    fn listColumns(
        &self,
        py: Python<'_>,
        tableName: &str,
        dbName: Option<&str>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        warn_db_name(py, "listColumns", dbName)?;
        named_tuples(
            py,
            self.detached(py, |catalog| {
                catalog
                    .list_columns_typed(tableName, dbName)
                    .map_err(Box::new)
            })?,
        )
    }

    /// Cache a table. Mirrors `Catalog.cacheTable(tableName, storageLevel=None) -> None`.
    ///
    /// A supplied pyspark `StorageLevel` is mapped onto the `spark.connect`
    /// `CacheTable.storage_level` field (the same field layout `DataFrame.persist` uses);
    /// `None` leaves it unset so the server default applies.
    #[pyo3(signature = (tableName, storageLevel=None))]
    fn cacheTable(
        &self,
        py: Python<'_>,
        tableName: &str,
        storageLevel: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let level = storageLevel
            .map(crate::storagelevel::storage_level_proto)
            .transpose()?;
        self.detached(py, |catalog| {
            { catalog.cache_table_with_storage_level(tableName, level) }.map_err(Box::new)
        })
    }

    /// Uncache a table.
    fn uncacheTable(&self, py: Python<'_>, tableName: &str) -> PyResult<()> {
        self.detached(py, |catalog| {
            catalog.uncache_table(tableName).map_err(Box::new)
        })
    }

    /// Drop temporary view.
    fn dropTempView(&self, py: Python<'_>, viewName: &str) -> PyResult<bool> {
        self.detached(py, |catalog| {
            catalog.drop_temp_view(viewName).map_err(Box::new)
        })
    }

    /// Drop a global temporary view.
    fn dropGlobalTempView(&self, py: Python<'_>, viewName: &str) -> PyResult<bool> {
        self.detached(py, |catalog| {
            catalog.drop_global_temp_view(viewName).map_err(Box::new)
        })
    }

    /// Whether a table/view is cached.
    fn isCached(&self, py: Python<'_>, tableName: &str) -> PyResult<bool> {
        self.detached(py, |catalog| catalog.is_cached(tableName).map_err(Box::new))
    }

    /// Remove all cached tables from the in-memory cache.
    fn clearCache(&self, py: Python<'_>) -> PyResult<()> {
        self.detached(py, |catalog| catalog.clear_cache().map_err(Box::new))
    }

    /// Invalidate and refresh cached metadata for a table.
    fn refreshTable(&self, py: Python<'_>, tableName: &str) -> PyResult<()> {
        self.detached(py, |catalog| {
            catalog.refresh_table(tableName).map_err(Box::new)
        })
    }

    /// Invalidate and refresh cached data for any table at a path.
    fn refreshByPath(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        self.detached(py, |catalog| {
            catalog.refresh_by_path(path).map_err(Box::new)
        })
    }

    /// Recover all the partitions of a table.
    fn recoverPartitions(&self, py: Python<'_>, tableName: &str) -> PyResult<()> {
        self.detached(py, |catalog| {
            catalog.recover_partitions(tableName).map_err(Box::new)
        })
    }

    /// Create a table from a data source, as PySpark's: of a StructType `schema` and string
    /// `**options`, created on return.
    #[pyo3(signature = (tableName, path=None, source=None, schema=None, description=None, **options))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn createTable(
        &self,
        py: Python<'_>,
        tableName: &str,
        path: Option<&str>,
        source: Option<&str>,
        schema: Option<&Bound<'_, PyAny>>,
        description: Option<&str>,
        options: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<PyDataFrame> {
        let schema = table_schema(schema)?;
        let options = table_options(options)?;
        py.detach(|| {
            self.catalog
                .create_table_with(
                    tableName,
                    path,
                    source,
                    schema.as_ref(),
                    description,
                    options,
                )
                .map_err(Box::new)
        })
        .map(PyDataFrame::new)
        .to_pyerr()
    }

    /// Create an external table from a data source: deprecated, as PySpark's, which warns and
    /// creates the table as `createTable` does.
    #[pyo3(signature = (tableName, path=None, source=None, schema=None, description=None, **options))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn createExternalTable(
        &self,
        py: Python<'_>,
        tableName: &str,
        path: Option<&str>,
        source: Option<&str>,
        schema: Option<&Bound<'_, PyAny>>,
        description: Option<&str>,
        options: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<PyDataFrame> {
        py.import("warnings")?.call_method1(
            "warn",
            (
                "createExternalTable is deprecated since Spark 4.0, please use createTable \
                 instead.",
                py.get_type::<pyo3::exceptions::PyFutureWarning>(),
            ),
        )?;
        self.createTable(py, tableName, path, source, schema, description, options)
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
        let warnings = py.import("warnings")?;
        let future_warning = py.import("builtins")?.getattr("FutureWarning")?;
        warnings.call_method1(
            "warn",
            (
                "Deprecated in 2.3.0. Use spark.udf.register instead.",
                future_warning,
            ),
        )?;
        let session = crate::session::session_object(py, self.catalog.spark_session())?;
        session
            .bind(py)
            .getattr("udf")?
            .call_method1("register", (name, f, returnType))
    }

    /// Create a database.
    #[pyo3(signature = (dbName, ifNotExists=false, properties=None))]
    #[allow(non_snake_case)]
    fn createDatabase(
        &self,
        py: Python<'_>,
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
        self.detached(py, |catalog| {
            { catalog.create_database(dbName, ifNotExists, props) }.map_err(Box::new)
        })
    }

    /// Drop a database.
    #[pyo3(signature = (dbName, ifExists=false, cascade=false))]
    #[allow(non_snake_case)]
    fn dropDatabase(
        &self,
        py: Python<'_>,
        dbName: &str,
        ifExists: bool,
        cascade: bool,
    ) -> PyResult<()> {
        self.detached(py, |catalog| {
            { catalog.drop_database(dbName, ifExists, cascade) }.map_err(Box::new)
        })
    }

    /// Drop a table.
    #[pyo3(signature = (tableName, ifExists=false, purge=false))]
    #[allow(non_snake_case)]
    fn dropTable(
        &self,
        py: Python<'_>,
        tableName: &str,
        ifExists: bool,
        purge: bool,
    ) -> PyResult<()> {
        self.detached(py, |catalog| {
            catalog
                .drop_table(tableName, ifExists, purge)
                .map_err(Box::new)
        })
    }

    /// Drop a view.
    #[pyo3(signature = (viewName, ifExists=false))]
    #[allow(non_snake_case)]
    fn dropView(&self, py: Python<'_>, viewName: &str, ifExists: bool) -> PyResult<()> {
        self.detached(py, |catalog| {
            catalog.drop_view(viewName, ifExists).map_err(Box::new)
        })
    }

    /// Truncate a table.
    #[allow(non_snake_case)]
    fn truncateTable(&self, py: Python<'_>, tableName: &str) -> PyResult<()> {
        self.detached(py, |catalog| {
            catalog.truncate_table(tableName).map_err(Box::new)
        })
    }

    /// Recover the statistics of a table.
    #[pyo3(signature = (tableName, noScan=false))]
    #[allow(non_snake_case)]
    fn analyzeTable(&self, py: Python<'_>, tableName: &str, noScan: bool) -> PyResult<()> {
        self.detached(py, |catalog| {
            catalog.analyze_table(tableName, noScan).map_err(Box::new)
        })
    }

    /// Get the `CREATE TABLE` string of a table.
    #[pyo3(signature = (tableName, asSerde=false))]
    #[allow(non_snake_case)]
    fn getCreateTableString(
        &self,
        py: Python<'_>,
        tableName: &str,
        asSerde: bool,
    ) -> PyResult<String> {
        self.detached(py, |catalog| {
            { catalog.get_create_table_string(tableName, asSerde) }.map_err(Box::new)
        })
    }

    /// Get the properties of a table as a dict.
    #[allow(non_snake_case)]
    fn getTableProperties<'py>(
        &self,
        py: Python<'py>,
        tableName: &str,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let props = self.detached(py, |catalog| {
            catalog.get_table_properties(tableName).map_err(Box::new)
        })?;
        let dict = pyo3::types::PyDict::new(py);
        for (k, v) in props {
            dict.set_item(k, v)?;
        }
        Ok(dict)
    }

    /// List the partitions of a table. Mirrors `Catalog.listPartitions -> List[TablePartition]`.
    #[allow(non_snake_case)]
    fn listPartitions(&self, py: Python<'_>, tableName: &str) -> PyResult<Vec<Py<PyAny>>> {
        named_tuples(
            py,
            self.detached(py, |catalog| {
                catalog.list_partitions_typed(tableName).map_err(Box::new)
            })?,
        )
    }

    /// List the views in a database. Mirrors `Catalog.listViews -> List[Table]`.
    #[pyo3(signature = (dbName=None, pattern=None))]
    #[allow(non_snake_case)]
    fn listViews(
        &self,
        py: Python<'_>,
        dbName: Option<&str>,
        pattern: Option<&str>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        named_tuples(
            py,
            self.detached(py, |catalog| {
                catalog.list_views_typed(dbName, pattern).map_err(Box::new)
            })?,
        )
    }

    fn __repr__(&self) -> String {
        "Catalog()".to_string()
    }
}
