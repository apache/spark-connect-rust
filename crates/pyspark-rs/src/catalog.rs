//! PyO3 wrapper for spark_connect::catalog::Catalog.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use spark_connect::catalog::{
    Catalog, CatalogMetadata, Column, Database, Function, Table, TablePartition,
};

use spark_connect_core::error::SparkError;

use crate::dataframe::PyDataFrame;
use crate::interrupt::interruptible;

/// PySpark's FutureWarning of the deprecated `dbName` of the catalog method `method`.
fn warn_db_name(py: Python<'_>, method: &str, db_name: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
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

// The names, patterns and paths of the catalog methods are set in the string fields of their
// plans as they are, which protobuf's default (upb) backend converts as `crate::str_argument`
// does: a str, or bytes decoded as UTF-8. An optional one is set only when it is not None.

/// The name of a catalog method that PySpark passes to the constructor of its message, which
/// leaves the string field unset for None, and converts any other value as
/// [`crate::str_argument`] does.
fn constructor_name(name: &Bound<'_, PyAny>) -> PyResult<String> {
    if name.is_none() {
        return Ok(String::new());
    }
    crate::str_argument(name)
}

/// A flag of a catalog method that PySpark passes to the constructor of its message, which
/// leaves the bool field unset (false) for None, as for an omitted argument, and converts any
/// other value as protobuf's default (upb) backend does ([`crate::streaming::proto_bool`]). The
/// flags that PySpark's plan sets instead are [`crate::streaming::ProtoBool`]s, of the
/// `TypeError` of None.
fn constructor_bool(flag: Option<&Bound<'_, PyAny>>) -> PyResult<bool> {
    Ok(flag
        .map(crate::streaming::proto_bool)
        .transpose()?
        .unwrap_or(false))
}

/// The `TypeError` that Python's binding of PySpark's `method` raises for the positional
/// arguments `extra` beyond its `count` parameters, of which the first, the table name, is
/// required, in Python's counts, which count `self`. PyO3 binds the other arguments, and raises
/// their `TypeError`s, as Python does.
fn check_extra_positional(
    method: &str,
    count: usize,
    extra: &Bound<'_, pyo3::types::PyTuple>,
) -> PyResult<()> {
    if extra.is_empty() {
        return Ok(());
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "{method}() takes from 2 to {} positional arguments but {} were given",
        count + 1,
        count + extra.len() + 1
    )))
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
    Err(crate::errors::not_expected_type(
        schema,
        "schema",
        "struct type",
    ))
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
    fn setCurrentCatalog(&self, py: Python<'_>, catalogName: &Bound<'_, PyAny>) -> PyResult<()> {
        let catalogName = crate::str_argument(catalogName)?;
        self.detached(py, |catalog| {
            catalog.set_current_catalog(&catalogName).map_err(Box::new)
        })
    }

    /// List catalogs. Mirrors `Catalog.listCatalogs(pattern=None) -> List[CatalogMetadata]`.
    #[pyo3(signature = (pattern=None))]
    fn listCatalogs(
        &self,
        py: Python<'_>,
        pattern: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let pattern = pattern.map(crate::str_argument).transpose()?;
        named_tuples(
            py,
            self.detached(py, |catalog| {
                { catalog.list_catalogs_typed(pattern.as_deref()) }.map_err(Box::new)
            })?,
        )
    }

    /// Get current database.
    fn currentDatabase(&self, py: Python<'_>) -> PyResult<String> {
        self.detached(py, |catalog| catalog.current_database().map_err(Box::new))
    }

    /// Set current database.
    fn setCurrentDatabase(&self, py: Python<'_>, dbName: &Bound<'_, PyAny>) -> PyResult<()> {
        let dbName = crate::str_argument(dbName)?;
        self.detached(py, |catalog| {
            catalog.set_current_database(&dbName).map_err(Box::new)
        })
    }

    /// List databases. Mirrors `Catalog.listDatabases(pattern=None) -> List[Database]`.
    #[pyo3(signature = (pattern=None))]
    fn listDatabases(
        &self,
        py: Python<'_>,
        pattern: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let pattern = pattern.map(crate::str_argument).transpose()?;
        named_tuples(
            py,
            self.detached(py, |catalog| {
                { catalog.list_databases_typed(pattern.as_deref()) }.map_err(Box::new)
            })?,
        )
    }

    /// Check if database exists.
    fn databaseExists(&self, py: Python<'_>, dbName: &Bound<'_, PyAny>) -> PyResult<bool> {
        let dbName = crate::str_argument(dbName)?;
        self.detached(py, |catalog| {
            catalog.database_exists(&dbName).map_err(Box::new)
        })
    }

    /// Get database info. Mirrors `Catalog.getDatabase(dbName) -> Database`.
    fn getDatabase(&self, py: Python<'_>, dbName: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let dbName = crate::str_argument(dbName)?;
        self.detached(py, |catalog| {
            catalog.get_database_typed(&dbName).map_err(Box::new)
        })?
        .named_tuple(py)
    }

    /// List tables. Mirrors `Catalog.listTables(dbName=None, pattern=None) -> List[Table]`.
    #[pyo3(signature = (dbName=None, pattern=None))]
    #[allow(non_snake_case)]
    fn listTables(
        &self,
        py: Python<'_>,
        dbName: Option<&Bound<'_, PyAny>>,
        pattern: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let dbName = dbName.map(crate::str_argument).transpose()?;
        let pattern = pattern.map(crate::str_argument).transpose()?;
        named_tuples(
            py,
            self.detached(py, |catalog| {
                { catalog.list_tables_typed(dbName.as_deref(), pattern.as_deref()) }
                    .map_err(Box::new)
            })?,
        )
    }

    /// Check if table exists. Mirrors `Catalog.tableExists(tableName, dbName=None) -> bool`.
    #[pyo3(signature = (tableName, dbName=None))]
    fn tableExists(
        &self,
        py: Python<'_>,
        tableName: &Bound<'_, PyAny>,
        dbName: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        warn_db_name(py, "tableExists", dbName)?;
        let tableName = crate::str_argument(tableName)?;
        let dbName = dbName.map(crate::str_argument).transpose()?;
        self.detached(py, |catalog| {
            { catalog.table_exists_with_database(&tableName, dbName.as_deref()) }.map_err(Box::new)
        })
    }

    /// Get table info. Mirrors `Catalog.getTable(tableName) -> Table`.
    fn getTable(&self, py: Python<'_>, tableName: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let tableName = crate::str_argument(tableName)?;
        self.detached(py, |catalog| {
            catalog.get_table_typed(&tableName).map_err(Box::new)
        })?
        .named_tuple(py)
    }

    /// List functions. Mirrors `Catalog.listFunctions(dbName=None, pattern=None) -> List[Function]`.
    #[pyo3(signature = (dbName=None, pattern=None))]
    #[allow(non_snake_case)]
    fn listFunctions(
        &self,
        py: Python<'_>,
        dbName: Option<&Bound<'_, PyAny>>,
        pattern: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let dbName = dbName.map(crate::str_argument).transpose()?;
        let pattern = pattern.map(crate::str_argument).transpose()?;
        named_tuples(
            py,
            self.detached(py, |catalog| {
                catalog
                    .list_functions_typed(dbName.as_deref(), pattern.as_deref())
                    .map_err(Box::new)
            })?,
        )
    }

    /// Check if function exists. Mirrors `Catalog.functionExists(functionName, dbName=None) -> bool`.
    #[pyo3(signature = (functionName, dbName=None))]
    fn functionExists(
        &self,
        py: Python<'_>,
        functionName: &Bound<'_, PyAny>,
        dbName: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        warn_db_name(py, "functionExists", dbName)?;
        let functionName = crate::str_argument(functionName)?;
        let dbName = dbName.map(crate::str_argument).transpose()?;
        self.detached(py, |catalog| {
            { catalog.function_exists_with_database(&functionName, dbName.as_deref()) }
                .map_err(Box::new)
        })
    }

    /// Get function info. Mirrors `Catalog.getFunction(functionName) -> Function`.
    fn getFunction(&self, py: Python<'_>, functionName: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let functionName = crate::str_argument(functionName)?;
        self.detached(py, |catalog| {
            catalog.get_function_typed(&functionName).map_err(Box::new)
        })?
        .named_tuple(py)
    }

    /// List columns of a table. Mirrors `Catalog.listColumns(tableName, dbName=None) -> List[Column]`.
    #[pyo3(signature = (tableName, dbName=None))]
    #[allow(non_snake_case)]
    fn listColumns(
        &self,
        py: Python<'_>,
        tableName: &Bound<'_, PyAny>,
        dbName: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        warn_db_name(py, "listColumns", dbName)?;
        let tableName = crate::str_argument(tableName)?;
        let dbName = dbName.map(crate::str_argument).transpose()?;
        named_tuples(
            py,
            self.detached(py, |catalog| {
                catalog
                    .list_columns_typed(&tableName, dbName.as_deref())
                    .map_err(Box::new)
            })?,
        )
    }

    /// Cache a table. Mirrors `Catalog.cacheTable(tableName, storageLevel=None) -> None`.
    ///
    /// A supplied pyspark `StorageLevel` is mapped onto the `spark.connect`
    /// `CacheTable.storage_level` field (the same field layout `DataFrame.persist` uses);
    /// `None` leaves it unset so the server default applies. As PySpark's `CacheTable`, which
    /// converts a storage level that is true (`if self._storage_level:`), any false value, such
    /// as 0, leaves it unset too, and another true value fails the assertion of the conversion,
    /// after the table name.
    #[pyo3(signature = (tableName, storageLevel=None))]
    fn cacheTable(
        &self,
        py: Python<'_>,
        tableName: &Bound<'_, PyAny>,
        storageLevel: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let tableName = constructor_name(tableName)?;
        let level = match storageLevel {
            Some(level) if level.is_truthy()? => {
                Some(crate::storagelevel::storage_level_proto(level)?)
            }
            _ => None,
        };
        self.detached(py, |catalog| {
            { catalog.cache_table_with_storage_level(&tableName, level) }.map_err(Box::new)
        })
    }

    /// Uncache a table.
    fn uncacheTable(&self, py: Python<'_>, tableName: &Bound<'_, PyAny>) -> PyResult<()> {
        let tableName = crate::str_argument(tableName)?;
        self.detached(py, |catalog| {
            catalog.uncache_table(&tableName).map_err(Box::new)
        })
    }

    /// Drop temporary view.
    fn dropTempView(&self, py: Python<'_>, viewName: &Bound<'_, PyAny>) -> PyResult<bool> {
        let viewName = crate::str_argument(viewName)?;
        self.detached(py, |catalog| {
            catalog.drop_temp_view(&viewName).map_err(Box::new)
        })
    }

    /// Drop a global temporary view.
    fn dropGlobalTempView(&self, py: Python<'_>, viewName: &Bound<'_, PyAny>) -> PyResult<bool> {
        let viewName = crate::str_argument(viewName)?;
        self.detached(py, |catalog| {
            catalog.drop_global_temp_view(&viewName).map_err(Box::new)
        })
    }

    /// Whether a table/view is cached.
    fn isCached(&self, py: Python<'_>, tableName: &Bound<'_, PyAny>) -> PyResult<bool> {
        let tableName = crate::str_argument(tableName)?;
        self.detached(py, |catalog| {
            catalog.is_cached(&tableName).map_err(Box::new)
        })
    }

    /// Remove all cached tables from the in-memory cache.
    fn clearCache(&self, py: Python<'_>) -> PyResult<()> {
        self.detached(py, |catalog| catalog.clear_cache().map_err(Box::new))
    }

    /// Invalidate and refresh cached metadata for a table.
    fn refreshTable(&self, py: Python<'_>, tableName: &Bound<'_, PyAny>) -> PyResult<()> {
        let tableName = crate::str_argument(tableName)?;
        self.detached(py, |catalog| {
            catalog.refresh_table(&tableName).map_err(Box::new)
        })
    }

    /// Invalidate and refresh cached data for any table at a path.
    fn refreshByPath(&self, py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<()> {
        let path = crate::str_argument(path)?;
        self.detached(py, |catalog| {
            catalog.refresh_by_path(&path).map_err(Box::new)
        })
    }

    /// Recover all the partitions of a table.
    fn recoverPartitions(&self, py: Python<'_>, tableName: &Bound<'_, PyAny>) -> PyResult<()> {
        let tableName = crate::str_argument(tableName)?;
        self.detached(py, |catalog| {
            catalog.recover_partitions(&tableName).map_err(Box::new)
        })
    }

    /// Create a table from a data source, as PySpark's: of a StructType `schema` and string
    /// `**options`, created on return.
    #[pyo3(
        signature = (tableName, path=None, source=None, schema=None, description=None, *extra, **options),
        text_signature = "($self, tableName, path=None, source=None, schema=None, description=None, **options)"
    )]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn createTable(
        &self,
        py: Python<'_>,
        tableName: &Bound<'_, PyAny>,
        path: Option<&Bound<'_, PyAny>>,
        source: Option<&Bound<'_, PyAny>>,
        schema: Option<&Bound<'_, PyAny>>,
        description: Option<&Bound<'_, PyAny>>,
        extra: &Bound<'_, pyo3::types::PyTuple>,
        options: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<PyDataFrame> {
        crate::options::check_keyword_keys(options)?;
        check_extra_positional("Catalog.createTable", 5, extra)?;
        let schema = table_schema(schema)?;
        // In the order of PySpark's `CreateTable` plan, after its check of the schema, and
        // before the options.
        let tableName = crate::str_argument(tableName)?;
        let path = path.map(crate::str_argument).transpose()?;
        let source = source.map(crate::str_argument).transpose()?;
        let description = description.map(crate::str_argument).transpose()?;
        let options = crate::options::str_kwargs_skip_none(options)?;
        interruptible(py, self.catalog.spark_session(), || {
            self.catalog
                .create_table_with(
                    &tableName,
                    path.as_deref(),
                    source.as_deref(),
                    schema.as_ref(),
                    description.as_deref(),
                    options,
                )
                .map_err(Box::new)
        })
        .map(PyDataFrame::new)
    }

    /// Create an external table from a data source: deprecated, as PySpark's
    /// `createExternalTable(tableName, path=None, source=None, schema=None, **options)`, which
    /// warns and creates the table as `createTable` does, of a `description` of the options.
    #[pyo3(
        signature = (tableName, path=None, source=None, schema=None, *extra, **options),
        text_signature = "($self, tableName, path=None, source=None, schema=None, **options)"
    )]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn createExternalTable(
        &self,
        py: Python<'_>,
        tableName: &Bound<'_, PyAny>,
        path: Option<&Bound<'_, PyAny>>,
        source: Option<&Bound<'_, PyAny>>,
        schema: Option<&Bound<'_, PyAny>>,
        extra: &Bound<'_, pyo3::types::PyTuple>,
        options: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<PyDataFrame> {
        crate::options::check_keyword_keys(options)?;
        check_extra_positional("Catalog.createExternalTable", 4, extra)?;
        py.import("warnings")?.call_method1(
            "warn",
            (
                "createExternalTable is deprecated since Spark 4.0, please use createTable \
                 instead.",
                py.get_type::<pyo3::exceptions::PyFutureWarning>(),
            ),
        )?;
        // As PySpark's `self.createTable(tableName, path, source, schema, **options)`, which
        // binds a `description` of the options to the parameter of createTable.
        let options = options.map(|options| options.copy()).transpose()?;
        let mut description = None;
        if let Some(options) = &options {
            description = options.get_item("description")?;
            if description.is_some() {
                options.del_item("description")?;
            }
        }
        let description = description.filter(|description| !description.is_none());
        let none = pyo3::types::PyTuple::empty(py);
        self.createTable(
            py,
            tableName,
            path,
            source,
            schema,
            description.as_ref(),
            &none,
            options.as_ref(),
        )
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

    /// Create a database, with the `properties` of PySpark's protobuf map of strings, whose keys
    /// and values are converted as `crate::str_argument` converts them. As PySpark's
    /// `CreateDatabase`, which takes `properties or {}`, and then passes the database and
    /// `ifNotExists` to the constructor of its message before it sets each of the
    /// `properties.items()`.
    #[pyo3(
        signature = (dbName, ifNotExists=None, properties=None),
        text_signature = "($self, dbName, ifNotExists=False, properties=None)"
    )]
    #[allow(non_snake_case)]
    fn createDatabase(
        &self,
        py: Python<'_>,
        dbName: &Bound<'_, PyAny>,
        ifNotExists: Option<&Bound<'_, PyAny>>,
        properties: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let properties = match properties {
            Some(properties) if properties.is_truthy()? => Some(properties),
            _ => None,
        };
        let dbName = constructor_name(dbName)?;
        let ifNotExists = constructor_bool(ifNotExists)?;
        let mut props = std::collections::HashMap::new();
        if let Some(properties) = properties {
            for item in properties.call_method0("items")?.try_iter()? {
                let (k, v): (Bound<'_, PyAny>, Bound<'_, PyAny>) = item?.extract()?;
                props.insert(crate::str_argument(&k)?, crate::str_argument(&v)?);
            }
        }
        self.detached(py, |catalog| {
            { catalog.create_database(&dbName, ifNotExists, props) }.map_err(Box::new)
        })
    }

    /// Drop a database.
    #[pyo3(
        signature = (dbName, ifExists=None, cascade=None),
        text_signature = "($self, dbName, ifExists=False, cascade=False)"
    )]
    #[allow(non_snake_case)]
    fn dropDatabase(
        &self,
        py: Python<'_>,
        dbName: &Bound<'_, PyAny>,
        ifExists: Option<&Bound<'_, PyAny>>,
        cascade: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let dbName = constructor_name(dbName)?;
        let (ifExists, cascade) = (constructor_bool(ifExists)?, constructor_bool(cascade)?);
        self.detached(py, |catalog| {
            { catalog.drop_database(&dbName, ifExists, cascade) }.map_err(Box::new)
        })
    }

    /// Drop a table.
    #[pyo3(
        signature = (tableName, ifExists=None, purge=None),
        text_signature = "($self, tableName, ifExists=False, purge=False)"
    )]
    #[allow(non_snake_case)]
    fn dropTable(
        &self,
        py: Python<'_>,
        tableName: &Bound<'_, PyAny>,
        ifExists: Option<&Bound<'_, PyAny>>,
        purge: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let tableName = constructor_name(tableName)?;
        let (ifExists, purge) = (constructor_bool(ifExists)?, constructor_bool(purge)?);
        self.detached(py, |catalog| {
            catalog
                .drop_table(&tableName, ifExists, purge)
                .map_err(Box::new)
        })
    }

    /// Drop a view.
    #[pyo3(
        signature = (viewName, ifExists=None),
        text_signature = "($self, viewName, ifExists=False)"
    )]
    #[allow(non_snake_case)]
    fn dropView(
        &self,
        py: Python<'_>,
        viewName: &Bound<'_, PyAny>,
        ifExists: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let viewName = constructor_name(viewName)?;
        let ifExists = constructor_bool(ifExists)?;
        self.detached(py, |catalog| {
            catalog.drop_view(&viewName, ifExists).map_err(Box::new)
        })
    }

    /// Truncate a table.
    #[allow(non_snake_case)]
    fn truncateTable(&self, py: Python<'_>, tableName: &Bound<'_, PyAny>) -> PyResult<()> {
        let tableName = crate::str_argument(tableName)?;
        self.detached(py, |catalog| {
            catalog.truncate_table(&tableName).map_err(Box::new)
        })
    }

    /// Recover the statistics of a table, of `noScan` as PySpark's plan sets it in its bool field.
    #[pyo3(
        signature = (tableName, noScan=crate::streaming::ProtoBool::default()),
        text_signature = "($self, tableName, noScan=False)"
    )]
    #[allow(non_snake_case)]
    fn analyzeTable(
        &self,
        py: Python<'_>,
        tableName: &Bound<'_, PyAny>,
        noScan: crate::streaming::ProtoBool,
    ) -> PyResult<()> {
        let tableName = crate::str_argument(tableName)?;
        let noScan = noScan.value(py)?;
        self.detached(py, |catalog| {
            catalog.analyze_table(&tableName, noScan).map_err(Box::new)
        })
    }

    /// Get the `CREATE TABLE` string of a table, of `asSerde` as PySpark's plan sets it in its
    /// bool field.
    #[pyo3(
        signature = (tableName, asSerde=crate::streaming::ProtoBool::default()),
        text_signature = "($self, tableName, asSerde=False)"
    )]
    #[allow(non_snake_case)]
    fn getCreateTableString(
        &self,
        py: Python<'_>,
        tableName: &Bound<'_, PyAny>,
        asSerde: crate::streaming::ProtoBool,
    ) -> PyResult<String> {
        let tableName = crate::str_argument(tableName)?;
        let asSerde = asSerde.value(py)?;
        self.detached(py, |catalog| {
            { catalog.get_create_table_string(&tableName, asSerde) }.map_err(Box::new)
        })
    }

    /// Get the properties of a table as a dict.
    #[allow(non_snake_case)]
    fn getTableProperties<'py>(
        &self,
        py: Python<'py>,
        tableName: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let tableName = crate::str_argument(tableName)?;
        let props = self.detached(py, |catalog| {
            catalog.get_table_properties(&tableName).map_err(Box::new)
        })?;
        let dict = pyo3::types::PyDict::new(py);
        for (k, v) in props {
            dict.set_item(k, v)?;
        }
        Ok(dict)
    }

    /// List the partitions of a table. Mirrors `Catalog.listPartitions -> List[TablePartition]`.
    #[allow(non_snake_case)]
    fn listPartitions(
        &self,
        py: Python<'_>,
        tableName: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let tableName = crate::str_argument(tableName)?;
        named_tuples(
            py,
            self.detached(py, |catalog| {
                catalog.list_partitions_typed(&tableName).map_err(Box::new)
            })?,
        )
    }

    /// List the views in a database. Mirrors `Catalog.listViews -> List[Table]`, which asks for
    /// the current database of a pattern without a database, before its plan sets either.
    #[pyo3(signature = (dbName=None, pattern=None))]
    #[allow(non_snake_case)]
    fn listViews(
        &self,
        py: Python<'_>,
        dbName: Option<&Bound<'_, PyAny>>,
        pattern: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let dbName = match (dbName, pattern) {
            (None, Some(_)) => Some(self.currentDatabase(py)?),
            (dbName, _) => dbName.map(crate::str_argument).transpose()?,
        };
        let pattern = pattern.map(crate::str_argument).transpose()?;
        named_tuples(
            py,
            self.detached(py, |catalog| {
                { catalog.list_views_typed(dbName.as_deref(), pattern.as_deref()) }
                    .map_err(Box::new)
            })?,
        )
    }

    fn __repr__(&self) -> String {
        "Catalog()".to_string()
    }
}
