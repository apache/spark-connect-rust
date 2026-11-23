//! PyO3 wrapper for spark_connect::catalog::Catalog.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use spark_connect::catalog::{
    Catalog, CatalogMetadata, Column, Database, Function, Table, TablePartition,
};

use spark_connect_core::error::SparkError;

use crate::dataframe::PyDataFrame;
use crate::interrupt::interruptible_without_parts;

/// PySpark's FutureWarning of the deprecated `dbName` of the catalog method `method`, of the line
/// `line` of its `warnings.warn` in `pyspark.sql.connect.catalog` (see `warn_as_pyspark`).
fn warn_db_name(
    py: Python<'_>,
    method: &str,
    line: u32,
    db_name: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    if db_name.is_some() {
        crate::errors::warn_as_pyspark(
            py,
            "pyspark.sql.connect.catalog",
            line,
            &format!(
                "`dbName` has been deprecated since Spark 3.4 and might be removed in a \
                 future version. Use {method}(`dbName.tableName`) instead."
            ),
            &py.get_type::<pyo3::exceptions::PyFutureWarning>(),
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

/// The schema of `createTable`, a StructType, as PySpark's, else `NOT_EXPECTED_TYPE`. Its proto
/// is made later, in the order of PySpark's `CreateTable` plan (see [`table_schema_proto`]).
fn table_schema<'a, 'py>(
    schema: Option<&'a Bound<'py, PyAny>>,
) -> PyResult<Option<&'a Bound<'py, PyAny>>> {
    let Some(schema) = schema.filter(|schema| !schema.is_none()) else {
        return Ok(None);
    };
    if schema.is_instance_of::<crate::types::PyStructType>() {
        return Ok(Some(schema));
    }
    Err(crate::errors::not_expected_type(
        schema,
        "schema",
        "struct type",
    ))
}

/// The core type of the StructType `schema` of `createTable`, of its fields as they are now, as
/// PySpark's `CreateTable` plan converts it with `pyspark_types_to_proto_types`: a type of a
/// field that has no proto, a `StructField` as a type or a type of its class only such as
/// `AtomicType()`, also nested, raises its `UNSUPPORTED_OPERATION`.
fn table_schema_proto(
    schema: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<spark_connect::types::DataType>> {
    schema.map(crate::types::py_to_proto_data_type).transpose()
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

// Python wrapper for Spark Catalog. It has the default repr, as PySpark Connect's Catalog, which
// defines none: `<pyspark.sql.connect.catalog.Catalog object at 0x...>`. As PySpark's plain class,
// it keeps its state in its `__dict__`, its `_sparkSession`, which its methods read when they make
// their requests, and it is subclassed and constructed of PySpark's arguments (see `crate::state`).
#[doc = connect_doc!(Catalog)]
#[pyclass(
    name = "Catalog",
    module = "pyspark.sql.connect.catalog",
    subclass,
    weakref,
    dict
)]
pub struct PyCatalog {}

impl PyCatalog {
    /// The catalog of `session`, as `spark.catalog` makes it: `Catalog(spark)`, of the session
    /// object, which PySpark's `__init__` keeps as its `_sparkSession`.
    pub(crate) fn of_session<'py>(session: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyCatalog>> {
        let catalog = Bound::new(session.py(), PyCatalog {})?;
        Self::__init__(&catalog, session)?;
        Ok(catalog)
    }

    /// The core catalog of the session of `_sparkSession`, which PySpark's methods read as they
    /// make their request, `DataFrame(plan, session=self._sparkSession)` of its `client`: of the
    /// AttributeError of a catalog whose `__init__` never ran, and of the object that it holds
    /// (see `state::session_of_value`).
    fn catalog_of(slf: &Bound<'_, Self>) -> PyResult<Catalog> {
        let py = slf.py();
        let session = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_sparkSession"))?;
        Ok(crate::state::session_of_value(&session, pyo3::intern!(py, "client"))?.catalog())
    }

    /// `f` of the catalog without the GIL, as PySpark's requests to the server release it
    /// while they wait, so that other Python threads run and a signal interrupts the wait.
    fn detached<T: Send>(
        slf: &Bound<'_, Self>,
        f: impl FnOnce(&Catalog) -> Result<T, Box<SparkError>> + Send,
    ) -> PyResult<T> {
        let catalog = Self::catalog_of(slf)?;
        interruptible_without_parts(slf.py(), catalog.spark_session(), || f(&catalog))
    }
}

#[pymethods]
impl PyCatalog {
    /// A catalog of no state, as the `object.__new__` of PySpark's plain class: of any arguments,
    /// which `__init__` takes, so that a subclass of another `__init__` is built as it is in
    /// PySpark. The class shows the signature of `__init__`, as `inspect.signature` of PySpark's
    /// class does.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs), text_signature = "(sparkSession)")]
    fn py_new(
        _args: &Bound<'_, pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> Self {
        PyCatalog {}
    }

    /// `Catalog(sparkSession)`, as PySpark Connect's `__init__`, which keeps its argument as it
    /// is, as the `_sparkSession` of the catalog: another catalog than `spark.catalog`, which the
    /// session keeps.
    #[allow(non_snake_case)]
    fn __init__(slf: &Bound<'_, Self>, sparkSession: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        crate::state::assign(
            slf.as_any(),
            &[(pyo3::intern!(py, "_sparkSession"), sparkSession)],
        )
    }

    /// The state of the catalog, its `__dict__`, as `object.__getstate__` gives that of PySpark's
    /// plain object, which a copy and a pickle keep (see `state::getstate`).
    fn __getstate__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::state::getstate(slf.as_any())
    }

    /// The reduce of the catalog, as PySpark's plain object's: `copyreg.__newobj__` of its class,
    /// of its `__dict__`, so a copy is of the same `_sparkSession`, a deep copy and a pickle are of
    /// the active session of its ID, and PySpark's pickle loads here (see `state::reduce_ex`).
    fn __reduce_ex__<'py>(slf: &Bound<'py, Self>, protocol: i32) -> PyResult<Bound<'py, PyAny>> {
        crate::state::reduce_ex(slf.as_any(), protocol)
    }

    // Get the current catalog.
    #[doc = connect_doc!(Catalog.currentCatalog)]
    fn currentCatalog(slf: &Bound<'_, Self>) -> PyResult<String> {
        Self::detached(slf, |catalog| catalog.current_catalog().map_err(Box::new))
    }

    // Set the current catalog.
    #[doc = connect_doc!(Catalog.setCurrentCatalog)]
    fn setCurrentCatalog(slf: &Bound<'_, Self>, catalogName: &Bound<'_, PyAny>) -> PyResult<()> {
        let catalogName = crate::str_argument(catalogName)?;
        Self::detached(slf, |catalog| {
            catalog.set_current_catalog(&catalogName).map_err(Box::new)
        })
    }

    // List catalogs. Mirrors `Catalog.listCatalogs(pattern=None) -> List[CatalogMetadata]`.
    #[pyo3(signature = (pattern=None))]
    #[doc = connect_doc!(Catalog.listCatalogs)]
    fn listCatalogs(
        slf: &Bound<'_, Self>,
        pattern: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let py = slf.py();
        let pattern = pattern.map(crate::str_argument).transpose()?;
        named_tuples(
            py,
            Self::detached(slf, |catalog| {
                { catalog.list_catalogs_typed(pattern.as_deref()) }.map_err(Box::new)
            })?,
        )
    }

    // Get current database.
    #[doc = connect_doc!(Catalog.currentDatabase)]
    fn currentDatabase(slf: &Bound<'_, Self>) -> PyResult<String> {
        Self::detached(slf, |catalog| catalog.current_database().map_err(Box::new))
    }

    // Set current database.
    #[doc = connect_doc!(Catalog.setCurrentDatabase)]
    fn setCurrentDatabase(slf: &Bound<'_, Self>, dbName: &Bound<'_, PyAny>) -> PyResult<()> {
        let dbName = crate::str_argument(dbName)?;
        Self::detached(slf, |catalog| {
            catalog.set_current_database(&dbName).map_err(Box::new)
        })
    }

    // List databases. Mirrors `Catalog.listDatabases(pattern=None) -> List[Database]`.
    #[pyo3(signature = (pattern=None))]
    #[doc = connect_doc!(Catalog.listDatabases)]
    fn listDatabases(
        slf: &Bound<'_, Self>,
        pattern: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let py = slf.py();
        let pattern = pattern.map(crate::str_argument).transpose()?;
        named_tuples(
            py,
            Self::detached(slf, |catalog| {
                { catalog.list_databases_typed(pattern.as_deref()) }.map_err(Box::new)
            })?,
        )
    }

    // Check if database exists.
    #[doc = connect_doc!(Catalog.databaseExists)]
    fn databaseExists(slf: &Bound<'_, Self>, dbName: &Bound<'_, PyAny>) -> PyResult<bool> {
        let dbName = crate::str_argument(dbName)?;
        Self::detached(slf, |catalog| {
            catalog.database_exists(&dbName).map_err(Box::new)
        })
    }

    // Get database info. Mirrors `Catalog.getDatabase(dbName) -> Database`.
    #[doc = connect_doc!(Catalog.getDatabase)]
    fn getDatabase(slf: &Bound<'_, Self>, dbName: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let dbName = crate::str_argument(dbName)?;
        Self::detached(slf, |catalog| {
            catalog.get_database_typed(&dbName).map_err(Box::new)
        })?
        .named_tuple(py)
    }

    // List tables. Mirrors `Catalog.listTables(dbName=None, pattern=None) -> List[Table]`.
    #[pyo3(signature = (dbName=None, pattern=None))]
    #[allow(non_snake_case)]
    #[doc = connect_doc!(Catalog.listTables)]
    fn listTables(
        slf: &Bound<'_, Self>,
        dbName: Option<&Bound<'_, PyAny>>,
        pattern: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let py = slf.py();
        let dbName = dbName.map(crate::str_argument).transpose()?;
        let pattern = pattern.map(crate::str_argument).transpose()?;
        named_tuples(
            py,
            Self::detached(slf, |catalog| {
                { catalog.list_tables_typed(dbName.as_deref(), pattern.as_deref()) }
                    .map_err(Box::new)
            })?,
        )
    }

    // Check if table exists. Mirrors `Catalog.tableExists(tableName, dbName=None) -> bool`.
    #[pyo3(signature = (tableName, dbName=None))]
    #[doc = connect_doc!(Catalog.tableExists)]
    fn tableExists(
        slf: &Bound<'_, Self>,
        tableName: &Bound<'_, PyAny>,
        dbName: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        let py = slf.py();
        warn_db_name(py, "tableExists", 218, dbName)?;
        let tableName = crate::str_argument(tableName)?;
        let dbName = dbName.map(crate::str_argument).transpose()?;
        Self::detached(slf, |catalog| {
            { catalog.table_exists_with_database(&tableName, dbName.as_deref()) }.map_err(Box::new)
        })
    }

    // Get table info. Mirrors `Catalog.getTable(tableName) -> Table`.
    #[doc = connect_doc!(Catalog.getTable)]
    fn getTable(slf: &Bound<'_, Self>, tableName: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let tableName = crate::str_argument(tableName)?;
        Self::detached(slf, |catalog| {
            catalog.get_table_typed(&tableName).map_err(Box::new)
        })?
        .named_tuple(py)
    }

    // List functions. Mirrors `Catalog.listFunctions(dbName=None, pattern=None) -> List[Function]`.
    #[pyo3(signature = (dbName=None, pattern=None))]
    #[allow(non_snake_case)]
    #[doc = connect_doc!(Catalog.listFunctions)]
    fn listFunctions(
        slf: &Bound<'_, Self>,
        dbName: Option<&Bound<'_, PyAny>>,
        pattern: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let py = slf.py();
        let dbName = dbName.map(crate::str_argument).transpose()?;
        let pattern = pattern.map(crate::str_argument).transpose()?;
        named_tuples(
            py,
            Self::detached(slf, |catalog| {
                catalog
                    .list_functions_typed(dbName.as_deref(), pattern.as_deref())
                    .map_err(Box::new)
            })?,
        )
    }

    // Check if function exists. Mirrors `Catalog.functionExists(functionName, dbName=None) -> bool`.
    #[pyo3(signature = (functionName, dbName=None))]
    #[doc = connect_doc!(Catalog.functionExists)]
    fn functionExists(
        slf: &Bound<'_, Self>,
        functionName: &Bound<'_, PyAny>,
        dbName: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        let py = slf.py();
        warn_db_name(py, "functionExists", 168, dbName)?;
        let functionName = crate::str_argument(functionName)?;
        let dbName = dbName.map(crate::str_argument).transpose()?;
        Self::detached(slf, |catalog| {
            { catalog.function_exists_with_database(&functionName, dbName.as_deref()) }
                .map_err(Box::new)
        })
    }

    // Get function info. Mirrors `Catalog.getFunction(functionName) -> Function`.
    #[doc = connect_doc!(Catalog.getFunction)]
    fn getFunction(slf: &Bound<'_, Self>, functionName: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let functionName = crate::str_argument(functionName)?;
        Self::detached(slf, |catalog| {
            catalog.get_function_typed(&functionName).map_err(Box::new)
        })?
        .named_tuple(py)
    }

    // List columns of a table. Mirrors `Catalog.listColumns(tableName, dbName=None) -> List[Column]`.
    #[pyo3(signature = (tableName, dbName=None))]
    #[allow(non_snake_case)]
    #[doc = connect_doc!(Catalog.listColumns)]
    fn listColumns(
        slf: &Bound<'_, Self>,
        tableName: &Bound<'_, PyAny>,
        dbName: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let py = slf.py();
        warn_db_name(py, "listColumns", 195, dbName)?;
        let tableName = crate::str_argument(tableName)?;
        let dbName = dbName.map(crate::str_argument).transpose()?;
        named_tuples(
            py,
            Self::detached(slf, |catalog| {
                catalog
                    .list_columns_typed(&tableName, dbName.as_deref())
                    .map_err(Box::new)
            })?,
        )
    }

    // Cache a table. Mirrors `Catalog.cacheTable(tableName, storageLevel=None) -> None`.
    //
    // A supplied pyspark `StorageLevel` is mapped onto the `spark.connect`
    // `CacheTable.storage_level` field (the same field layout `DataFrame.persist` uses);
    // `None` leaves it unset so the server default applies. As PySpark's `CacheTable`, which
    // converts a storage level that is true (`if self._storage_level:`), any false value, such
    // as 0, leaves it unset too, and another true value fails the assertion of the conversion,
    // after the table name.
    #[pyo3(signature = (tableName, storageLevel=None))]
    #[doc = connect_doc!(Catalog.cacheTable)]
    fn cacheTable(
        slf: &Bound<'_, Self>,
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
        Self::detached(slf, |catalog| {
            { catalog.cache_table_with_storage_level(&tableName, level) }.map_err(Box::new)
        })
    }

    // Uncache a table.
    #[doc = connect_doc!(Catalog.uncacheTable)]
    fn uncacheTable(slf: &Bound<'_, Self>, tableName: &Bound<'_, PyAny>) -> PyResult<()> {
        let tableName = crate::str_argument(tableName)?;
        Self::detached(slf, |catalog| {
            catalog.uncache_table(&tableName).map_err(Box::new)
        })
    }

    // Drop temporary view.
    #[doc = connect_doc!(Catalog.dropTempView)]
    fn dropTempView(slf: &Bound<'_, Self>, viewName: &Bound<'_, PyAny>) -> PyResult<bool> {
        let viewName = crate::str_argument(viewName)?;
        Self::detached(slf, |catalog| {
            catalog.drop_temp_view(&viewName).map_err(Box::new)
        })
    }

    // Drop a global temporary view.
    #[doc = connect_doc!(Catalog.dropGlobalTempView)]
    fn dropGlobalTempView(slf: &Bound<'_, Self>, viewName: &Bound<'_, PyAny>) -> PyResult<bool> {
        let viewName = crate::str_argument(viewName)?;
        Self::detached(slf, |catalog| {
            catalog.drop_global_temp_view(&viewName).map_err(Box::new)
        })
    }

    // Whether a table/view is cached.
    #[doc = connect_doc!(Catalog.isCached)]
    fn isCached(slf: &Bound<'_, Self>, tableName: &Bound<'_, PyAny>) -> PyResult<bool> {
        let tableName = crate::str_argument(tableName)?;
        Self::detached(slf, |catalog| {
            catalog.is_cached(&tableName).map_err(Box::new)
        })
    }

    // Remove all cached tables from the in-memory cache.
    #[doc = connect_doc!(Catalog.clearCache)]
    fn clearCache(slf: &Bound<'_, Self>) -> PyResult<()> {
        Self::detached(slf, |catalog| catalog.clear_cache().map_err(Box::new))
    }

    // Invalidate and refresh cached metadata for a table.
    #[doc = connect_doc!(Catalog.refreshTable)]
    fn refreshTable(slf: &Bound<'_, Self>, tableName: &Bound<'_, PyAny>) -> PyResult<()> {
        let tableName = crate::str_argument(tableName)?;
        Self::detached(slf, |catalog| {
            catalog.refresh_table(&tableName).map_err(Box::new)
        })
    }

    // Invalidate and refresh cached data for any table at a path.
    #[doc = connect_doc!(Catalog.refreshByPath)]
    fn refreshByPath(slf: &Bound<'_, Self>, path: &Bound<'_, PyAny>) -> PyResult<()> {
        let path = crate::str_argument(path)?;
        Self::detached(slf, |catalog| {
            catalog.refresh_by_path(&path).map_err(Box::new)
        })
    }

    // Recover all the partitions of a table.
    #[doc = connect_doc!(Catalog.recoverPartitions)]
    fn recoverPartitions(slf: &Bound<'_, Self>, tableName: &Bound<'_, PyAny>) -> PyResult<()> {
        let tableName = crate::str_argument(tableName)?;
        Self::detached(slf, |catalog| {
            catalog.recover_partitions(&tableName).map_err(Box::new)
        })
    }

    // Create a table from a data source, as PySpark's: of a StructType `schema` and string
    // `**options`, created on return.
    #[pyo3(
        signature = (tableName, path=None, source=None, schema=None, description=None, *extra, **options),
        text_signature = "($self, tableName, path=None, source=None, schema=None, description=None, **options)"
    )]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    #[doc = connect_doc!(Catalog.createTable)]
    fn createTable(
        slf: &Bound<'_, Self>,
        tableName: &Bound<'_, PyAny>,
        path: Option<&Bound<'_, PyAny>>,
        source: Option<&Bound<'_, PyAny>>,
        schema: Option<&Bound<'_, PyAny>>,
        description: Option<&Bound<'_, PyAny>>,
        extra: &Bound<'_, pyo3::types::PyTuple>,
        options: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<PyDataFrame> {
        let py = slf.py();
        crate::options::check_keyword_keys(options)?;
        check_extra_positional("Catalog.createTable", 5, extra)?;
        let schema = table_schema(schema)?;
        // In the order of PySpark's `CreateTable` plan, after its check of the schema: the name,
        // the path, the source and the description, then the schema, and the options.
        let tableName = crate::str_argument(tableName)?;
        let path = path.map(crate::str_argument).transpose()?;
        let source = source.map(crate::str_argument).transpose()?;
        let description = description.map(crate::str_argument).transpose()?;
        let schema = table_schema_proto(schema)?;
        let options = crate::options::str_kwargs_skip_none(options)?;
        // Of the session that PySpark's `DataFrame(plan, session=self._sparkSession)` reads then.
        let catalog = Self::catalog_of(slf)?;
        interruptible_without_parts(py, catalog.spark_session(), || {
            catalog
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

    // Create an external table from a data source: deprecated, as PySpark's
    // `createExternalTable(tableName, path=None, source=None, schema=None, **options)`, which
    // warns and creates the table as `createTable` does, of a `description` of the options.
    #[pyo3(
        signature = (tableName, path=None, source=None, schema=None, *extra, **options),
        text_signature = "($self, tableName, path=None, source=None, schema=None, **options)"
    )]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    #[doc = connect_doc!(Catalog.createExternalTable)]
    fn createExternalTable<'py>(
        slf: &Bound<'py, Self>,
        tableName: &Bound<'py, PyAny>,
        path: Option<&Bound<'py, PyAny>>,
        source: Option<&Bound<'py, PyAny>>,
        schema: Option<&Bound<'py, PyAny>>,
        extra: &Bound<'py, pyo3::types::PyTuple>,
        options: Option<&Bound<'py, pyo3::types::PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        crate::options::check_keyword_keys(options)?;
        check_extra_positional("Catalog.createExternalTable", 4, extra)?;
        // Of the line of PySpark's `warnings.warn` in `pyspark.sql.connect.catalog`.
        crate::errors::warn_as_pyspark(
            py,
            "pyspark.sql.connect.catalog",
            236,
            "createExternalTable is deprecated since Spark 4.0, please use createTable instead.",
            &py.get_type::<pyo3::exceptions::PyFutureWarning>(),
        )?;
        // As PySpark's `self.createTable(tableName, path, source, schema, **options)`, a call of
        // the method of the object, of a subclass too, which binds a `description` of the options
        // to the parameter of createTable.
        slf.call_method(
            pyo3::intern!(py, "createTable"),
            (tableName, path, source, schema),
            options,
        )
    }

    // Register a Python function as a UDF, returning the registered UDF. Deprecated
    // in pyspark in favour of `spark.udf.register`; mirrors `Catalog.registerFunction`.
    #[pyo3(signature = (name, f, returnType=None))]
    #[allow(non_snake_case)]
    #[doc = connect_doc!(Catalog.registerFunction)]
    fn registerFunction<'a>(
        slf: &Bound<'a, Self>,
        name: &str,
        f: Bound<'a, PyAny>,
        returnType: Option<Bound<'a, PyAny>>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let py = slf.py();
        // Of the line of PySpark's `warnings.warn` in `pyspark.sql.connect.catalog`.
        crate::errors::warn_as_pyspark(
            py,
            "pyspark.sql.connect.catalog",
            403,
            "Deprecated in 2.3.0. Use spark.udf.register instead.",
            &py.get_type::<pyo3::exceptions::PyFutureWarning>(),
        )?;
        // As PySpark's `self._sparkSession.udf.register(name, f, returnType)`.
        crate::state::attr(slf.as_any(), pyo3::intern!(py, "_sparkSession"))?
            .getattr(pyo3::intern!(py, "udf"))?
            .call_method1(pyo3::intern!(py, "register"), (name, f, returnType))
    }

    // Create a database, with the `properties` of PySpark's protobuf map of strings, whose keys
    // and values are converted as `crate::str_argument` converts them. As PySpark's
    // `CreateDatabase`, which takes `properties or {}`, and then passes the database and
    // `ifNotExists` to the constructor of its message before it sets each of the
    // `properties.items()`.
    #[pyo3(
        signature = (dbName, ifNotExists=None, properties=None),
        text_signature = "($self, dbName, ifNotExists=False, properties=None)"
    )]
    #[allow(non_snake_case)]
    #[doc = connect_doc!(Catalog.createDatabase)]
    fn createDatabase(
        slf: &Bound<'_, Self>,
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
        Self::detached(slf, |catalog| {
            { catalog.create_database(&dbName, ifNotExists, props) }.map_err(Box::new)
        })
    }

    // Drop a database.
    #[pyo3(
        signature = (dbName, ifExists=None, cascade=None),
        text_signature = "($self, dbName, ifExists=False, cascade=False)"
    )]
    #[allow(non_snake_case)]
    #[doc = connect_doc!(Catalog.dropDatabase)]
    fn dropDatabase(
        slf: &Bound<'_, Self>,
        dbName: &Bound<'_, PyAny>,
        ifExists: Option<&Bound<'_, PyAny>>,
        cascade: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let dbName = constructor_name(dbName)?;
        let (ifExists, cascade) = (constructor_bool(ifExists)?, constructor_bool(cascade)?);
        Self::detached(slf, |catalog| {
            { catalog.drop_database(&dbName, ifExists, cascade) }.map_err(Box::new)
        })
    }

    // Drop a table.
    #[pyo3(
        signature = (tableName, ifExists=None, purge=None),
        text_signature = "($self, tableName, ifExists=False, purge=False)"
    )]
    #[allow(non_snake_case)]
    #[doc = connect_doc!(Catalog.dropTable)]
    fn dropTable(
        slf: &Bound<'_, Self>,
        tableName: &Bound<'_, PyAny>,
        ifExists: Option<&Bound<'_, PyAny>>,
        purge: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let tableName = constructor_name(tableName)?;
        let (ifExists, purge) = (constructor_bool(ifExists)?, constructor_bool(purge)?);
        Self::detached(slf, |catalog| {
            catalog
                .drop_table(&tableName, ifExists, purge)
                .map_err(Box::new)
        })
    }

    // Drop a view.
    #[pyo3(
        signature = (viewName, ifExists=None),
        text_signature = "($self, viewName, ifExists=False)"
    )]
    #[allow(non_snake_case)]
    #[doc = connect_doc!(Catalog.dropView)]
    fn dropView(
        slf: &Bound<'_, Self>,
        viewName: &Bound<'_, PyAny>,
        ifExists: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let viewName = constructor_name(viewName)?;
        let ifExists = constructor_bool(ifExists)?;
        Self::detached(slf, |catalog| {
            catalog.drop_view(&viewName, ifExists).map_err(Box::new)
        })
    }

    // Truncate a table.
    #[allow(non_snake_case)]
    #[doc = connect_doc!(Catalog.truncateTable)]
    fn truncateTable(slf: &Bound<'_, Self>, tableName: &Bound<'_, PyAny>) -> PyResult<()> {
        let tableName = crate::str_argument(tableName)?;
        Self::detached(slf, |catalog| {
            catalog.truncate_table(&tableName).map_err(Box::new)
        })
    }

    // Recover the statistics of a table, of `noScan` as PySpark's plan sets it in its bool field.
    #[pyo3(
        signature = (tableName, noScan=crate::streaming::ProtoBool::default()),
        text_signature = "($self, tableName, noScan=False)"
    )]
    #[allow(non_snake_case)]
    #[doc = connect_doc!(Catalog.analyzeTable)]
    fn analyzeTable(
        slf: &Bound<'_, Self>,
        tableName: &Bound<'_, PyAny>,
        noScan: crate::streaming::ProtoBool,
    ) -> PyResult<()> {
        let py = slf.py();
        let tableName = crate::str_argument(tableName)?;
        let noScan = noScan.value(py)?;
        Self::detached(slf, |catalog| {
            catalog.analyze_table(&tableName, noScan).map_err(Box::new)
        })
    }

    // Get the `CREATE TABLE` string of a table, of `asSerde` as PySpark's plan sets it in its
    // bool field.
    #[pyo3(
        signature = (tableName, asSerde=crate::streaming::ProtoBool::default()),
        text_signature = "($self, tableName, asSerde=False)"
    )]
    #[allow(non_snake_case)]
    #[doc = connect_doc!(Catalog.getCreateTableString)]
    fn getCreateTableString(
        slf: &Bound<'_, Self>,
        tableName: &Bound<'_, PyAny>,
        asSerde: crate::streaming::ProtoBool,
    ) -> PyResult<String> {
        let py = slf.py();
        let tableName = crate::str_argument(tableName)?;
        let asSerde = asSerde.value(py)?;
        Self::detached(slf, |catalog| {
            { catalog.get_create_table_string(&tableName, asSerde) }.map_err(Box::new)
        })
    }

    // Get the properties of a table as a dict.
    #[allow(non_snake_case)]
    #[doc = connect_doc!(Catalog.getTableProperties)]
    fn getTableProperties<'py>(
        slf: &Bound<'py, Self>,
        tableName: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let py = slf.py();
        let tableName = crate::str_argument(tableName)?;
        let props = Self::detached(slf, |catalog| {
            catalog.get_table_properties(&tableName).map_err(Box::new)
        })?;
        let dict = pyo3::types::PyDict::new(py);
        for (k, v) in props {
            dict.set_item(k, v)?;
        }
        Ok(dict)
    }

    // List the partitions of a table. Mirrors `Catalog.listPartitions -> List[TablePartition]`.
    #[allow(non_snake_case)]
    #[doc = connect_doc!(Catalog.listPartitions)]
    fn listPartitions(
        slf: &Bound<'_, Self>,
        tableName: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let py = slf.py();
        let tableName = crate::str_argument(tableName)?;
        named_tuples(
            py,
            Self::detached(slf, |catalog| {
                catalog.list_partitions_typed(&tableName).map_err(Box::new)
            })?,
        )
    }

    // List the views in a database. Mirrors `Catalog.listViews -> List[Table]`, which asks for
    // the current database of a pattern without a database, before its plan sets either.
    #[pyo3(signature = (dbName=None, pattern=None))]
    #[allow(non_snake_case)]
    #[doc = connect_doc!(Catalog.listViews)]
    fn listViews(
        slf: &Bound<'_, Self>,
        dbName: Option<&Bound<'_, PyAny>>,
        pattern: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let py = slf.py();
        let dbName = match (dbName, pattern) {
            // As PySpark's `self.currentDatabase()`, of the method of the object.
            (None, Some(_)) => Some(
                slf.call_method0(pyo3::intern!(py, "currentDatabase"))?
                    .extract::<String>()?,
            ),
            (dbName, _) => dbName.map(crate::str_argument).transpose()?,
        };
        let pattern = pattern.map(crate::str_argument).transpose()?;
        named_tuples(
            py,
            Self::detached(slf, |catalog| {
                { catalog.list_views_typed(dbName.as_deref(), pattern.as_deref()) }
                    .map_err(Box::new)
            })?,
        )
    }
}
