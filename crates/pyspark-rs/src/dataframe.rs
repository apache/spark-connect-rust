//! PyO3 wrapper for spark_connect::dataframe::DataFrame.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt, PyList, PyTuple};
use spark_connect::dataframe::{AsOfJoinOptions, DataFrame, LocalRowIterator, PreparedExecution};
use spark_connect::plan::{AsOfJoinDirection, AsOfJoinType, JoinType};
use spark_connect::udf::{eval_type, CommonInlineUserDefinedFunctionExpression, PythonUDFPayload};

use crate::column::PyColumn;
use crate::errors::{not_expected_type, ResultExt};
use crate::group::PyGroupedData;
use crate::interrupt::{interruptible, interruptible_of};
use crate::options::DefaultedOption;
use crate::row::{FieldConverters, NestedFieldNames};
use crate::stat_arguments::proto_int;
use crate::streaming::{proto_bool, ProtoBool, PyDataStreamWriter};
use crate::types::PyDataType;

// Python wrapper for a Spark DataFrame.
#[doc = connect_doc!(DataFrame)]
#[pyclass(
    name = "DataFrame",
    module = "pyspark.sql.connect.dataframe",
    subclass,
    weakref,
    dict
)]
pub struct PyDataFrame {
    pub(crate) dataframe: DataFrame,
    /// Whether `_repr_html_` was called, as PySpark's `_support_repr_html`: then `__repr__` is
    /// the string of the schema, so that a notebook does not show the DataFrame twice.
    support_repr_html: std::sync::atomic::AtomicBool,
    /// Set once the value is written. Python's garbage collector may traverse the object while it
    /// is created: the allocation already tracks it, and `object.__new__` allocates the instance
    /// dictionary, which can start a collection, before PyO3 writes the value. Until then the
    /// memory is zeroed, so this flag is false and `__traverse__` reads nothing else (see
    /// [`crate::gc::GcReady`]).
    gc_ready: crate::gc::GcReady,
    /// The reference of this DataFrame to the parts of its plan, which `__traverse__` reports, as
    /// PySpark's DataFrame holds its plan object (see [`crate::gc::Parts`]).
    parts: crate::gc::Parts,
    /// The answers that PySpark keeps for each DataFrame once it has them, `isStreaming` (a
    /// `cached_property`), `semanticHash`, `isEmpty` and `inputFiles` (`functools.cache`): later
    /// calls do not ask again, also after the session stopped. (`isLocal` is cached in the core.)
    is_streaming: std::sync::OnceLock<bool>,
    semantic_hash: std::sync::OnceLock<i32>,
    is_empty: std::sync::OnceLock<bool>,
    input_files: std::sync::OnceLock<Vec<String>>,
    /// The schema of a DataFrame of `to`, such as a DDL string or a DataType, or the `StructType`
    /// of a map function, which PySpark keeps as the `_cached_schema` of the DataFrame, as the
    /// object that it is: `schema` returns a copy of it as it is then, and `columns` its names
    /// then. A schema that is not a DataType fails the DataFrame at each action, as its plan does
    /// not build.
    schema_object: Option<Py<PyAny>>,
    /// The `ExecutionInfo` that `executionInfo` made, with the information of the execution of
    /// the core that it is of: PySpark keeps the one object of the last execution of a DataFrame
    /// (`_execution_info`), which each access returns until another execution replaces it.
    execution_info_object:
        std::sync::Mutex<Option<(spark_connect::session::ExecutionInfo, Py<PyAny>)>>,
    /// The `ExecutionInfo` of the last write of the DataFrame, which the callback of a writer of it
    /// set (`_set_execution_info`), with the information of the execution of the core when it was
    /// set: `executionInfo` gives it while no later execution of the DataFrame replaced that.
    written_execution_info:
        std::sync::Mutex<Option<(Option<spark_connect::session::ExecutionInfo>, Py<PyAny>)>>,
    /// The schema that the copy, the deep copy or the unpickle of which this DataFrame is made
    /// requested, as PySpark's `_cached_schema_serialized`: its reconstruction reads the columns
    /// of the new DataFrame, which requests the schema and keeps it, before the state sets the
    /// `_cached_schema` of the DataFrame that it is of. `schema` gives it, and so do
    /// `printSchema`, `toArrow`, the schema that a set operation keeps and the numeric
    /// aggregations of grouped data, which read `schema`; `columns`, `dtypes`, `__repr__`, the
    /// column lookups and the DataFrames of methods that copy the cached schema read the cached
    /// one, as PySpark's `_schema`. It is kept with a cached schema only: without one, PySpark's
    /// next read of `_schema` requests the schema and keeps it as both.
    copy_schema: Option<spark_connect::types::DataType>,
}

/// The answer kept in `cell`, else that of `ask`, which is kept unless it fails, as
/// `functools.cache` keeps the result of a call.
fn cached<T: Clone>(
    cell: &std::sync::OnceLock<T>,
    ask: impl FnOnce() -> PyResult<T>,
) -> PyResult<T> {
    if let Some(value) = cell.get() {
        return Ok(value.clone());
    }
    let value = ask()?;
    Ok(cell.get_or_init(|| value).clone())
}

/// The settings of the eager evaluation of `__repr__` and `_repr_html_`.
const EAGER_EVAL: [&str; 3] = [
    "spark.sql.repl.eagerEval.enabled",
    "spark.sql.repl.eagerEval.maxNumRows",
    "spark.sql.repl.eagerEval.truncate",
];

/// The relation of a DataFrame as its pickle carries it, the description of the error of a plan
/// that does not build instead, and the holes of the functions of its UDFs (see
/// `PyDataFrame::pickled_relation`).
type PickledRelation<'py> = (
    Bound<'py, pyo3::types::PyBytes>,
    Option<Bound<'py, PyTuple>>,
    Bound<'py, PyList>,
);

impl PyDataFrame {
    /// The names of the columns of the schema of this DataFrame, as Rust callers need them. Not
    /// a method of the Python class, as PySpark's DataFrame has none of this name: so
    /// `df.column_names` is the column of that name, of `__getattr__`.
    pub(crate) fn column_names(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        self.own_schema(py)?.columns(py)?.extract()
    }

    /// The schema of this DataFrame, as PySpark's `_cached_schema` is: the object of `to` that is
    /// not a DataType, which it keeps (see `schema_object`), or else the schema of the server, as
    /// PySpark's client converts it, of the errors of that conversion (see `check_python_schema`).
    /// The schema is cached, so it is asked of the server once.
    fn own_schema<'py>(&self, py: Python<'py>) -> PyResult<PropertySchema<'py>> {
        if let Some(schema) = &self.schema_object {
            return Ok(PropertySchema::Object(schema.bind(py).clone()));
        }
        let schema = interruptible_of(py, &self.dataframe.spark_session(), &[&self.parts], || {
            self.dataframe.schema()
        })?;
        check_python_schema(py, &schema)?;
        Ok(PropertySchema::Core(schema))
    }

    /// Raises as PySpark's `schema` of this DataFrame does, which its grouped data reads, of the
    /// conversion of the schema that it caches (see `own_schema`): of a copy or an unpickle,
    /// whose `schema` is the schema that it requested and converted then (see `copy_schema`),
    /// nothing.
    pub(crate) fn check_schema_read(&self, py: Python<'_>) -> PyResult<()> {
        if self.copy_schema.is_none() {
            self.own_schema(py)?;
        }
        Ok(())
    }

    /// The schema that the properties `schema`, `columns` and `dtypes` read, as PySpark's read
    /// `self._schema`: `own_schema`, or, when the plan fails to build with an `AttributeError`, as
    /// of a join on a tuple, the object that Python then gives of `self._schema`, as PySpark's
    /// property `_schema` raises it: of the `__getattr__` of the DataFrame (or of its Python
    /// subclass), the Column of `_schema`.
    fn property_schema<'py>(slf: &Bound<'py, Self>) -> PyResult<PropertySchema<'py>> {
        let py = slf.py();
        let schema = slf.borrow().own_schema(py);
        match schema {
            Err(err) if err.is_instance_of::<pyo3::exceptions::PyAttributeError>(py) => Ok(
                PropertySchema::Object(slf.getattr(pyo3::intern!(py, "_schema"))?),
            ),
            schema => schema,
        }
    }

    /// `other`, a DataFrame of the session of this one, as PySpark's `_check_same_session`
    /// takes the DataFrame of a method: else `SESSION_NOT_SAME`, as for what is not a
    /// DataFrame.
    pub(crate) fn same_session<'py>(
        &self,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<PyRef<'py, PyDataFrame>> {
        match other.extract::<PyRef<'py, PyDataFrame>>() {
            Ok(df)
                if df.dataframe.spark_session().session_id()
                    == self.dataframe.spark_session().session_id() =>
            {
                Ok(df)
            }
            _ => Err(crate::column::raise_pyspark(
                other.py(),
                "SessionNotSameException",
                "SESSION_NOT_SAME",
                &[],
            )),
        }
    }

    /// The pandas DataFrame of the result, as PySpark's `_to_pandas` makes it: the plan is built
    /// first (`to_proto`), so a plan that does not build raises before any request, and the
    /// first plan of the session reads the plan compression settings then; then
    /// `SparkConnectClient.to_pandas` reads the settings of the conversion, in one request,
    /// and sends the plan; the Arrow result with the schema that the server sends is converted
    /// by `pyspark.sql.connect.client.core._to_pandas`. The DataFrame keeps the information of the
    /// execution once the result is converted, as `_to_pandas` sets `_execution_info` after
    /// `SparkConnectClient.to_pandas` returns: a conversion that fails leaves `executionInfo` as
    /// it was.
    pub(crate) fn to_pandas(
        &self,
        py: Python<'_>,
        struct_handling_mode: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let session = self.dataframe.spark_session();
        let execution = interruptible_of(py, &session, &[&self.parts], || {
            let execution = self.dataframe.prepare_execution().map_err(Box::new)?;
            // As PySpark's `to_pandas`, which logs the plan before it reads its settings.
            execution.log_plan_of_conversion().map_err(Box::new)?;
            Ok::<_, Box<spark_connect_core::error::SparkError>>(execution)
        })?;
        let settings = crate::interrupt::interruptible_without_parts(py, &session, || {
            session.conf().get_many(&[
                "spark.sql.session.timeZone",
                "spark.sql.execution.pandas.structHandlingMode",
                "spark.sql.execution.arrow.pyspark.selfDestruct.enabled",
            ])
        })?;
        let (batches, schema, info) = interruptible_of(py, &session, &[&self.parts], || {
            execution
                .collect_record_batches_and_info(check_schema_at_receipt)
                .map_err(Box::new)
        })?;
        check_schema_conversion(py, schema.as_ref())?;
        // The conversion of pyarrow and pandas, of calls of Python of many frames, on a stack with
        // room for them, of a thread of a small stack too.
        crate::on_grown_stack(|| {
            self.to_pandas_of(py, batches, schema, info, settings, struct_handling_mode)
        })
    }

    /// The pandas DataFrame of the batches of the result of [`Self::to_pandas`], of its settings.
    fn to_pandas_of(
        &self,
        py: Python<'_>,
        batches: Vec<arrow::record_batch::RecordBatch>,
        schema: Option<spark_connect::types::DataType>,
        info: spark_connect::session::ExecutionInfo,
        settings: Vec<Option<String>>,
        struct_handling_mode: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let mut table = crate::pyarrow_export::record_batches_to_pyarrow(py, batches)?;
        // As SparkConnectClient._execute_and_fetch with self_destruct, which copies each column
        // of each batch into its own allocation before the conversion, with its errors.
        if settings[2].as_deref() == Some("true") {
            table = py
                .import("pyspark.sql.connect.client.core")?
                .call_method1("_self_destruct_table", (table,))?
                .unbind();
        }
        let schema = match &schema {
            Some(schema) => crate::types::schema_to_py(py, schema)?,
            None => py.None(),
        };
        let (metrics, observed) = execution_metrics(py, &info)?;
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("timezone", &settings[0])?;
        match struct_handling_mode {
            Some(mode) => kwargs.set_item("struct_handling_mode", mode.str()?)?,
            None => kwargs.set_item("struct_handling_mode", &settings[1])?,
        }
        kwargs.set_item("self_destruct", &settings[2])?;
        let pdf = py.import("pyspark.sql.connect.client.core")?.call_method(
            "_to_pandas",
            (table, schema, metrics, observed),
            Some(&kwargs),
        )?;
        self.dataframe.keep_execution_info(info);
        Ok(pdf.unbind())
    }

    /// The table of `show` and `_show_string`, for the arguments of `function`.
    fn show_table(
        &self,
        py: Python<'_>,
        function: &str,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<String> {
        crate::options::check_keyword_keys(kwargs)?;
        let [n, truncate, vertical] = bind_arguments(
            function,
            args,
            kwargs,
            ["n", "truncate", "vertical"],
            [
                20i32.into_pyobject(py)?.into_any(),
                pyo3::types::PyBool::new(py, true).to_owned().into_any(),
                pyo3::types::PyBool::new(py, false).to_owned().into_any(),
            ],
        )?;
        if n.is_instance_of::<pyo3::types::PyBool>() || !n.is_instance_of::<pyo3::types::PyInt>() {
            return Err(not_expected_type(&n, "n", "int"));
        }
        if !vertical.is_instance_of::<pyo3::types::PyBool>() {
            return Err(not_expected_type(&vertical, "vertical", "bool"));
        }
        let truncate =
            if truncate.is_instance_of::<pyo3::types::PyBool>() && truncate.is_truthy()? {
                20i32.into_pyobject(py)?.into_any()
            } else {
                // As `int(truncate)`, whose ValueError is PySpark's NOT_EXPECTED_TYPE.
                match py.import("builtins")?.getattr("int")?.call1((&truncate,)) {
                    Ok(value) => value,
                    Err(err) if err.is_instance_of::<pyo3::exceptions::PyValueError>(py) => {
                        return Err(not_expected_type(&truncate, "truncate", "bool"));
                    }
                    Err(err) => return Err(err),
                }
            };
        // The proto's int32 fields, as PySpark's `ShowString` sets them: `num_rows`, then
        // `truncate`.
        let n = proto_int::<i32>(&n, "spark.connect.ShowString.num_rows")?;
        let truncate = proto_int::<i32>(&truncate, "spark.connect.ShowString.truncate")?;
        let vertical = vertical.extract::<bool>()?;
        interruptible_of(py, &self.dataframe.spark_session(), &[&self.parts], || {
            self.dataframe
                .show_string(n, truncate, vertical)
                .map_err(Box::new)
        })
    }

    pub fn new(dataframe: DataFrame) -> Self {
        PyDataFrame {
            parts: Python::attach(|py| crate::gc::Parts::of(py, dataframe.held_parts())),
            dataframe,
            support_repr_html: std::sync::atomic::AtomicBool::new(false),
            gc_ready: crate::gc::GcReady::new(),
            is_streaming: std::sync::OnceLock::new(),
            semantic_hash: std::sync::OnceLock::new(),
            is_empty: std::sync::OnceLock::new(),
            input_files: std::sync::OnceLock::new(),
            schema_object: None,
            execution_info_object: std::sync::Mutex::new(None),
            written_execution_info: std::sync::Mutex::new(None),
            copy_schema: None,
        }
    }

    /// The schema that the copy or the unpickle of which this DataFrame is made requested, which
    /// PySpark's `schema` gives (see `copy_schema`): the one place that reads it, for
    /// `schema`, `printSchema`, the merge of a set operation and the numeric aggregations of
    /// grouped data.
    pub(crate) fn copy_schema(&self) -> Option<spark_connect::types::DataType> {
        self.copy_schema.clone()
    }

    /// Keep `requested`, the schema that the copy or the unpickle of which this DataFrame is made
    /// requested, if this DataFrame has a cached schema, as PySpark's `_cached_schema_serialized`
    /// is read only then (see `copy_schema`).
    fn keep_copy_schema(&mut self, requested: spark_connect::types::DataType) {
        if self.schema_object.is_some() || self.dataframe.known_schema().is_some() {
            self.copy_schema = Some(requested);
        }
    }

    /// The relation of this DataFrame as its pickle carries it, the error of a plan that does not
    /// build instead, and the holes of the functions of its UDFs: the relation is built of holes
    /// in place of their commands, as PySpark's pickle carries the functions of its plan objects
    /// ([`crate::udf_command::PickleHoles`]); a plan that does not build, of a value that PySpark
    /// converts when it builds the plan, pickles of its error ([`crate::column::described_error`]),
    /// and a plan of the Python client whose build raised, of that plan, which the unpickle builds
    /// again, as PySpark's does.
    fn pickled_relation<'py>(&self, py: Python<'py>) -> PyResult<PickledRelation<'py>> {
        use pyo3::types::PyBytes;
        let (holes, copies) = crate::udf_command::PickleHoles::of(py, self.dataframe.picklers());
        let built = if copies.is_empty() {
            crate::origin::pickling(|| self.dataframe.to_relation())
        } else {
            let marked = self.dataframe.with_pickler_copies(&copies);
            crate::origin::pickling(|| marked.to_relation())
        };
        Ok(match built {
            Ok(relation) => {
                let relation = crate::encode_on_grown_stack(relation);
                let holes = holes.holes_of(py, &relation)?;
                (PyBytes::new(py, &relation), None, holes)
            }
            // Of the DataFrame of the plan of a kept error itself, or of one over it.
            Err(error) => match crate::column::described_error_of(
                py,
                &error,
                !self.dataframe.is_unbuilt_with(&error),
            )? {
                Some(description) => (PyBytes::new(py, b""), Some(description), PyList::empty(py)),
                None => return Err(crate::errors::spark_error_to_py_exception(error)),
            },
        })
    }

    /// Whether this DataFrame is streaming, of the answer that `isStreaming` keeps, asked once,
    /// as PySpark's `functools.cached_property` of it, which its `Observation._on` reads too. It
    /// waits for the server without the GIL.
    pub(crate) fn streaming_once(&self) -> spark_connect::Result<bool> {
        if let Some(streaming) = self.is_streaming.get() {
            return Ok(*streaming);
        }
        let streaming = self.dataframe.is_streaming()?;
        Ok(*self.is_streaming.get_or_init(|| streaming))
    }

    /// `result`, of a plan over this DataFrame of the same schema, with the schema of `to` that
    /// this one keeps apart from the core, as PySpark's `alias`, `filter`, `limit`, `sort`, the
    /// repartitions and the other methods that copy the `_cached_schema` of their input copy it
    /// whatever it is. The core keeps a cached struct itself.
    pub(crate) fn of_same_schema(&self, mut result: PyDataFrame) -> PyDataFrame {
        if let Some(schema) = &self.schema_object {
            result.schema_object = Some(Python::attach(|py| schema.clone_ref(py)));
        }
        result
    }

    /// The DataFrame of `dataframe`, a copy of this one, with the `_support_repr_html` of this
    /// one, as the state of PySpark's DataFrame carries it to its copies. Its other answers are
    /// its own, as PySpark keeps them for each DataFrame.
    fn copy_of(&self, dataframe: DataFrame) -> PyDataFrame {
        let copy = PyDataFrame::new(dataframe);
        copy.support_repr_html.store(
            self.support_repr_html
                .load(std::sync::atomic::Ordering::SeqCst),
            std::sync::atomic::Ordering::SeqCst,
        );
        copy
    }

    /// The schema request that PySpark's `copy` of a DataFrame makes for the new DataFrame of its
    /// reconstruction, `hasattr(copy, "__setstate__")`, which reads the columns of a DataFrame that
    /// has no schema yet, even when this one has one: the schema that it gives, which the copy
    /// keeps (see `copy_schema`). The request is that of a DataFrame of the same relation,
    /// which has no schema cached, so that it fails as PySpark's does, of the error of its plan
    /// as the read of the columns raises it ([`columns_read_error`]).
    fn fresh_schema_request(&self, py: Python<'_>) -> PyResult<spark_connect::types::DataType> {
        let session = self.dataframe.spark_session();
        let relation =
            crate::errors::of_request_session(&session, || self.dataframe.to_relation().to_pyerr())
                .map_err(|err| columns_read_error(py, err))?;
        let fresh = DataFrame::from_sent_relation(session.clone(), relation);
        // Of the plan of the new DataFrame, of the relation as it was sent.
        let schema =
            crate::interrupt::interruptible_of_values(py, &session, fresh.held_parts(), || {
                fresh.schema()
            })?;
        // As the read of the columns, which converts the schema (see `check_python_schema`).
        check_python_schema(py, &schema)?;
        Ok(schema)
    }

    /// The number of rows and the truncation of the eager evaluation, as PySpark's `int` of
    /// each setting, if it is enabled.
    fn eager_eval(&self, py: Python<'_>) -> PyResult<Option<(i32, i32)>> {
        let session = self.dataframe.spark_session();
        let values = crate::interrupt::interruptible_without_parts(py, &session, || {
            session.conf().get_many(&EAGER_EVAL)
        })?;
        if values[0].as_deref() != Some("true") {
            return Ok(None);
        }
        let int = py.import("builtins")?.getattr("int")?;
        let n: i32 = int.call1((values[1].as_deref(),))?.extract()?;
        let truncate: i32 = int.call1((values[2].as_deref(),))?.extract()?;
        Ok(Some((n, truncate)))
    }

    /// The column at the 0-based index `index`, as PySpark's `df[index]`: `col` of the name of
    /// the column at that index of `columns`, of Python's list indexing, and of the origin of
    /// `col`.
    fn column_at(&self, py: Python<'_>, index: i64) -> PyResult<spark_connect::column::Column> {
        let columns = self.column_names(py)?;
        let name: String = PyList::new(py, columns)?
            .as_any()
            .get_item(index)?
            .extract()?;
        let _origin = crate::origin::with_origin("col")?;
        Ok(spark_connect::functions::col(&name))
    }

    /// The grouping columns of `groupBy`, `rollup`, `cube` and `groupingSets`, as PySpark's
    /// `_to_cols` of ordinals: Columns, column names and the 1-based ordinals of columns, or
    /// a list or tuple of them.
    fn grouping_columns(
        &self,
        py: Python<'_>,
        cols: Vec<Bound<'_, PyAny>>,
        arg_name: &str,
    ) -> PyResult<Vec<spark_connect::column::Column>> {
        self.grouping_column_list(py, unwrap_sequence(cols)?, arg_name)
    }

    /// The grouping columns of `cols`, each a Column, a column name or the 1-based ordinal of
    /// a column, as PySpark's `_to_cols` of ordinals takes the items of a sequence.
    fn grouping_column_list(
        &self,
        py: Python<'_>,
        cols: Vec<Bound<'_, PyAny>>,
        arg_name: &str,
    ) -> PyResult<Vec<spark_connect::column::Column>> {
        use pyo3::types::{PyBool, PyString};
        cols.iter()
            .map(|c| {
                if let Ok(column) = c.extract::<PyRef<'_, PyColumn>>() {
                    return Ok(column.column.clone());
                }
                if c.is_instance_of::<PyString>() {
                    // As PySpark's `F._to_col`, by `col`, of its origin.
                    let _origin = crate::origin::with_origin("col")?;
                    return crate::plan_args::col(c);
                }
                if c.is_instance_of::<PyInt>() && !c.is_instance_of::<PyBool>() {
                    let ordinal: i64 = c.extract()?;
                    if ordinal < 1 {
                        return Err(crate::column::raise_pyspark(
                            py,
                            "PySparkIndexError",
                            "INDEX_NOT_POSITIVE",
                            &[("index", &ordinal.to_string())],
                        ));
                    }
                    return self.column_at(py, ordinal - 1);
                }
                Err(not_expected_type(c, arg_name, "Column, int or str"))
            })
            .collect()
    }

    /// The sort orders of `sort`, `orderBy` and `sortWithinPartitions`, as PySpark's
    /// `_preapare_cols_for_sort`: Columns, column names and the 1-based ordinals of columns,
    /// descending for a negative one, or a list or tuple of them, each descending for a
    /// false `ascending`, or for a false one of a list of them.
    fn sort_columns(
        &self,
        py: Python<'_>,
        cols: Vec<Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<spark_connect::expression::Expression>> {
        use pyo3::types::{PyBool, PyString};
        crate::options::check_keyword_keys(kwargs)?;
        if cols.is_empty() {
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkValueError",
                "CANNOT_BE_EMPTY",
                &[("item", "cols")],
            ));
        }
        let mut columns = Vec::new();
        for c in unwrap_sequence(cols)? {
            let column = if c.is_instance_of::<PyInt>() && !c.is_instance_of::<PyBool>() {
                let ordinal: i64 = c.extract()?;
                match ordinal {
                    0 => {
                        return Err(crate::column::raise_pyspark(
                            py,
                            "PySparkIndexError",
                            "ZERO_INDEX",
                            &[],
                        ));
                    }
                    ordinal if ordinal > 0 => self.column_at(py, ordinal - 1)?,
                    ordinal => self.column_at(py, -ordinal - 1)?.desc(),
                }
            } else if let Ok(column) = c.extract::<PyRef<'_, PyColumn>>() {
                column.column.clone()
            } else if c.is_instance_of::<PyString>() {
                // As PySpark's `F._to_col`, by `col`, of its origin.
                let _origin = crate::origin::with_origin("col")?;
                crate::plan_args::col(&c)?
            } else {
                return Err(not_expected_type(&c, "col", "Column, int or str"));
            };
            columns.push(column);
        }
        let ascending = match kwargs {
            Some(kwargs) => kwargs.get_item("ascending")?,
            None => None,
        };
        if let Some(ascending) = ascending {
            if ascending.is_instance_of::<PyInt>() {
                if !ascending.is_truthy()? {
                    columns = columns.into_iter().map(|c| c.desc()).collect();
                }
            } else if ascending.is_instance_of::<PyList>() {
                // As PySpark's zip of them, which keeps a column for each flag.
                let flags: Vec<bool> = ascending
                    .try_iter()?
                    .map(|flag| flag?.is_truthy())
                    .collect::<PyResult<_>>()?;
                columns = columns
                    .into_iter()
                    .zip(flags)
                    .map(|(c, asc)| if asc { c } else { c.desc() })
                    .collect();
            } else {
                return Err(not_expected_type(
                    &ascending,
                    "ascending",
                    "Column, int or str",
                ));
            }
        }
        Ok(columns.iter().map(|c| c.expression().clone()).collect())
    }
}

/// The arguments of a method of `*cols`, the items of a single sequence of them, as PySpark's
/// `_to_cols` takes them: a sequence that is not an int, a str or a Column, which is then
/// `tuple(cols[0])`. Any `collections.abc.Sequence` counts, as a range, bytes or a user class.
fn unwrap_sequence(cols: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<Bound<'_, PyAny>>> {
    if let [only] = cols.as_slice() {
        if is_sequence_argument(only)? {
            return only.try_iter()?.collect();
        }
    }
    Ok(cols)
}

/// Whether `only`, the single argument of `*cols`, is a sequence that PySpark unpacks: an
/// `isinstance` of `collections.abc.Sequence`, except for an int, a str and a Column.
fn is_sequence_argument(only: &Bound<'_, PyAny>) -> PyResult<bool> {
    use pyo3::types::{PyInt, PyString, PyTuple};
    if only.is_instance_of::<PyInt>()
        || only.is_instance_of::<PyString>()
        || only.is_instance_of::<PyColumn>()
    {
        return Ok(false);
    }
    if only.is_instance_of::<PyList>() || only.is_instance_of::<PyTuple>() {
        return Ok(true);
    }
    let sequence = only.py().import("collections.abc")?.getattr("Sequence")?;
    only.is_instance(&sequence)
}

/// PySpark's check of the `numPartitions` of `repartition`, `repartitionByRange` and `coalesce`,
/// `not numPartitions > 0`: `VALUE_NOT_POSITIVE`, or the `TypeError` of the comparison, as for
/// None in `coalesce`. The count is converted after, as the proto's int32 ([`proto_int`]).
fn positive_partitions(num_partitions: &Bound<'_, PyAny>) -> PyResult<()> {
    if num_partitions.gt(0)? {
        return Ok(());
    }
    Err(not_positive_partitions(num_partitions)?)
}

/// PySpark's `VALUE_NOT_POSITIVE` error of `numPartitions`, of its `str`.
fn not_positive_partitions(num_partitions: &Bound<'_, PyAny>) -> PyResult<PyErr> {
    Ok(crate::column::raise_pyspark(
        num_partitions.py(),
        "PySparkValueError",
        "VALUE_NOT_POSITIVE",
        &[
            ("arg_name", "numPartitions"),
            ("arg_value", &num_partitions.str()?.to_cow()?),
        ],
    ))
}

/// The `subset` of `dropDuplicates` and `dropDuplicatesWithinWatermark`, as PySpark's: None,
/// or a list or tuple of column names, which its `Deduplicate` plan sets when it builds the plan
/// (`args`).
fn deduplicate_subset(
    args: &mut crate::plan_args::PlanArgs,
    subset: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Vec<String>>> {
    let Some(subset) = subset else {
        return Ok(None);
    };
    if !(subset.is_instance_of::<PyList>() || subset.is_instance_of::<PyTuple>()) {
        return Err(not_expected_type(subset, "subset", "list or tuple"));
    }
    subset
        .try_iter()?
        .map(|column| {
            let column = column?;
            if column.is_instance_of::<pyo3::types::PyString>() {
                args.str(&column)
            } else {
                Err(not_expected_type(&column, "subset", "str"))
            }
        })
        .collect::<PyResult<Vec<String>>>()
        .map(Some)
}

/// Helper to convert arguments to a vector of Columns.
pub(crate) fn to_column_list(
    args: Vec<Bound<'_, PyAny>>,
) -> PyResult<Vec<spark_connect::column::Column>> {
    // pyspark unpacks a single sequence argument: df.select(["a", "b"]) == df.select("a","b").
    // (A single Column/str stays as-is.)
    to_columns_as_given(unwrap_sequence(args)?)
}

/// The columns of `args`, each a Column or a column name, else `NOT_EXPECTED_TYPE`, without
/// unpacking a single list: as PySpark's `_to_cols(cols)` of the tuple of `*cols` itself, which
/// `repartition` and `repartitionByRange` call, so that a list among their columns is an error.
pub(crate) fn to_columns_as_given(
    args: Vec<Bound<'_, PyAny>>,
) -> PyResult<Vec<spark_connect::column::Column>> {
    let mut cols = vec![];
    for arg in args {
        // Try as PyColumn first
        if let Ok(pycol) = arg.extract::<Bound<'_, PyColumn>>() {
            cols.push(pycol.borrow().column.clone());
        } else if arg.is_instance_of::<pyo3::types::PyString>() {
            // A column name, as PySpark's `F._to_col`, by `col`, of its origin.
            let _origin = crate::origin::with_origin("col")?;
            cols.push(crate::plan_args::col(&arg)?);
        } else {
            // As PySpark's `_to_cols`.
            return Err(not_expected_type(&arg, "cols", "Column or str"));
        }
    }
    Ok(cols)
}

/// The join types that `DataFrame.join` accepts, as the message of its `UNSUPPORTED_JOIN_TYPE`.
const JOIN_TYPES_SUPPORTED: &str = "'inner', 'outer', 'full', 'fullouter', 'full_outer', 'leftouter', 'left', 'left_outer', 'rightouter', 'right', 'right_outer', 'leftsemi', 'left_semi', 'semi', 'leftanti', 'left_anti', 'anti', 'cross'";

/// The join types that `DataFrame.lateralJoin` accepts, as the message of its
/// `UNSUPPORTED_JOIN_TYPE`.
const LATERAL_JOIN_TYPES_SUPPORTED: &str = "'inner', 'leftouter', 'left', 'left_outer', 'cross'";

/// Map a pyspark `how` to a JoinType, as the connect client's `plan.Join` (or `plan.LateralJoin`
/// when `lateral`) maps it. The `how` is lowercased with its underscores removed, and no `how`
/// is an inner join. One that is not accepted raises the `AnalysisException` of
/// `UNSUPPORTED_JOIN_TYPE` that PySpark raises locally, naming the normalized `how` (or the
/// `str()` of a `how` that is not a str).
fn parse_join_type(
    py: Python<'_>,
    how: Option<&Bound<'_, PyAny>>,
    lateral: bool,
) -> PyResult<JoinType> {
    let Some(how) = how.filter(|how| !how.is_none()) else {
        return Ok(JoinType::Inner);
    };
    // As PySpark's `how.lower().replace("_", "")` of a str, in Python: of a str that UTF-8
    // cannot encode too, which is no join type, and which the error names as it is.
    let normalized = if how.is_instance_of::<pyo3::types::PyString>() {
        Some(
            how.call_method0("lower")?
                .call_method1("replace", ("_", ""))?,
        )
    } else {
        None
    };
    let text = normalized.as_ref().and_then(|normalized| {
        normalized
            .cast::<pyo3::types::PyString>()
            .ok()?
            .to_cow()
            .ok()
    });
    let join_type = match text.as_deref() {
        Some("inner") => Some(JoinType::Inner),
        Some("cross") => Some(JoinType::Cross),
        Some("left" | "leftouter") => Some(JoinType::LeftOuter),
        Some("right" | "rightouter") if !lateral => Some(JoinType::RightOuter),
        Some("outer" | "full" | "fullouter") if !lateral => Some(JoinType::FullOuter),
        Some("leftsemi" | "semi") if !lateral => Some(JoinType::LeftSemi),
        Some("leftanti" | "anti") if !lateral => Some(JoinType::LeftAnti),
        _ => None,
    };
    if let Some(join_type) = join_type {
        return Ok(join_type);
    }
    let typ = normalized.unwrap_or_else(|| how.clone());
    let supported = if lateral {
        LATERAL_JOIN_TYPES_SUPPORTED
    } else {
        JOIN_TYPES_SUPPORTED
    };
    Err(crate::errors::raise_pyspark_of(
        py,
        "AnalysisException",
        "UNSUPPORTED_JOIN_TYPE",
        &[
            ("typ", &typ),
            (
                "supported",
                pyo3::types::PyString::new(py, supported).as_any(),
            ),
        ],
    ))
}

/// The `AttributeError` that PySpark's join raises for an `on` that is neither a name, a
/// condition nor a list of them: the plan reads `to_plan` of it.
fn no_to_plan(obj: &Bound<'_, PyAny>) -> PyErr {
    let typ = obj
        .get_type()
        .name()
        .map(|name| name.to_string())
        .unwrap_or_default();
    pyo3::exceptions::PyAttributeError::new_err(format!(
        "'{typ}' object has no attribute 'to_plan'"
    ))
}

/// [`no_to_plan`] of `obj` as the error of the core, which the requests of a plan raise, as
/// PySpark's `Join` plan reads `to_plan` of `on` when it builds the plan.
fn no_to_plan_error(obj: &Bound<'_, PyAny>) -> PyResult<spark_connect_core::error::SparkError> {
    let typ = obj.get_type().name()?.to_string();
    Ok(spark_connect_core::error::SparkError::builtin_attribute(
        format!("'{typ}' object has no attribute 'to_plan'"),
    ))
}

/// The join condition of `_joinAsOf` for its `on`, as PySpark's `AsOfJoin` plan sends it: a
/// name or a list of names (`using_columns`), or a Column or a list of Columns, which are
/// joined on their conjunction (`join_expr`). The plan builds them when it builds the plan
/// (`args`), as `DataFrame.join` does its condition: the names in the repeated string field as
/// protobuf converts them, the conjunction of a list as `functools.reduce` of `&`, and the
/// `to_plan` of an `on` of another type, which only a Column has.
fn as_of_join_condition(
    args: &mut crate::plan_args::PlanArgs,
    on: &Bound<'_, PyAny>,
) -> PyResult<(Option<spark_connect::column::Column>, Vec<String>)> {
    if let Ok(column) = on.extract::<PyRef<PyColumn>>() {
        return Ok((Some(column.column.clone()), Vec::new()));
    }
    if on.is_instance_of::<pyo3::types::PyString>() {
        return Ok((None, vec![args.str(on)?]));
    }
    // As PySpark's `_collect_references`, which iterates `on` first, and then its `to_plan`.
    if !on.is_instance_of::<PyList>() {
        on.try_iter()?;
        args.record(no_to_plan_error(on)?);
        return Ok((None, Vec::new()));
    }
    let list = on.cast::<PyList>()?;
    // An empty list is no condition at all.
    let Some(first) = list.iter().next() else {
        return Ok((None, Vec::new()));
    };
    if first.is_instance_of::<pyo3::types::PyString>() {
        return Ok((None, args.strs(list.iter())?));
    }
    Ok((list_conjunction(args, list)?, Vec::new()))
}

/// The conjunction of the conditions of `list`, the `on` of a join or an as-of join whose first
/// item is not a name, as PySpark's `Join` and `AsOfJoin` plans make it when they are built,
/// `functools.reduce(lambda c1, c2: c1 & c2, on)`: of Columns, the [`plan_conjunction`] of them,
/// of a Column and an item that the `&` of a Column takes as a literal, the `and` of it, and of
/// an item that is not a Column, its own `&`. The error of an `&`, such as the `AttributeError`
/// of the `_expr` of an object that is neither a Column nor a literal, or of the `to_plan` of a
/// conjunction that is not a Column, fails the requests of the plan (`args`).
fn list_conjunction(
    args: &mut crate::plan_args::PlanArgs,
    list: &Bound<'_, PyList>,
) -> PyResult<Option<spark_connect::column::Column>> {
    let py = list.py();
    let mut items = list.iter();
    let Some(mut reduced) = items.next() else {
        return Ok(None);
    };
    let mut column = reduced
        .extract::<PyRef<PyColumn>>()
        .ok()
        .map(|column| column.column.clone());
    for item in items {
        match column.take() {
            Some(left) => {
                let right = match item.extract::<PyRef<PyColumn>>() {
                    Ok(right) => right.column.clone(),
                    Err(_) => match crate::column::operand_column(&item) {
                        Ok(right) => right,
                        Err(err) => {
                            args.raised(py, err)?;
                            return Ok(None);
                        }
                    },
                };
                column = Some(plan_conjunction(left, right));
            }
            None => match reduced.bitand(&item) {
                Ok(result) => {
                    column = result
                        .extract::<PyRef<PyColumn>>()
                        .ok()
                        .map(|column| column.column.clone());
                    reduced = result;
                }
                Err(err) => {
                    args.raised(py, err)?;
                    return Ok(None);
                }
            },
        }
    }
    if column.is_none() {
        args.record(no_to_plan_error(&reduced)?);
    }
    Ok(column)
}

/// The conjunction `left & right` of the conditions of a join of a list of Columns, as PySpark's
/// `Join` and `AsOfJoin` plans make it when they are built, `functools.reduce(lambda c1, c2:
/// c1 & c2, on)`: so the `and` is of the origin of `__and__` at the call site of each request.
fn plan_conjunction(
    left: spark_connect::column::Column,
    right: spark_connect::column::Column,
) -> spark_connect::column::Column {
    let and = left.and(right).expression().clone();
    spark_connect::column::Column::new(
        spark_connect::expression::Expression::with_plan_build_origin(and, "__and__"),
    )
}

/// The as-of column of `_joinAsOf` of `df`: a Column as it is, or the column of `df` that
/// has this name, as PySpark's `df._col(name)`, whose `ColumnReference` asserts that the name
/// is a str: any other object is a bare `AssertionError`.
fn as_of_column(
    df: &DataFrame,
    column: &Bound<'_, PyAny>,
) -> PyResult<spark_connect::column::Column> {
    if let Ok(column) = column.extract::<PyRef<PyColumn>>() {
        return Ok(column.column.clone());
    }
    crate::plan_args::df_col(df, column, false)
}

/// Resolve a filter/where condition: a Column, or a SQL string parsed via `expr`, as PySpark's
/// `filter`, whose `Filter` plan collects the subqueries of anything else by its `foreach`: an
/// object without one raises the `AttributeError` of it.
fn cond_column(condition: &Bound<'_, PyAny>) -> PyResult<spark_connect::column::Column> {
    if let Ok(c) = condition.extract::<PyRef<PyColumn>>() {
        return Ok(c.column.clone());
    }
    if condition.is_instance_of::<pyo3::types::PyString>() {
        return crate::plan_args::expr(condition);
    }
    condition.getattr("foreach")?;
    Err(no_to_plan(condition))
}

/// An argument of `withWatermark`, as PySpark's: `NOT_EXPECTED_TYPE` for an empty one, or one
/// whose type is not str. Its `WithWatermark` plan sets it when it builds the plan (`args`).
fn watermark_argument(
    args: &mut crate::plan_args::PlanArgs,
    value: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<String> {
    if !value.is_truthy()?
        || !value
            .get_type()
            .is(value.py().get_type::<pyo3::types::PyString>())
    {
        return Err(not_expected_type(value, name, "str"));
    }
    args.str(value)
}

#[pymethods]
impl PyDataFrame {
    /// A DataFrame of `plan`, a plan of the Python client (`pyspark.sql.connect.plan`), such as
    /// the plans of `pyspark.ml`, as PySpark Connect's `DataFrame(plan, session)`: of the
    /// relation of `plan.plan(session.client)`, which the client sends as it is, of the
    /// `_plan_id` of the plan, as PySpark's DataFrame keeps its plan and so its plan ID, which its
    /// columns refer to. The relation is the root of the `to_proto` of the plan, as PySpark's
    /// requests build it. A plan whose build raises makes a DataFrame whose requests raise the
    /// exception of the build, as PySpark's DataFrame keeps the plan and its requests build it
    /// (see `errors::kept_error`); no plan (`None`) is PySpark's `MISSING_VALID_PLAN`. The plan
    /// of a DataFrame of the core (`DataFrame._plan`) gives a DataFrame of the plan of that
    /// DataFrame, as PySpark's is of the plan object of the DataFrame (`DataFrame::of_plan_in`):
    /// its requests build the plan, and so pickle the functions of its UDFs and raise the errors
    /// of its build, at each request.
    #[new]
    fn py_new(
        plan: &Bound<'_, PyAny>,
        session: &Bound<'_, crate::session::PySparkSession>,
    ) -> PyResult<Self> {
        let py = plan.py();
        if plan.is_none() {
            return Err(crate::errors::raise_pyspark(
                py,
                "PySparkRuntimeError",
                "MISSING_VALID_PLAN",
                &[("operator", "__init__")],
            ));
        }
        if let Some(of) = dataframe_of_plan(plan)? {
            let dataframe = of
                .borrow()
                .dataframe
                .of_plan_in(session.borrow().session.clone());
            return Ok(PyDataFrame::new(dataframe));
        }
        let (dataframe, _) = dataframe_of_python_plan(plan, session)?;
        Ok(PyDataFrame::new(dataframe))
    }

    /// The DataFrame of `plan`, a plan of the Python client that a pickled DataFrame of it carried
    /// in place of the relation that its build did not give, as PySpark's pickle carries its plan
    /// objects: its build again, as PySpark's unpickle builds the plan, of its error at each
    /// request if it raises again (see [`dataframe_of_python_plan`]). Of a DataFrame over the
    /// plan (`derived`), whose relation the pickle could not carry, one that builds now gives a
    /// DataFrame whose requests raise the `RuntimeError` of [`derived_plan_error`], rather than
    /// the DataFrame of the plan alone. Not a stable API.
    #[staticmethod]
    fn _of_python_plan(
        plan: &Bound<'_, PyAny>,
        session: &Bound<'_, crate::session::PySparkSession>,
        derived: bool,
    ) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(
            rebuilt_python_plan(plan, session, derived)?.0,
        ))
    }

    /// The plan ID of the plan of this DataFrame, by which its columns refer to it, and that of its
    /// relation within the `WithRelations` that sends it with the plans of its subqueries, if it
    /// has subqueries, for the `_plan_id` and `_root_plan_id` of `DataFrame._plan`, as those of
    /// PySpark's plan of a DataFrame.
    fn _plan_ids(&self) -> (i64, Option<i64>) {
        self.dataframe.plan_id_and_root()
    }

    /// As a DataFrame of PySpark Connect, a Python object whose plan Python's garbage collector
    /// sees: the parts of its plan are reported (see [`crate::gc`]), of the plans of the Python
    /// client, such as of `pyspark.ml`, that `DataFrame(plan, session)` keeps, also through the
    /// subqueries of its expressions, of the plans that the errors of its expressions keep, of the
    /// functions of its UDFs, which a request pickles when it builds the plan, as the plan of
    /// PySpark keeps them, and of the Python Observations of its `observe`, which its plan holds,
    /// as PySpark's plan holds them. Each is reported through every object that holds it, as
    /// PySpark's DataFrames, Columns and the objects that hold them hold their plan objects, so a
    /// reference cycle through any of them is freed, as PySpark frees it: a `pyspark.ml` model
    /// whose summary keeps `model.transform(dataset)`, whose plan keeps the model, is released
    /// then, and so is a bound method of an object that keeps both a DataFrame and one derived
    /// from it.
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        if !self.gc_ready.get() {
            return Ok(());
        }
        self.parts.visit(&visit)?;
        if let Some(schema) = &self.schema_object {
            visit.call(schema)?;
        }
        // The ExecutionInfo of `executionInfo`, which may keep this DataFrame, as an attribute
        // that a user sets on it. Its lock is held only while no Python code runs.
        if let Ok(kept) = self.execution_info_object.try_lock() {
            if let Some((_, info)) = kept.as_ref() {
                visit.call(info)?;
            }
        }
        if let Ok(written) = self.written_execution_info.try_lock() {
            if let Some((_, info)) = written.as_ref() {
                visit.call(info)?;
            }
        }
        Ok(())
    }

    /// Break a reference cycle that the garbage collector found unreachable: the DataFrame drops
    /// its plan and its reference to the parts of it, and so the Python objects that
    /// `__traverse__` reported.
    fn __clear__(&mut self) {
        if self.gc_ready.get() {
            self.parts.release();
            self.dataframe.clear_plan();
            self.schema_object = None;
            let kept = self
                .execution_info_object
                .get_mut()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .take();
            drop(kept);
            let written = self
                .written_execution_info
                .get_mut()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .take();
            drop(written);
        }
    }

    /// The bytes of the relation of this DataFrame as it is sent, for the plan of `_plan`, built of
    /// its session (see `errors::of_request_session`).
    fn _relation_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let session = self.dataframe.spark_session();
        let relation = crate::errors::of_request_session(&session, || {
            self.dataframe.to_relation().to_pyerr()
        })?;
        Ok(pyo3::types::PyBytes::new(
            py,
            &crate::encode_on_grown_stack(relation),
        ))
    }

    /// The bytes of the placeholder of the relation of this DataFrame, which the plan of `_plan`
    /// sends in its place: one extension that holds the relation encoded, which the core builds
    /// now, as PySpark's `plan(session)` builds the relation, with its errors, and replaces the
    /// placeholder with when the plan reaches it, also after this DataFrame is gone. It is built
    /// of the session of `client`, the client that the plan is built of, as PySpark's plan builds
    /// the plan objects of the subqueries of its Columns of it, or else of the session of this
    /// DataFrame (see `errors::of_request_session`).
    #[pyo3(signature = (client=None))]
    fn _relation_placeholder<'py>(
        &self,
        py: Python<'py>,
        client: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        use prost::Message;
        let session = client
            .and_then(crate::session::session_of_client)
            .unwrap_or_else(|| self.dataframe.spark_session());
        let placeholder = crate::errors::of_request_session(&session, || {
            self.dataframe.standalone_placeholder_relation().to_pyerr()
        })?;
        Ok(pyo3::types::PyBytes::new(py, &placeholder.encode_to_vec()))
    }

    /// A DataFrame pickles, as PySpark Connect's, of its plan, here the bytes of its relation and
    /// the functions of its UDFs, of its session, which gives back the active session of its ID,
    /// and of the `_support_repr_html` and the `_cached_schema` of its state. The functions
    /// unpickle first and then the session, as PySpark's plan and session: so on a Python worker
    /// of the server, which has no active session, a pickled DataFrame raises
    /// `NO_ACTIVE_SESSION`, as PySpark's does, and a plain pickle fails at the first function
    /// that it cannot pickle, a lambda, and else at the session. The unpickled DataFrame has
    /// copies of the Observations of this one (see `DataFrame::from_sent_relation`).
    ///
    /// PySpark's pickle keeps the objects of the plan, which it does not build, and its unpickle
    /// builds the plan, as pickle looks the `__setstate__` of the new DataFrame up through its
    /// `__getattr__`, which reads its columns. So the relation is of holes in place of the
    /// commands of the UDFs, and the pickle carries the functions, the objects of their picklers,
    /// which the requests of the unpickled DataFrame pickle, as PySpark's unpickled plan pickles
    /// them at each request ([`crate::udf_command::PickleHoles`]). A plan that does not build, of
    /// a value that PySpark converts when it builds the plan, pickles of its error instead of its
    /// relation, which the unpickle raises ([`crate::column::described_error`]); of a plan of the
    /// Python client whose build raised, of that plan, which the unpickle builds again, as
    /// PySpark's does, of its own error then.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let this = slf.borrow();
        let session = this.spark_session(py)?;
        let (relation, build_error, holes) = this.pickled_relation(py)?;
        // The schema that this DataFrame cached, of the types of PySpark, when they hold it: the
        // `StructType` that it keeps as an object, of `to`, of a map function or of an unpickle,
        // as it is now, as PySpark's state carries the object; else that of the server, of new
        // objects of its user-defined types, as PySpark's `_schema` made them.
        let cached_schema = match &this.schema_object {
            Some(schema)
                if schema
                    .bind(py)
                    .is_instance_of::<crate::types::PyStructType>() =>
            {
                Some(schema.clone_ref(py))
            }
            _ => known_schema_object(py, &this.dataframe)?,
        };
        py.import("pyspark.sql.connect.dataframe")?
            .getattr("_reduce_dataframe")?
            .call1((
                session,
                relation,
                this.dataframe.plan_levels(),
                this.support_repr_html
                    .load(std::sync::atomic::Ordering::SeqCst),
                cached_schema,
                build_error,
                holes,
            ))
    }

    /// The reduce of the plan of this DataFrame (`DataFrame._plan`), as PySpark's plan object of a
    /// DataFrame pickles of its fields and of no session: the plan of `_column_reduce.sent_plan`
    /// of the relation of this DataFrame as it was sent, of its holes and of the error of a plan
    /// that does not build, as its pickle carries them (see `__reduce__`), a `LogicalPlan` on the
    /// client and on a Python worker of the server, whose PySpark has no class of the drop-in,
    /// which `DataFrame(plan, session)` and PySpark's DataFrame of a plan over it build alike, of
    /// `observations`, the Observations of the plan by their names. So a plan of the Python client
    /// over the plan of this DataFrame, such as a `SubqueryAlias`, pickles plainly too, and
    /// unpickles on the worker, as PySpark's plan does; one of a model fails as the DataFrame's
    /// pickle does (see `__reduce_ex__`). Not a stable API.
    #[pyo3(signature = (observations=None))]
    fn _plan_reduce<'py>(
        slf: &Bound<'py, Self>,
        observations: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let (relation, build_error, holes) = slf.borrow().pickled_relation(py)?;
        let reduce_module = py.import("pyspark.sql.connect._column_reduce")?;
        let fill = if holes.is_empty() {
            py.None().into_bound(py)
        } else {
            reduce_module.getattr("fill_holes")?
        };
        let observations = observations.unwrap_or_else(|| py.None().into_bound(py));
        let args = (relation, build_error, holes, fill, observations).into_pyobject(py)?;
        let reduce = PyTuple::new(py, [reduce_module.getattr("sent_plan")?, args.into_any()])?;
        // As PySpark's pickle of the plan of a model, which fails at the objects of `pyspark.ml`
        // that it keeps, as the pickle of the DataFrame does (see `__reduce_ex__`).
        let dataframe = slf.borrow().dataframe.clone();
        let reach = crate::ml::object::pickle_reach(py, &dataframe)?;
        crate::ml::object::reach_reduce(py, &reach, reduce.into_any())
    }

    /// The reduce of a pickle, as PySpark Connect's DataFrame pickles its plan before its session,
    /// of the objects of `pyspark.ml` that the plan keeps (see `ml::object::pickle_reach`, of the
    /// plans of its subqueries and of the Python plans of PySpark's own classes of `pyspark.ml`
    /// too): of a plan that keeps a model or a summary (a transform of a model, a DataFrame of a
    /// method of a model or of a summary), the reduce of the objects that PySpark's pickle of the
    /// plan reaches first, the functions of its UDFs before them and then the lock of the model's
    /// `RemoteModelRef`, or the summary dataset of a training summary, whose session a plain
    /// pickle refuses, as PySpark's refuses the session of the summary's `_summary_dataset`, and
    /// then the lock, which cloudpickle, of the function of the session, reaches next (see
    /// `ml::object::reach_reduce`); so a pickle fails at the first object that it cannot pickle,
    /// of the error of its protocol (the TypeError of a lock at 2 and up, its PicklingError below).
    /// Else the reduce of `__reduce__`.
    fn __reduce_ex__<'py>(
        slf: &Bound<'py, Self>,
        protocol: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = protocol;
        let py = slf.py();
        let dataframe = slf.borrow().dataframe.clone();
        let reach = crate::ml::object::pickle_reach(py, &dataframe)?;
        let reduced = slf.call_method0(pyo3::intern!(py, "__reduce__"))?;
        crate::ml::object::reach_reduce(py, &reach, reduced)
    }

    /// The shallow copy of a DataFrame, as PySpark's `copy.copy` of its `__reduce__`: the same
    /// plan, and so the same Observations, as PySpark's copy shares its plan, with the
    /// `_support_repr_html` and the schema that its state carries, the one that it keeps as an
    /// object, of `to` or of a map function, as the same object, the schema that its
    /// reconstruction requests (see `copy_schema`), and an `executionInfo` of its own, None
    /// until it executes, as that of PySpark's `DataFrame(plan, session)`.
    fn __copy__(slf: &Bound<'_, Self>) -> PyResult<PyDataFrame> {
        let py = slf.py();
        let this = slf.borrow();
        let requested = this.fresh_schema_request(py)?;
        let mut copy = PyDataFrame::new(this.dataframe.shallow_copy());
        copy.support_repr_html.store(
            this.support_repr_html
                .load(std::sync::atomic::Ordering::SeqCst),
            std::sync::atomic::Ordering::SeqCst,
        );
        copy.schema_object = this
            .schema_object
            .as_ref()
            .map(|schema| schema.clone_ref(py));
        copy.keep_copy_schema(requested);
        Ok(copy)
    }

    /// The deep copy of a DataFrame, as PySpark's `copy.deepcopy` of its `__reduce__`, which
    /// copies its plan and then its session with the memo of the call: the same relation and
    /// plan ID, with the `_support_repr_html` and the schema that its state carries and an
    /// `executionInfo` of its own, whose Observations are the copies of those of this DataFrame
    /// in `memo` (see `observation::deep_copy_in`). So an Observation that the call copies too,
    /// in a tuple or an object that holds both, is copied once, and the actions of the copied
    /// DataFrame set the copy; the Observations of this DataFrame stay as they are. The copy of
    /// the session is that of its `__reduce__`, the active session of its ID, once in a call: so
    /// a deep copy raises `NO_ACTIVE_SESSION` where no session is active, as on another thread,
    /// and `PySparkAssertionError` where another one is, while `copy.copy` keeps the session.
    /// PySpark's copy of the plan deep-copies the function of each UDF of it, with `memo`, before
    /// the session: so do these (`deep_copied_picklers`), of which, and of the copies of the
    /// Observations, the plan of the copy is made, which its requests build, as PySpark's copy
    /// does: they pickle the copied functions, warn of a window of no partitions, and give the
    /// `and` of the conditions of a join the origin of its call site then. A copy in another
    /// session, of a memo that maps the session or of another active session of its ID, has the
    /// plan IDs of this DataFrame there (`deep_copy_with_pickler_copies`).
    fn __deepcopy__(slf: &Bound<'_, Self>, memo: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let py = slf.py();
        let this = slf.borrow();
        let this = &*this;
        // As PySpark's: `copy.deepcopy` looks up `__deepcopy__` through `__getattr__`, whose check
        // reads the columns, so the schema is requested unless it is cached, and converted, of the
        // errors of its conversion (see `own_schema`), before the functions of the UDFs and the
        // session are copied. On a stopped session, that request raises the SparkConnectException
        // of its closed client, before the session is copied, which raises a PySparkRuntimeError.
        this.own_schema(py)
            .map_err(|err| columns_read_error(py, err))?;
        // As PySpark's copy of the plan, which copies the objects of `pyspark.ml` that it keeps
        // (see `__reduce_ex__`): the TypeError of the lock of a model, after copies of the
        // functions of the UDFs before it and of the `_summary_dataset` of the training summary of
        // a DataFrame of a method of the summary, of its requests of its schema.
        let reach = crate::ml::object::pickle_reach(py, &this.dataframe)?;
        if let Some(error) = crate::ml::object::reach_deepcopy_error(py, &reach, memo)? {
            return Err(error);
        }
        let observations = this
            .dataframe
            .observations()
            .into_iter()
            .map(|(name, observation)| {
                Ok((
                    name,
                    crate::observation::deep_copy_in(py, &observation, memo)?,
                ))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let copies = crate::udf_command::deep_copied_picklers(py, this.dataframe.picklers(), memo)?;
        let session = py
            .import("copy")?
            .call_method1(
                pyo3::intern!(py, "deepcopy"),
                (this.spark_session(py)?, memo),
            )?
            .extract::<PyRef<'_, crate::session::PySparkSession>>()?
            .session
            .clone();
        let copy = crate::on_grown_stack(|| {
            this.dataframe
                .deep_copy_with_pickler_copies(session, observations, &copies)
        })
        .to_pyerr()?;
        // Of the plan of this DataFrame when nothing of it is copied, which both hold, as
        // PySpark's copies of one plan object.
        let mut copy = this.copy_of(copy);
        // As PySpark's reconstruction of the copy, which requests its schema, as the copy has none,
        // and keeps it, before it sets the deep copy of the state, of the schema that it keeps as
        // an object too, which the deep copy of the plan shares in `memo`.
        let requested = copy.fresh_schema_request(py)?;
        if let Some(schema) = &this.schema_object {
            copy.schema_object = Some(
                py.import("copy")?
                    .call_method1(pyo3::intern!(py, "deepcopy"), (schema.bind(py), memo))?
                    .unbind(),
            );
        }
        copy.keep_copy_schema(requested);
        Ok(copy)
    }

    /// The DataFrame of `relation`, the bytes of a relation of `levels` levels of `session`, as a
    /// pickled DataFrame unpickles, with the `_support_repr_html` and the `_cached_schema` of its
    /// state. As PySpark's unpickle, whose lookup of the `__setstate__` of the new DataFrame reads
    /// its columns, it requests the schema of the relation, whose error it raises, also that of
    /// its conversion (see `check_python_schema`), and keeps it (see `copy_schema`), and then the
    /// DataFrame has the cached schema of the pickled one, the objects that it is of, or none
    /// when it had none, so that its `schema` requests it again then. Without
    /// `request_schema`, of an object that PySpark pickles of the plan of the DataFrame, as a
    /// DataStreamWriter, the schema is not requested. `holes` are those of the relation, of the
    /// functions of its UDFs, which each request pickles (see `udf_command::sent_holes`).
    #[staticmethod]
    #[pyo3(signature = (
        session,
        relation,
        levels=1,
        support_repr_html=false,
        cached_schema=None,
        request_schema=true,
        holes=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn _of_relation(
        py: Python<'_>,
        session: &Bound<'_, crate::session::PySparkSession>,
        relation: &[u8],
        levels: usize,
        support_repr_html: bool,
        cached_schema: Option<&Bound<'_, PyAny>>,
        request_schema: bool,
        holes: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let holes = crate::udf_command::sent_holes(holes)?;
        let session = session.borrow().session.clone();
        let known_schema = cached_schema
            .filter(|schema| !schema.is_none())
            .map(crate::types::py_to_data_type)
            .transpose()?;
        use prost::Message;
        // Decoded, and then built or dropped, on a stack that holds its levels.
        // The errors become Python exceptions on the caller's stack, which runs no Python code
        // on the grown one.
        let dataframe = crate::on_grown_stack(|| {
            spark_connect_proto::Relation::decode(relation).map(|relation| {
                DataFrame::from_sent_relation_with_holes(session, relation, levels, holes)
            })
        })
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?
        .to_pyerr()?;
        let requested = if request_schema {
            // Of the plan of the unpickled DataFrame, which holds the functions of its holes, and
            // the Observations of a live DataFrame of its plan ID.
            let schema = crate::interrupt::interruptible_of_values(
                py,
                &dataframe.spark_session(),
                dataframe.held_parts(),
                || dataframe.schema(),
            )?;
            check_python_schema(py, &schema)?;
            Some(schema)
        } else {
            None
        };
        let mut dataframe = PyDataFrame::new(dataframe.with_known_schema(known_schema));
        // The `StructType` of the state as the object that it is, as PySpark's `_cached_schema`,
        // which `columns`, `dtypes` and the methods that copy the cached schema read: of the
        // objects of `to`, of a map function or of the server that the pickled DataFrame held.
        dataframe.schema_object = cached_schema
            .filter(|schema| schema.is_instance_of::<crate::types::PyStructType>())
            .map(|schema| schema.clone().unbind());
        if let Some(requested) = requested {
            dataframe.keep_copy_schema(requested);
        }
        dataframe
            .support_repr_html
            .store(support_repr_html, std::sync::atomic::Ordering::SeqCst);
        Ok(dataframe)
    }

    /// The plan of this DataFrame, as PySpark Connect's `DataFrame._plan`, whose
    /// `plan(client)` is its relation, over which `pyspark.ml` builds its plans
    /// (`pyspark.sql.connect.plan`).
    #[getter]
    fn _plan(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        Ok(slf
            .py()
            .import("pyspark.sql.connect.plan")?
            .getattr("_DataFramePlan")?
            .call1((slf,))?
            .unbind())
    }

    /// The Observations of the plan of this DataFrame by their names, as PySpark Connect's
    /// `LogicalPlan.observations` of its `DataFrame._plan`, which its DataFrame gives the
    /// executions of `SparkSession.client` (`client.to_table(plan, df._plan.observations)`): those
    /// of `observe` on this DataFrame and on the inputs that PySpark's plans hold as their child,
    /// both sides of a join and of a set operation, and those of the DataFrames under a plan of
    /// Python (`DataFrame(plan, session)`), such as a `SubqueryAlias`, in the order of PySpark's
    /// dict. A DataFrame observed by a name has none, as in PySpark.
    fn _observations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let observations = PyDict::new(py);
        // The core gives them in the reverse of the order of PySpark's dict, which merges the
        // Observations of a plan after those of its inputs (see `DataFrame::observations`).
        for (_, observation) in self.dataframe.observations().into_iter().rev() {
            let python = crate::observation::python_observation_of(py, &observation)?;
            // As PySpark's key, `str(observation._name)`, of a name that UTF-8 cannot encode too.
            let name = python.getattr(pyo3::intern!(py, "_name"))?.str()?;
            observations.set_item(name, python)?;
        }
        Ok(observations)
    }

    // Select columns (accepts Column objects or string names).
    #[pyo3(signature = (*cols))]
    #[doc = connect_doc!(DataFrame.select)]
    fn select(&self, _py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        let columns = to_column_list(cols)?;
        Ok(PyDataFrame::new(self.dataframe.select(columns)))
    }

    // Select using SQL expressions, as PySpark's `DataFrame.selectExpr`: strings, or lists of
    // them.
    #[pyo3(signature = (*expr))]
    #[doc = connect_doc!(DataFrame.selectExpr)]
    fn selectExpr(&self, expr: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        let expr = match expr.as_slice() {
            [only] if only.is_instance_of::<PyList>() => {
                only.try_iter()?.collect::<PyResult<Vec<_>>>()?
            }
            _ => expr,
        };
        // Each of `F.expr`, which asserts a str.
        let mut columns = Vec::new();
        for element in &expr {
            if element.is_instance_of::<pyo3::types::PyString>() {
                columns.push(crate::plan_args::expr(element)?);
            } else {
                for e in element.try_iter()? {
                    columns.push(crate::plan_args::expr(&e?)?);
                }
            }
        }
        Ok(PyDataFrame::new(self.dataframe.select(columns)))
    }

    // Filter rows by a Column condition or a SQL-string condition.
    #[doc = connect_doc!(DataFrame.filter)]
    fn filter(&self, condition: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        Ok(self.of_same_schema(PyDataFrame::new(
            self.dataframe.filter(cond_column(condition)?),
        )))
    }

    // Alias for filter (reference pyspark name is `where`).
    #[pyo3(name = "where")]
    #[doc = connect_doc!(DataFrame.where)]
    fn where_(&self, condition: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        // As PySpark's: a Column or a str, else NOT_EXPECTED_TYPE.
        if !(condition.is_instance_of::<PyColumn>()
            || condition.is_instance_of::<pyo3::types::PyString>())
        {
            return Err(not_expected_type(condition, "condition", "Column or str"));
        }
        Ok(self.of_same_schema(PyDataFrame::new(
            self.dataframe.filter(cond_column(condition)?),
        )))
    }

    // Add or replace a column, of a Column, as PySpark's `DataFrame.withColumn`, which checks
    // the Column first. Its `WithColumns` plan asserts a str name, and sets it when it builds
    // the plan, after the Column.
    #[doc = connect_doc!(DataFrame.withColumn)]
    fn withColumn(
        &self,
        colName: &Bound<'_, PyAny>,
        col: &Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        let Ok(column) = col.extract::<PyRef<'_, PyColumn>>() else {
            return Err(not_expected_type(col, "col", "Column"));
        };
        if !colName.is_instance_of::<pyo3::types::PyString>() {
            return Err(pyo3::exceptions::PyAssertionError::new_err(()));
        }
        let mut args = crate::plan_args::PlanArgs::new();
        let name = args.str(colName)?;
        Ok(PyDataFrame::new(args.plan(
            self.dataframe.with_column(&name, column.column.clone()),
        )))
    }

    // Rename a column, as PySpark's, which renames of the dict `{existing: new}`, so an
    // `existing` that is not hashable raises the `TypeError` of the dict.
    #[doc = connect_doc!(DataFrame.withColumnRenamed)]
    fn withColumnRenamed(
        &self,
        existing: &Bound<'_, PyAny>,
        new: &Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        let renames = PyDict::new(existing.py());
        renames.set_item(existing, new)?;
        self.with_columns_renamed(renames.as_any())
    }

    // Add or replace multiple columns from a {name: Column} mapping, as PySpark's
    // `DataFrame.withColumns`, whose `WithColumns` plan asserts, as the call builds it, that the
    // mapping is not empty, and of str names and Columns.
    #[pyo3(name = "withColumns", signature = (*colsMap))]
    #[doc = connect_doc!(DataFrame.withColumns)]
    fn with_columns(&self, colsMap: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        let [cols_map] = colsMap.as_slice() else {
            return Err(pyo3::exceptions::PyAssertionError::new_err(()));
        };
        let Ok(cols_map) = cols_map.cast::<PyDict>() else {
            return Err(not_expected_type(cols_map, "colsMap", "dict"));
        };
        let valid = !cols_map.is_empty()
            && cols_map.iter().all(|(name, column)| {
                name.is_instance_of::<pyo3::types::PyString>()
                    && column.is_instance_of::<PyColumn>()
            });
        if !valid {
            return Err(pyo3::exceptions::PyAssertionError::new_err(()));
        }
        // A name that UTF-8 cannot encode fails the plan when it is sent, with the error of its
        // encoding, which the first of them gives.
        let mut args = crate::plan_args::PlanArgs::new();
        let mut pairs = Vec::with_capacity(cols_map.len());
        for (name, column) in cols_map.iter() {
            let column = column.extract::<PyRef<'_, PyColumn>>()?;
            pairs.push((args.str(&name)?, column.column.clone()));
        }
        Ok(PyDataFrame::new(
            args.plan(self.dataframe.with_columns(pairs)),
        ))
    }

    // Specify a relation hint (e.g. "broadcast"), with optional parameters. Reference
    // hint params may be ints (e.g. `df.hint("rebalance", 3)`), not just strings, so
    // accept any value and stringify it; the core parses a numeric string back into a
    // long literal, matching reference semantics.
    #[pyo3(signature = (name, *parameters))]
    #[doc = connect_doc!(DataFrame.hint)]
    fn hint(
        &self,
        py: Python<'_>,
        name: &Bound<'_, PyAny>,
        parameters: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        use pyo3::types::{PyFloat, PyString};
        // As PySpark's: a single list is the parameters, each a str, a float, an int, a
        // Column or a list of str, float or int, sent as `lit` of it.
        let parameters = match parameters.as_slice() {
            [only] if only.is_instance_of::<PyList>() => {
                only.try_iter()?.collect::<PyResult<Vec<_>>>()?
            }
            _ => parameters,
        };
        if !name.is_instance_of::<PyString>() {
            return Err(not_expected_type(name, "name", "str"));
        }
        let primitive = |p: &Bound<'_, PyAny>| {
            p.is_instance_of::<PyString>()
                || p.is_instance_of::<PyFloat>()
                || p.is_instance_of::<PyInt>()
        };
        let invalid = |item_type: &str| {
            crate::column::raise_pyspark(
                py,
                "PySparkTypeError",
                "INVALID_ITEM_FOR_CONTAINER",
                &[
                    ("arg_name", "parameters"),
                    (
                        "allowed_types",
                        "str, float, int, Column, list[str], list[float], list[int]",
                    ),
                    ("item_type", item_type),
                ],
            )
        };
        let type_name =
            |p: &Bound<'_, PyAny>| -> PyResult<String> { Ok(p.get_type().name()?.to_string()) };
        let mut expressions = Vec::with_capacity(parameters.len());
        for p in &parameters {
            if p.is_instance_of::<PyList>() {
                let items: Vec<Bound<'_, PyAny>> = p.try_iter()?.collect::<PyResult<_>>()?;
                if !items.iter().all(primitive) {
                    // As PySpark's, which names the type of the first item.
                    let first = items
                        .first()
                        .map(type_name)
                        .transpose()?
                        .unwrap_or_default();
                    return Err(invalid(&format!("list[{first}]")));
                }
            } else if !(primitive(p) || p.extract::<PyRef<'_, PyColumn>>().is_ok()) {
                return Err(invalid(&type_name(p)?));
            }
            expressions.push(crate::functions::to_column(p)?.expression().clone());
        }
        // Its `Hint` plan sets the name when it builds the plan, before the parameters.
        let mut args = crate::plan_args::PlanArgs::new();
        let name = args.str(name)?;
        let args = args.before(&mut expressions);
        Ok(self.of_same_schema(PyDataFrame::new(
            args.plan(self.dataframe.hint_expressions(&name, expressions)),
        )))
    }

    // Drop columns, of Columns or column names, as PySpark's `DataFrame.drop`.
    #[pyo3(signature = (*cols))]
    #[doc = connect_doc!(DataFrame.drop)]
    fn drop(&self, py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        // Its `Drop` plan sets the names when it builds the plan.
        let mut args = crate::plan_args::PlanArgs::new();
        let (mut columns, mut names) = (Vec::new(), Vec::new());
        for column in &cols {
            if let Ok(column) = column.extract::<PyRef<'_, PyColumn>>() {
                columns.push(column.column.clone());
            } else if column.is_instance_of::<pyo3::types::PyString>() {
                names.push(args.str(column)?);
            } else {
                return Err(crate::column::raise_pyspark(
                    py,
                    "PySparkTypeError",
                    "NOT_EXPECTED_TYPE",
                    &[
                        ("expected_type", "Column or str"),
                        ("arg_name", "cols"),
                        ("arg_type", "tuple"),
                    ],
                ));
            }
        }
        Ok(PyDataFrame::new(
            args.plan(self.dataframe.drop_columns(columns, names)),
        ))
    }

    // Limit the number of rows, as PySpark's `limit`, which does not check `num`: its plan sets
    // `num` in the proto's int32, whose errors ([`proto_int`]) PySpark raises once it sends the
    // plan, so the DataFrame raises them at each request. The server checks the rest.
    #[doc = connect_doc!(DataFrame.limit)]
    fn limit(&self, num: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let py = num.py();
        let value = proto_int(num, "spark.connect.Limit.limit");
        let df = converted_plan(py, &self.dataframe, value, |num| self.dataframe.limit(num))?;
        Ok(self.of_same_schema(PyDataFrame::new(df)))
    }

    // Skip the first `num` rows, as PySpark's `offset`, which sends `num` as `limit` does, but
    // does not copy the `_cached_schema` of its input, also when the count fails the plan.
    #[doc = connect_doc!(DataFrame.offset)]
    fn offset(&self, num: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let py = num.py();
        let df = match proto_int(num, "spark.connect.Offset.offset") {
            Ok(num) => self.dataframe.offset(num),
            Err(err) => self.dataframe.unbuildable(unbuildable_error(py, err)?),
        };
        Ok(PyDataFrame::new(df))
    }

    // Remove duplicate rows.
    #[doc = connect_doc!(DataFrame.distinct)]
    fn distinct(&self) -> PyDataFrame {
        self.of_same_schema(PyDataFrame::new(self.dataframe.distinct()))
    }

    // Remove duplicate rows, optionally on specific columns.
    #[pyo3(signature = (subset=None))]
    #[doc = connect_doc!(DataFrame.dropDuplicates)]
    fn dropDuplicates(&self, subset: Option<&Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        let mut args = crate::plan_args::PlanArgs::new();
        let subset = deduplicate_subset(&mut args, subset)?;
        let col_refs: Option<Vec<&str>> = subset
            .as_ref()
            .map(|names| names.iter().map(|s| s.as_str()).collect());
        Ok(self.of_same_schema(PyDataFrame::new(
            args.plan(self.dataframe.drop_duplicates(col_refs)),
        )))
    }

    // Sort rows. `ascending` (a bool or list of bools, via kwargs) mirrors reference
    // pyspark: it wraps each sort column with asc()/desc().
    #[pyo3(signature = (*cols, **kwargs))]
    #[doc = connect_doc!(DataFrame.sort)]
    fn sort(
        &self,
        _py: Python<'_>,
        cols: Vec<Bound<'_, PyAny>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        let exprs = self.sort_columns(_py, cols, kwargs.as_ref())?;
        Ok(self.of_same_schema(PyDataFrame::new(self.dataframe.sort(exprs))))
    }

    // Alias for sort.
    #[pyo3(signature = (*cols, **kwargs))]
    #[doc = connect_doc!(DataFrame.orderBy)]
    fn orderBy(
        &self,
        _py: Python<'_>,
        cols: Vec<Bound<'_, PyAny>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        let exprs = self.sort_columns(_py, cols, kwargs.as_ref())?;
        Ok(self.of_same_schema(PyDataFrame::new(self.dataframe.order_by(exprs))))
    }

    // Join with another DataFrame.
    //
    // `on` is a Column condition, a column name, or a list of column names (a "using"
    // join) or of Column conditions (joined on their conjunction), as the connect client's
    // `plan.Join` takes it. Mirrors `pyspark.sql.DataFrame.join(other, on=None, how=None)`
    // (default inner), including its errors.
    #[pyo3(signature = (other, on=None, how=None))]
    #[doc = connect_doc!(DataFrame.join)]
    fn join(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        on: Option<Bound<'_, PyAny>>,
        how: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        // As PySpark's `_check_same_session`, first.
        let other = self.same_session(other)?;
        // As PySpark's `plan.Join`, `on` is iterated before `how` is checked, so an `on`
        // that is not iterable is the error that is raised first.
        if let Some(on) = &on {
            if !on.is_instance_of::<PyColumn>() && !on.is_instance_of::<pyo3::types::PyString>() {
                on.try_iter()?;
            }
        }
        let join_type = parse_join_type(py, how.as_ref(), false)?;
        let Some(on) = on else {
            return Ok(PyDataFrame::new(self.dataframe.join(
                &other.dataframe,
                None,
                join_type,
            )));
        };
        if let Ok(col) = on.extract::<PyRef<PyColumn>>() {
            return Ok(PyDataFrame::new(self.dataframe.join(
                &other.dataframe,
                Some(col.column.clone()),
                join_type,
            )));
        }
        // The names are set in the repeated string field of its `Join` plan as they are when it
        // builds the plan, as protobuf takes them: bytes of UTF-8 too.
        let mut args = crate::plan_args::PlanArgs::new();
        if on.is_instance_of::<pyo3::types::PyString>() {
            let name = args.str(&on)?;
            return Ok(PyDataFrame::new(args.plan(self.dataframe.join_using(
                &other.dataframe,
                vec![name],
                join_type,
            ))));
        }
        if let Ok(list) = on.cast::<PyList>() {
            // An empty list is no condition at all.
            let Some(first) = list.iter().next() else {
                return Ok(PyDataFrame::new(self.dataframe.join(
                    &other.dataframe,
                    None,
                    join_type,
                )));
            };
            if first.is_instance_of::<pyo3::types::PyString>() {
                let names = args.strs(list.iter())?;
                return Ok(PyDataFrame::new(args.plan(self.dataframe.join_using(
                    &other.dataframe,
                    names,
                    join_type,
                ))));
            }
            // A list of conditions is their conjunction, which its `Join` plan computes when it
            // builds the plan.
            let condition = list_conjunction(&mut args, list)?;
            return Ok(PyDataFrame::new(args.plan(self.dataframe.join(
                &other.dataframe,
                condition,
                join_type,
            ))));
        }
        // A tuple, or another iterable that is not a list: its plan reads `to_plan` of it when
        // it builds the plan.
        args.record(no_to_plan_error(&on)?);
        Ok(PyDataFrame::new(args.plan(self.dataframe.join(
            &other.dataframe,
            None,
            join_type,
        ))))
    }

    /// `DataFrame._joinAsOf(other, leftAsOfColumn, rightAsOfColumn, on=None, how=None, *,
    /// tolerance=None, allowExactMatches=True, direction="backward")`, the as-of join of
    /// pandas-on-Spark's `merge_asof`, as PySpark's. The as-of columns are Columns, or names of
    /// their DataFrame. `on` is a name, a Column condition, or a list of either. `how` (inner by
    /// default) and `direction` are not checked, as PySpark does not check them: a string that
    /// is none of the join types or directions of the core is sent as it is, and the server
    /// refuses what it does not read.
    #[pyo3(
        name = "_joinAsOf",
        signature = (
            other,
            leftAsOfColumn,
            rightAsOfColumn,
            on=None,
            how=None,
            *,
            tolerance=None,
            allowExactMatches=DefaultedOption::Omitted,
            direction=DefaultedOption::Omitted
        ),
        text_signature = "($self, other, leftAsOfColumn, rightAsOfColumn, on=None, how=None, *, tolerance=None, allowExactMatches=True, direction='backward')"
    )]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn join_as_of(
        &self,
        other: &Bound<'_, PyAny>,
        leftAsOfColumn: &Bound<'_, PyAny>,
        rightAsOfColumn: &Bound<'_, PyAny>,
        on: Option<Bound<'_, PyAny>>,
        how: Option<Bound<'_, PyAny>>,
        tolerance: Option<Bound<'_, PyAny>>,
        allowExactMatches: DefaultedOption,
        direction: DefaultedOption,
    ) -> PyResult<PyDataFrame> {
        // As PySpark's `_check_same_session`, first.
        let other = self.same_session(other)?;
        // As PySpark's `assert isinstance(how, str)`. A str that UTF-8 cannot encode is one, and
        // fails the plan with the error of its encoding, when PySpark sends it.
        let how = how.filter(|how| !how.is_none());
        if let Some(how) = &how {
            if !how.is_instance_of::<pyo3::types::PyString>() {
                return Err(pyo3::exceptions::PyAssertionError::new_err(
                    "how should be a string",
                ));
            }
        }
        let tolerance = match tolerance.filter(|tolerance| !tolerance.is_none()) {
            None => None,
            Some(tolerance) => {
                let column = tolerance.extract::<PyRef<PyColumn>>().map_err(|_| {
                    pyo3::exceptions::PyAssertionError::new_err("tolerance should be Column")
                })?;
                Some(column.column.clone())
            }
        };
        // The as-of columns before `on`, as PySpark converts them before its `AsOfJoin` plan
        // iterates `on`.
        let left_as_of = as_of_column(&self.dataframe, leftAsOfColumn)?;
        let right_as_of = as_of_column(&other.dataframe, rightAsOfColumn)?;
        // The plan's values are converted in the order of its protobuf fields, in which its
        // `AsOfJoin` plan sets them when it builds the plan, which gives the first error.
        let mut args = crate::plan_args::PlanArgs::new();
        let (on, using_columns) = match on.filter(|on| !on.is_none()) {
            None => (None, Vec::new()),
            Some(on) => as_of_join_condition(&mut args, &on)?,
        };
        let join_type = match &how {
            None => AsOfJoinType::Inner,
            Some(how) => AsOfJoinType::parse(&args.str(how)?),
        };
        // The flag is a protobuf bool, which PySpark sets as the plan is sent.
        let py = leftAsOfColumn.py();
        let allow_exact_matches = match &allowExactMatches {
            DefaultedOption::Omitted => true,
            DefaultedOption::Given(allow) => args.converted(py, proto_bool(allow.bind(py)))?,
        };
        // PySpark does not check `direction`, so its errors come last, as those of protobuf's
        // string field when the plan is sent.
        let direction = match &direction {
            DefaultedOption::Omitted => AsOfJoinDirection::Backward,
            DefaultedOption::Given(direction) => {
                AsOfJoinDirection::parse(&args.str(direction.bind(py))?)
            }
        };
        let options = AsOfJoinOptions {
            left_as_of,
            right_as_of,
            on,
            using_columns,
            join_type,
            tolerance,
            allow_exact_matches,
            direction,
        };
        let df = self
            .dataframe
            .as_of_join(&other.dataframe, options)
            .to_pyerr()?;
        Ok(PyDataFrame::new(args.plan(df)))
    }

    // Cross join.
    #[doc = connect_doc!(DataFrame.crossJoin)]
    fn crossJoin(&self, other: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let other = self.same_session(other)?;
        Ok(PyDataFrame::new(
            self.dataframe.cross_join(&other.dataframe),
        ))
    }

    // Lateral (correlated) join. `on` is an optional Column condition; `how` is inner by
    // default and takes the join types of a lateral join. Mirrors `DataFrame.lateralJoin`.
    #[pyo3(signature = (other, on=None, how=None))]
    #[doc = connect_doc!(DataFrame.lateralJoin)]
    fn lateralJoin(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        on: Option<Bound<'_, PyAny>>,
        how: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        // As PySpark's `_check_same_session`, first. Its `LateralJoin` plan then collects the
        // subqueries of `on` by its `foreach`, which an object that is not a Column lacks, before
        // it checks `how`.
        let other = self.same_session(other)?;
        let on_col = match on {
            None => None,
            Some(on) => match on.extract::<PyRef<PyColumn>>() {
                Ok(column) => Some(column.column.clone()),
                Err(_) => {
                    on.getattr("foreach")?;
                    return Err(no_to_plan(&on));
                }
            },
        };
        let join_type = parse_join_type(py, how.as_ref(), true)?;
        Ok(PyDataFrame::new(self.dataframe.lateral_join(
            &other.dataframe,
            on_col,
            join_type,
        )))
    }

    // Transpose (swap rows/columns). `indexColumn` optionally names the header
    // column. Mirrors `DataFrame.transpose(indexColumn=None)`.
    #[pyo3(signature = (indexColumn=None))]
    #[allow(non_snake_case)]
    #[doc = connect_doc!(DataFrame.transpose)]
    fn transpose(&self, indexColumn: Option<&Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        // As PySpark's `F._to_col`: a Column or a column name, else NOT_EXPECTED_TYPE of `col`.
        let index = match indexColumn {
            None => None,
            Some(c) => Some(if let Ok(column) = c.extract::<PyRef<'_, PyColumn>>() {
                column.column.clone()
            } else if c.is_instance_of::<pyo3::types::PyString>() {
                let _origin = crate::origin::with_origin("col")?;
                crate::plan_args::col(c)?
            } else {
                return Err(not_expected_type(c, "col", "Column or str"));
            }),
        };
        let df = match index {
            Some(column) => self.dataframe.transpose_with_index(column),
            None => self.dataframe.transpose(),
        }
        .to_pyerr()?;
        Ok(PyDataFrame::new(df))
    }

    // Repartition into `numPartitions` by a range partitioning on the given
    // columns. Mirrors `DataFrame.repartitionByRange(numPartitions, *cols)`.
    #[pyo3(signature = (numPartitions, *cols))]
    #[allow(non_snake_case)]
    #[doc = connect_doc!(DataFrame.repartitionByRange)]
    fn repartitionByRange(
        &self,
        py: Python<'_>,
        numPartitions: Bound<'_, PyAny>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        // As PySpark's: a positive partition count and columns, or columns alone. The columns
        // are sorted ascending unless they are sort orders, by `repartition_by_range_opt`.
        let (num_partitions, cols) = if numPartitions.is_instance_of::<PyInt>() {
            positive_partitions(&numPartitions)?;
            if cols.is_empty() {
                return Err(crate::column::raise_pyspark(
                    py,
                    "PySparkValueError",
                    "CANNOT_BE_EMPTY",
                    &[("item", "cols")],
                ));
            }
            (Some(numPartitions), cols)
        } else if numPartitions.is_instance_of::<pyo3::types::PyString>()
            || numPartitions.extract::<PyRef<'_, PyColumn>>().is_ok()
        {
            // No partition count is sent, so the server picks it, as PySpark's.
            let mut all = vec![numPartitions];
            all.extend(cols);
            (None, all)
        } else {
            return Err(not_expected_type(
                &numPartitions,
                "numPartitions",
                "Column, int or str",
            ));
        };
        // As PySpark's `_to_cols(cols, sort=True)`, which does not unpack a list among them, and
        // sorts a column that is not a sort order by its `asc`, which is `asc_nulls_first`, of
        // the origin of that.
        let mut exprs = Vec::with_capacity(cols.len());
        for column in to_columns_as_given(cols)? {
            let sorted = match column.expression().node() {
                spark_connect::ExpressionNode::SortOrder(_) => column,
                _ => {
                    let _origin = crate::origin::with_origin("asc_nulls_first")?;
                    column.asc_nulls_first()
                }
            };
            exprs.push(sorted.expression().clone());
        }
        // The count is the proto's int32 of the plan, which PySpark converts once it sends the
        // plan, after the columns: a count that protobuf refuses fails the plan, which is built
        // here with no count, so that the errors of the columns come first.
        let (num_partitions, plan_error) = match num_partitions
            .map(|n| proto_int(&n, "spark.connect.RepartitionByExpression.num_partitions"))
            .transpose()
        {
            Ok(num_partitions) => (num_partitions, None),
            Err(err) => (None, Some(unbuildable_error(py, err)?)),
        };
        let df = self
            .dataframe
            .repartition_by_range_opt(num_partitions, exprs)
            .to_pyerr()?;
        // Of the schema that the input cached, as PySpark copies it.
        Ok(self.of_same_schema(PyDataFrame::new(match plan_error {
            Some(error) => df.unbuildable_with_cached_schema(error),
            None => df,
        })))
    }

    // Repartition into `numPartitions` using a column's value directly as the shuffle
    // partition id. Mirrors `DataFrame.repartitionById(numPartitions, partitionIdCol)`.
    #[pyo3(signature = (numPartitions, partitionIdCol))]
    #[allow(non_snake_case)]
    #[doc = connect_doc!(DataFrame.repartitionById)]
    fn repartitionById(
        &self,
        numPartitions: &Bound<'_, PyAny>,
        partitionIdCol: &Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        // As PySpark's: an int that is not a bool, and not `numPartitions <= 0`.
        if !numPartitions.is_instance_of::<PyInt>()
            || numPartitions.is_instance_of::<pyo3::types::PyBool>()
        {
            return Err(not_expected_type(numPartitions, "numPartitions", "int"));
        }
        if numPartitions.le(0)? {
            return Err(not_positive_partitions(numPartitions)?);
        }
        // A `ColumnOrName`, as PySpark's `F._to_col`: a string is a column name, by `col`, of its
        // origin, not a literal, and another value is NOT_EXPECTED_TYPE of `col`.
        let col = if let Ok(c) = partitionIdCol.extract::<Bound<'_, PyColumn>>() {
            c.borrow().column.clone()
        } else if partitionIdCol.is_instance_of::<pyo3::types::PyString>() {
            let _origin = crate::origin::with_origin("col")?;
            crate::plan_args::col(partitionIdCol)?
        } else {
            return Err(not_expected_type(partitionIdCol, "col", "Column or str"));
        };
        // The count is the proto's int32 of the plan, which PySpark's `RepartitionByExpression`
        // sets after the column and the input, as in `repartition`: a count that protobuf refuses
        // fails the plan, which is built here with no count.
        let df = match proto_int(
            numPartitions,
            "spark.connect.RepartitionByExpression.num_partitions",
        ) {
            Ok(num_partitions) => self.dataframe.repartition_by_id(num_partitions, col),
            // A count of the plan that is never built: one that the core takes, which refuses a
            // count that is not positive.
            Err(err) => self
                .dataframe
                .repartition_by_id(1, col)
                .unbuildable_with_cached_schema(unbuildable_error(numPartitions.py(), err)?),
        };
        Ok(self.of_same_schema(PyDataFrame::new(df)))
    }

    // Append a distributed monotonically-increasing index column, as PySpark's
    // `DataFrame.zipWithIndex(indexColName="index")`: `self.select(F.col("*"),
    // F._invoke_function("distributed_sequence_id").alias(indexColName))`, of the origins of
    // `col` and `alias`. The name is that of the alias of a Column ([`PyColumn::alias`]), which
    // the plan sets when it is built: a str or bytes of UTF-8, and another value fails each
    // request of the DataFrame, not the call.
    #[pyo3(
        signature = (indexColName=DefaultedOption::Omitted),
        text_signature = "($self, indexColName='index')"
    )]
    #[allow(non_snake_case)]
    #[doc = connect_doc!(DataFrame.zipWithIndex)]
    fn zipWithIndex(&self, py: Python<'_>, indexColName: DefaultedOption) -> PyResult<PyDataFrame> {
        let star = {
            let _origin = crate::origin::with_origin("col")?;
            spark_connect::functions::col("*")
        };
        let name = match &indexColName {
            DefaultedOption::Omitted => pyo3::types::PyString::new(py, "index").into_any(),
            DefaultedOption::Given(name) => name.bind(py).clone(),
        };
        let index = PyColumn::new(spark_connect::column::Column::new(
            spark_connect::expression::Expression::function("distributed_sequence_id", vec![]),
        ))
        .alias(py, &PyTuple::new(py, [name])?, None)?;
        Ok(PyDataFrame::new(
            self.dataframe.select(vec![star, index.column]),
        ))
    }

    // Zip this DataFrame with another column-wise (side-by-side). Mirrors
    // `DataFrame.zip(other)`.
    #[doc = connect_doc!(DataFrame.zip)]
    fn zip(&self, other: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let other = self.same_session(other)?;
        Ok(PyDataFrame::new(
            self.dataframe.zip(&other.dataframe).to_pyerr()?,
        ))
    }

    // Union with another DataFrame.
    #[doc = connect_doc!(DataFrame.union)]
    fn union(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let other = self.same_session(other)?;
        let dataframe = self.dataframe.union(&other.dataframe);
        set_operation_result(py, self, &other, dataframe)
    }

    // Union by name, optionally filling columns missing on one side with null. The flag is
    // converted as PySpark's set operation sets it: None is a `TypeError`.
    #[pyo3(
        signature = (other, allowMissingColumns=ProtoBool::default()),
        text_signature = "($self, other, allowMissingColumns=False)"
    )]
    #[doc = connect_doc!(DataFrame.unionByName)]
    fn unionByName(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        allowMissingColumns: ProtoBool,
    ) -> PyResult<PyDataFrame> {
        let other = self.same_session(other)?;
        // As PySpark's set operation, which takes the flag when a request sends the plan: a
        // flag that a protobuf bool does not take fails each request, not this call.
        let dataframe = match allowMissingColumns.value(py) {
            Ok(allow_missing_columns) => self
                .dataframe
                .union_by_name_opt(&other.dataframe, allow_missing_columns),
            Err(err) => self
                .dataframe
                .union_by_name_unbuildable(&other.dataframe, unbuildable_error(py, err)?),
        };
        set_operation_result(py, self, &other, dataframe)
    }

    // Intersect with another DataFrame.
    #[doc = connect_doc!(DataFrame.intersect)]
    fn intersect(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let other = self.same_session(other)?;
        let dataframe = self.dataframe.intersect(&other.dataframe);
        set_operation_result(py, self, &other, dataframe)
    }

    // Intersect keeping duplicates (reference pyspark `intersectAll`).
    #[pyo3(name = "intersectAll")]
    #[doc = connect_doc!(DataFrame.intersectAll)]
    fn intersect_all(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let other = self.same_session(other)?;
        let dataframe = self.dataframe.intersect_all(&other.dataframe);
        set_operation_result(py, self, &other, dataframe)
    }

    // Subtract another DataFrame.
    #[doc = connect_doc!(DataFrame.subtract)]
    fn subtract(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let other = self.same_session(other)?;
        let dataframe = self.dataframe.subtract(&other.dataframe);
        set_operation_result(py, self, &other, dataframe)
    }

    // Repartition. Mirrors reference `repartition(numPartitions, *cols)` where the
    // first argument may instead be a Column/str - `df.repartition("country")` /
    // `df.repartition(col("country"))` repartition by that column at the default
    // partition count (no explicit number).
    #[pyo3(signature = (numPartitions, *cols))]
    #[doc = connect_doc!(DataFrame.repartition)]
    fn repartition(
        &self,
        numPartitions: Bound<'_, PyAny>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        // As PySpark's: a positive partition count, or a partition column (str/Column), at
        // the server's count (passed as 0 = unset). The count is the proto's int32 of the plan,
        // converted after the columns, as in `repartitionByRange`.
        if numPartitions.is_instance_of::<PyInt>() {
            positive_partitions(&numPartitions)?;
            let py = numPartitions.py();
            if cols.is_empty() {
                let n = proto_int(&numPartitions, "spark.connect.Repartition.num_partitions");
                let df = converted_plan(py, &self.dataframe, n, |n| self.dataframe.repartition(n))?;
                return Ok(self.of_same_schema(PyDataFrame::new(df)));
            }
            // As PySpark's `_to_cols(cols)`, which does not unpack a list among them.
            let exprs: Vec<_> = to_columns_as_given(cols)?
                .iter()
                .map(|c| c.expression().clone())
                .collect();
            // PySpark's `RepartitionByExpression.plan` builds the partition expressions, then the
            // input, and sets the count last: a count that protobuf refuses fails the plan, which
            // is built here with no count, so that the errors of the columns and of the input come
            // first, as in `repartitionByRange`.
            let df = match proto_int(
                &numPartitions,
                "spark.connect.RepartitionByExpression.num_partitions",
            ) {
                Ok(n) => self.dataframe.repartition_by_expressions(n, exprs),
                // A count of the plan that is never built, which the core takes.
                Err(err) => self
                    .dataframe
                    .repartition_by_expressions(1, exprs)
                    .unbuildable_with_cached_schema(unbuildable_error(py, err)?),
            };
            return Ok(self.of_same_schema(PyDataFrame::new(df)));
        }
        if !(numPartitions.is_instance_of::<pyo3::types::PyString>()
            || numPartitions.extract::<PyRef<'_, PyColumn>>().is_ok())
        {
            return Err(not_expected_type(
                &numPartitions,
                "numPartitions",
                "Column or str",
            ));
        }
        let mut all = vec![numPartitions];
        all.extend(cols);
        let exprs: Vec<_> = to_columns_as_given(all)?
            .iter()
            .map(|c| c.expression().clone())
            .collect();
        Ok(self.of_same_schema(PyDataFrame::new(
            self.dataframe.repartition_by_expressions(0, exprs),
        )))
    }

    // Coalesce into `numPartitions` partitions, as PySpark's `coalesce`: any value that is
    // `> 0`, which is then the proto's int32 of the plan, as in `repartitionByRange`.
    #[doc = connect_doc!(DataFrame.coalesce)]
    fn coalesce(&self, numPartitions: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        positive_partitions(numPartitions)?;
        let value = proto_int(numPartitions, "spark.connect.Repartition.num_partitions");
        let df = converted_plan(
            numPartitions.py(),
            &self.dataframe,
            value,
            |num_partitions| self.dataframe.coalesce(num_partitions),
        )?;
        Ok(self.of_same_schema(PyDataFrame::new(df)))
    }

    // Convert to DataFrame with new column names.
    #[pyo3(signature = (*cols))]
    #[doc = connect_doc!(DataFrame.toDF)]
    fn toDF(&self, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        // As PySpark's, of strings, which its `ToDF` plan sets when it builds the plan.
        let mut args = crate::plan_args::PlanArgs::new();
        let mut names = Vec::with_capacity(cols.len());
        for col in &cols {
            if !col.is_instance_of::<pyo3::types::PyString>() {
                return Err(not_expected_type(col, "cols", "list[str]"));
            }
            names.push(args.str(col)?);
        }
        Ok(PyDataFrame::new(args.plan(self.dataframe.to_df(names))))
    }

    // Alias this DataFrame, as PySpark's, whose `SubqueryAlias` plan sets the alias when it
    // builds the plan, as protobuf's string field takes it: bytes of UTF-8 too, and anything
    // else fails the requests of the DataFrame.
    #[doc = connect_doc!(DataFrame.alias)]
    fn alias(&self, alias: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let mut args = crate::plan_args::PlanArgs::new();
        let alias = args.str(alias)?;
        Ok(self.of_same_schema(PyDataFrame::new(args.plan(self.dataframe.alias(&alias)))))
    }

    // Group by columns for aggregation, as PySpark's `GroupedData(df=self, group_type="groupby",
    // grouping_cols=self._to_cols(*cols, allow_ordinal=True))`, of this DataFrame object.
    #[pyo3(signature = (*cols))]
    #[doc = connect_doc!(DataFrame.groupBy)]
    fn groupBy<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
        cols: Vec<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyGroupedData>> {
        let columns = slf.borrow().grouping_columns(py, cols, "cols")?;
        PyGroupedData::of_grouping(slf.as_any(), "groupby", &columns, None)
    }

    // Rollup multi-dimensional aggregation, as PySpark's `GroupedData(df=self,
    // group_type="rollup", ...)`.
    #[pyo3(signature = (*cols))]
    #[doc = connect_doc!(DataFrame.rollup)]
    fn rollup<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
        cols: Vec<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyGroupedData>> {
        let columns = slf.borrow().grouping_columns(py, cols, "cols")?;
        PyGroupedData::of_grouping(slf.as_any(), "rollup", &columns, None)
    }

    // Cube multi-dimensional aggregation, as PySpark's `GroupedData(df=self, group_type="cube",
    // ...)`.
    #[pyo3(signature = (*cols))]
    #[doc = connect_doc!(DataFrame.cube)]
    fn cube<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
        cols: Vec<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyGroupedData>> {
        let columns = slf.borrow().grouping_columns(py, cols, "cols")?;
        PyGroupedData::of_grouping(slf.as_any(), "cube", &columns, None)
    }

    // Grouping sets aggregation: `groupingSets([[a, b], [a], []], "a", "b")`.
    //
    // As PySpark's: `*cols` are the grouping columns, the columns of the result besides the
    // aggregations, of Columns, names and ordinals, as are those of the sets, of
    // `GroupedData(df=self, group_type="grouping_sets", grouping_cols=..., grouping_sets=...)`,
    // whose `_grouping_sets` are lists of the Columns of each set.
    #[pyo3(name = "groupingSets", signature = (groupingSets, *cols))]
    #[doc = connect_doc!(DataFrame.groupingSets)]
    fn grouping_sets<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
        groupingSets: &Bound<'py, PyAny>,
        cols: Vec<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyGroupedData>> {
        let this = slf.borrow();

        use pyo3::types::{PyInt, PyString};
        // As PySpark's `_to_cols(grouping_set)`: a set is a sequence of columns, or one column.
        let mut sets = Vec::new();
        for set in groupingSets.try_iter()? {
            let set = set?;
            let one = set.is_instance_of::<PyString>()
                || set.is_instance_of::<PyInt>()
                || set.is_instance_of::<PyColumn>();
            let sequence = !one
                && py
                    .import("collections.abc")?
                    .getattr("Sequence")
                    .and_then(|sequence| set.is_instance(&sequence))?;
            let items = if sequence {
                set.try_iter()?.collect::<PyResult<Vec<_>>>()?
            } else {
                vec![set]
            };
            sets.push(this.grouping_column_list(py, items, "groupingSets")?);
        }
        // As PySpark's `_to_cols(cols)`, of the columns themselves, not a list of them.
        let columns = this.grouping_column_list(py, cols, "cols")?;
        drop(this);
        PyGroupedData::of_grouping(slf.as_any(), "grouping_sets", &columns, Some(&sets[..]))
    }

    // Drop duplicates within the event-time watermark.
    #[pyo3(name = "dropDuplicatesWithinWatermark", signature = (subset=None))]
    #[doc = connect_doc!(DataFrame.dropDuplicatesWithinWatermark)]
    fn drop_duplicates_within_watermark(
        &self,
        subset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let mut args = crate::plan_args::PlanArgs::new();
        let subset = deduplicate_subset(&mut args, subset)?;
        let refs: Option<Vec<&str>> = subset
            .as_ref()
            .map(|names| names.iter().map(|s| s.as_str()).collect());
        Ok(PyDataFrame::new(args.plan(
            self.dataframe.drop_duplicates_within_watermark(refs),
        )))
    }

    // `DataFrame.toJSON`, as PySpark Connect's: a DataFrame of each row as a JSON object
    // string, in a column `value`, of `self.select(F.to_json(F.struct(F.col("*")))
    // .alias("value"))`, so its expressions have the origins of `col` and `alias`.
    #[pyo3(name = "toJSON")]
    #[doc = connect_doc!(DataFrame.toJSON)]
    fn to_json(&self) -> PyResult<PyDataFrame> {
        use spark_connect::functions::{col, r#struct, to_json};
        let star = {
            let _origin = crate::origin::with_origin("col")?;
            col("*")
        };
        let json = to_json(r#struct([star]));
        let value = {
            let _origin = crate::origin::with_origin("alias")?;
            json.alias("value")
        };
        Ok(PyDataFrame::new(self.dataframe.select([value])))
    }

    // Whether this DataFrame is semantically equal to `other`.
    #[pyo3(name = "sameSemantics")]
    #[doc = connect_doc!(DataFrame.sameSemantics)]
    fn same_semantics(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        // As PySpark's: a DataFrame, else NOT_EXPECTED_TYPE, of the same session.
        if !other.is_instance_of::<PyDataFrame>() {
            return Err(not_expected_type(other, "other", "DataFrame"));
        }
        let other_frame = self.same_session(other)?;
        let other = other_frame.dataframe.clone();
        interruptible_of(
            py,
            &self.dataframe.spark_session(),
            &[&self.parts, &other_frame.parts],
            || self.dataframe.same_semantics(&other),
        )
    }

    // The server-side hash of this DataFrame's logical plan, asked once, as PySpark's.
    #[pyo3(name = "semanticHash")]
    fn semantic_hash(&self, py: Python<'_>) -> PyResult<i32> {
        cached(&self.semantic_hash, || {
            interruptible_of(py, &self.dataframe.spark_session(), &[&self.parts], || {
                self.dataframe.semantic_hash().map_err(Box::new)
            })
        })
    }

    // Sort within each partition.
    #[pyo3(name = "sortWithinPartitions", signature = (*cols, **kwargs))]
    #[doc = connect_doc!(DataFrame.sortWithinPartitions)]
    fn sort_within_partitions(
        &self,
        _py: Python<'_>,
        cols: Vec<Bound<'_, PyAny>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        let exprs = self.sort_columns(_py, cols, kwargs.as_ref())?;
        Ok(self.of_same_schema(PyDataFrame::new(
            self.dataframe.sort_within_partitions(exprs),
        )))
    }

    // Except all (keeps duplicates).
    #[pyo3(name = "exceptAll")]
    #[doc = connect_doc!(DataFrame.exceptAll)]
    fn except_all(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let other = self.same_session(other)?;
        let dataframe = self.dataframe.except_all(&other.dataframe);
        set_operation_result(py, self, &other, dataframe)
    }

    // Rename multiple columns from an {old: new} mapping.
    #[pyo3(name = "withColumnsRenamed")]
    #[doc = connect_doc!(DataFrame.withColumnsRenamed)]
    fn with_columns_renamed(&self, colsMap: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        // As PySpark's, of a dict, whose renames it sends in order.
        let Ok(cols_map) = colsMap.cast::<PyDict>() else {
            return Err(not_expected_type(colsMap, "colsMap", "dict"));
        };
        // Its `WithColumnsRenamed` plan sets the names of each rename, the old one first, in its
        // string fields when it builds the plan, as protobuf takes them: bytes of UTF-8 too.
        let mut args = crate::plan_args::PlanArgs::new();
        let mut renames = Vec::with_capacity(cols_map.len());
        for (existing, new) in cols_map.iter() {
            let existing = args.str(&existing)?;
            renames.push((existing, args.str(&new)?));
        }
        Ok(PyDataFrame::new(
            args.plan(self.dataframe.with_columns_renamed(renames)),
        ))
    }

    // The columns of this DataFrame whose names match a regex, as a Column. Mirrors
    // `DataFrame.colRegex`.
    #[pyo3(name = "colRegex")]
    #[doc = connect_doc!(DataFrame.colRegex)]
    fn col_regex(&self, colName: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        // As PySpark's: a str, else NOT_EXPECTED_TYPE.
        if !colName.is_instance_of::<pyo3::types::PyString>() {
            return Err(not_expected_type(colName, "colName", "str"));
        }
        // Its `UnresolvedRegex` sets the name when it builds the plan.
        let mut args = crate::plan_args::PlanArgs::new();
        let name = args.str(colName)?;
        Ok(PyColumn::new(args.column(self.dataframe.col_regex(&name))))
    }

    // Basic statistics for the given columns (count/mean/stddev/min/max), of column names or
    // a list of them, as PySpark's `DataFrame.describe`.
    #[pyo3(signature = (*cols))]
    #[doc = connect_doc!(DataFrame.describe)]
    fn describe(&self, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        let mut args = crate::plan_args::PlanArgs::new();
        let cols = crate::stat_arguments::describe_columns(&mut args, &cols)?;
        Ok(PyDataFrame::new(args.plan(self.dataframe.describe(cols))))
    }

    // Summary statistics (percentiles etc.), of strings, as PySpark's `DataFrame.summary`.
    #[pyo3(signature = (*statistics))]
    #[doc = connect_doc!(DataFrame.summary)]
    fn summary(&self, statistics: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        let mut args = crate::plan_args::PlanArgs::new();
        let statistics = crate::stat_arguments::summary_statistics(&mut args, &statistics)?;
        Ok(PyDataFrame::new(
            args.plan(self.dataframe.summary(statistics)),
        ))
    }

    // Persist with the default MEMORY_AND_DISK storage level.
    #[doc = connect_doc!(DataFrame.cache)]
    fn cache<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<PyRef<'py, Self>> {
        // As PySpark's, which returns the DataFrame itself.
        let df = slf.dataframe.clone();
        interruptible_of(py, &df.spark_session(), &[&slf.parts], || {
            df.cache().map_err(Box::new)
        })?;
        Ok(slf)
    }

    // Mark the DataFrame as non-persistent. As PySpark's `unpersist(blocking=False)`, which sets
    // `blocking` in the bool field of its request unless it is None, as
    // [`crate::streaming::proto_bool`] converts it: None sends no `blocking`.
    #[pyo3(
        signature = (blocking=ProtoBool::default()),
        text_signature = "($self, blocking=False)"
    )]
    #[doc = connect_doc!(DataFrame.unpersist)]
    fn unpersist<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        blocking: ProtoBool,
    ) -> PyResult<PyRef<'py, Self>> {
        let blocking = blocking.value_unless_none(py)?;
        // As PySpark's, which returns the DataFrame itself.
        let df = slf.dataframe.clone();
        interruptible_of(py, &df.spark_session(), &[&slf.parts], || {
            df.unpersist_opt(blocking).map_err(Box::new)
        })?;
        Ok(slf)
    }

    // Print the plan with Python's `print`, as PySpark's `DataFrame.explain`, which prints
    // `_explain_string(extended, mode)`.
    #[pyo3(signature = (extended=None, mode=None))]
    #[doc = connect_doc!(DataFrame.explain)]
    fn explain(
        &self,
        py: Python<'_>,
        extended: Option<&Bound<'_, PyAny>>,
        mode: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let plan = self.explain_string(py, extended, mode)?;
        py.import("builtins")?.call_method1("print", (plan,))?;
        Ok(())
    }

    /// The plan that `explain` prints, as PySpark's `DataFrame._explain_string`: `extended` is
    /// a bool, for the "extended" or the "simple" mode, or a mode, and `mode` a mode, but not
    /// both; the mode is "simple" without either.
    #[pyo3(name = "_explain_string", signature = (extended=None, mode=None))]
    fn explain_string(
        &self,
        py: Python<'_>,
        extended: Option<&Bound<'_, PyAny>>,
        mode: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        use pyo3::types::{PyBool, PyString};
        let explain_mode: String = match (extended, mode) {
            (Some(_), Some(_)) => {
                return Err(crate::column::raise_pyspark(
                    py,
                    "PySparkValueError",
                    "CANNOT_SET_TOGETHER",
                    &[("arg_list", "extended and mode")],
                ));
            }
            (None, None) => "simple".to_string(),
            (Some(extended), None) if extended.is_instance_of::<PyBool>() => {
                if extended.extract::<bool>()? {
                    "extended"
                } else {
                    "simple"
                }
                .to_string()
            }
            // A mode that UTF-8 cannot encode is none of the modes, which the core refuses.
            (Some(arg), None) | (None, Some(arg)) if arg.is_instance_of::<PyString>() => arg
                .cast::<PyString>()?
                .to_cow()
                .map(|mode| mode.into_owned())
                .unwrap_or_default(),
            (Some(arg), None) | (None, Some(arg)) => {
                return Err(crate::column::raise_pyspark(
                    py,
                    "PySparkTypeError",
                    "NOT_EXPECTED_TYPE",
                    &[
                        ("expected_type", "bool or str"),
                        ("arg_name", "extended (optional) and mode (optional)"),
                        ("arg_type", &arg.get_type().str()?.to_string()),
                    ],
                ));
            }
        };
        interruptible_of(py, &self.dataframe.spark_session(), &[&self.parts], || {
            self.dataframe
                .explain_string(&explain_mode)
                .map_err(Box::new)
        })
    }

    // Register this DataFrame as a temporary view, of a name that PySpark's `CreateView` sets in
    // its string field as it is, as [`crate::str_argument`] converts it, as the other views do.
    #[pyo3(name = "createOrReplaceTempView")]
    #[doc = connect_doc!(DataFrame.createOrReplaceTempView)]
    fn create_or_replace_temp_view(&self, py: Python<'_>, name: &Bound<'_, PyAny>) -> PyResult<()> {
        let name = crate::str_argument(name)?;
        interruptible_of(py, &self.dataframe.spark_session(), &[&self.parts], || {
            self.dataframe
                .create_or_replace_temp_view(&name)
                .map_err(Box::new)
        })
    }

    // Drop rows containing nulls, as PySpark's `DataFrame.dropna`.
    #[pyo3(
        signature = (how=None, thresh=None, subset=None),
        text_signature = "($self, how='any', thresh=None, subset=None)"
    )]
    #[doc = connect_doc!(DataFrame.dropna)]
    fn dropna(
        &self,
        how: Option<&Bound<'_, PyAny>>,
        thresh: Option<&Bound<'_, PyAny>>,
        subset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        crate::stat_arguments::dropna(&self.dataframe, how, thresh, subset)
    }

    // Fill null values, as PySpark's `DataFrame.fillna`: `value` is a scalar, over
    // `subset`, or a {column: value} mapping.
    #[pyo3(signature = (value, subset=None))]
    #[doc = connect_doc!(DataFrame.fillna)]
    fn fillna(
        &self,
        value: &Bound<'_, PyAny>,
        subset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        crate::stat_arguments::fillna(&self.dataframe, value, subset)
    }

    // Statistical functions (`df.stat`): a new object at each access, as PySpark's
    // `DataFrameStatFunctions(self)`, of this DataFrame object, its `df`.
    #[getter]
    #[pyo3(name = "stat")]
    #[doc = connect_doc!(DataFrame.stat)]
    fn stat<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, crate::stat::PyStatFunctions>> {
        crate::stat::PyStatFunctions::of_frame(slf.as_any())
    }

    // `df.crosstab(col1, col2)` - pyspark exposes it on DataFrame as well as
    // `df.stat.crosstab`.
    #[doc = connect_doc!(DataFrame.crosstab)]
    fn crosstab(&self, col1: &Bound<'_, PyAny>, col2: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        crate::stat_arguments::crosstab(&self.dataframe.stat(), col1, col2)
    }

    // `df.freqItems(cols, support=None)` (also on df.stat).
    #[pyo3(name = "freqItems", signature = (cols, support=None))]
    #[doc = connect_doc!(DataFrame.freqItems)]
    fn freq_items(
        &self,
        cols: &Bound<'_, PyAny>,
        support: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        crate::stat_arguments::freq_items(&self.dataframe.stat(), cols, support)
    }

    // `df.approxQuantile(col, probabilities, relativeError)` (also on df.stat): the
    // quantiles of a column, or a list of them of each of a list or tuple of columns.
    #[pyo3(name = "approxQuantile")]
    #[doc = connect_doc!(DataFrame.approxQuantile)]
    fn approx_quantile(
        &self,
        py: Python<'_>,
        col: &Bound<'_, PyAny>,
        probabilities: &Bound<'_, PyAny>,
        relativeError: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        crate::stat_arguments::approx_quantile(
            py,
            &self.dataframe.stat(),
            col,
            probabilities,
            relativeError,
        )
    }

    // Observe metrics, as PySpark's `DataFrame.observe(observation, *exprs)`: the Columns
    // `exprs`, which an Observation receives by its `_on`, of a subclass too, or which are
    // observed by a name.
    #[pyo3(signature = (observation, *exprs))]
    #[doc = connect_doc!(DataFrame.observe)]
    fn observe<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
        observation: &Bound<'py, PyAny>,
        exprs: Vec<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if exprs.is_empty() {
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkValueError",
                "CANNOT_BE_EMPTY",
                &[("item", "exprs")],
            ));
        }
        let mut columns = Vec::with_capacity(exprs.len());
        for expr in &exprs {
            match expr.extract::<PyRef<'_, PyColumn>>() {
                Ok(column) => columns.push(column.column.expression().clone()),
                Err(_) => {
                    return Err(crate::column::raise_pyspark(
                        py,
                        "PySparkTypeError",
                        "NOT_EXPECTED_TYPE",
                        &[
                            ("arg_name", "exprs"),
                            ("expected_type", "list[Column]"),
                            ("arg_type", "tuple"),
                        ],
                    ));
                }
            }
        }
        if observation.is_instance_of::<crate::observation::PyObservation>() {
            // As PySpark's, which gives the DataFrame of the `_on` of the Observation, of a
            // subclass too.
            let mut arguments = vec![slf.clone().into_any()];
            arguments.extend(exprs);
            return observation.call_method1(
                pyo3::intern!(py, "_on"),
                pyo3::types::PyTuple::new(py, arguments)?,
            );
        }
        let this = slf.borrow();
        if observation.is_instance_of::<pyo3::types::PyString>() {
            // Its `CollectMetrics` plan sets the name when it builds the plan, before the metrics.
            let mut args = crate::plan_args::PlanArgs::new();
            let name = args.str(observation)?;
            let args = args.before(&mut columns);
            let observed = this.of_same_schema(PyDataFrame::new(
                args.plan(this.dataframe.observe(&name, columns)),
            ));
            return Ok(Bound::new(py, observed)?.into_any());
        }
        let type_name = observation.get_type().name()?.to_string();
        Err(crate::column::raise_pyspark(
            py,
            "PySparkTypeError",
            "NOT_EXPECTED_TYPE",
            &[
                ("arg_name", "observation"),
                ("expected_type", "Observation or str"),
                ("arg_type", &type_name),
            ],
        ))
    }

    // Register as a temporary view (session-scoped).
    #[pyo3(name = "createTempView")]
    #[doc = connect_doc!(DataFrame.createTempView)]
    fn create_temp_view(&self, py: Python<'_>, name: &Bound<'_, PyAny>) -> PyResult<()> {
        let name = crate::str_argument(name)?;
        interruptible_of(py, &self.dataframe.spark_session(), &[&self.parts], || {
            self.dataframe.create_temp_view(&name).map_err(Box::new)
        })
    }

    // Register as a global temporary view.
    #[pyo3(name = "createGlobalTempView")]
    #[doc = connect_doc!(DataFrame.createGlobalTempView)]
    fn create_global_temp_view(&self, py: Python<'_>, name: &Bound<'_, PyAny>) -> PyResult<()> {
        let name = crate::str_argument(name)?;
        interruptible_of(py, &self.dataframe.spark_session(), &[&self.parts], || {
            self.dataframe
                .create_global_temp_view(&name)
                .map_err(Box::new)
        })
    }

    // Register as (or replace) a global temporary view.
    #[pyo3(name = "createOrReplaceGlobalTempView")]
    #[doc = connect_doc!(DataFrame.createOrReplaceGlobalTempView)]
    fn create_or_replace_global_temp_view(
        &self,
        py: Python<'_>,
        name: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let name = crate::str_argument(name)?;
        interruptible_of(py, &self.dataframe.spark_session(), &[&self.parts], || {
            self.dataframe
                .create_or_replace_global_temp_view(&name)
                .map_err(Box::new)
        })
    }

    // Deprecated alias for createOrReplaceTempView, as PySpark's, which warns that it is
    // deprecated, of its line of `pyspark.sql.connect.dataframe` (see `warn_as_pyspark`), and
    // then replaces the view as `createOrReplaceTempView` does.
    #[pyo3(name = "registerTempTable")]
    #[doc = connect_doc!(DataFrame.registerTempTable)]
    fn register_temp_table(&self, py: Python<'_>, name: &Bound<'_, PyAny>) -> PyResult<()> {
        crate::errors::warn_as_pyspark(
            py,
            "pyspark.sql.connect.dataframe",
            2259,
            "Deprecated in 2.0, use createOrReplaceTempView instead.",
            &py.get_type::<pyo3::exceptions::PyFutureWarning>(),
        )?;
        self.create_or_replace_temp_view(py, name)
    }

    // Reliable checkpoint (materialize + truncate lineage), at once if `eager`, as PySpark's.
    #[pyo3(signature = (eager=DefaultedOption::Omitted), text_signature = "($self, eager=True)")]
    #[doc = connect_doc!(DataFrame.checkpoint)]
    fn checkpoint(&self, py: Python<'_>, eager: DefaultedOption) -> PyResult<PyDataFrame> {
        let eager = checkpoint_eager(py, &eager)?;
        Ok(PyDataFrame::new(interruptible_of(
            py,
            &self.dataframe.spark_session(),
            &[&self.parts],
            || self.dataframe.checkpoint_with(eager).map_err(Box::new),
        )?))
    }

    // Local checkpoint, at once if `eager`, kept at `storageLevel`, else at the server's
    // default, as PySpark's.
    #[pyo3(
        name = "localCheckpoint",
        signature = (eager=DefaultedOption::Omitted, storageLevel=None),
        text_signature = "($self, eager=True, storageLevel=None)"
    )]
    #[doc = connect_doc!(DataFrame.localCheckpoint)]
    fn local_checkpoint(
        &self,
        py: Python<'_>,
        eager: DefaultedOption,
        storageLevel: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let eager = checkpoint_eager(py, &eager)?;
        let level = storageLevel
            .filter(|level| !level.is_none())
            .map(|level| crate::storagelevel::storage_level_proto(&level))
            .transpose()?;
        Ok(PyDataFrame::new(interruptible_of(
            py,
            &self.dataframe.spark_session(),
            &[&self.parts],
            || {
                self.dataframe
                    .local_checkpoint_with(eager, level)
                    .map_err(Box::new)
            },
        )?))
    }

    // Pearson correlation between two columns; PySpark's `method` is only "pearson".
    #[pyo3(signature = (col1, col2, method=None))]
    #[doc = connect_doc!(DataFrame.corr)]
    fn corr(
        &self,
        py: Python<'_>,
        col1: &Bound<'_, PyAny>,
        col2: &Bound<'_, PyAny>,
        method: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<f64> {
        crate::stat_arguments::corr(py, &self.dataframe.stat(), col1, col2, method)
    }

    // Sample covariance between two columns.
    #[doc = connect_doc!(DataFrame.cov)]
    fn cov(
        &self,
        py: Python<'_>,
        col1: &Bound<'_, PyAny>,
        col2: &Bound<'_, PyAny>,
    ) -> PyResult<f64> {
        crate::stat_arguments::cov(py, &self.dataframe.stat(), col1, col2)
    }

    // Stratified sample without replacement per key. Without a seed, the client draws one
    // from Python's `random`, as PySpark Connect's `sampleBy` does.
    #[pyo3(name = "sampleBy", signature = (col, fractions, seed=None))]
    #[doc = connect_doc!(DataFrame.sampleBy)]
    fn sample_by(
        &self,
        col: &Bound<'_, PyAny>,
        fractions: &Bound<'_, PyAny>,
        seed: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let py = col.py();
        let (col, fractions) = crate::stat::sample_by_arguments(col, fractions)?;
        // The seed is converted as the proto's long as PySpark sends the plan, after its checks.
        let (seed, plan_error) = match crate::stat::sample_by_seed(py, seed) {
            Ok(seed) => (seed, None),
            Err(err) => (0, Some(unbuildable_error(py, err)?)),
        };
        let df = self
            .dataframe
            .stat()
            .sample_by(col, fractions, Some(seed))
            .to_pyerr()?;
        Ok(PyDataFrame::new(match plan_error {
            Some(error) => df.unbuildable(error),
            None => df,
        }))
    }

    // Replace values, as PySpark's `DataFrame.replace`: `to_replace` is a {old: new} dict, or
    // a value or a list or tuple of them with `value`, a value or a list or tuple of them.
    #[pyo3(signature = (to_replace, value=ReplaceValue::Omitted, subset=None))]
    #[doc = connect_doc!(DataFrame.replace)]
    fn replace(
        &self,
        to_replace: &Bound<'_, PyAny>,
        value: ReplaceValue,
        subset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        replace_values(&self.dataframe, to_replace, value, subset)
    }

    // Reconcile to a target schema (StructType or DDL string). As PySpark's, a schema that is
    // not a DataType is kept as it is, and the DataFrame fails at each action, which PySpark's
    // client raises when it builds the plan; a DDL string is not parsed. As PySpark's, the
    // schema is the cached schema of the DataFrame whatever it is, as the object that it is, so a
    // DataType that is not a struct, which the core does not cache, is kept as it is too, while
    // its plan is sent; and as PySpark's `ToSchema` converts the type object when it builds the
    // plan, a `StructType` changed in place after the call is the schema of the DataFrame, and
    // the schema that its requests send, as it is then.
    #[doc = connect_doc!(DataFrame.to)]
    fn to(&self, schema: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        if schema.is_none() {
            // As PySpark's `assert schema is not None`.
            return Err(pyo3::exceptions::PyAssertionError::new_err(()));
        }
        // As PySpark's `ToSchema`, which converts the schema object when it builds the plan: each
        // request reads the object as it is then (see `crate::udf_command::data_type_pickler`),
        // of the errors of PySpark's `pyspark_types_to_proto_types`, its `UNSUPPORTED_OPERATION`
        // and those of the reads of the elements of a struct whose fields hold one that is not a
        // `StructField`. The DataFrame caches the type that stands for the object at the call
        // (see `crate::types::lenient_type`), and its `schema` is the object.
        if schema.is_instance_of::<crate::types::PyDataType>() {
            let stand_in = crate::types::lenient_type(schema)?;
            let reader = crate::udf_command::data_type_pickler(schema.clone().unbind());
            let mut result = PyDataFrame::new(self.dataframe.to_read_at_build(stand_in, reader));
            result.schema_object = Some(schema.clone().unbind());
            return Ok(result);
        }
        // Of a value that is not a data type, such as a str, PySpark's conversion of the plan
        // raises its `UNSUPPORTED_OPERATION` at each request.
        let operation = crate::plan_args::data_type_operation(schema)?;
        let error = spark_connect_core::error::SparkError::classed(
            spark_connect_core::error::SparkErrorKind::ValueError,
            "UNSUPPORTED_OPERATION",
            &[("operation", &operation)],
        );
        let mut result = PyDataFrame::new(self.dataframe.unbuildable(error));
        result.schema_object = Some(schema.clone().unbind());
        Ok(result)
    }

    // Set the metadata of a column, as PySpark's: a dict of any JSON values, which it sends as
    // `json.dumps(metadata)`, and `NOT_EXPECTED_TYPE` for one that is not a dict.
    #[pyo3(name = "withMetadata")]
    #[doc = connect_doc!(DataFrame.withMetadata)]
    fn with_metadata(
        &self,
        py: Python<'_>,
        columnName: &Bound<'_, PyAny>,
        metadata: &Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        if !metadata.is_instance_of::<PyDict>() {
            return Err(not_expected_type(metadata, "metadata", "dict"));
        }
        // As PySpark's, which takes `self[columnName]` and `json.dumps(metadata)`, in this order,
        // for its `WithColumns` plan, which asserts a str name, and sets it when it builds the
        // plan. So an item that `__getitem__` does not take fails first, and an int reads the
        // columns before the assertion fails.
        self.__getitem__(py, columnName)?;
        let json: String = py
            .import("json")?
            .call_method1("dumps", (metadata,))?
            .extract()?;
        if !columnName.is_instance_of::<pyo3::types::PyString>() {
            return Err(pyo3::exceptions::PyAssertionError::new_err(()));
        }
        let mut args = crate::plan_args::PlanArgs::new();
        let name = args.str(columnName)?;
        Ok(PyDataFrame::new(
            args.plan(self.dataframe.with_metadata_json(&name, &json)),
        ))
    }

    // Define an event-time watermark, as PySpark's: `NOT_EXPECTED_TYPE` for an empty event time
    // or delay, or one that is not a str.
    #[pyo3(name = "withWatermark")]
    #[doc = connect_doc!(DataFrame.withWatermark)]
    fn with_watermark(
        &self,
        eventTime: &Bound<'_, PyAny>,
        delayThreshold: &Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        let mut args = crate::plan_args::PlanArgs::new();
        let event_time = watermark_argument(&mut args, eventTime, "eventTime")?;
        let delay_threshold = watermark_argument(&mut args, delayThreshold, "delayThreshold")?;
        Ok(PyDataFrame::new(args.plan(
            self.dataframe.with_watermark(&event_time, &delay_threshold),
        )))
    }

    // Randomly split into multiple DataFrames by the given weights.
    // As PySpark's: `VALUE_NOT_POSITIVE` for a negative weight and for weights of a sum that is
    // not positive, of their `str`; without a seed, the client chooses one of every split; a
    // seed goes through `int()`, so a float or a str is accepted.
    #[pyo3(name = "randomSplit", signature = (weights, seed=None))]
    #[doc = connect_doc!(DataFrame.randomSplit)]
    fn random_split(
        &self,
        weights: &Bound<'_, PyAny>,
        seed: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<PyDataFrame>> {
        let py = weights.py();
        let not_positive = |arg_name: &str, value: &Bound<'_, PyAny>| -> PyResult<PyErr> {
            Ok(crate::column::raise_pyspark(
                py,
                "PySparkValueError",
                "VALUE_NOT_POSITIVE",
                &[
                    ("arg_name", arg_name),
                    ("arg_value", &value.str()?.to_cow()?),
                ],
            ))
        };
        // Keep Python's checks so arg_value uses str() of the original objects, as in
        // PySpark. The core also validates the extracted weights for native Rust callers.
        for weight in weights.try_iter()? {
            let weight = weight?;
            if weight.lt(0.0)? {
                return Err(not_positive("weights", &weight)?);
            }
        }
        // Drawn before the sum is checked, as PySpark does, so that a failing sum uses one too.
        let seed = match seed {
            Some(seed) => seed.clone(),
            None => python_random_seed(py)?.into_pyobject(py)?.into_any(),
        };
        let total = py.import("builtins")?.call_method1("sum", (weights,))?;
        if total.le(0)? {
            return Err(not_positive("sum(weights)", &total)?);
        }
        // As PySpark's, which iterates the weights once more for their proportions, and adds
        // them up from 0.0 for the bounds of the splits, in Python: so any iterable is taken, as
        // a set or the keys of a dict, and a weight that does not add to a float, such as a
        // Decimal, raises the TypeError of the addition. The core computes the same bounds of
        // the weights as floats.
        let mut given = Vec::new();
        let mut proportions = Vec::new();
        for weight in weights.try_iter()? {
            let weight = weight?;
            proportions.push(weight.div(&total)?);
            given.push(weight);
        }
        let mut bound = pyo3::types::PyFloat::new(py, 0.0).into_any();
        for proportion in &proportions {
            bound = bound.add(proportion)?;
        }
        let float_weights = given
            .iter()
            .map(|weight| weight.extract::<f64>())
            .collect::<PyResult<Vec<_>>>()?;
        // Converted once the sum is checked, as PySpark's plans do with int(seed). The long of the
        // proto is converted as the plan is sent: a value out of its range fails each split.
        let seed = py.import("builtins")?.getattr("int")?.call1((&seed,))?;
        let (seed, plan_error) = match seed.extract::<i64>() {
            Ok(seed) => (seed, None),
            Err(_) => (
                0,
                Some(unbuildable_error(
                    py,
                    crate::session::value_out_of_range(&seed),
                )?),
            ),
        };
        Ok(self
            .dataframe
            .random_split(float_weights, Some(seed))
            .to_pyerr()?
            .into_iter()
            .map(|df| {
                // Of the schema that the input cached, as PySpark copies it to each split.
                self.of_same_schema(PyDataFrame::new(match &plan_error {
                    Some(error) => df.unbuildable_with_cached_schema(error.clone()),
                    None => df,
                }))
            })
            .collect())
    }

    // The list of files that make up this DataFrame, asked once, as PySpark's.
    #[pyo3(name = "inputFiles")]
    fn input_files(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        cached(&self.input_files, || {
            interruptible_of(py, &self.dataframe.spark_session(), &[&self.parts], || {
                self.dataframe.input_files().map_err(Box::new)
            })
        })
    }

    // Whether the server answers that this DataFrame is local (a local relation).
    #[pyo3(name = "isLocal")]
    fn is_local(&self, py: Python<'_>) -> PyResult<bool> {
        interruptible_of(py, &self.dataframe.spark_session(), &[&self.parts], || {
            self.dataframe.is_local_on_server().map_err(Box::new)
        })
    }

    // Whether this (cached) DataFrame is cached.
    #[pyo3(name = "is_cached")]
    #[getter]
    fn is_cached(&self, py: Python<'_>) -> PyResult<bool> {
        interruptible_of(py, &self.dataframe.spark_session(), &[&self.parts], || {
            self.dataframe.is_cached().map_err(Box::new)
        })
    }

    // Whether this DataFrame contains one or more sources that continuously return data, asked
    // once, as PySpark's.
    #[pyo3(name = "isStreaming")]
    #[getter]
    fn is_streaming(&self, py: Python<'_>) -> PyResult<bool> {
        cached(&self.is_streaming, || {
            // A `functools.cached_property` of PySpark's.
            crate::client_log::of_cached_property("isStreaming", || {
                interruptible_of(py, &self.dataframe.spark_session(), &[&self.parts], || {
                    self.dataframe.is_streaming().map_err(Box::new)
                })
            })
        })
    }

    // The storage level currently in use for this DataFrame.
    #[pyo3(name = "storageLevel")]
    #[getter]
    #[doc = connect_doc!(DataFrame.storageLevel)]
    fn storage_level(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let sl = interruptible_of(py, &self.dataframe.spark_session(), &[&self.parts], || {
            self.dataframe.storage_level().map_err(Box::new)
        })?;
        let cls = py.import("pyspark.storagelevel")?.getattr("StorageLevel")?;
        Ok(cls
            .call1((
                sl.use_disk,
                sl.use_memory,
                sl.use_off_heap,
                sl.deserialized,
                sl.replication,
            ))?
            .unbind())
    }

    // The SparkSession that created this DataFrame.
    #[pyo3(name = "sparkSession")]
    #[getter]
    #[doc = connect_doc!(DataFrame.sparkSession)]
    fn spark_session(&self, py: Python<'_>) -> PyResult<Py<crate::session::PySparkSession>> {
        crate::session::session_object(py, &self.dataframe.spark_session())
    }

    /// The SparkSession of this DataFrame, as PySpark Connect's `DataFrame._session`, which the
    /// modules of PySpark read, such as `pyspark.ml`, `pyspark.sql.plot` and `pyspark.pandas`.
    #[getter]
    fn _session(&self, py: Python<'_>) -> PyResult<Py<crate::session::PySparkSession>> {
        self.spark_session(py)
    }

    // The EXISTS subquery of this DataFrame, as a Column. Mirrors `DataFrame.exists()`.
    #[doc = connect_doc!(DataFrame.exists)]
    fn exists(&self) -> PyColumn {
        PyColumn::new(self.dataframe.exists())
    }

    // The scalar subquery of this DataFrame, as a Column. Mirrors `DataFrame.scalar()`.
    #[doc = connect_doc!(DataFrame.scalar)]
    fn scalar(&self) -> PyColumn {
        PyColumn::new(self.dataframe.scalar())
    }

    // Return a `TableArg` wrapping this DataFrame, for use as a table-valued function
    // argument. Mirrors `DataFrame.asTable()`.
    #[pyo3(name = "asTable")]
    #[doc = connect_doc!(DataFrame.asTable)]
    fn as_table(&self) -> crate::tablearg::PyTableArg {
        crate::tablearg::PyTableArg::new(
            spark_connect::table_arg::TableArg::new(self.dataframe.clone()),
            self.dataframe.spark_session().session_id().to_string(),
        )
    }

    // Deprecated alias for dropDuplicates.
    #[pyo3(name = "drop_duplicates", signature = (subset=None))]
    #[doc = connect_doc!(DataFrame.drop_duplicates)]
    fn drop_duplicates(&self, subset: Option<&Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        self.dropDuplicates(subset)
    }

    // Alias for groupBy.
    #[pyo3(signature = (*cols))]
    #[doc = connect_doc!(DataFrame.groupby)]
    fn groupby<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
        cols: Vec<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyGroupedData>> {
        Self::groupBy(slf, py, cols)
    }

    // Apply a function to this DataFrame and return its result:
    // `df.transform(func, *args, **kwargs)` == `func(df, *args, **kwargs)`.
    #[pyo3(signature = (func, *args, **kwargs))]
    #[doc = connect_doc!(DataFrame.transform)]
    fn transform(
        slf: &Bound<'_, Self>,
        func: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        // As PySpark's, which passes `self`: a subclass of DataFrame keeps its methods and type.
        let py = slf.py();
        let mut call_args: Vec<Py<PyAny>> = vec![slf.clone().into_any().unbind()];
        for a in args.iter() {
            call_args.push(a.unbind());
        }
        let tuple = PyTuple::new(py, call_args)?;
        let result = func.call(tuple, kwargs)?;
        // As PySpark's assert that the function returned a DataFrame.
        if !result.is_instance_of::<PyDataFrame>() {
            return Err(pyo3::exceptions::PyAssertionError::new_err(format!(
                "Func returned an instance of type [{}], should have been DataFrame.",
                result.get_type().repr()?
            )));
        }
        Ok(result.unbind())
    }

    // Unpivot (wide-to-long), as PySpark's `DataFrame.unpivot`: `ids` and `values` are each a
    // column or a column name, or a list or tuple of them, and `values=None` unpivots all
    // non-id columns.
    #[pyo3(signature = (ids, values, variableColumnName, valueColumnName))]
    #[doc = connect_doc!(DataFrame.unpivot)]
    fn unpivot(
        &self,
        ids: &Bound<'_, PyAny>,
        values: Option<&Bound<'_, PyAny>>,
        variableColumnName: &Bound<'_, PyAny>,
        valueColumnName: &Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        if ids.is_none() {
            return Err(pyo3::exceptions::PyAssertionError::new_err(
                "ids must not be None",
            ));
        }
        let ids = unpivot_columns(ids)?;
        let values = values.map(unpivot_columns).transpose()?;
        // The names are of the string fields of its `Unpivot` plan, which sets them as they
        // are when it builds the plan, after the columns.
        let mut args = crate::plan_args::PlanArgs::new();
        let variable = args.str(variableColumnName)?;
        let value = args.str(valueColumnName)?;
        Ok(PyDataFrame::new(args.plan(
            self.dataframe.unpivot(ids, values, &variable, &value),
        )))
    }

    // The same function as `unpivot`, as PySpark's `DataFrame.melt`.
    #[pyo3(signature = (ids, values, variableColumnName, valueColumnName))]
    #[doc = connect_doc!(DataFrame.melt)]
    fn melt(
        &self,
        ids: &Bound<'_, PyAny>,
        values: Option<&Bound<'_, PyAny>>,
        variableColumnName: &Bound<'_, PyAny>,
        valueColumnName: &Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        self.unpivot(ids, values, variableColumnName, valueColumnName)
    }

    // Persist the DataFrame at `storageLevel`, as PySpark's `DataFrame.persist()`:
    // MEMORY_AND_DISK_DESER when it is omitted, and no storage level for None, so that the
    // server caches at its `spark.sql.defaultCacheStorageLevel`.
    #[pyo3(signature = (storageLevel=crate::storagelevel::PersistLevel::Omitted))]
    #[doc = connect_doc!(DataFrame.persist)]
    fn persist<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        storageLevel: crate::storagelevel::PersistLevel,
    ) -> PyResult<PyRef<'py, Self>> {
        let level = storageLevel.proto(py)?;
        // As PySpark's, which returns the DataFrame itself.
        let df = slf.dataframe.clone();
        interruptible_of(py, &df.spark_session(), &[&slf.parts], || {
            df.persist(level).map_err(Box::new)
        })?;
        Ok(slf)
    }

    /// Column access by attribute (`df.colname` -> Column), by this DataFrame's plan ID.
    /// JVM attribute names raise JVM_ATTRIBUTE_NOT_SUPPORTED. Dunder names that are not
    /// columns raise ATTRIBUTE_NOT_SUPPORTED; with PYSPARK_VALIDATE_COLUMN_NAME_LEGACY=1,
    /// all non-column names do. Single-underscore names otherwise resolve to columns.
    fn __getattr__(
        &self,
        py: Python<'_>,
        name: &Bound<'_, pyo3::types::PyString>,
    ) -> PyResult<PyColumn> {
        // A name that UTF-8 cannot encode is none of those below, nor a column of the server,
        // which the validation reads: it is the column of the name, which fails its requests, as
        // PySpark's `_col`, unless it is validated.
        let Ok(text) = name.to_cow() else {
            let legacy = std::env::var("PYSPARK_VALIDATE_COLUMN_NAME_LEGACY").as_deref() == Ok("1");
            if legacy || name.call_method1("startswith", ("__",))?.is_truthy()? {
                self.column_names(py)?;
                return Err(crate::errors::raise_pyspark_of(
                    py,
                    "PySparkAttributeError",
                    "ATTRIBUTE_NOT_SUPPORTED",
                    &[("attr_name", name.as_any())],
                ));
            }
            return Ok(PyColumn::new(crate::plan_args::df_col(
                &self.dataframe,
                name.as_any(),
                false,
            )?));
        };
        let name = text.as_ref();
        // As PySpark Connect's: the attributes of the JVM of a classic DataFrame are not
        // supported, and a name is validated as a column only for a dunder name, which Python
        // may look up for an attribute, or with PYSPARK_VALIDATE_COLUMN_NAME_LEGACY.
        if ["_jseq", "_jdf", "_jmap", "_jcols", "rdd"].contains(&name) {
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkAttributeError",
                "JVM_ATTRIBUTE_NOT_SUPPORTED",
                &[("attr_name", name)],
            ));
        }
        let legacy = std::env::var("PYSPARK_VALIDATE_COLUMN_NAME_LEGACY").as_deref() == Ok("1");
        if (legacy || name.starts_with("__")) && !self.column_names(py)?.iter().any(|c| c == name) {
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkAttributeError",
                "ATTRIBUTE_NOT_SUPPORTED",
                &[("attr_name", name)],
            ));
        }
        // As PySpark's `self._col(name)`, of the name `*` too.
        Ok(PyColumn::new(self.dataframe.column_reference(name)))
    }

    /// Private bridge used by pyspark.sql.dataframe.DataFrame.__getattr__ for
    /// explicit column lookup. The __getattr__ special method itself is exposed
    /// only as a slot; this ordinary method remains callable by the public base.
    fn _get_column_by_attribute(
        &self,
        py: Python<'_>,
        name: &Bound<'_, pyo3::types::PyString>,
    ) -> PyResult<PyColumn> {
        self.__getattr__(py, name)
    }

    /// Class attributes and all column names, matching Python Connect, of its `self.columns`,
    /// which an AttributeError of the property makes a Column, which raises `NOT_ITERABLE`.
    fn __dir__(slf: &Bound<'_, Self>) -> PyResult<Vec<String>> {
        let py = slf.py();
        // Inspect the class rather than the instance to avoid calling __dir__ again.
        let mut attrs = py.get_type::<Self>().dir()?.extract::<Vec<String>>()?;
        for name in slf.getattr(pyo3::intern!(py, "columns"))?.try_iter()? {
            attrs.push(name?.extract()?);
        }
        attrs.sort();
        attrs.dedup();
        Ok(attrs)
    }

    /// Bracket completion keeps column order, duplicates and non-identifiers, as PySpark's
    /// `self.columns`.
    fn _ipython_key_completions_<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        slf.getattr(pyo3::intern!(slf.py(), "columns"))
    }

    // Aggregate over the whole DataFrame (shorthand for `groupBy().agg(...)`).
    //
    // Mirrors `pyspark.sql.DataFrame.agg(*exprs)`.
    #[pyo3(signature = (*exprs))]
    #[doc = connect_doc!(DataFrame.agg)]
    fn agg(&self, py: Python<'_>, exprs: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        if exprs.is_empty() {
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkValueError",
                "CANNOT_BE_EMPTY",
                &[("item", "exprs")],
            ));
        }
        // As PySpark's: a dict of column names and functions, the aggregation of each, as of
        // `groupBy().agg`, else Columns, which it asserts.
        if let [only] = exprs.as_slice() {
            if let Ok(dict) = only.cast::<PyDict>() {
                // As PySpark's, whose `groupBy().agg()` of none of the entries asserts that
                // it has expressions.
                if dict.is_empty() {
                    return Err(pyo3::exceptions::PyAssertionError::new_err(
                        "exprs should not be empty",
                    ));
                }
                let expressions = crate::group::dict_aggregates(dict)?;
                return Ok(PyDataFrame::new(self.dataframe.agg(expressions)));
            }
        }
        let mut expressions = Vec::with_capacity(exprs.len());
        for expr in &exprs {
            let Ok(column) = expr.extract::<PyRef<'_, PyColumn>>() else {
                return Err(pyo3::exceptions::PyAssertionError::new_err(
                    "all exprs should be Expression",
                ));
            };
            expressions.push(column.column.expression().clone());
        }
        Ok(PyDataFrame::new(self.dataframe.agg(expressions)))
    }

    // Collect all rows into memory.
    #[doc = connect_doc!(DataFrame.collect)]
    fn collect(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        // Release the GIL across the blocking RPC so other Python threads run.
        let (rows, schema) =
            interruptible_of(py, &self.dataframe.spark_session(), &[&self.parts], || {
                self.dataframe
                    .collect_with_schema_checked(check_schema_at_receipt)
                    .map_err(Box::new)
            })?;
        rows_to_py(py, &self.dataframe, rows, schema.as_ref())
    }

    // Return an iterator over rows without materializing the entire result in memory.
    //
    // Mirrors `pyspark.sql.DataFrame.toLocalIterator(prefetchPartitions=False)`, a generator:
    // the execution starts with the first row, and its errors are raised by the iteration.
    // Yields Row objects lazily as batches are fetched from the server. As PySpark Connect's,
    // which ignores `prefetchPartitions`, a batch is fetched when the iteration reaches it, so
    // the progress handlers of the execution are called by `next()`, on the iterating thread.
    #[pyo3(name = "toLocalIterator")]
    #[pyo3(
        signature = (prefetchPartitions=None),
        text_signature = "($self, prefetchPartitions=False)"
    )]
    #[doc = connect_doc!(DataFrame.toLocalIterator)]
    fn to_local_iterator<'py>(
        slf: &Bound<'py, Self>,
        prefetchPartitions: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Any value, as PySpark Connect's, which ignores it.
        let _ = prefetchPartitions;
        let py = slf.py();
        // The rows hold this DataFrame, as PySpark's generator holds `self` until it ends.
        let rows = PyLocalRowIterator::new(slf);
        py.import("pyspark.sql.connect.dataframe")?
            .getattr("_local_rows")?
            .call1((rows,))
    }

    /// The `pyarrow.Table` of the result as the server sends it, uncast, and the schema that the
    /// server sends with it, a `StructType`, or None: PySpark Connect's private `_to_table`, of
    /// which its `toArrow` is made. A result without batches raises the `AssertionError` of its
    /// `assert table is not None`; a Spark server sends at least one batch. The metadata of the
    /// schema and of the fields of the table are those that PySpark reads of the server's stream
    /// (`pyspark.sql.connect.client.core._table_as_received`).
    fn _to_table(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let (batches, schema) =
            interruptible_of(py, &self.dataframe.spark_session(), &[&self.parts], || {
                self.dataframe
                    .collect_record_batches_with_schema_checked(check_schema_at_receipt)
                    .map_err(Box::new)
            })?;
        if batches.is_empty() {
            return Err(pyo3::exceptions::PyAssertionError::new_err(()));
        }
        let table = crate::pyarrow_export::record_batches_to_pyarrow(py, batches)?;
        let table = py
            .import("pyspark.sql.connect.client.core")?
            .call_method1("_table_as_received", (table,))?
            .unbind();
        let schema = match &schema {
            Some(schema) => crate::types::schema_to_py(py, schema)?,
            None => py.None(),
        };
        Ok((table, schema))
    }

    // Collect the DataFrame into a `pyarrow.Table`.
    //
    // Mirrors PySpark Connect's `toArrow()`: the Arrow schema of the `schema` of the DataFrame
    // first, `to_arrow_schema(self.schema, error_on_duplicated_field_names_in_struct=True,
    // timezone="UTC")`, then the table of the result, cast to it. The schema is that of the
    // analysis of the plan, which the DataFrame keeps, so `columns` asks nothing afterwards; a
    // type that PySpark does not convert to Arrow, and duplicate names of the fields of a
    // nested struct, raise before the query runs, so before any error of its execution.
    //
    // The result is decoded from the server's Arrow IPC stream and handed to `pyarrow` through
    // arrow-rs's reader conversion, sharing compatible decoded Arrow value buffers. Bitmap
    // alignment and schemas requiring the IPC compatibility fallback can still copy.
    #[pyo3(name = "toArrow")]
    #[doc = connect_doc!(DataFrame.toArrow)]
    fn to_arrow(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let kwargs = PyDict::new(py);
        kwargs.set_item("error_on_duplicated_field_names_in_struct", true)?;
        kwargs.set_item("timezone", "UTC")?;
        // Of `self.schema`, as PySpark's (see `property_schema`).
        let arrow_schema = py
            .import("pyspark.sql.pandas.types")?
            .getattr("to_arrow_schema")?
            .call((slf.getattr(pyo3::intern!(py, "schema"))?,), Some(&kwargs))?;
        let this = slf.borrow();
        let dataframe = &this.dataframe;
        let (batches, _) =
            interruptible_of(py, &dataframe.spark_session(), &[&this.parts], || {
                dataframe
                    .collect_record_batches_with_schema_checked(check_schema_at_receipt)
                    .map_err(Box::new)
            })?;
        let table = crate::pyarrow_export::record_batches_to_pyarrow(py, batches)?;
        let table = table.bind(py);
        // A result without batches has no columns: the empty table of the schema.
        if table.getattr("num_columns")?.extract::<usize>()? == 0 && arrow_schema.len()? > 0 {
            return Ok(arrow_schema.call_method0("empty_table")?.unbind());
        }
        Ok(table.call_method1("cast", (arrow_schema,))?.unbind())
    }

    /// The Arrow C stream of the DataFrame, of the Arrow PyCapsule interface, as PySpark's
    /// `__arrow_c_stream__`: `pyspark.sql.interchange.SparkArrowCStreamer`, which reads the rows of
    /// one partition at a time, of the Arrow schema of the schema, and `requested_schema`, a
    /// PyCapsule of a schema, if any, which the stream is cast to as far as Arrow can.
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__(
        slf: &Bound<'_, Self>,
        requested_schema: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        Ok(py
            .import("pyspark.sql.interchange")?
            .getattr("SparkArrowCStreamer")?
            .call1((slf,))?
            .call_method1("__arrow_c_stream__", (requested_schema,))?
            .unbind())
    }

    // Collect the DataFrame into a pandas DataFrame.
    //
    // Mirrors `pyspark.sql.DataFrame.toPandas()`, as `SparkConnectClient.to_pandas` converts
    // the Arrow result: each column by its Spark type, in the session time zone.
    #[doc = connect_doc!(DataFrame.toPandas)]
    fn toPandas(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.to_pandas(py, None)
    }

    /// Internal Arrow-based pandas conversion used by pandas-on-Spark
    /// (`pyspark.sql.connect.dataframe.DataFrame._to_pandas`): `toPandas`, with
    /// `pandasStructHandlingMode` overriding `spark.sql.execution.pandas.structHandlingMode`,
    /// as in `SparkConnectClient.to_pandas`.
    #[pyo3(signature = (**kwargs))]
    fn _to_pandas(
        &self,
        py: Python<'_>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        crate::options::check_keyword_keys(kwargs)?;
        let mode = match kwargs {
            Some(kwargs) => kwargs.get_item("pandasStructHandlingMode")?,
            None => None,
        };
        self.to_pandas(py, mode)
    }

    // Get the count of rows.
    #[doc = connect_doc!(DataFrame.count)]
    fn count(&self, py: Python<'_>) -> PyResult<i64> {
        interruptible_of(py, &self.dataframe.spark_session(), &[&self.parts], || {
            self.dataframe.count().map_err(Box::new)
        })
    }

    // Print the first `n` rows as the server formats them. Mirrors `DataFrame.show`, which
    // prints the table of `_show_string` with Python's `print`.
    #[pyo3(signature = (*args, **kwargs), text_signature = "($self, n=20, truncate=True, vertical=False)")]
    #[doc = connect_doc!(DataFrame.show)]
    fn show(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let table = self.show_table(py, "DataFrame.show", args, kwargs)?;
        py.import("builtins")?.call_method1("print", (table,))?;
        Ok(())
    }

    /// The table that `show` prints: the first `n` rows (20 by default), values truncated
    /// to 20 characters for `truncate=True`, to `int(truncate)` otherwise (0 for none), and
    /// vertical with a line per value. Mirrors `DataFrame._show_string`, with its errors,
    /// for any value given, None included.
    #[pyo3(name = "_show_string", signature = (*args, **kwargs), text_signature = "($self, n=20, truncate=True, vertical=False)")]
    fn show_string(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<String> {
        self.show_table(py, "DataFrame._show_string", args, kwargs)
    }

    // Execution info from the most recent action, as a `pyspark.sql.metrics.ExecutionInfo`
    // (plan metrics, observed-metric flows, and the ExecutePlan `operation_id`). Mirrors
    // `DataFrame.executionInfo`; `None` before any action ran. As PySpark's, which returns the
    // `_execution_info` that the last execution of the DataFrame set, it is one object until
    // another execution replaces it: the one that the first access after an execution makes.
    #[getter]
    #[pyo3(name = "executionInfo")]
    #[doc = connect_doc!(DataFrame.executionInfo)]
    fn execution_info(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // A copy of the observed metrics, of up to `spark_connect_proto::MAX_KEPT_DEPTH` levels.
        let core = crate::on_grown_stack(|| self.dataframe.execution_info()).ok();
        // The information of a write of the DataFrame, of a writer's callback, until a later
        // execution of the DataFrame replaces it.
        let written = {
            let mut written = self
                .written_execution_info
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let current = match (written.as_ref(), core.as_ref()) {
                (Some((None, _)), None) => true,
                (Some((Some(at), _)), Some(now)) => same_execution(at, now),
                _ => false,
            };
            if current {
                return Ok(written
                    .as_ref()
                    .expect("a written execution")
                    .1
                    .clone_ref(py));
            }
            written.take()
        };
        // Released after the lock, as a finalizer of the replaced object may run Python code.
        drop(written);
        let Some(info) = core else {
            return Ok(py.None());
        };
        // The lock is not held while Python code runs, which may read this property again.
        let kept_of = |info: &spark_connect::session::ExecutionInfo| {
            let kept = self
                .execution_info_object
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            match kept.as_ref() {
                Some((of, object)) if same_execution(of, info) => Some(object.clone_ref(py)),
                _ => None,
            }
        };
        if let Some(object) = kept_of(&info) {
            return Ok(object);
        }
        let metrics_mod = py.import("pyspark.sql.metrics")?;
        let (plan_metrics, observed) = execution_metrics(py, &info)?;
        let ei = metrics_mod
            .getattr("ExecutionInfo")?
            .call1((plan_metrics, observed, info.operation_id.clone()))?
            .unbind();
        // Another thread may have made one of the same execution meanwhile, which is kept.
        if let Some(object) = kept_of(&info) {
            return Ok(object);
        }
        let replaced = self
            .execution_info_object
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .replace((info, ei.clone_ref(py)));
        // Released after the lock, as a finalizer of the replaced object may run Python code.
        drop(replaced);
        Ok(ei)
    }

    // Convert to a pandas-on-Spark DataFrame, as PySpark's `DataFrame.pandas_api`: of the
    // `InternalFrame` of this DataFrame, of the index of the columns of `index_col`
    // (`pyspark.pandas.namespace._get_index_map`), of the vendored `pyspark.pandas`.
    #[pyo3(name = "pandas_api")]
    #[pyo3(signature = (index_col=None))]
    #[doc = connect_doc!(DataFrame.pandas_api)]
    fn pandas_api(
        slf: Bound<'_, Self>,
        py: Python<'_>,
        index_col: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let (index_spark_columns, index_names) = py
            .import("pyspark.pandas.namespace")?
            .getattr("_get_index_map")?
            .call1((&slf, index_col))?
            .extract::<(Bound<'_, PyAny>, Bound<'_, PyAny>)>()?;
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("spark_frame", &slf)?;
        kwargs.set_item("index_spark_columns", index_spark_columns)?;
        kwargs.set_item("index_names", index_names)?;
        let internal = py
            .import("pyspark.pandas.internal")?
            .getattr("InternalFrame")?
            .call((), Some(&kwargs))?;
        Ok(py
            .import("pyspark.pandas.frame")?
            .getattr("DataFrame")?
            .call1((internal,))?
            .unbind())
    }

    // The plot accessor of the DataFrame, as PySpark's `DataFrame.plot`: the
    // `PySparkPlotAccessor` of `pyspark.sql.plot`, of this DataFrame.
    #[getter]
    #[doc = connect_doc!(DataFrame.plot)]
    fn plot(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(py
            .import("pyspark.sql.plot")?
            .getattr("PySparkPlotAccessor")?
            .call1((slf,))?
            .unbind())
    }

    // The schema of this DataFrame as a `StructType` (a property, mirroring
    // `DataFrame.schema`) — materialized into the proper Python type classes so
    // `df.schema.fields`, `df.schema["c"]`, `df.schema.fieldNames()` all work. Of a schema
    // that is an object, a deep copy of it, as PySpark's (see `property_schema`).
    #[getter]
    #[doc = connect_doc!(DataFrame.schema)]
    fn schema(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        // As PySpark's loads of `_cached_schema_serialized`, of the schema that the copy or the
        // unpickle of which this DataFrame is made requested, which the server gave: of a new
        // object of each user-defined type (see `schema_to_py`).
        if let Some(schema) = slf.borrow().copy_schema() {
            check_schema_conversion(slf.py(), Some(&schema))?;
            return crate::types::schema_to_py(slf.py(), &schema);
        }
        Self::property_schema(slf)?.schema(slf.py())
    }

    // Get the first row.
    #[doc = connect_doc!(DataFrame.first)]
    fn first(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let (row, schema) =
            interruptible_of(py, &self.dataframe.spark_session(), &[&self.parts], || {
                self.dataframe
                    .first_with_schema_checked(check_schema_at_receipt)
                    .map_err(Box::new)
            })?;
        Ok(rows_to_py(
            py,
            &self.dataframe,
            row.into_iter().collect(),
            schema.as_ref(),
        )?
        .pop())
    }

    // The first row, or None, or with `n`, a list of the first `n` rows, as PySpark's `head`,
    // whose `head(n)` is `take(n)`.
    #[pyo3(signature = (n=None))]
    #[doc = connect_doc!(DataFrame.head)]
    fn head(&self, py: Python<'_>, n: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        use pyo3::IntoPyObjectExt;
        match n {
            None => self.first(py)?.into_py_any(py),
            Some(n) => self.take(py, n)?.into_py_any(py),
        }
    }

    // Get the first n rows, as PySpark's `DataFrame.take`, which is `limit(num).collect()`: `num`
    // is converted as `limit` converts it, and the server checks the rest.
    #[doc = connect_doc!(DataFrame.take)]
    fn take(&self, py: Python<'_>, num: &Bound<'_, PyAny>) -> PyResult<Vec<Py<PyAny>>> {
        let num = proto_int(num, "spark.connect.Limit.limit")?;
        let (rows, schema) =
            interruptible_of(py, &self.dataframe.spark_session(), &[&self.parts], || {
                self.dataframe
                    .limit(num)
                    .collect_with_schema_checked(check_schema_at_receipt)
                    .map_err(Box::new)
            })?;
        rows_to_py(py, &self.dataframe, rows, schema.as_ref())
    }

    // Column names (a property, mirroring `DataFrame.columns`). A schema that is an object, of
    // `to` that is not a DataType or of a plan that fails to build (see `property_schema`), gives
    // the names of its fields as it is, as PySpark's `[field.name for field in
    // self._schema.fields]`: an AttributeError of it falls back to a Column, as in PySpark, and
    // a Column raises `NOT_ITERABLE`.
    #[getter]
    #[doc = connect_doc!(DataFrame.columns)]
    fn columns(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        Ok(Self::property_schema(slf)?.columns(py)?.into_any().unbind())
    }

    // Get the last `num` rows as a list of Rows (eager, like pyspark `tail`), of `num` as the
    // proto's int32 of the plan, converted as `limit` converts it.
    #[doc = connect_doc!(DataFrame.tail)]
    fn tail(&self, py: Python<'_>, num: &Bound<'_, PyAny>) -> PyResult<Vec<Py<PyAny>>> {
        let num = proto_int(num, "spark.connect.Tail.limit")?;
        let (rows, schema) =
            interruptible_of(py, &self.dataframe.spark_session(), &[&self.parts], || {
                self.dataframe
                    .tail(num)
                    .collect_with_schema_checked(check_schema_at_receipt)
                    .map_err(Box::new)
            })?;
        rows_to_py(py, &self.dataframe, rows, schema.as_ref())
    }

    // Sample a fraction of rows, as PySpark's `DataFrame.sample(withReplacement=None,
    // fraction=None, seed=None)`, which takes `sample(fraction[, seed])` positionally as
    // well: a bool and a float, a float, or a float and the seed.
    #[pyo3(signature = (withReplacement=None, fraction=None, seed=None))]
    #[doc = connect_doc!(DataFrame.sample)]
    fn sample(
        &self,
        py: Python<'_>,
        withReplacement: Option<&Bound<'_, PyAny>>,
        fraction: Option<&Bound<'_, PyAny>>,
        seed: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        use pyo3::types::{PyBool, PyFloat};
        // The seed is `int(seed)`, as PySpark takes it, which fails at the call; the long of the
        // proto is converted as the plan is sent, so that its error is raised then.
        let int = |value: &Bound<'_, PyAny>| -> PyResult<Result<i64, PyErr>> {
            let int = py.import("builtins")?.getattr("int")?.call1((value,))?;
            Ok(int
                .extract::<i64>()
                .map_err(|_| crate::session::value_out_of_range(&int)))
        };
        let (replace, frac, seed) = match (withReplacement, fraction) {
            (Some(w), Some(f)) if w.is_instance_of::<PyBool>() && f.is_instance_of::<PyFloat>() => {
                (w.extract()?, f.extract()?, seed.map(int).transpose()?)
            }
            (None, Some(f)) if f.is_instance_of::<PyFloat>() => {
                (false, f.extract()?, seed.map(int).transpose()?)
            }
            // `sample(fraction[, seed])`: the fraction is the first argument, and the seed the
            // second.
            (Some(w), f) if w.is_instance_of::<PyFloat>() => {
                (false, w.extract()?, f.map(int).transpose()?)
            }
            _ => {
                let type_name = |value: Option<&Bound<'_, PyAny>>| -> PyResult<String> {
                    Ok(match value {
                        Some(value) => value.get_type().name()?.to_string(),
                        None => "NoneType".to_string(),
                    })
                };
                let arg_types = [withReplacement, fraction, seed]
                    .into_iter()
                    .map(type_name)
                    .collect::<PyResult<Vec<_>>>()?
                    .join(", ");
                return Err(crate::column::raise_pyspark(
                    py,
                    "PySparkTypeError",
                    "NOT_EXPECTED_TYPE",
                    &[
                        ("expected_type", "bool, float or int"),
                        (
                            "arg_name",
                            "withReplacement (optional), fraction (required) and seed (optional)",
                        ),
                        ("arg_type", &arg_types),
                    ],
                ));
            }
        };
        let (seed, plan_error) = match seed {
            Some(Ok(seed)) => (seed, None),
            Some(Err(err)) => (0, Some(unbuildable_error(py, err)?)),
            None => (python_random_seed(py)?, None),
        };
        let df = self.dataframe.sample_opt(frac, replace, Some(seed));
        // Of the schema that the input cached, as PySpark copies it.
        Ok(self.of_same_schema(PyDataFrame::new(match plan_error {
            Some(error) => df.unbuildable_with_cached_schema(error),
            None => df,
        })))
    }

    // Check if the DataFrame is empty, asked once, as PySpark's.
    fn isEmpty(&self, py: Python<'_>) -> PyResult<bool> {
        cached(&self.is_empty, || {
            let empty =
                interruptible_of(py, &self.dataframe.spark_session(), &[&self.parts], || {
                    self.dataframe.is_empty().map_err(Box::new)
                })?;
            // As PySpark's `select().take(1)`, whose `collect` reads the binary setting once the
            // rows are fetched, though they have no columns.
            read_binary_as_bytes(py, &self.dataframe)?;
            Ok(empty)
        })
    }

    // Print the schema through Python stdout, honoring the maximum nesting depth.
    #[pyo3(signature = (level=None))]
    #[doc = connect_doc!(DataFrame.printSchema)]
    fn printSchema(slf: &Bound<'_, Self>, level: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let py = slf.py();
        // Reference's `if level:` selects the default for false values before
        // resolving the schema; positive fractional depths behave as ceil(level).
        let level = match level {
            Some(level) if level.is_truthy()? => Some(level),
            _ => None,
        };
        // `self.schema` is the schema that a copy or an unpickle requested, if any (see
        // `copy_schema`), else that of `property_schema`.
        let property = match slf.borrow().copy_schema() {
            Some(schema) => {
                check_schema_conversion(py, Some(&schema))?;
                PropertySchema::Core(schema)
            }
            None => Self::property_schema(slf)?,
        };
        let tree = match property {
            // A schema that is an object (see `property_schema`) is printed as PySpark's
            // `print(self.schema.treeString(level))`, of a copy of it, whose `treeString` raises
            // its own AttributeError, and that of a Column is a Column, whose call raises.
            schema @ PropertySchema::Object(_) => {
                let schema = schema.schema(py)?.into_bound(py);
                match level {
                    Some(level) => schema.call_method1("treeString", (level,))?,
                    None => schema.call_method0("treeString")?,
                }
            }
            PropertySchema::Core(schema) => {
                let depth = level
                    .map(crate::types::normalize_tree_depth)
                    .transpose()?
                    .unwrap_or(i32::MAX);
                schema
                    .tree_string_with_depth(depth)
                    .to_pyerr()?
                    .into_pyobject(py)?
                    .into_any()
            }
        };
        py.import("builtins")?.getattr("print")?.call1((tree,))?;
        Ok(())
    }

    // Get the schema as a list of (name, type) tuples. A schema that is an object (see
    // `property_schema`) gives them as it is, as PySpark's `dtypes` reads its fields.
    #[getter]
    #[doc = connect_doc!(DataFrame.dtypes)]
    fn dtypes(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        Ok(Self::property_schema(slf)?.dtypes(py)?.into_any().unbind())
    }

    // Additional set operation: unionAll (alias for union).
    #[doc = connect_doc!(DataFrame.unionAll)]
    fn unionAll(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let other = self.same_session(other)?;
        let dataframe = self.dataframe.union(&other.dataframe);
        set_operation_result(py, self, &other, dataframe)
    }

    // The V1 writer interface (`df.write`): PySpark Connect's `DataFrameWriter(self._plan,
    // self._session, cb)`, whose callback sets the execution information of this DataFrame.
    #[getter]
    #[doc = connect_doc!(DataFrame.write)]
    fn write<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let module = py.import("pyspark.sql.connect.readwriter")?;
        let callback = module.call_method1("_execution_info_callback", (slf,))?;
        let plan = slf.getattr(pyo3::intern!(py, "_plan"))?;
        let session = slf.getattr(pyo3::intern!(py, "_session"))?;
        module
            .getattr(pyo3::intern!(py, "DataFrameWriter"))?
            .call1((plan, session, callback))
    }

    // The V2 writer interface (`df.writeTo("table")`): PySpark Connect's
    // `DataFrameWriterV2(self._plan, self._session, table, cb)`.
    #[pyo3(name = "writeTo")]
    #[doc = connect_doc!(DataFrame.writeTo)]
    fn write_to<'py>(
        slf: &Bound<'py, Self>,
        table: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let module = py.import("pyspark.sql.connect.readwriter")?;
        let callback = module.call_method1("_execution_info_callback", (slf,))?;
        let plan = slf.getattr(pyo3::intern!(py, "_plan"))?;
        let session = slf.getattr(pyo3::intern!(py, "_session"))?;
        module
            .getattr(pyo3::intern!(py, "DataFrameWriterV2"))?
            .call1((plan, session, table, callback))
    }

    /// Set the execution information of the DataFrame, as the callback of PySpark's writers of it
    /// sets its `_execution_info`: the `ExecutionInfo` of the command of a write, which
    /// `executionInfo` gives until a later execution of the DataFrame replaces it.
    fn _set_execution_info(&self, py: Python<'_>, info: Py<PyAny>) {
        let at = crate::on_grown_stack(|| self.dataframe.execution_info()).ok();
        let replaced = self
            .written_execution_info
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .replace((at, info));
        // Released after the lock, as a finalizer of the replaced object may run Python code.
        drop(replaced);
        let _ = py;
    }

    // The streaming writer interface (`df.writeStream`).
    #[getter]
    #[pyo3(name = "writeStream")]
    #[doc = connect_doc!(DataFrame.writeStream)]
    fn write_stream(slf: &Bound<'_, Self>) -> PyDataStreamWriter {
        let writer = slf.borrow().dataframe.write_stream();
        PyDataStreamWriter::new(writer)
    }

    // Null-handling functions (`df.na.drop()/fill()/replace()`): a new object at each access,
    // as PySpark's `DataFrameNaFunctions(self)`, of this DataFrame object, its `df`.
    #[getter]
    #[doc = connect_doc!(DataFrame.na)]
    fn na<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDataFrameNaFunctions>> {
        PyDataFrameNaFunctions::of_frame(slf.as_any())
    }

    // Merge-into (`df.mergeInto("table", cond)`), returning a V2 merge builder, of the table as
    // [`written_table`] takes it, and of the condition as PySpark's builder keeps it, which the
    // merge builds ([`merge_condition`]). The builder keeps this DataFrame, as PySpark's keeps it
    // in the callback of its execution.
    #[pyo3(name = "mergeInto")]
    #[doc = connect_doc!(DataFrame.mergeInto)]
    fn merge_into(
        slf: &Bound<'_, Self>,
        table: &Bound<'_, PyAny>,
        condition: &Bound<'_, PyAny>,
    ) -> PyResult<PyMergeIntoWriter> {
        let (name, unsent_table) = written_table(table);
        let frame = slf.clone().unbind();
        let inner = std::sync::Arc::new(WriterState::of_parts(
            slf.borrow()
                .dataframe
                .merge_into(&name, merge_condition(condition)?),
            spark_connect::MergeIntoWriter::held_parts,
        ));
        Ok(PyMergeIntoWriter {
            parts: writer_parts(&inner),
            inner,
            schema_evolution: std::sync::atomic::AtomicBool::new(false),

            unsent_table,
            frame,
            gc_ready: crate::gc::GcReady::new(),
        })
    }

    /// As PySpark's `DataFrame.__getitem__`: a column by name (`df["col"]`), of this DataFrame
    /// by its plan ID, as its star of `df["*"]`, or by position (`df[0]`), the rows that match
    /// a Column (`df[df.a > 1]`, as `filter`), or the projection of a list or tuple of columns
    /// (`df[["a", "b"]]`, as `select`). With `PYSPARK_VALIDATE_COLUMN_NAME_LEGACY=1`, a name is
    /// validated first, as PySpark's: by `verify_col_name` of the cached schema, which is
    /// requested unless it is cached, and else by the server, of `select(item).isLocal()`, whose
    /// request raises its error, such as `UNRESOLVED_COLUMN.WITH_SUGGESTION`.
    fn __getitem__(&self, py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        use pyo3::IntoPyObjectExt;

        let name = if item.is_instance_of::<pyo3::types::PyString>() {
            // The star of this DataFrame, or its column of the name, `_col(item)`, which sets the
            // name when it builds the plan.
            let column = if item.eq("*")? {
                self.dataframe.star()
            } else {
                let legacy =
                    std::env::var("PYSPARK_VALIDATE_COLUMN_NAME_LEGACY").as_deref() == Ok("1");
                if legacy
                    && !self
                        .spark_session(py)?
                        .bind(py)
                        .hasattr("is_mock_session")?
                {
                    // `self._schema`: the schema that it keeps as a Python object, as it is, or
                    // that of the server, of a new object of each user-defined type (see
                    // `schema_to_py`).
                    let schema = match self.own_schema(py)? {
                        PropertySchema::Object(schema) => schema.unbind(),
                        PropertySchema::Core(schema) => crate::types::schema_to_py(py, &schema)?,
                    };
                    let verified = py
                        .import("pyspark.sql.connect.types")?
                        .getattr("verify_col_name")?
                        .call1((item, schema))?
                        .is_truthy()?;
                    if !verified {
                        self.select(py, vec![item.clone()])?.is_local(py)?;
                    }
                }
                crate::plan_args::df_col(&self.dataframe, item, false)?
            };
            return PyColumn::new(column).into_py_any(py);
        } else if item.is_instance_of::<PyColumn>() {
            return self.filter(item)?.into_py_any(py);
        } else if item.is_instance_of::<PyList>() || item.is_instance_of::<PyTuple>() {
            let cols = item.try_iter()?.collect::<PyResult<Vec<_>>>()?;
            return self.select(py, cols)?.into_py_any(py);
        } else if item.is_instance_of::<PyInt>() {
            let columns = self.column_names(py)?;
            // Use Python list indexing for negative indices, bools (an int
            // subclass), and IndexError even for integers larger than isize.
            PyList::new(py, columns)?
                .as_any()
                .get_item(item)?
                .extract::<String>()?
        } else {
            let arg_type = item.get_type().name()?.to_string();
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkTypeError",
                "NOT_EXPECTED_TYPE",
                &[
                    ("expected_type", "Column, int, list, str or tuple"),
                    ("arg_name", "item"),
                    ("arg_type", &arg_type),
                ],
            ));
        };
        // As PySpark's `F.col(self.columns[item])`, of the origin of `col`.
        let _origin = crate::origin::with_origin("col")?;
        PyColumn::new(spark_connect::functions::col(&name)).into_py_any(py)
    }

    /// Private column accessor used by pandas-on-Spark (`sdf._col(name)`); mirrors the
    /// connect DataFrame's `_col`, the column of this DataFrame by its plan ID. Backtick-quoted
    /// names arrive from `scol_for`.
    fn _col(&self, item: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        Ok(PyColumn::new(crate::plan_args::df_col(
            &self.dataframe,
            item,
            false,
        )?))
    }

    /// As PySpark's `DataFrame.__repr__`: the table of `_show_string` when the session's
    /// `spark.sql.repl.eagerEval.enabled` is `true`, with its `maxNumRows` rows and its
    /// `truncate`, and `DataFrame[name: type, ...]` otherwise, of its `dtypes`. Like PySpark, it
    /// reads the three settings, in one request, on every call.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        {
            let this = slf.borrow();
            // As PySpark's: once `_repr_html_` was called, the string of the schema.
            if !this
                .support_repr_html
                .load(std::sync::atomic::Ordering::Relaxed)
            {
                if let Some((n, truncate)) = this.eager_eval(py)? {
                    let dataframe = &this.dataframe;
                    return interruptible_of(
                        py,
                        &dataframe.spark_session(),
                        &[&this.parts],
                        || dataframe.show_string(n, truncate, false).map_err(Box::new),
                    );
                }
            }
        }
        // As PySpark's `", ".join("%s: %s" % c for c in self.dtypes)`: an AttributeError of the
        // property, of a schema that is an object (see `property_schema`), makes it a Column, which
        // the join cannot iterate, as it cannot iterate the Column of `_schema`.
        let mut fields = Vec::new();
        for pair in slf.getattr(pyo3::intern!(py, "dtypes"))?.try_iter()? {
            let pair = pair?;
            let name = pair.get_item(0)?.str()?.to_string();
            let dtype = pair.get_item(1)?.str()?.to_string();
            fields.push(format!("{name}: {dtype}"));
        }
        Ok(format!("DataFrame[{}]", fields.join(", ")))
    }

    /// The HTML table of the DataFrame, which notebooks show, as PySpark Connect's
    /// `_repr_html_`: with `spark.sql.repl.eagerEval.enabled`, the table of the first
    /// `maxNumRows` rows, values truncated to `truncate` characters, which the server formats
    /// (`HtmlString`); else None. Once called, `__repr__` is the string of the schema.
    fn _repr_html_(&self, py: Python<'_>) -> PyResult<Option<String>> {
        self.support_repr_html
            .store(true, std::sync::atomic::Ordering::Relaxed);
        match self.eager_eval(py)? {
            Some((n, truncate)) => {
                interruptible_of(py, &self.dataframe.spark_session(), &[&self.parts], || {
                    self.dataframe.html_string(n, truncate).map_err(Box::new)
                })
                .map(Some)
            }
            None => Ok(None),
        }
    }

    // `DataFrame.mapInPandas` — map an iterator of pandas DataFrames, in a stage of the
    // resource profile `profile`, if any, as PySpark's.
    #[pyo3(
        name = "mapInPandas",
        signature = (func, schema, barrier = ProtoBool::default(), profile = None),
        text_signature = "($self, func, schema, barrier=False, profile=None)"
    )]
    #[doc = connect_doc!(DataFrame.mapInPandas)]
    fn map_in_pandas(
        &self,
        py: Python<'_>,
        func: Bound<'_, PyAny>,
        schema: Bound<'_, PyAny>,
        barrier: ProtoBool,
        profile: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        validate_map_function(py, &func, eval_type::SQL_MAP_PANDAS_ITER_UDF)?;
        // As PySpark's `_map_partitions`: a DDL string is parsed before the columns of the
        // input are read.
        let session = self.dataframe.spark_session();
        let return_type = resolve_datatype(py, &session, &schema)?;
        let cols = self.column_names(py)?;
        let udf = map_udf_of(
            py,
            &func,
            &schema,
            return_type,
            eval_type::SQL_MAP_PANDAS_ITER_UDF,
            &cols,
        )?;
        Ok(map_result_of(
            &schema,
            map_partitions_of(py, &self.dataframe, udf, &barrier, profile)?,
        ))
    }

    // `DataFrame.mapInArrow` — map an iterator of Arrow batches, in a stage of the
    // resource profile `profile`, if any, as PySpark's.
    #[pyo3(
        name = "mapInArrow",
        signature = (func, schema, barrier = ProtoBool::default(), profile = None),
        text_signature = "($self, func, schema, barrier=False, profile=None)"
    )]
    #[doc = connect_doc!(DataFrame.mapInArrow)]
    fn map_in_arrow(
        &self,
        py: Python<'_>,
        func: Bound<'_, PyAny>,
        schema: Bound<'_, PyAny>,
        barrier: ProtoBool,
        profile: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        validate_map_function(py, &func, eval_type::SQL_MAP_ARROW_ITER_UDF)?;
        // As PySpark's `_map_partitions`: a DDL string is parsed before the columns of the
        // input are read.
        let session = self.dataframe.spark_session();
        let return_type = resolve_datatype(py, &session, &schema)?;
        let cols = self.column_names(py)?;
        let udf = map_udf_of(
            py,
            &func,
            &schema,
            return_type,
            eval_type::SQL_MAP_ARROW_ITER_UDF,
            &cols,
        )?;
        Ok(map_result_of(
            &schema,
            map_partitions_of(py, &self.dataframe, udf, &barrier, profile)?,
        ))
    }

    // `DataFrame.foreach`, as PySpark Connect's: a UDF over a struct of the columns of each
    // row calls `f` with the row. The UDF is Python, as it runs on the server
    // (`pyspark.sql._foreach`).
    #[doc = connect_doc!(DataFrame.foreach)]
    fn foreach(slf: &Bound<'_, Self>, f: &Bound<'_, PyAny>) -> PyResult<()> {
        slf.py()
            .import("pyspark.sql._foreach")?
            .call_method1("foreach", (slf, f))?;
        Ok(())
    }

    // `DataFrame.foreachPartition`, as PySpark Connect's: a `mapInArrow` function calls `f`
    // with an iterator of the rows of each partition.
    #[pyo3(name = "foreachPartition")]
    #[doc = connect_doc!(DataFrame.foreachPartition)]
    fn foreach_partition(slf: &Bound<'_, Self>, f: &Bound<'_, PyAny>) -> PyResult<()> {
        slf.py()
            .import("pyspark.sql._foreach")?
            .call_method1("foreachPartition", (slf, f))?;
        Ok(())
    }

    // `DataFrame.nearestByJoin(other, rankingExpression, numResults, mode, direction, *,
    // joinType="inner")`, as PySpark's, which checks its arguments as the Scala
    // `Dataset.validateNearestByJoinArgs` does: an `AnalysisException` of
    // `NEAREST_BY_JOIN.NUM_RESULTS_OUT_OF_RANGE`, `UNSUPPORTED_JOIN_TYPE`, `UNSUPPORTED_MODE` or
    // `UNSUPPORTED_DIRECTION`. As PySpark, it compares `joinType.lower().replace("_", "")`,
    // `mode.lower()` and `direction.lower()` to the values that it takes, so a value that has no
    // `lower` raises its `AttributeError`, bytes as the join type the `TypeError` of `replace`,
    // and another value, such as bytes as the mode or a str of a lone surrogate, is named as it
    // is in the error.
    #[pyo3(
        name = "nearestByJoin",
        signature = (other, rankingExpression, numResults, mode, direction, *, joinType = DefaultedOption::Omitted),
        text_signature = "($self, other, rankingExpression, numResults, mode, direction, *, joinType='inner')"
    )]
    #[allow(clippy::too_many_arguments)]
    #[doc = connect_doc!(DataFrame.nearestByJoin)]
    fn nearest_by_join(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        rankingExpression: &PyColumn,
        numResults: &Bound<'_, PyAny>,
        mode: &Bound<'_, PyAny>,
        direction: &Bound<'_, PyAny>,
        joinType: DefaultedOption,
    ) -> PyResult<PyDataFrame> {
        const MAX_NUM_RESULTS: i64 = 100_000;
        const MODES: [&str; 2] = ["approx", "exact"];
        const DIRECTIONS: [&str; 2] = ["distance", "similarity"];
        let other = self.same_session(other)?;
        let join_type = match &joinType {
            DefaultedOption::Omitted => pyo3::types::PyString::new(py, "inner").into_any(),
            DefaultedOption::Given(join_type) => join_type.bind(py).clone(),
        };
        // As PySpark's `int(numResults)`.
        let num_results = py
            .import("builtins")?
            .getattr("int")?
            .call1((numResults,))?;
        let analysis = |class: &str, params: &[(&str, &str)]| {
            crate::column::raise_pyspark(py, "AnalysisException", class, params)
        };
        let in_range = num_results
            .extract::<i64>()
            .is_ok_and(|n| (1..=MAX_NUM_RESULTS).contains(&n));
        if !in_range {
            return Err(analysis(
                "NEAREST_BY_JOIN.NUM_RESULTS_OUT_OF_RANGE",
                &[
                    ("numResults", &num_results.str()?.to_cow()?),
                    ("min", "1"),
                    ("max", &MAX_NUM_RESULTS.to_string()),
                ],
            ));
        }
        // Whether `value` is one of `values`, by Python's `in` of the tuple of them.
        let one_of = |value: &Bound<'_, PyAny>, values: &[&str]| -> PyResult<bool> {
            PyTuple::new(py, values)?.contains(value)
        };
        let unsupported = |class: &str, params: &[(&str, &Bound<'_, PyAny>)]| {
            crate::errors::raise_pyspark_of(py, "AnalysisException", class, params)
        };
        let normalized = join_type
            .call_method0("lower")?
            .call_method1("replace", ("_", ""))?;
        if !one_of(&normalized, &["inner", "leftouter", "left"])? {
            let supported = pyo3::types::PyString::new(py, "'INNER', 'LEFT OUTER'");
            return Err(unsupported(
                "NEAREST_BY_JOIN.UNSUPPORTED_JOIN_TYPE",
                &[("joinType", &join_type), ("supported", supported.as_any())],
            ));
        }
        let supported = |values: &[&str]| {
            pyo3::types::PyString::new(py, &format!("'{}'", values.join("', '"))).into_any()
        };
        if !one_of(&mode.call_method0("lower")?, &MODES)? {
            return Err(unsupported(
                "NEAREST_BY_JOIN.UNSUPPORTED_MODE",
                &[("mode", mode), ("supported", &supported(&MODES))],
            ));
        }
        if !one_of(&direction.call_method0("lower")?, &DIRECTIONS)? {
            return Err(unsupported(
                "NEAREST_BY_JOIN.UNSUPPORTED_DIRECTION",
                &[
                    ("direction", direction),
                    ("supported", &supported(&DIRECTIONS)),
                ],
            ));
        }
        Ok(PyDataFrame::new(self.dataframe.nearest_by_join(
            &other.dataframe,
            rankingExpression.column.clone(),
            num_results.extract::<i32>()?,
            &mode.extract::<String>()?,
            &direction.extract::<String>()?,
            &join_type.extract::<String>()?,
        )))
    }

    // `DataFrame.metadataColumn`.
    #[pyo3(name = "metadataColumn")]
    #[doc = connect_doc!(DataFrame.metadataColumn)]
    fn metadata_column(&self, colName: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        // As PySpark's: a str, else NOT_EXPECTED_TYPE.
        if !colName.is_instance_of::<pyo3::types::PyString>() {
            return Err(not_expected_type(colName, "colName", "str"));
        }
        Ok(PyColumn::new(crate::plan_args::df_col(
            &self.dataframe,
            colName,
            true,
        )?))
    }
}

// Python wrapper for `DataFrameNaFunctions` (`df.na`). As PySpark's plain class, it keeps its
// state in its `__dict__`, its `df`, whose methods its methods call, as PySpark's do, and it is
// subclassed and constructed of PySpark's arguments (see `crate::state`).
#[doc = connect_doc!(DataFrameNaFunctions)]
#[pyclass(
    name = "DataFrameNaFunctions",
    module = "pyspark.sql.connect.dataframe",
    subclass,
    weakref,
    dict
)]
pub struct PyDataFrameNaFunctions {}

impl PyDataFrameNaFunctions {
    /// The null functions of `df`, as `df.na` makes them: `DataFrameNaFunctions(df)`, of the
    /// DataFrame object, which PySpark's `__init__` keeps as its `df`.
    fn of_frame<'py>(df: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDataFrameNaFunctions>> {
        let na = Bound::new(df.py(), PyDataFrameNaFunctions {})?;
        Self::__init__(&na, df)?;
        Ok(na)
    }

    /// The `df` of the functions, which each of PySpark's methods reads, `self.df`, to call its
    /// method: of the `AttributeError` of functions whose `__init__` never ran.
    fn frame<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::state::attr(slf.as_any(), pyo3::intern!(slf.py(), "df"))
    }
}

#[pymethods]
impl PyDataFrameNaFunctions {
    /// Null functions of no state, as the `object.__new__` of PySpark's plain class: of any
    /// arguments, which `__init__` takes, so that a subclass of another `__init__` is built as it
    /// is in PySpark. The class shows the signature of `__init__`, as `inspect.signature` of
    /// PySpark's class does.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs), text_signature = "(df)")]
    fn py_new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        PyDataFrameNaFunctions {}
    }

    /// `DataFrameNaFunctions(df)`, as PySpark Connect's `__init__`, which keeps its argument as
    /// it is, as the `df` of the functions: `df.na.df` is `df`.
    fn __init__(slf: &Bound<'_, Self>, df: &Bound<'_, PyAny>) -> PyResult<()> {
        crate::state::assign(slf.as_any(), &[(pyo3::intern!(slf.py(), "df"), df)])
    }

    /// The state of the functions, their `__dict__`, as `object.__getstate__` gives that of
    /// PySpark's plain object, which a copy and a pickle keep (see `state::getstate`).
    fn __getstate__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::state::getstate(slf.as_any())
    }

    /// The reduce of the functions, as PySpark's plain object's: `copyreg.__newobj__` of their
    /// class, of their `__dict__`, so a copy is of the same DataFrame, and a deep copy and a pickle
    /// of a copy of it (see `state::reduce_ex`).
    fn __reduce_ex__<'py>(slf: &Bound<'py, Self>, protocol: i32) -> PyResult<Bound<'py, PyAny>> {
        crate::state::reduce_ex(slf.as_any(), protocol)
    }

    // Drop rows containing nulls, as PySpark's `DataFrameNaFunctions.drop`, which is
    // `self.df.dropna(how=how, thresh=thresh, subset=subset)`, a call of the method of the
    // DataFrame, of a subclass or of any object too, of its default `how` of "any".
    #[pyo3(
        signature = (how=crate::options::DefaultedOption::Omitted, thresh=None, subset=None),
        text_signature = "($self, how='any', thresh=None, subset=None)"
    )]
    #[doc = connect_doc!(DataFrameNaFunctions.drop)]
    fn drop<'py>(
        slf: &Bound<'py, Self>,
        how: crate::options::DefaultedOption,
        thresh: Option<&Bound<'py, PyAny>>,
        subset: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let how = match how {
            crate::options::DefaultedOption::Omitted => pyo3::intern!(py, "any").clone().into_any(),
            crate::options::DefaultedOption::Given(how) => how.into_bound(py),
        };
        let kwargs = PyDict::new(py);
        kwargs.set_item(pyo3::intern!(py, "how"), how)?;
        kwargs.set_item(pyo3::intern!(py, "thresh"), thresh)?;
        kwargs.set_item(pyo3::intern!(py, "subset"), subset)?;
        Self::frame(slf)?.call_method(pyo3::intern!(py, "dropna"), (), Some(&kwargs))
    }

    // Fill nulls, as PySpark's `DataFrameNaFunctions.fill`, which is
    // `self.df.fillna(value=value, subset=subset)`.
    #[pyo3(signature = (value, subset=None))]
    #[doc = connect_doc!(DataFrameNaFunctions.fill)]
    fn fill<'py>(
        slf: &Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
        subset: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let kwargs = PyDict::new(py);
        kwargs.set_item(pyo3::intern!(py, "value"), value)?;
        kwargs.set_item(pyo3::intern!(py, "subset"), subset)?;
        Self::frame(slf)?.call_method(pyo3::intern!(py, "fillna"), (), Some(&kwargs))
    }

    // Replace values, as PySpark's `DataFrameNaFunctions.replace`, which is
    // `self.df.replace(to_replace, value, subset)`, of PySpark's `_NoValue` of an omitted value.
    #[pyo3(signature = (to_replace, value=ReplaceValue::Omitted, subset=None))]
    #[doc = connect_doc!(DataFrameNaFunctions.replace)]
    fn replace<'py>(
        slf: &Bound<'py, Self>,
        to_replace: &Bound<'py, PyAny>,
        value: ReplaceValue,
        subset: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let value = match value {
            ReplaceValue::Omitted => no_value(py)?.clone(),
            ReplaceValue::Given(value) => value.into_bound(py),
        };
        Self::frame(slf)?.call_method1(pyo3::intern!(py, "replace"), (to_replace, value, subset))
    }
}

/// PySpark's `pyspark._globals._NoValue`, the default `value` of `replace`, which tells an
/// omitted value apart from None.
fn no_value(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    static NO_VALUE: pyo3::sync::PyOnceLock<Py<PyAny>> = pyo3::sync::PyOnceLock::new();
    NO_VALUE.import(py, "pyspark._globals", "_NoValue")
}

/// The `value` argument of `DataFrame.replace`, where PySpark tells an omitted value apart
/// from None: `replace(to_replace)` requires a dict `to_replace`, while
/// `replace(to_replace, None)` replaces with nulls. PySpark's `_NoValue`, its default, is an
/// omitted value too, as `DataFrameNaFunctions.replace` gives it.
pub enum ReplaceValue {
    Omitted,
    Given(Py<PyAny>),
}

impl<'a, 'py> FromPyObject<'a, 'py> for ReplaceValue {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if obj.is(no_value(obj.py())?) {
            return Ok(ReplaceValue::Omitted);
        }
        Ok(ReplaceValue::Given(obj.to_owned().unbind()))
    }
}

/// `df.replace(to_replace, value, subset)`, checked and built as PySpark's
/// `DataFrame.replace`: each value of `to_replace` (a {old: new} dict, or a value or a list or
/// tuple of them) is replaced by `value` (a value or a list or tuple of them, or None), all
/// bools, numbers or strings, sent as literals with an int as a float, in the columns of
/// `subset` (a column name, or a list or tuple of them).
fn replace_values(
    dataframe: &DataFrame,
    to_replace: &Bound<'_, PyAny>,
    value: ReplaceValue,
    subset: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyDataFrame> {
    use pyo3::types::{PyBool, PyFloat, PyString};
    use spark_connect::expression::LiteralExpression;

    let py = to_replace.py();
    let raise = |class_name: &str, error_class: &str, params: &[(&str, &str)]| {
        crate::column::raise_pyspark(py, class_name, error_class, params)
    };
    let type_name = |v: &Bound<'_, PyAny>| -> String {
        v.get_type()
            .name()
            .map(|name| name.to_string())
            .unwrap_or_default()
    };
    let is_sequence =
        |v: &Bound<'_, PyAny>| v.is_instance_of::<PyList>() || v.is_instance_of::<PyTuple>();
    // A bool, a float, an int or a str: a bool is an int.
    let is_scalar = |v: &Bound<'_, PyAny>| {
        v.is_instance_of::<PyFloat>()
            || v.is_instance_of::<PyInt>()
            || v.is_instance_of::<PyString>()
    };
    let is_dict = to_replace.is_instance_of::<PyDict>();
    let value = match value {
        ReplaceValue::Given(value) => value.into_bound(py),
        ReplaceValue::Omitted if is_dict => py.None().into_bound(py),
        ReplaceValue::Omitted => {
            return Err(raise(
                "PySparkTypeError",
                "ARGUMENT_REQUIRED",
                &[("arg_name", "value"), ("condition", "`to_replace` is dict")],
            ));
        }
    };

    if !(is_scalar(to_replace) || is_sequence(to_replace) || is_dict) {
        return Err(raise(
            "PySparkTypeError",
            "NOT_EXPECTED_TYPE",
            &[
                ("expected_type", "bool, dict, float, int, str or tuple"),
                ("arg_name", "to_replace"),
                ("arg_type", &type_name(to_replace)),
            ],
        ));
    }
    if !(is_scalar(&value) || is_sequence(&value) || value.is_none() || is_dict) {
        return Err(raise(
            "PySparkTypeError",
            "NOT_EXPECTED_TYPE",
            &[
                (
                    "expected_type",
                    "bool, float, int, list, None, str or tuple",
                ),
                ("arg_name", "value"),
                ("arg_type", &type_name(&value)),
            ],
        ));
    }
    if is_sequence(to_replace) && is_sequence(&value) && to_replace.len()? != value.len()? {
        return Err(raise(
            "PySparkValueError",
            "LENGTH_SHOULD_BE_THE_SAME",
            &[
                ("arg1", "to_replace"),
                ("arg2", "value"),
                ("arg1_length", &to_replace.len()?.to_string()),
                ("arg2_length", &value.len()?.to_string()),
            ],
        ));
    }
    // The names of a list or a tuple are set in the repeated string field of the plan as they
    // are, as protobuf converts them, when its `NAReplace` plan builds the plan, before the
    // replacements (`args`).
    let mut args = crate::plan_args::PlanArgs::new();
    let subset: Option<Bound<'_, PyAny>> = match subset {
        None => None,
        Some(subset) if subset.is_instance_of::<PyString>() || is_sequence(subset) => {
            Some(subset.clone())
        }
        Some(subset) => {
            return Err(raise(
                "PySparkTypeError",
                "NOT_EXPECTED_TYPE",
                &[
                    ("arg_name", "subset"),
                    ("expected_type", "list, str or tuple"),
                    ("arg_type", &type_name(subset)),
                ],
            ));
        }
    };

    // The replacements as a dict, as PySpark builds it, so that equal values are one key. Of a
    // dict and a value, PySpark's `DataFrame.replace`, which `na.replace` calls, warns of the line
    // of `pyspark.sql.connect.dataframe` of its `warnings.warn`.
    let replacements = if is_dict {
        if !value.is_none() {
            crate::errors::warn_as_pyspark(
                py,
                "pyspark.sql.connect.dataframe",
                1541,
                "to_replace is a dict and value is not None. value will be ignored.",
                &py.get_type::<pyo3::exceptions::PyUserWarning>(),
            )?;
        }
        to_replace.cast::<PyDict>()?.clone()
    } else {
        let olds: Vec<Bound<'_, PyAny>> = if is_scalar(to_replace) {
            vec![to_replace.clone()]
        } else {
            to_replace.try_iter()?.collect::<PyResult<_>>()?
        };
        let news: Vec<Bound<'_, PyAny>> = if is_scalar(&value) || value.is_none() {
            vec![value.clone(); olds.len()]
        } else {
            value.try_iter()?.collect::<PyResult<_>>()?
        };
        let replacements = PyDict::new(py);
        for (old, new) in olds.iter().zip(news.iter()) {
            replacements.set_item(old, new)?;
        }
        replacements
    };

    // All of the values to replace, and all of their replacements but None, are bools,
    // numbers or strings.
    let olds: Vec<Bound<'_, PyAny>> = replacements.keys().iter().collect();
    let news: Vec<Bound<'_, PyAny>> = replacements
        .values()
        .iter()
        .filter(|new| !new.is_none())
        .collect();
    let all_of = |is_type: &dyn Fn(&Bound<'_, PyAny>) -> bool| {
        olds.iter().all(is_type) && news.iter().all(is_type)
    };
    if !(all_of(&|v| v.is_instance_of::<PyBool>())
        || all_of(&|v| v.is_instance_of::<PyString>())
        || all_of(&|v| v.is_instance_of::<PyFloat>() || v.is_instance_of::<PyInt>()))
    {
        return Err(raise("PySparkValueError", "MIXED_TYPE_REPLACEMENT", &[]));
    }

    let subset = match subset {
        None => None,
        Some(subset) if subset.is_instance_of::<PyString>() => Some(vec![args.str(&subset)?]),
        Some(subset) => Some(args.strs(subset.try_iter()?.collect::<PyResult<Vec<_>>>()?)?),
    };
    // The literals of `F.lit`, whose strings the plan sets when it builds the plan.
    let mut literal = |v: &Bound<'_, PyAny>| -> PyResult<LiteralExpression> {
        Ok(if v.is_none() {
            LiteralExpression::null(spark_connect::types::DataType::Null)
        } else if v.is_instance_of::<PyBool>() {
            LiteralExpression::boolean(v.extract()?)
        } else if v.is_instance_of::<PyString>() {
            LiteralExpression::string(args.str(v)?)
        } else {
            // As `float(v)`, for an int as well.
            LiteralExpression::double(v.extract()?)
        })
    };
    let replacements = replacements
        .iter()
        .map(|(old, new)| Ok((literal(&old)?, literal(&new)?)))
        .collect::<PyResult<Vec<_>>>()?;
    let subset: Option<Vec<&str>> = subset
        .as_ref()
        .map(|subset| subset.iter().map(String::as_str).collect());
    Ok(PyDataFrame::new(
        args.plan(dataframe.replace_literals(replacements, subset)),
    ))
}

/// The columns of the `ids` or `values` of `df.unpivot`, as PySpark's: a list or tuple of
/// columns or column names, or one of them.
fn unpivot_columns(cols: &Bound<'_, PyAny>) -> PyResult<Vec<spark_connect::column::Column>> {
    if cols.is_instance_of::<PyList>() || cols.is_instance_of::<PyTuple>() {
        cols.try_iter()?.map(|col| column_or_name(&col?)).collect()
    } else {
        Ok(vec![column_or_name(cols)?])
    }
}

/// A column, or a column by its name, as PySpark's `_to_col`.
pub(crate) fn column_or_name(col: &Bound<'_, PyAny>) -> PyResult<spark_connect::column::Column> {
    if let Ok(column) = col.extract::<PyRef<'_, PyColumn>>() {
        return Ok(column.column.clone());
    }
    if col.is_instance_of::<pyo3::types::PyString>() {
        // As PySpark's `F._to_col`, by `col`, of its origin.
        let _origin = crate::origin::with_origin("col")?;
        return crate::plan_args::col(col);
    }
    Err(not_expected_type(col, "col", "Column or str"))
}

/// The table of `writeTo` and `mergeInto`, which PySpark keeps as it is given and sets in the
/// string field of the command of each write, which leaves the field unset for None: its name,
/// as [`crate::str_argument`] converts it, or "" with the table that the field does not take,
/// whose error each write raises again with [`check_written_table`], and not the call.
fn written_table(table: &Bound<'_, PyAny>) -> (String, Option<Py<PyAny>>) {
    if table.is_none() {
        return (String::new(), None);
    }
    match crate::str_argument(table) {
        Ok(name) => (name, None),
        Err(_) => (String::new(), Some(table.clone().unbind())),
    }
}

/// The error of a write of a table that its string field does not take ([`written_table`]),
/// which PySpark's command raises after its input plan, before it sets the other fields.
fn check_written_table(py: Python<'_>, unsent_table: &Option<Py<PyAny>>) -> PyResult<()> {
    match unsent_table {
        Some(table) => crate::str_argument(table.bind(py)).map(drop),
        None => Ok(()),
    }
}

// Python wrapper for `MergeIntoWriter` (`df.mergeInto(...)`).
//
// Mirrors PySpark's mutable writer: `withSchemaEvolution` and every clause action
// (`whenMatched().updateAll()`, ...) change this object and return it, and `merge`
// leaves it usable. The core writer is a consuming builder, so the wrapped value is
// replaced in place and `merge` runs on a clone.
#[doc = connect_doc!(MergeIntoWriter)]
#[pyclass(
    name = "MergeIntoWriter",
    module = "pyspark.sql.connect.merge",
    weakref,
    dict
)]
pub struct PyMergeIntoWriter {
    inner: std::sync::Arc<WriterState<spark_connect::MergeIntoWriter>>,
    /// The object of the parts of the state, which the shallow copies of this writer share and
    /// report, which reports them (see [`WriterParts`]).
    parts: Option<Py<WriterParts>>,
    /// Whether `withSchemaEvolution` was called on this builder, as PySpark's
    /// `_schema_evolution_enabled` attribute, which a shallow copy takes by value and then
    /// keeps apart from the copied builder.
    schema_evolution: std::sync::atomic::AtomicBool,

    /// The table of `mergeInto` that its string field does not take, which Python's garbage
    /// collector sees ([`written_table`]).
    unsent_table: Option<Py<PyAny>>,
    /// The DataFrame of `mergeInto`, which PySpark's builder keeps in the callback of its
    /// execution.
    frame: Py<PyDataFrame>,
    /// Set once the value is written, as `__traverse__` reads `inner` and `frame`, which are not
    /// valid while the object is created (see [`crate::gc::GcReady`]).
    gc_ready: crate::gc::GcReady,
}

/// Apply a clause action to the parent writer in place and return that same writer
/// (PySpark's clause methods append to `self.writer` and return it).
fn merge_clause_action(
    py: Python<'_>,
    writer: &Py<PyMergeIntoWriter>,
    action: impl FnOnce(spark_connect::MergeIntoWriter) -> spark_connect::MergeIntoWriter,
) -> PyResult<Py<PyMergeIntoWriter>> {
    writer.bind(py).borrow().inner.update(action);
    Ok(writer.clone_ref(py))
}

/// The condition of `mergeInto`, which PySpark's builder keeps as it is given and builds with its
/// `to_plan` when the merge runs: a Column, or else a Column whose build raises the error of the
/// `to_plan` that the object lacks, its `AttributeError`, or the `PySparkAttributeError` of a
/// Row, after the source of the merge is built, as PySpark's merge raises it
/// ([`crate::plan_args::PlanArgs::raised`]). An object that has a `to_plan` is not a Column of
/// this client, which raises PySpark's `NOT_COLUMN`.
fn merge_condition(condition: &Bound<'_, PyAny>) -> PyResult<spark_connect::column::Column> {
    if let Ok(column) = condition.extract::<PyRef<'_, PyColumn>>() {
        return Ok(column.column.clone());
    }
    match condition.getattr("to_plan") {
        Ok(_) => Err(not_column(condition, "condition")),
        Err(err) => {
            let mut args = crate::plan_args::PlanArgs::new();
            args.raised(condition.py(), err)?;
            Ok(spark_connect::column::Column::new(args.expression(
                spark_connect::expression::Expression::literal(
                    spark_connect::expression::LiteralExpression::null(
                        spark_connect::types::DataType::Null,
                    ),
                ),
            )))
        }
    }
}

/// PySpark's `NOT_COLUMN` of the argument `name`, `value`, an object of a `to_plan` that this
/// client does not build.
fn not_column(value: &Bound<'_, PyAny>, name: &str) -> PyErr {
    let type_name = value
        .get_type()
        .name()
        .map(|name| name.to_string())
        .unwrap_or_default();
    crate::column::raise_pyspark(
        value.py(),
        "PySparkTypeError",
        "NOT_COLUMN",
        &[("arg_name", name), ("arg_type", &type_name)],
    )
}

/// The condition of a merge clause, which PySpark's clause keeps as it is given and builds with its
/// `to_plan` when an action adds the clause: none, a Column, of the reference of the clause to the
/// parts of its expression (see [`crate::gc::Parts`]), or another object.
enum ClauseCondition {
    None,
    Column(spark_connect::column::Column, crate::gc::Parts),
    Object(Py<PyAny>),
}

impl ClauseCondition {
    fn of(condition: Option<&Bound<'_, PyAny>>) -> Self {
        match condition.filter(|condition| !condition.is_none()) {
            None => ClauseCondition::None,
            Some(condition) => match condition.extract::<PyRef<'_, PyColumn>>() {
                Ok(column) => ClauseCondition::of_column(condition.py(), column.column.clone()),
                Err(_) => ClauseCondition::Object(condition.clone().unbind()),
            },
        }
    }

    /// The condition `column`, of its parts.
    fn of_column(py: Python<'_>, column: spark_connect::column::Column) -> Self {
        let parts = crate::gc::Parts::of(
            py,
            [spark_connect::gc::PartRef::Expression(column.expression())],
        );
        ClauseCondition::Column(column, parts)
    }

    fn clone_ref(&self, py: Python<'_>) -> Self {
        match self {
            ClauseCondition::None => ClauseCondition::None,
            ClauseCondition::Column(column, _) => ClauseCondition::of_column(py, column.clone()),
            ClauseCondition::Object(object) => ClauseCondition::Object(object.clone_ref(py)),
        }
    }

    /// Report the condition to Python's garbage collector, as PySpark's clause holds it: an
    /// object as it is, and the parts of the expression of a Column, as the Column reports them.
    fn visit(&self, visit: &pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        match self {
            ClauseCondition::None => Ok(()),
            ClauseCondition::Column(_, parts) => parts.visit(visit),
            ClauseCondition::Object(object) => visit.call(object),
        }
    }
}

/// The condition and the assignments of a merge clause, in the order of the assignments.
type MergeClause = (
    Option<spark_connect::column::Column>,
    Vec<(String, spark_connect::column::Column)>,
);

/// The condition and the assignments of a merge clause, as PySpark's `_build_merge_action` builds
/// them when its clause method adds the clause, of which it raises the errors: first each
/// assignment of `assignments.items()`, in their order, its key the SQL expression of `expr(k)`,
/// which asserts a str and sets it in a string field, and then its value, whose `to_plan` an
/// object that is not a Column lacks; and then the condition. A Column that does not build raises
/// its error there, such as a str that UTF-8 cannot encode, or the `RecursionError` of PySpark's
/// serializer of one too deep to be sent, and the clause is not added. The clause keeps them as
/// they were built then ([`built_column`]), which the merge sends, as PySpark's merge sends the
/// protos that its clause built: so the functions of their UDFs are pickled once, when the clause
/// is added, and the warning of a window of no partitions is given then. The assignments keep
/// their order, in which the merge sends them.
fn merge_clause(
    py: Python<'_>,
    condition: &ClauseCondition,
    assignments: Option<&Bound<'_, PyAny>>,
) -> PyResult<MergeClause> {
    let mut built = Vec::new();
    if let Some(assignments) = assignments {
        for item in assignments.call_method0("items")?.try_iter()? {
            let (key, value): (Bound<'_, PyAny>, Bound<'_, PyAny>) = item?.extract()?;
            if !key.is_instance_of::<pyo3::types::PyString>() {
                return Err(pyo3::exceptions::PyAssertionError::new_err(()));
            }
            let key = crate::str_argument(&key)?;
            let value = match value.extract::<PyRef<'_, PyColumn>>() {
                Ok(column) => column.column.clone(),
                Err(_) => {
                    value.getattr("to_plan")?;
                    return Err(not_column(&value, "value"));
                }
            };
            built.push((key, built_column(&value)?));
        }
    }
    let condition = match condition {
        ClauseCondition::None => None,
        ClauseCondition::Column(column, _) => Some(built_column(column)?),
        ClauseCondition::Object(object) => {
            let object = object.bind(py);
            object.getattr("to_plan")?;
            return Err(not_column(object, "condition"));
        }
    };
    Ok((condition, built))
}

/// The Column of the `spark.connect.Expression` of `column` as a request builds it now, of its
/// rendering and its levels, which sends that expression as it is (`Column::sent`), as the
/// `MergeAction` that PySpark's `_build_merge_action` builds holds the protos of its Columns; or
/// the error of the build.
fn built_column(column: &spark_connect::column::Column) -> PyResult<spark_connect::column::Column> {
    let (expression, levels) = column.try_to_proto_with_levels().to_pyerr()?;
    spark_connect::column::Column::sent(expression, column.expression().render(), levels).to_pyerr()
}

#[pymethods]
impl PyMergeIntoWriter {
    /// The shallow copy of a merge builder, as PySpark's `copy.copy`: the copy shares the state
    /// of this one, its table, its condition and its actions, as the copy of PySpark's shares
    /// its attributes, and the instance attributes of this one.
    fn __copy__(slf: &Bound<'_, Self>) -> PyResult<Py<Self>> {
        let py = slf.py();
        let this = slf.borrow();
        let copy = Py::new(
            py,
            PyMergeIntoWriter {
                inner: std::sync::Arc::clone(&this.inner),
                parts: this.parts.as_ref().map(|parts| parts.clone_ref(py)),
                schema_evolution: std::sync::atomic::AtomicBool::new(
                    this.schema_evolution
                        .load(std::sync::atomic::Ordering::SeqCst),
                ),

                unsent_table: this.unsent_table.as_ref().map(|table| table.clone_ref(py)),
                frame: this.frame.clone_ref(py),
                gc_ready: crate::gc::GcReady::new(),
            },
        )?;
        crate::reduce::copy_instance_dict(slf.as_any(), copy.bind(py).as_any())?;
        Ok(copy)
    }

    /// A merge builder does not pickle, as PySpark's, whose client holds a thread-local state
    /// that pickle cannot copy.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::reduce::thread_local_reduce(slf.as_any())
    }

    /// Nested builder-class references, mirroring pyspark's `MergeIntoWriter.WhenMatched`
    /// etc. (the objects returned by `whenMatched()` / `whenNotMatched()` /
    /// `whenNotMatchedBySource()`).
    #[classattr]
    #[allow(non_snake_case)]
    fn WhenMatched(py: Python<'_>) -> Py<pyo3::types::PyType> {
        py.get_type::<PyWhenMatched>().unbind()
    }
    #[classattr]
    #[allow(non_snake_case)]
    fn WhenNotMatched(py: Python<'_>) -> Py<pyo3::types::PyType> {
        py.get_type::<PyWhenNotMatched>().unbind()
    }
    #[classattr]
    #[allow(non_snake_case)]
    fn WhenNotMatchedBySource(py: Python<'_>) -> Py<pyo3::types::PyType> {
        py.get_type::<PyWhenNotMatchedBySource>().unbind()
    }

    // A clause of the condition `condition`, which PySpark keeps as it is given: its action builds
    // it ([`merge_clause`]).
    #[pyo3(name = "whenMatched", signature = (condition=None))]
    #[doc = connect_doc!(MergeIntoWriter.whenMatched)]
    fn when_matched(slf: &Bound<'_, Self>, condition: Option<&Bound<'_, PyAny>>) -> PyWhenMatched {
        PyWhenMatched::new(slf.clone().unbind(), ClauseCondition::of(condition))
    }

    #[pyo3(name = "whenNotMatched", signature = (condition=None))]
    #[doc = connect_doc!(MergeIntoWriter.whenNotMatched)]
    fn when_not_matched(
        slf: &Bound<'_, Self>,
        condition: Option<&Bound<'_, PyAny>>,
    ) -> PyWhenNotMatched {
        PyWhenNotMatched::new(slf.clone().unbind(), ClauseCondition::of(condition))
    }

    #[pyo3(name = "whenNotMatchedBySource", signature = (condition=None))]
    #[doc = connect_doc!(MergeIntoWriter.whenNotMatchedBySource)]
    fn when_not_matched_by_source(
        slf: &Bound<'_, Self>,
        condition: Option<&Bound<'_, PyAny>>,
    ) -> PyWhenNotMatchedBySource {
        PyWhenNotMatchedBySource::new(slf.clone().unbind(), ClauseCondition::of(condition))
    }

    #[pyo3(name = "withSchemaEvolution")]
    #[doc = connect_doc!(MergeIntoWriter.withSchemaEvolution)]
    fn with_schema_evolution<'py>(slf: &Bound<'py, Self>) -> Bound<'py, Self> {
        slf.borrow()
            .schema_evolution
            .store(true, std::sync::atomic::Ordering::SeqCst);

        slf.clone()
    }

    #[doc = connect_doc!(MergeIntoWriter.merge)]
    fn merge(&self, py: Python<'_>) -> PyResult<()> {
        check_written_table(py, &self.unsent_table)?;
        let mut writer = self.inner.get();
        if self
            .schema_evolution
            .load(std::sync::atomic::Ordering::SeqCst)
        {
            writer = writer.with_schema_evolution();
        }
        // Of the parts of the copy of the state, of the source, the condition and the clauses,
        // which the command is built of.
        let _in_use = crate::gc::in_use_of_values(py, writer.held_parts());
        interruptible(py, &writer.spark_session(), || {
            writer.merge().map_err(Box::new)
        })
    }

    /// The DataFrame and the table that the merge keeps, Python objects, which Python's garbage
    /// collector sees, and the parts of its source plan, its condition and its clauses (see
    /// [`crate::gc`]), as PySpark's merge builder, a Python object, holds the DataFrame and the
    /// Columns: so a reference cycle through them is collected, as PySpark collects it. The parts
    /// of a state that a copy shares, or that another thread holds, are not visited, so that
    /// nothing is counted twice.
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        if !self.gc_ready.get() {
            return Ok(());
        }
        visit.call(&self.frame)?;
        if let Some(table) = &self.unsent_table {
            visit.call(table)?;
        }
        if let Some(parts) = &self.parts {
            visit.call(parts)?;
        }
        Ok(())
    }

    /// Drop this builder's references to its table and to its state, of which it keeps a cleared
    /// copy; a copy of the builder keeps the state.
    fn __clear__(&mut self) {
        if self.gc_ready.get() {
            self.unsent_table = None;
            self.parts = None;
            self.inner = std::sync::Arc::new(
                self.inner
                    .cleared(spark_connect::MergeIntoWriter::clear_plan),
            );
        }
    }
}

// `whenMatched(...)` clause builder. Holds its parent writer, like PySpark's
// `MergeIntoWriter.WhenMatched`; each action may be called more than once.
#[doc = connect_doc!(MergeIntoWriter.WhenMatched)]
#[pyclass(
    name = "WhenMatched",
    module = "pyspark.sql.connect.merge",
    weakref,
    dict
)]
pub struct PyWhenMatched {
    writer: Py<PyMergeIntoWriter>,
    condition: ClauseCondition,
    /// Set once the value is written, as `__traverse__` reads `writer` and `condition`, which are
    /// not valid while the object is created (see [`crate::gc::GcReady`]).
    gc_ready: crate::gc::GcReady,
}

impl PyWhenMatched {
    fn new(writer: Py<PyMergeIntoWriter>, condition: ClauseCondition) -> Self {
        PyWhenMatched {
            writer,
            condition,
            gc_ready: crate::gc::GcReady::new(),
        }
    }
}

#[pymethods]
impl PyWhenMatched {
    /// The merge builder of the clause, as PySpark's `writer` attribute of the clause.
    #[getter]
    fn writer(&self, py: Python<'_>) -> Py<PyMergeIntoWriter> {
        self.writer.clone_ref(py)
    }

    /// The shallow copy of a merge action, as PySpark's `copy.copy`: of the same merge builder
    /// and condition, with the instance attributes of this one.
    fn __copy__(slf: &Bound<'_, Self>) -> PyResult<Py<Self>> {
        let py = slf.py();
        let this = slf.borrow();
        let copy = Py::new(
            py,
            PyWhenMatched::new(this.writer.clone_ref(py), this.condition.clone_ref(py)),
        )?;
        crate::reduce::copy_instance_dict(slf.as_any(), copy.bind(py).as_any())?;
        Ok(copy)
    }

    /// As PySpark's clause, a Python object that holds its merge builder and its condition,
    /// which Python's garbage collector sees (see [`ClauseCondition::visit`]).
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        if !self.gc_ready.get() {
            return Ok(());
        }
        visit.call(&self.writer)?;
        self.condition.visit(&visit)
    }

    /// Break a reference cycle that the garbage collector found unreachable: the clause drops its
    /// condition.
    fn __clear__(&mut self) {
        if self.gc_ready.get() {
            self.condition = ClauseCondition::None;
        }
    }

    /// A merge action does not pickle, as PySpark's, whose merge builder holds a client with a
    /// thread-local state that pickle cannot copy.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::reduce::thread_local_reduce(slf.as_any())
    }

    #[pyo3(name = "updateAll")]
    #[doc = connect_doc!(MergeIntoWriter.WhenMatched.updateAll)]
    fn update_all(&self, py: Python<'_>) -> PyResult<Py<PyMergeIntoWriter>> {
        let (cond, _) = merge_clause(py, &self.condition, None)?;
        merge_clause_action(py, &self.writer, |w| w.when_matched(cond).update_all())
    }

    #[doc = connect_doc!(MergeIntoWriter.WhenMatched.update)]
    fn update(
        &self,
        py: Python<'_>,
        assignments: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyMergeIntoWriter>> {
        let (cond, assignments) = merge_clause(py, &self.condition, Some(assignments))?;
        merge_clause_action(py, &self.writer, |w| {
            w.when_matched(cond).update(assignments)
        })
    }

    #[doc = connect_doc!(MergeIntoWriter.WhenMatched.delete)]
    fn delete(&self, py: Python<'_>) -> PyResult<Py<PyMergeIntoWriter>> {
        let (cond, _) = merge_clause(py, &self.condition, None)?;
        merge_clause_action(py, &self.writer, |w| w.when_matched(cond).delete())
    }
}

// `whenNotMatched(...)` clause builder. Holds its parent writer, like PySpark's
// `MergeIntoWriter.WhenNotMatched`.
#[doc = connect_doc!(MergeIntoWriter.WhenNotMatched)]
#[pyclass(
    name = "WhenNotMatched",
    module = "pyspark.sql.connect.merge",
    weakref,
    dict
)]
pub struct PyWhenNotMatched {
    writer: Py<PyMergeIntoWriter>,
    condition: ClauseCondition,
    /// Set once the value is written, as `__traverse__` reads `writer` and `condition`, which are
    /// not valid while the object is created (see [`crate::gc::GcReady`]).
    gc_ready: crate::gc::GcReady,
}

impl PyWhenNotMatched {
    fn new(writer: Py<PyMergeIntoWriter>, condition: ClauseCondition) -> Self {
        PyWhenNotMatched {
            writer,
            condition,
            gc_ready: crate::gc::GcReady::new(),
        }
    }
}

#[pymethods]
impl PyWhenNotMatched {
    /// The merge builder of the clause, as PySpark's `writer` attribute of the clause.
    #[getter]
    fn writer(&self, py: Python<'_>) -> Py<PyMergeIntoWriter> {
        self.writer.clone_ref(py)
    }

    /// The shallow copy of a merge action, as PySpark's `copy.copy`: of the same merge builder
    /// and condition, with the instance attributes of this one.
    fn __copy__(slf: &Bound<'_, Self>) -> PyResult<Py<Self>> {
        let py = slf.py();
        let this = slf.borrow();
        let copy = Py::new(
            py,
            PyWhenNotMatched::new(this.writer.clone_ref(py), this.condition.clone_ref(py)),
        )?;
        crate::reduce::copy_instance_dict(slf.as_any(), copy.bind(py).as_any())?;
        Ok(copy)
    }

    /// As PySpark's clause, a Python object that holds its merge builder and its condition,
    /// which Python's garbage collector sees (see [`ClauseCondition::visit`]).
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        if !self.gc_ready.get() {
            return Ok(());
        }
        visit.call(&self.writer)?;
        self.condition.visit(&visit)
    }

    /// Break a reference cycle that the garbage collector found unreachable: the clause drops its
    /// condition.
    fn __clear__(&mut self) {
        if self.gc_ready.get() {
            self.condition = ClauseCondition::None;
        }
    }

    /// A merge action does not pickle, as PySpark's, whose merge builder holds a client with a
    /// thread-local state that pickle cannot copy.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::reduce::thread_local_reduce(slf.as_any())
    }

    #[pyo3(name = "insertAll")]
    #[doc = connect_doc!(MergeIntoWriter.WhenNotMatched.insertAll)]
    fn insert_all(&self, py: Python<'_>) -> PyResult<Py<PyMergeIntoWriter>> {
        let (cond, _) = merge_clause(py, &self.condition, None)?;
        merge_clause_action(py, &self.writer, |w| w.when_not_matched(cond).insert_all())
    }

    #[doc = connect_doc!(MergeIntoWriter.WhenNotMatched.insert)]
    fn insert(
        &self,
        py: Python<'_>,
        assignments: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyMergeIntoWriter>> {
        let (cond, assignments) = merge_clause(py, &self.condition, Some(assignments))?;
        merge_clause_action(py, &self.writer, |w| {
            w.when_not_matched(cond).insert(assignments)
        })
    }
}

// `whenNotMatchedBySource(...)` clause builder. Holds its parent writer, like
// PySpark's `MergeIntoWriter.WhenNotMatchedBySource`.
#[doc = connect_doc!(MergeIntoWriter.WhenNotMatchedBySource)]
#[pyclass(
    name = "WhenNotMatchedBySource",
    module = "pyspark.sql.connect.merge",
    weakref,
    dict
)]
pub struct PyWhenNotMatchedBySource {
    writer: Py<PyMergeIntoWriter>,
    condition: ClauseCondition,
    /// Set once the value is written, as `__traverse__` reads `writer` and `condition`, which are
    /// not valid while the object is created (see [`crate::gc::GcReady`]).
    gc_ready: crate::gc::GcReady,
}

impl PyWhenNotMatchedBySource {
    fn new(writer: Py<PyMergeIntoWriter>, condition: ClauseCondition) -> Self {
        PyWhenNotMatchedBySource {
            writer,
            condition,
            gc_ready: crate::gc::GcReady::new(),
        }
    }
}

#[pymethods]
impl PyWhenNotMatchedBySource {
    /// The merge builder of the clause, as PySpark's `writer` attribute of the clause.
    #[getter]
    fn writer(&self, py: Python<'_>) -> Py<PyMergeIntoWriter> {
        self.writer.clone_ref(py)
    }

    /// The shallow copy of a merge action, as PySpark's `copy.copy`: of the same merge builder
    /// and condition, with the instance attributes of this one.
    fn __copy__(slf: &Bound<'_, Self>) -> PyResult<Py<Self>> {
        let py = slf.py();
        let this = slf.borrow();
        let copy = Py::new(
            py,
            PyWhenNotMatchedBySource::new(this.writer.clone_ref(py), this.condition.clone_ref(py)),
        )?;
        crate::reduce::copy_instance_dict(slf.as_any(), copy.bind(py).as_any())?;
        Ok(copy)
    }

    /// As PySpark's clause, a Python object that holds its merge builder and its condition,
    /// which Python's garbage collector sees (see [`ClauseCondition::visit`]).
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        if !self.gc_ready.get() {
            return Ok(());
        }
        visit.call(&self.writer)?;
        self.condition.visit(&visit)
    }

    /// Break a reference cycle that the garbage collector found unreachable: the clause drops its
    /// condition.
    fn __clear__(&mut self) {
        if self.gc_ready.get() {
            self.condition = ClauseCondition::None;
        }
    }

    /// A merge action does not pickle, as PySpark's, whose merge builder holds a client with a
    /// thread-local state that pickle cannot copy.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::reduce::thread_local_reduce(slf.as_any())
    }

    #[pyo3(name = "updateAll")]
    #[doc = connect_doc!(MergeIntoWriter.WhenNotMatchedBySource.updateAll)]
    fn update_all(&self, py: Python<'_>) -> PyResult<Py<PyMergeIntoWriter>> {
        let (cond, _) = merge_clause(py, &self.condition, None)?;
        merge_clause_action(py, &self.writer, |w| {
            w.when_not_matched_by_source(cond).update_all()
        })
    }

    #[doc = connect_doc!(MergeIntoWriter.WhenNotMatchedBySource.update)]
    fn update(
        &self,
        py: Python<'_>,
        assignments: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyMergeIntoWriter>> {
        let (cond, assignments) = merge_clause(py, &self.condition, Some(assignments))?;
        merge_clause_action(py, &self.writer, |w| {
            w.when_not_matched_by_source(cond).update(assignments)
        })
    }

    #[doc = connect_doc!(MergeIntoWriter.WhenNotMatchedBySource.delete)]
    fn delete(&self, py: Python<'_>) -> PyResult<Py<PyMergeIntoWriter>> {
        let (cond, _) = merge_clause(py, &self.condition, None)?;
        merge_clause_action(py, &self.writer, |w| {
            w.when_not_matched_by_source(cond).delete()
        })
    }
}

/// The parts of the state of a writer, as Python's garbage collector sees them: a Python object
/// that each writer that holds the state, and each shallow copy of it, which shares the state,
/// holds a reference to and reports, and which reports the parts of the state once, as PySpark's
/// copies of a writer share the plan object of its DataFrame (see [`WriterState::try_parts`]).
#[pyclass(frozen, name = "WriterParts", module = "pyspark._pyspark")]
pub(crate) struct WriterParts {
    state: std::sync::Arc<dyn VisitParts>,
}

/// A state whose parts a [`WriterParts`] reports.
trait VisitParts: Send + Sync {
    fn visit_parts(&self, visit: &pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError>;
}

impl<T: Clone + Send> VisitParts for WriterState<T> {
    /// Report the parts of the state, unless another thread holds their lock, or they are of an
    /// earlier state than the state's, until its change gives the state its own (see
    /// [`WriterState::refresh_parts`]).
    fn visit_parts(&self, visit: &pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        match self.try_parts() {
            Some(parts) if parts.0 == self.generation.load(std::sync::atomic::Ordering::SeqCst) => {
                parts.1.visit(visit)
            }
            _ => Ok(()),
        }
    }
}

#[pymethods]
impl WriterParts {
    /// The parts of the state, which the writers that share it reach through this object.
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        self.state.visit_parts(&visit)
    }
}

/// The [`WriterParts`] of `state`, for a writer that holds it, which its shallow copies share;
/// None if the object is not made, as on an error of memory, of which the parts are not reported.
pub(crate) fn writer_parts<T: Clone + Send + 'static>(
    state: &std::sync::Arc<WriterState<T>>,
) -> Option<Py<WriterParts>> {
    let state: std::sync::Arc<dyn VisitParts> = state.clone();
    Python::attach(|py| Py::new(py, WriterParts { state }).ok())
}

/// The state of a Python writer (`df.write`, `df.writeTo`, `df.mergeInto`, `df.writeStream`).
/// PySpark's writers are plain Python objects, so their methods may run on several threads at
/// once; a method that releases the GIL for its request must not hold a Rust borrow of the
/// object (that would make the calls of the other threads fail with "Already borrowed"). So
/// the lock is held only to copy or replace the state, and a request runs on its own copy.
pub(crate) struct WriterState<T> {
    state: std::sync::Mutex<T>,
    /// The number of the changes of the state, which a change counts under the lock of the
    /// state, so that the parts are known to be of the state as it is (see `refresh_parts`).
    generation: std::sync::atomic::AtomicU64,
    /// The parts of the values of the core that the state holds (see [`crate::gc::Parts`]), of a
    /// state of a writer of a plan, which gives them (`held`), and the generation of the state
    /// that they are of: the writer that alone holds the state reports them, while they are of
    /// the state as it is, as PySpark's writer holds the plan of its DataFrame and its Columns.
    /// The lock is held only to replace them or to report them.
    parts: std::sync::Mutex<(u64, crate::gc::Parts)>,
    held: Option<HeldParts<T>>,
}

/// The references of a state of a writer to the parts of the values of the core that it holds.
type HeldParts<T> = for<'a> fn(&'a T) -> Vec<spark_connect::gc::PartRef<'a>>;

impl<T: Clone> WriterState<T> {
    /// The state of a writer of a plan, of the parts that `held` gives of it.
    pub(crate) fn of_parts(state: T, held: HeldParts<T>) -> Self {
        let parts = Python::attach(|py| crate::gc::Parts::of(py, held(&state)));
        WriterState {
            state: std::sync::Mutex::new(state),
            generation: std::sync::atomic::AtomicU64::new(0),
            parts: std::sync::Mutex::new((0, parts)),
            held: Some(held),
        }
    }

    /// A state of a copy of this one that `clear` cleared, for the `__clear__` of a writer, whose
    /// copies keep this one, of no parts.
    pub(crate) fn cleared(&self, clear: impl FnOnce(&mut T)) -> Self {
        let mut state = self.get();
        clear(&mut state);
        WriterState {
            state: std::sync::Mutex::new(state),
            generation: std::sync::atomic::AtomicU64::new(0),
            parts: std::sync::Mutex::new((0, crate::gc::Parts::none())),
            held: self.held,
        }
    }

    /// The parts of the state and their generation, unless another thread holds their lock, for
    /// the `__traverse__` of the writer that alone holds this state, which must not wait for it.
    fn try_parts(&self) -> Option<std::sync::MutexGuard<'_, (u64, crate::gc::Parts)>> {
        self.parts.try_lock().ok()
    }

    /// Give the state the parts of the values that it holds now, after it was replaced: they are
    /// those of a copy of it, which holds the same references, made with no lock held, as the
    /// [`crate::gc::SharedPart`]s that they make are Python objects, of which a collection may
    /// clear a copy of a writer of this state, which reads it. They take the place of parts of
    /// an earlier generation of the state only, as another thread may have changed the state and
    /// given it its parts meanwhile, which Python code that `Parts::of` runs lets it do; until
    /// then, those of an earlier generation are not reported (see `visit_parts`).
    fn refresh_parts(&self) {
        let Some(held) = self.held else {
            return;
        };
        let (generation, state) = {
            let state = self.lock();
            (
                self.generation.load(std::sync::atomic::Ordering::SeqCst),
                state.clone(),
            )
        };
        let parts = Python::attach(|py| crate::gc::Parts::of(py, held(&state)));
        drop(state);
        self.give_parts(generation, parts);
    }

    /// Give the state `parts`, of its generation `generation`, unless it has those of a later
    /// one; the parts that are left are dropped once the lock is released.
    fn give_parts(&self, generation: u64, parts: crate::gc::Parts) {
        let left = {
            let mut slot = self
                .parts
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if generation > slot.0 {
                std::mem::replace(&mut *slot, (generation, parts)).1
            } else {
                parts
            }
        };
        drop(left);
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, T> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// A copy of the state, to run a request or a builder step on.
    pub(crate) fn get(&self) -> T {
        self.lock().clone()
    }

    pub(crate) fn set(&self, state: T) {
        let replaced = {
            let mut current = self.lock();
            self.generation
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            std::mem::replace(&mut *current, state)
        };
        drop(replaced);
        self.refresh_parts();
    }

    /// Replace the state by `f` of it, as the builder methods do (`f` only builds, no Python).
    pub(crate) fn update(&self, f: impl FnOnce(T) -> T) {
        let replaced = {
            let mut state = self.lock();
            let updated = f(state.clone());
            self.generation
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            std::mem::replace(&mut *state, updated)
        };
        drop(replaced);
        self.refresh_parts();
    }
}

// ---- Shared helpers for closure-based (UDF) DataFrame/GroupedData methods ----

/// cloudpickle an arbitrary Python object to bytes (mirrors the scalar UDF path,
/// where the client serializes the callable for server-side execution).
///
/// Uses `CloudPickleSerializer().dumps` as PySpark does, which raises a `pickle.PicklingError`
/// ("Could not serialize object: ...") for an object that does not pickle, rather than the
/// bare pickling error of `cloudpickle`.
pub(crate) fn py_cloudpickle(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    let serializer = py
        .import("pyspark.serializers")?
        .getattr("CloudPickleSerializer")?
        .call0()?;
    serializer.call_method1("dumps", (obj,))?.extract()
}

/// The running Python's "major.minor" (recorded in the UDF payload).
pub(crate) fn py_version(py: Python<'_>) -> String {
    (|| -> Option<String> {
        let vi = py.import("sys").ok()?.getattr("version_info").ok()?;
        let major: i64 = vi.getattr("major").ok()?.extract().ok()?;
        let minor: i64 = vi.getattr("minor").ok()?.extract().ok()?;
        Some(format!("{major}.{minor}"))
    })()
    .unwrap_or_else(|| "3.11".to_string())
}

/// The arguments of a call, bound to `names` positionally and by keyword as Python binds a
/// signature of those names with `defaults`, raising Python's `TypeError`s of `function`
/// for too many, repeated or unexpected arguments. For a method that tells a value of None
/// from an argument not given.
fn bind_arguments<'py, const N: usize>(
    function: &str,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
    names: [&str; N],
    defaults: [Bound<'py, PyAny>; N],
) -> PyResult<[Bound<'py, PyAny>; N]> {
    if args.len() > N {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{function}() takes from 1 to {} positional arguments but {} were given",
            N + 1,
            args.len() + 1
        )));
    }
    let mut values: [Option<Bound<'py, PyAny>>; N] = std::array::from_fn(|_| None);
    for (i, value) in args.iter().enumerate() {
        values[i] = Some(value);
    }
    if let Some(kwargs) = kwargs {
        for (key, value) in kwargs.iter() {
            let key: String = key.extract()?;
            let Some(i) = names.iter().position(|name| *name == key) else {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "{function}() got an unexpected keyword argument '{key}'"
                )));
            };
            if values[i].is_some() {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "{function}() got multiple values for argument '{key}'"
                )));
            }
            values[i] = Some(value);
        }
    }
    let mut defaults = defaults.into_iter();
    Ok(values.map(|value| {
        let default = defaults.next().expect("a default per name");
        value.unwrap_or(default)
    }))
}

/// Resolve the return type of a map or group function (a `DataType` wrapper or a DDL string): a
/// DDL string into the type that the server of `session` parses it to, as PySpark's `_parse_ddl`,
/// and a type object into None, which [`map_udf_of`] lowers when it builds the function, as the
/// `returnType` of PySpark's `UserDefinedFunction` reads it then, after the columns of the input.
/// Another object raises as [`check_return_type_argument`].
pub(crate) fn resolve_datatype(
    py: Python<'_>,
    session: &spark_connect::session::SparkSession,
    schema: &Bound<'_, PyAny>,
) -> PyResult<Option<spark_connect::types::DataType>> {
    if schema.extract::<PyRef<PyDataType>>().is_ok() {
        return Ok(None);
    }
    if schema.is_instance_of::<pyo3::types::PyString>() {
        return crate::types::schema_data_type(py, session, schema).map(Some);
    }
    Err(not_expected_type(schema, "returnType", "DataType or str"))
}

/// Raises as the constructor of PySpark's `UserDefinedFunction` does for a return type that is
/// neither a `DataType` nor a str, `NOT_EXPECTED_TYPE` of `returnType`, before any request.
pub(crate) fn check_return_type_argument(schema: &Bound<'_, PyAny>) -> PyResult<()> {
    if schema.extract::<PyRef<PyDataType>>().is_ok()
        || schema.is_instance_of::<pyo3::types::PyString>()
    {
        return Ok(());
    }
    Err(not_expected_type(schema, "returnType", "DataType or str"))
}

/// The DataFrame of `mapInPandas` or `mapInArrow` of `df`, of the function `udf`, as PySpark's
/// `MapPartitions` plan sets its values when it is built: the function, then the flag `barrier`,
/// as protobuf converts it, and then the id of the resource profile `profile`, if any, which it
/// reads of `profile.id` then, so the profile is created on the server by the first request that
/// builds the plan, as `collect` (`schema` is of the DataFrame's cached schema), not by this call.
/// A flag that protobuf does not take fails each request, after the function is pickled, and the
/// id is not read, as PySpark's build stops at the flag; the DataFrame keeps its cached schema,
/// as PySpark's `_map_partitions` sets it of the schema.
fn map_partitions_of(
    py: Python<'_>,
    df: &DataFrame,
    udf: CommonInlineUserDefinedFunctionExpression,
    barrier: &ProtoBool,
    profile: Option<Bound<'_, PyAny>>,
) -> PyResult<DataFrame> {
    let mut args = crate::plan_args::PlanArgs::new();
    let is_barrier = args.converted(py, barrier.value(py))?;
    let reader = match profile.filter(|profile| !profile.is_none()) {
        Some(profile) if args.error().is_none() => Some(profile_id_reader(profile.unbind())),
        _ => None,
    };
    Ok(args.plan(df.map_in_python_with_profile_reader(udf, is_barrier, reader)))
}

/// The reader of `profile.id`, which a request calls when it builds the plan of `mapInPandas` or
/// `mapInArrow`, as PySpark's `MapPartitions` reads it then, of the int32 field of the proto: the
/// `AttributeError` of an object that has no `id`, the error of the command that creates the
/// profile, or that of a value that the field does not take, is the request's, as it is.
fn profile_id_reader(profile: Py<PyAny>) -> spark_connect::udf::CommandPickler {
    spark_connect::udf::CommandPickler::new(profile, |profile: &Py<PyAny>| {
        Python::attach(|py| {
            let id = profile.bind(py).getattr("id")?;
            let id: i32 = proto_int(&id, "spark.connect.MapPartitions.profile_id")?;
            Ok(id.to_le_bytes().to_vec())
        })
        .map_err(crate::errors::callback_error)
    })
}

/// The DataFrame `dataframe` of a map or group-map function of `schema`, the schema that the call
/// was given, as PySpark's `_map_partitions`, `applyInPandas`, `applyInArrow` and those of a
/// cogroup cache it: a `StructType` is the cached schema of the DataFrame as the object that it
/// is, so `schema` gives a copy of it as it is then and `columns` its names then, also once it
/// was changed in place, as the plan sends it as it is then (`udf_command::udf_pickler`). The
/// core keeps the type of a DDL string, which PySpark parses into an object of its own.
pub(crate) fn map_result_of(schema: &Bound<'_, PyAny>, dataframe: DataFrame) -> PyDataFrame {
    let mut result = PyDataFrame::new(dataframe);
    if schema.is_instance_of::<crate::types::PyStructType>() {
        result.schema_object = Some(schema.clone().unbind());
    }
    result
}

/// Validate the function of a map UDF for its eval type with
/// `pyspark.sql.pandas.functions._validate_vectorized_udf`, as the reference client does
/// before it builds the UDF.
pub(crate) fn validate_map_function(
    py: Python<'_>,
    func: &Bound<'_, PyAny>,
    eval_type_id: i32,
) -> PyResult<()> {
    py.import("pyspark.sql.pandas.functions")?
        .call_method1("_validate_vectorized_udf", (func, eval_type_id))?;
    Ok(())
}

/// The eval type of a grouped map function from its type hints, by `infer` of
/// `pyspark.sql.pandas.typehints` (the iterator forms, for example), as the reference
/// client infers it. `default` is used when the hints give none, and when they cannot be
/// read, after the reference client's `warning`, of the line `line` of its
/// `pyspark.sql.connect.group` (see `warn_as_pyspark`).
pub(crate) fn infer_group_map_eval_type(
    py: Python<'_>,
    func: &Bound<'_, PyAny>,
    infer: &str,
    warning: &str,
    line: u32,
    default: i32,
) -> PyResult<i32> {
    match py
        .import("pyspark.sql.pandas.typehints")?
        .call_method1(infer, (func,))
    {
        Ok(inferred) if inferred.is_none() => Ok(default),
        Ok(inferred) => inferred.extract(),
        Err(err) if err.is_instance_of::<pyo3::exceptions::PyException>(py) => {
            crate::errors::warn_as_pyspark(
                py,
                "pyspark.sql.connect.group",
                line,
                warning,
                &py.get_type::<pyo3::exceptions::PyUserWarning>(),
            )?;
            Ok(default)
        }
        Err(err) => Err(err),
    }
}

/// The name of the UDF of `func`, as PySpark's `UserDefinedFunction` names it: the
/// function's `__name__`, or the name of its class for a callable object without one.
fn udf_name(func: &Bound<'_, PyAny>) -> PyResult<String> {
    if func.hasattr("__name__")? {
        func.getattr("__name__")?.extract()
    } else {
        Ok(func.get_type().name()?.to_string())
    }
}

/// Build a map/group UDF expression: its command is `(func, schema)`, cloudpickled by each
/// request that builds the plan, and it carries the resolved schema as the output type (mirrors
/// the scalar UDF command layout).
/// The return type is resolved here, after the caller read the columns of the input, as the
/// `returnType` of PySpark's `UserDefinedFunction` parses a DDL string when the map node builds
/// the function, which `applyInPandasWithState` and `transformWithState` do.
pub(crate) fn build_map_udf(
    py: Python<'_>,
    session: &spark_connect::session::SparkSession,
    func: &Bound<'_, PyAny>,
    schema: &Bound<'_, PyAny>,
    eval_type_id: i32,
    input_cols: &[String],
) -> PyResult<CommonInlineUserDefinedFunctionExpression> {
    let return_type = resolve_datatype(py, session, schema)?;
    map_udf_of(py, func, schema, return_type, eval_type_id, input_cols)
}

/// The map/group UDF expression of [`build_map_udf`], of `return_type`, the type of `schema`
/// that [`resolve_datatype`] gives: for a function whose DDL string PySpark parses before it
/// reads the columns of the input, as `mapInPandas`, `applyInPandas` and their Arrow forms do,
/// and cogroups, which bind no columns.
pub(crate) fn map_udf_of(
    py: Python<'_>,
    func: &Bound<'_, PyAny>,
    schema: &Bound<'_, PyAny>,
    return_type: Option<spark_connect::types::DataType>,
    eval_type_id: i32,
    input_cols: &[String],
) -> PyResult<CommonInlineUserDefinedFunctionExpression> {
    let name = udf_name(func)?;
    // As PySpark's `MapPartitions`, `GroupMap` and the other map nodes do when they build the
    // UDF, the return type is checked for the eval type: a map or group function returns a
    // struct, so another type raises `INVALID_RETURN_TYPE_FOR_PANDAS_UDF` here, not at the action.
    // A type object is checked as it is, whose Arrow type of metadata that `json.dumps` refuses is
    // its `NOT_IMPLEMENTED`, and then lowered, as the live type of the object: an edit in place
    // of a StructType's fields or metadata after it was built is kept, as PySpark does.
    let checked = match &return_type {
        Some(return_type) => crate::types::data_type_to_py(py, return_type)?.into_bound(py),
        None => schema.clone(),
    };
    py.import("pyspark.sql.udf")?
        .getattr("UserDefinedFunction")?
        .call_method1("_check_return_type", (checked, eval_type_id))?;
    let return_type = match return_type {
        Some(return_type) => return_type,
        // A struct whose fields hold an element that is not a `StructField`, which the build
        // reads as PySpark's does (see `crate::types::py_to_proto_data_type`), stands as the types
        // that stand for its elements until then.
        None => {
            let (lowered, maybe_foreign) =
                crate::types::lower_noting_subclasses(schema, crate::types::MetadataOf::Strict);
            match crate::types::foreign_of_lowering(schema, maybe_foreign)? {
                true => crate::types::lenient_type(schema)?,
                false => lowered?,
            }
        }
    };
    // The worker reads the pickled type as a `DataType`, so a DDL string is pickled as the
    // type it parses to, as the reference client pickles the type of `_parse_ddl`.
    let pickled_type = if schema.is_instance_of::<pyo3::types::PyString>() {
        crate::types::data_type_to_py(py, &return_type)?.into_bound(py)
    } else {
        schema.clone()
    };
    // As PySpark's `PythonUDF.to_plan`, each request that builds the plan converts the output
    // type of the schema and pickles it with the function, not this call: so a `StructType`
    // changed in place after the call is the type that the server gets, and the command's.
    let tup = PyTuple::new(py, [func.clone(), pickled_type])?;
    let pickler = crate::udf_command::udf_pickler(tup.into_any().unbind());
    let payload =
        PythonUDFPayload::pickled_at_build(return_type, eval_type_id, pickler, py_version(py));
    // The worker reconstructs the input pandas/Arrow DataFrame of a map or group function from
    // the UDF's argument columns; without them the input columns are unnamed (KeyError in the
    // user fn). Mirror the reference Connect client, which passes the child's columns as the
    // function arguments, and none for a cogroup function, which takes the whole cogroups: each
    // the `ColumnReference` of the name, of `to_expr` of a UDF call, so a column named `*` or
    // `s.*` is that column, not a star.
    let arguments = input_cols
        .iter()
        .map(spark_connect::expression::Expression::column)
        .collect();
    Ok(CommonInlineUserDefinedFunctionExpression::new(
        name, true, arguments, payload,
    ))
}

/// The converters of the fields of a result of `schema` that deserialize its values of
/// user-defined types, as PySpark does when it collects them
/// (`pyspark.sql.types._udt_field_converters`); None without a user-defined type.
fn udt_field_converters(
    py: Python<'_>,
    schema: Option<&spark_connect::types::DataType>,
) -> PyResult<Option<FieldConverters>> {
    let Some(schema) = schema.filter(|schema| crate::types::has_converted_values(schema)) else {
        return Ok(None);
    };
    // As PySpark's, which converts the rows of the schema of the server, of a new object of each
    // user-defined type (see `schema_to_py`).
    let schema = crate::types::schema_to_py(py, schema)?;
    let converters = py
        .import("pyspark.sql.types")?
        .call_method1("_udt_field_converters", (schema,))?;
    if converters.is_none() {
        return Ok(None);
    }
    Ok(Some(std::sync::Arc::new(converters.extract()?)))
}

/// Whether `data_type` holds a year-month interval, also nested in an array, a map, a struct or
/// a user-defined type (`pyspark.sql.types._has_type`).
fn has_year_month_interval(data_type: &spark_connect::types::DataType) -> bool {
    has_type(data_type, &|t| {
        matches!(t, spark_connect::types::DataType::YearMonthInterval { .. })
    })
}

/// Whether `data_type` is of `is`, also nested in an array, a map, a struct or a user-defined
/// type.
fn has_type(
    data_type: &spark_connect::types::DataType,
    is: &dyn Fn(&spark_connect::types::DataType) -> bool,
) -> bool {
    use spark_connect::types::DataType;
    if is(data_type) {
        return true;
    }
    match data_type {
        DataType::Struct { fields } => fields.iter().any(|field| has_type(&field.data_type, is)),
        DataType::Array { element_type, .. } => has_type(element_type, is),
        DataType::Map {
            key_type,
            value_type,
            ..
        } => has_type(key_type, is) || has_type(value_type, is),
        DataType::Udt {
            sql_type: Some(sql_type),
            ..
        } => has_type(sql_type, is),
        _ => false,
    }
}

/// Raises as PySpark's `collect` does for a result of `schema` with a calendar interval: it
/// checks the Arrow schema of the result with `from_arrow_schema`, which has no type of the
/// Arrow `month_day_nano_interval` of a calendar interval. Its `toLocalIterator` does not.
fn check_arrow_schema(
    py: Python<'_>,
    schema: Option<&spark_connect::types::DataType>,
) -> PyResult<()> {
    let calendar = |t: &spark_connect::types::DataType| {
        matches!(t, spark_connect::types::DataType::CalendarInterval)
    };
    if schema.is_some_and(|schema| has_type(schema, &calendar)) {
        return Err(crate::column::raise_pyspark(
            py,
            "PySparkTypeError",
            "UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION",
            &[("data_type", "month_day_nano_interval")],
        ));
    }
    Ok(())
}

/// The proto text of the first nanosecond-precision timestamp of `data_type`, `TIMESTAMP_NTZ(p)`
/// or `TIMESTAMP_LTZ(p)` with p > 6, in the order of PySpark's conversion of a schema from its
/// proto: the message that PySpark's `proto_schema_to_pyspark_data_type` names when it fails.
pub(crate) fn nanosecond_timestamp_text(
    data_type: &spark_connect::types::DataType,
) -> Option<String> {
    use spark_connect::types::DataType;
    match data_type {
        DataType::TimestampNtzNanos { precision } => Some(format!(
            "timestamp_ntz_nanos {{\n  precision: {precision}\n}}\n"
        )),
        DataType::TimestampLtzNanos { precision } => Some(format!(
            "timestamp_ltz_nanos {{\n  precision: {precision}\n}}\n"
        )),
        DataType::Struct { fields } => fields
            .iter()
            .find_map(|field| nanosecond_timestamp_text(&field.data_type)),
        DataType::Array { element_type, .. } => nanosecond_timestamp_text(element_type),
        DataType::Map {
            key_type,
            value_type,
            ..
        } => nanosecond_timestamp_text(key_type).or_else(|| nanosecond_timestamp_text(value_type)),
        DataType::Udt {
            sql_type: Some(sql_type),
            ..
        } => nanosecond_timestamp_text(sql_type),
        _ => None,
    }
}

/// A type that PySpark's conversion of a schema from its proto, `proto_schema_to_pyspark_data_type`,
/// fails to build.
enum UnconvertibleType {
    /// A nanosecond-precision timestamp, which PySpark has no type of: the proto text of it.
    NanosecondTimestamp(String),
    /// A spatial type of an SRID that PySpark's `GeometryType` or `GeographyType` does not take.
    SpatialSrid(i32),
}

/// The first type of `data_type` that PySpark's conversion of a schema from its proto fails to
/// build, in the order in which it converts them (the fields of a struct, the element of an
/// array, the key and then the value of a map): a nanosecond-precision timestamp (see
/// [`nanosecond_timestamp_text`]), or a spatial type of an SRID that PySpark does not take (see
/// [`crate::types::pyspark_converts_srid`]). PySpark builds a user-defined type of its Python
/// class, not of its SQL type, of which only a nanosecond timestamp counts.
fn unconvertible_type(data_type: &spark_connect::types::DataType) -> Option<UnconvertibleType> {
    use spark_connect::types::DataType;
    let spatial = |srid: i32, geographic: bool| {
        (!crate::types::pyspark_converts_srid(srid, geographic))
            .then_some(UnconvertibleType::SpatialSrid(srid))
    };
    match data_type {
        DataType::Geometry { srid } => spatial(*srid, false),
        DataType::Geography { srid } => spatial(*srid, true),
        DataType::Struct { fields } => fields
            .iter()
            .find_map(|field| unconvertible_type(&field.data_type)),
        DataType::Array { element_type, .. } => unconvertible_type(element_type),
        DataType::Map {
            key_type,
            value_type,
            ..
        } => unconvertible_type(key_type).or_else(|| unconvertible_type(value_type)),
        _ => nanosecond_timestamp_text(data_type).map(UnconvertibleType::NanosecondTimestamp),
    }
}

/// Raises as PySpark does for a schema that it cannot convert from the proto of the schema (see
/// [`unconvertible_type`]): `UNSUPPORTED_OPERATION` for a nanosecond-precision timestamp, and
/// `ST_INVALID_SRID_VALUE` for a spatial type of an SRID that PySpark does not take. The
/// collecting APIs and every schema surface raise (the schema, columns, dtypes, printSchema,
/// repr, a DDL string of a schema), though the Rust core keeps these values, and count, show and
/// describe work.
pub(crate) fn check_schema_conversion(
    py: Python<'_>,
    schema: Option<&spark_connect::types::DataType>,
) -> PyResult<()> {
    match schema.and_then(unconvertible_type) {
        Some(UnconvertibleType::NanosecondTimestamp(text)) => Err(crate::column::raise_pyspark(
            py,
            "PySparkValueError",
            "UNSUPPORTED_OPERATION",
            &[("operation", &format!("data type {text}"))],
        )),
        Some(UnconvertibleType::SpatialSrid(srid)) => {
            Err(crate::types::invalid_schema_srid(py, srid))
        }
        None => Ok(()),
    }
}

/// Raises as PySpark's conversion of `schema`, a schema of the server, raises where its
/// `_schema` makes the type objects of it: the errors of [`check_schema_conversion`], and those of
/// its user-defined types, of which `UserDefinedType.fromJson` makes `UDT()` of the class that
/// the `pyClass` names (see `types::schema_to_py`), so that a class whose `__init__` takes
/// arguments raises its `TypeError`. The properties that read the schema, the column lookups,
/// the copies and the unpickles raise it so.
pub(crate) fn check_python_schema(
    py: Python<'_>,
    schema: &spark_connect::types::DataType,
) -> PyResult<()> {
    check_schema_conversion(py, Some(schema))?;
    if crate::types::has_udt(schema) {
        crate::types::schema_to_py(py, schema)?;
    }
    Ok(())
}

/// The `_cached_schema` of PySpark's DataFrame of `dataframe`, of the schema that the core caches,
/// which is of the server: its type objects, as PySpark converts the schema of the server, of a
/// new object of each user-defined type (see `types::schema_to_py`), and not of the objects that
/// the client was given. None when it has none, or when PySpark would not have one, as its
/// conversion of the schema raised (see [`check_python_schema`]).
fn known_schema_object(py: Python<'_>, dataframe: &DataFrame) -> PyResult<Option<Py<PyAny>>> {
    let Some(schema) = dataframe.known_schema() else {
        return Ok(None);
    };
    if check_schema_conversion(py, Some(&schema)).is_err() {
        return Ok(None);
    }
    match crate::types::schema_to_py(py, &schema) {
        Ok(schema) => Ok(Some(schema)),
        Err(err) if err.is_instance_of::<pyo3::exceptions::PyException>(py) => Ok(None),
        Err(err) => Err(err),
    }
}

/// The result of a set operation of `this` and `other`, `dataframe`, which keeps the schema that
/// PySpark's `_merge_cached_schema` keeps: `this.schema` when the cached schemas of both are
/// equal, else none. The core keeps that of two cached structs; of a schema that a DataFrame
/// keeps as a Python object, the schema of `to` that is not a DataType or the `StructType` of a
/// map function, the objects are compared as they are now, as PySpark compares them with `==`,
/// with the type objects of the cached schema of the other, which of the server holds new objects
/// of its user-defined types (see `known_schema_object`), and the result keeps a deep copy of
/// that of `this`, as `this.schema` gives. Of a copy or an unpickle, `this.schema` is the schema
/// that it requested (see `copy_schema`).
fn set_operation_result(
    py: Python<'_>,
    this: &PyDataFrame,
    other: &PyDataFrame,
    dataframe: DataFrame,
) -> PyResult<PyDataFrame> {
    if this.schema_object.is_none() && other.schema_object.is_none() {
        return Ok(PyDataFrame::new(
            match (this.copy_schema(), dataframe.known_schema()) {
                (Some(requested), Some(_)) => dataframe.with_known_schema(Some(requested)),
                _ => dataframe,
            },
        ));
    }
    let cached = |df: &PyDataFrame| -> PyResult<Option<Py<PyAny>>> {
        match &df.schema_object {
            Some(schema) => Ok(Some(schema.clone_ref(py))),
            None => known_schema_object(py, &df.dataframe),
        }
    };
    if let (Some(left), Some(right)) = (cached(this)?, cached(other)?) {
        if left.bind(py).eq(right.bind(py))? {
            if let Some(requested) = this.copy_schema() {
                return Ok(PyDataFrame::new(
                    dataframe.with_known_schema(Some(requested)),
                ));
            }
            return Ok(match &this.schema_object {
                Some(schema) => {
                    let mut result = PyDataFrame::new(dataframe.with_known_schema(None));
                    result.schema_object = Some(
                        py.import("copy")?
                            .call_method1(pyo3::intern!(py, "deepcopy"), (schema.bind(py),))?
                            .unbind(),
                    );
                    result
                }
                None => {
                    PyDataFrame::new(dataframe.with_known_schema(this.dataframe.known_schema()))
                }
            });
        }
    }
    Ok(PyDataFrame::new(dataframe.with_known_schema(None)))
}

/// The schema that the properties of a DataFrame read, as PySpark's read its `_schema` (see
/// `PyDataFrame::property_schema`).
enum PropertySchema<'py> {
    /// The schema of the server.
    Core(spark_connect::types::DataType),
    /// An object, which PySpark's properties read as they read any `_schema`: the schema of `to`
    /// that is not a DataType, or the Column of `_schema` of the `__getattr__` of the DataFrame.
    Object(Bound<'py, PyAny>),
}

impl<'py> PropertySchema<'py> {
    /// As PySpark's `schema`: the type object of the schema of the server, of a new object of each
    /// user-defined type (see `schema_to_py`), or a deep copy of the object, as PySpark's
    /// `copy.deepcopy` of its `_schema`.
    fn schema(self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        match self {
            PropertySchema::Core(schema) => crate::types::schema_to_py(py, &schema),
            PropertySchema::Object(schema) => Ok(py
                .import("copy")?
                .call_method1("deepcopy", (schema,))?
                .unbind()),
        }
    }

    /// As PySpark's `columns`, `[field.name for field in self._schema.fields]`.
    fn columns(self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        match self {
            PropertySchema::Core(spark_connect::types::DataType::Struct { fields }) => {
                PyList::new(py, fields.into_iter().map(|field| field.name))
            }
            PropertySchema::Core(_) => unreachable!("schema() validates struct responses"),
            PropertySchema::Object(schema) => {
                let names = PyList::empty(py);
                for field in schema.getattr("fields")?.try_iter()? {
                    names.append(field?.getattr("name")?)?;
                }
                Ok(names)
            }
        }
    }

    /// As PySpark's `dtypes`, of the `simpleString()` of the type of each field: of a user-defined
    /// type, such as `vector`, which only its Python class knows, of a new object of it, as
    /// PySpark's `dtypes` reads its `_schema` (see `schema_to_py`).
    fn dtypes(self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        match self {
            PropertySchema::Core(spark_connect::types::DataType::Struct { fields }) => {
                let pairs = fields
                    .into_iter()
                    .map(|field| {
                        let name = if crate::types::has_udt(&field.data_type) {
                            crate::types::schema_to_py(py, &field.data_type)?
                                .bind(py)
                                .call_method0("simpleString")?
                                .extract()?
                        } else {
                            crate::types::simple_string_with_udts(py, &field.data_type)?
                        };
                        Ok((field.name, name))
                    })
                    .collect::<PyResult<Vec<(String, String)>>>()?;
                PyList::new(py, pairs)
            }
            PropertySchema::Core(_) => unreachable!("schema() validates struct responses"),
            PropertySchema::Object(schema) => object_dtypes(py, &schema),
        }
    }
}

/// The `(str(field.name), field.dataType.simpleString())` of each field of `schema`, a schema of
/// `to` that is not a DataType, as PySpark's `dtypes` reads them: an object without `fields`, or
/// whose fields lack these attributes, raises the error of Python for it.
fn object_dtypes<'py>(py: Python<'py>, schema: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let pairs = PyList::empty(py);
    for field in schema.getattr("fields")?.try_iter()? {
        let field = field?;
        let name = field.getattr("name")?.str()?;
        let simple = field.getattr("dataType")?.call_method0("simpleString")?;
        pairs.append((name, simple))?;
    }
    Ok(pairs)
}

/// The error that PySpark's read of the columns of a DataFrame, `[field.name for field in
/// self._schema.fields]`, raises of `err`, the error of its schema: of an `AttributeError`, of a
/// plan that does not build, the property `_schema` falls back to the Column of the `__getattr__`
/// of the DataFrame, whose `fields` is a Column, which is not iterable: `NOT_ITERABLE` of
/// `Column`. Another error is raised as it is.
fn columns_read_error(py: Python<'_>, err: PyErr) -> PyErr {
    if !err.is_instance_of::<pyo3::exceptions::PyAttributeError>(py) {
        return err;
    }
    crate::column::raise_pyspark(
        py,
        "PySparkTypeError",
        "NOT_ITERABLE",
        &[("objectName", "Column")],
    )
}

/// The DataFrame of `plan`, a plan of the Python client of `session`, as `DataFrame(plan, session)`
/// makes it, and the error that it keeps when its build failed: of the relation of
/// `plan.to_proto(session.client)`, of
/// the plan ID of the plan. A plan whose build raises an `Exception` gives a DataFrame that keeps
/// it, of which each request raises a new exception ([`crate::errors::kept_error`]), as each of
/// PySpark's requests builds the plan and raises; another exception of the build, such as the
/// `KeyboardInterrupt` of a Ctrl-C while it runs or a `SystemExit`, is raised now, as it would be
/// where PySpark's program is then, which goes on after it.
pub(crate) fn dataframe_of_python_plan(
    plan: &Bound<'_, PyAny>,
    session: &Bound<'_, crate::session::PySparkSession>,
) -> PyResult<(DataFrame, Option<spark_connect_core::error::SparkError>)> {
    use prost::Message;
    let py = plan.py();
    let core_session = session.borrow().session.clone();
    let client = session.getattr("client")?;
    // The plan, as PySpark Connect's DataFrame keeps it, keeps the models whose cached objects on
    // the server the relation refers to (`__source_instance__`).
    let owner: std::sync::Arc<dyn std::any::Any + Send + Sync> =
        std::sync::Arc::new(plan.clone().unbind());
    // Of the session of the build, the plans that the Columns of the subqueries of the plan build
    // again (`errors::of_request_session`), as PySpark's plan builds them of its client.
    let built = crate::errors::of_request_session(&core_session, || {
        plan.call_method1("to_proto", (client,))
            .and_then(|built| built.getattr("root"))
            .and_then(|relation| relation.call_method0("SerializeToString"))
            .and_then(|relation| {
                // With the type URLs of placeholders of its encoding, of which the core finds
                // those of its relations as it walks them (see `plan::BuiltOutside`).
                spark_connect::plan::BuiltOutside::decoded(
                    relation.cast::<pyo3::types::PyBytes>()?.as_bytes(),
                    spark_connect_proto::Relation::decode,
                )
                .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
            })
            .and_then(|relation| {
                DataFrame::from_relation(
                    core_session.clone(),
                    relation,
                    Some(std::sync::Arc::clone(&owner)),
                )
                .to_pyerr()
            })
    });
    let (dataframe, kept) = match built {
        Ok(dataframe) => (dataframe, None),
        Err(exception) if exception.is_instance_of::<pyo3::exceptions::PyException>(py) => {
            let (error, owner) = crate::errors::kept_error(py, plan, exception)?;
            (
                DataFrame::from_unbuilt_relation(core_session, Some(owner), error.clone()),
                Some(error),
            )
        }
        Err(exception) => return Err(exception),
    };
    // The plan ID of the plan, of the counter of the core (`LogicalPlan._fresh_plan_id`), of no
    // other node; one that is not an int, of an object that is not such a plan, leaves the plan
    // ID of the core.
    if let Some(plan_id) = plan
        .getattr(pyo3::intern!(py, "_plan_id"))
        .ok()
        .filter(|id| id.is_exact_instance_of::<pyo3::types::PyInt>())
        .and_then(|id| id.extract::<i64>().ok())
    {
        dataframe.assign_plan_id(plan_id);
    }
    Ok((dataframe, kept))
}

/// The DataFrame of `plan`, a plan of the Python client whose build failed when an object over it
/// was pickled, which the pickle carried, as PySpark's pickle carries its plan objects: built
/// again, as PySpark builds the plan when the unpickled object is used, of its error if it raises
/// again ([`dataframe_of_python_plan`]). Of an object `derived` from the DataFrame of the plan,
/// such as a DataFrame over it or a Column of its subquery, whose own plan the pickle could not
/// carry, a plan that builds now gives a DataFrame whose requests raise the error of
/// [`derived_plan_error`]. With the error of each request of the DataFrame, if it has one.
pub(crate) fn rebuilt_python_plan(
    plan: &Bound<'_, PyAny>,
    session: &Bound<'_, crate::session::PySparkSession>,
    derived: bool,
) -> PyResult<(DataFrame, Option<spark_connect_core::error::SparkError>)> {
    let (dataframe, kept) = dataframe_of_python_plan(plan, session)?;
    if kept.is_some() || !derived {
        return Ok((dataframe, kept));
    }
    let error = derived_plan_error();
    Ok((
        DataFrame::from_unbuilt_relation(session.borrow().session.clone(), None, error.clone()),
        Some(error),
    ))
}

/// The error of the requests of an unpickled object over a plan of the Python client that failed
/// to build when the object was pickled and builds when it was unpickled: the pickle carries the
/// plan, and not the plan of the object over it, which it could not build.
pub(crate) fn derived_plan_error() -> spark_connect_core::error::SparkError {
    spark_connect_core::error::SparkError::builtin_runtime_msg(
        "the plan of the Python client that the pickled object was made over failed to build \
         when it was pickled and builds when it is unpickled, so the pickle has no plan of the \
         object itself",
    )
}

/// The DataFrame of `plan`, if it is the plan of a DataFrame of the core, `DataFrame._plan`
/// (`pyspark.sql.connect.plan._DataFramePlan`, of that class itself): a DataFrame of it is of the
/// plan of that DataFrame, as PySpark's `DataFrame(df._plan, session)` is of the plan object of
/// `df`.
fn dataframe_of_plan<'py>(plan: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyDataFrame>>> {
    let py = plan.py();
    static PLAN_OF_DATAFRAME: pyo3::sync::PyOnceLock<Py<pyo3::types::PyType>> =
        pyo3::sync::PyOnceLock::new();
    let class = PLAN_OF_DATAFRAME.import(py, "pyspark.sql.connect.plan", "_DataFramePlan")?;
    if !plan.get_type().is(class) {
        return Ok(None);
    }
    Ok(plan
        .getattr(pyo3::intern!(py, "_df"))
        .ok()
        .and_then(|df| df.cast_into::<PyDataFrame>().ok()))
}

/// The error of a value that protobuf refuses in a field of a plan, which PySpark's plan raises as
/// a request sends it: a builtin `TypeError`, `OverflowError` or `ValueError`, as [`proto_int`]
/// and [`proto_bool`] raise them. Another error is raised at the call.
pub(crate) fn unbuildable_error(
    py: Python<'_>,
    err: PyErr,
) -> PyResult<spark_connect_core::error::SparkError> {
    use spark_connect_core::error::{SparkError, SparkErrorKind};
    let kind = if err.is_instance_of::<pyo3::exceptions::PyTypeError>(py) {
        SparkErrorKind::BuiltinTypeError
    } else if err.is_instance_of::<pyo3::exceptions::PyOverflowError>(py) {
        SparkErrorKind::BuiltinOverflowError
    } else if err.is_instance_of::<pyo3::exceptions::PyValueError>(py)
        && !err.is_instance_of::<pyo3::exceptions::PyUnicodeError>(py)
    {
        SparkErrorKind::BuiltinValueError
    } else {
        return Err(err);
    };
    let message = err.value(py).str()?.to_string();
    Ok(SparkError {
        kind,
        ..SparkError::value_msg(message)
    })
}

/// The plan of `df` with `plan` of the value `value`, which PySpark converts as it sends the plan:
/// a value that protobuf refuses makes `df` fail each request with its error, as
/// [`unbuildable_error`] gives it, with the schema that `df` cached, as PySpark's `limit`,
/// `coalesce`, `repartition` and `repartitionById` copy the `_cached_schema` of their input.
/// Another error is raised at the call.
fn converted_plan<T>(
    py: Python<'_>,
    df: &DataFrame,
    value: PyResult<T>,
    plan: impl FnOnce(T) -> DataFrame,
) -> PyResult<DataFrame> {
    match value {
        Ok(value) => Ok(plan(value)),
        Err(err) => Ok(df.unbuildable_with_cached_schema(unbuildable_error(py, err)?)),
    }
}

/// The error that PySpark's plan raises when it sends the str `text` of a protobuf string field,
/// which fails to convert as [`crate::options::str_argument`] gives it: a str that UTF-8 cannot
/// encode, bytes that are not UTF-8, or an object that is neither. None for a value that converts.
pub(crate) fn unbuildable_str_error(
    text: &Bound<'_, PyAny>,
) -> PyResult<Option<spark_connect_core::error::SparkError>> {
    use pyo3::types::PyBytes;
    use spark_connect_core::error::SparkError;
    let err = match crate::options::str_argument(text) {
        Ok(_) => return Ok(None),
        Err(err) => err,
    };
    let py = text.py();
    if err.is_instance_of::<pyo3::exceptions::PyUnicodeEncodeError>(py) {
        // The UTF-32 bytes of the str, which keep the surrogates, as `unencodable_str` takes them.
        let utf32le = text.call_method1("encode", ("utf-32-le", "surrogatepass"))?;
        let utf32le = utf32le.cast::<PyBytes>()?;
        return Ok(Some(SparkError::unencodable_str(utf32le.as_bytes())));
    }
    if err.is_instance_of::<pyo3::exceptions::PyUnicodeDecodeError>(py) {
        let bytes = text.cast::<PyBytes>()?;
        return Ok(Some(SparkError::invalid_utf8(0, bytes.as_bytes())));
    }
    Ok(Some(unbuildable_error(py, err)?))
}

/// The `eager` flag of a checkpoint, which PySpark passes to the constructor of its command: true
/// when it is omitted, and false for None, which the constructor leaves unset. Another value is
/// converted as [`proto_bool`] converts it.
fn checkpoint_eager(py: Python<'_>, eager: &DefaultedOption) -> PyResult<bool> {
    match eager {
        DefaultedOption::Omitted => Ok(true),
        DefaultedOption::Given(value) => {
            let value = value.bind(py);
            if value.is_none() {
                Ok(false)
            } else {
                proto_bool(value)
            }
        }
    }
}

/// The error that [`check_schema_conversion`] raises for a schema that it cannot convert, as the
/// core error of the same PySpark error class and parameters.
fn unconvertible_schema_error(
    schema: &spark_connect::types::DataType,
) -> Option<spark_connect_core::error::SparkError> {
    use spark_connect_core::error::{SparkError, SparkErrorKind};
    unconvertible_type(schema).map(|unconvertible| match unconvertible {
        UnconvertibleType::NanosecondTimestamp(text) => SparkError::classed(
            SparkErrorKind::ValueError,
            "UNSUPPORTED_OPERATION",
            &[("operation", &format!("data type {text}"))],
        ),
        UnconvertibleType::SpatialSrid(srid) => SparkError::classed(
            SparkErrorKind::IllegalArgument,
            "ST_INVALID_SRID_VALUE",
            &[("srid", &srid.to_string())],
        ),
    })
}

/// The check of the schema that the collecting APIs run as the server sends it, as
/// [`unconvertible_schema_error`]: PySpark raises the error of a schema that it cannot convert
/// when it receives the schema, before the rows and any later failure of the execution.
fn check_schema_at_receipt(
    schema: &spark_connect::types::DataType,
) -> spark_connect_core::error::Result<()> {
    match unconvertible_schema_error(schema) {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// Raises as PySpark's Arrow conversion of the rows does for a result of `schema` with a
/// year-month interval, which Spark Connect cannot collect: the error is raised even when the
/// result has no rows.
fn check_collectable(
    py: Python<'_>,
    schema: Option<&spark_connect::types::DataType>,
) -> PyResult<()> {
    if schema.is_some_and(has_year_month_interval) {
        return Err(crate::column::raise_pyspark(
            py,
            "PySparkNotImplementedError",
            "NOT_IMPLEMENTED",
            &[(
                "feature",
                "Collecting a year-month interval value in Spark Connect",
            )],
        ));
    }
    Ok(())
}

/// The binary result setting, as PySpark's `DataFrame._get_binary_as_bytes` reads it,
/// including when the server sends only an Arrow schema or the result has no binary fields.
/// Collect reads it after execution; a local iterator reads it once its plan is built, before
/// the execution starts.
fn read_binary_as_bytes(py: Python<'_>, dataframe: &DataFrame) -> PyResult<bool> {
    let session = dataframe.spark_session();
    let value = crate::interrupt::interruptible_without_parts(py, &session, || {
        session
            .conf()
            .get_with_default("spark.sql.execution.pyspark.binaryAsBytes", Some("true"))
    })?;
    Ok(value.is_some_and(|value| value.eq_ignore_ascii_case("true")))
}

/// The seed of a sample without one, drawn from Python's `random` as PySpark's
/// `random.randint(0, sys.maxsize)`, so that `random.seed` makes the rows repeat.
pub(crate) fn python_random_seed(py: Python<'_>) -> PyResult<i64> {
    let max_size = py.import("sys")?.getattr("maxsize")?;
    py.import("random")?
        .call_method1("randint", (0, max_size))?
        .extract()
}

/// The collected `rows` of a result of `schema`, with configured binary values and
/// user-defined values materialized before constructing the tuples returned to Python.
/// Reads the binary setting after execution, including with only an Arrow schema.
fn rows_to_py(
    py: Python<'_>,
    dataframe: &DataFrame,
    rows: Vec<spark_connect::row::Row>,
    schema: Option<&spark_connect::types::DataType>,
) -> PyResult<Vec<Py<PyAny>>> {
    check_schema_conversion(py, schema)?;
    check_arrow_schema(py, schema)?;
    let binary_as_bytes = read_binary_as_bytes(py, dataframe)?;
    check_collectable(py, schema)?;
    let converters = udt_field_converters(py, schema)?;
    let Some(first) = rows.first() else {
        return Ok(Vec::new());
    };
    let fields = PyList::new(py, first.fields())?;
    let mut nested_fields = NestedFieldNames::default();
    if let Some(schema) = schema {
        nested_fields.note_schema(schema);
    }
    // Drop each native payload after conversion instead of retaining two full results.
    rows.into_iter()
        .map(|row| {
            crate::row::row_values_to_py(
                py,
                &fields,
                row.values(),
                converters.as_ref(),
                binary_as_bytes,
                &mut nested_fields,
            )
        })
        .collect()
}

/// Python wrapper for LocalRowIterator yielding Row objects lazily: the rows that the generator
/// of `toLocalIterator` yields from. Implements `__iter__`, `__next__` and `close`.
#[pyclass(name = "LocalRowIterator", module = "pyspark.sql.dataframe")]
pub struct PyLocalRowIterator {
    /// The reference of the rows to the parts of the plan of `frame`, of which `dataframe` is a
    /// clone, which `__traverse__` reports (see [`crate::gc::Parts`]). The clone that the
    /// execution holds while it is prepared is not covered by it, which keeps the parts alive
    /// meanwhile.
    parts: crate::gc::Parts,
    dataframe: DataFrame,
    /// The DataFrame of `toLocalIterator`, as the generator of PySpark's holds `self` until it
    /// ends: the generator of the drop-in holds this iterator until then.
    frame: Py<PyDataFrame>,
    /// The execution of the rows once its plan is built, which the first row sends.
    prepared: Option<PreparedExecution>,
    /// The rows of the execution, which the first row starts, as PySpark's generator does.
    iterator: Option<LocalRowIterator>,
    /// Whether the iteration ended, with its last row or an error, as a generator ends.
    finished: bool,
    /// The converters of the fields of user-defined types of the rows, once the schema of the
    /// result is known: with the first row.
    converters: Option<Option<FieldConverters>>,
    /// Whether the conversion of the first record batch of the result was checked, once it
    /// arrived.
    batch_checked: bool,
    /// Read before the execution starts, after the plan is built, as the body of PySpark's
    /// generator reads it.
    binary_as_bytes: Option<bool>,
    /// Top-level names shared only by rows of the current Arrow record batch.
    fields: Option<(std::sync::Arc<[String]>, Py<PyList>)>,
    /// Nested names belong to schema paths and stay shared across iterator batches.
    nested_fields: NestedFieldNames,
}

impl PyLocalRowIterator {
    /// The rows of the DataFrame `frame`, of a clone of its plan.
    pub fn new(frame: &Bound<'_, PyDataFrame>) -> Self {
        let dataframe = frame.borrow().dataframe.clone();
        let parts = crate::gc::Parts::of(frame.py(), dataframe.held_parts());
        let frame = frame.clone().unbind();
        PyLocalRowIterator {
            parts,
            dataframe,
            frame,
            prepared: None,
            iterator: None,
            finished: false,
            converters: None,
            batch_checked: false,
            binary_as_bytes: None,
            fields: None,
            nested_fields: NestedFieldNames::default(),
        }
    }

    /// The next row, sending the prepared execution for the first; it fetches the next batch of
    /// the result when the batch that the iterator holds is used up.
    fn next_row(&mut self) -> Option<spark_connect_core::error::Result<spark_connect::row::Row>> {
        if let Some(prepared) = self.prepared.take() {
            match prepared.into_local_iterator(false) {
                Ok(iterator) => self.iterator = Some(iterator),
                Err(error) => return Some(Err(error)),
            }
        }
        self.iterator.as_mut()?.next()
    }

    /// Prepare and fetch one Python row; the public wrapper closes on exhaustion or error.
    fn next_python_row(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let session = self.dataframe.spark_session();
        let binary_as_bytes = match self.binary_as_bytes {
            Some(value) => value,
            None => {
                // As the body of PySpark's generator: the plan is built first (`to_proto`), so a
                // plan that does not build raises before any request, then the binary setting is
                // read, and then the execution starts.
                let prepared = interruptible_of(py, &session, &[&self.parts], || {
                    self.dataframe.prepare_execution().map_err(Box::new)
                })?;
                let value = read_binary_as_bytes(py, &self.dataframe)?;
                self.prepared = Some(prepared);
                self.binary_as_bytes = Some(value);
                value
            }
        };
        // A row of the batch that the iterator holds is taken with the GIL held: only the fetch
        // of a batch waits for the server, which releases it.
        let next = match self
            .iterator
            .as_mut()
            .and_then(LocalRowIterator::next_decoded)
        {
            Some(row) => Some(Ok(row)),
            // The first fetch sends the execution, of the request context of the client; the
            // others continue it, as PySpark's generator reads nothing of the client again.
            None if self.prepared.is_some() => {
                let _in_use = crate::gc::in_use(py, &[&self.parts]);
                interruptible(py, &session, || out_of_the_wait(self.next_row()))?
            }
            None => {
                let _in_use = crate::gc::in_use(py, &[&self.parts]);
                crate::interrupt::interruptible_continuation(py, &session, || {
                    out_of_the_wait(self.next_row())
                })?
            }
        };
        if let Some(iterator) = self.iterator.as_ref() {
            if self.converters.is_none() {
                if let Some(schema) = iterator.schema() {
                    // As PySpark's conversion of the schema of the response that carries it,
                    // before the rows and a later failure of the execution.
                    check_schema_conversion(py, Some(&schema))?;
                    self.converters = Some(udt_field_converters(py, Some(&schema))?);
                    self.nested_fields.note_schema(&schema);
                }
            }
            if !self.batch_checked && iterator.has_received_batch() {
                // As PySpark's `ArrowTableToRowsConversion.convert` of each record batch as it
                // arrives, also of one without rows, as the server sends for an empty result: its
                // year-month error is raised with the first record batch, before the error of a
                // later response, and not for an execution that fails before any.
                self.batch_checked = true;
                check_collectable(py, iterator.schema().as_ref())?;
            }
        }
        match next {
            Some(Ok(row)) => {
                let converters = self.converters.clone().flatten();
                let fields = match row.shared_fields() {
                    Some(native_fields) => match self.fields.as_ref() {
                        Some((cached, fields)) if std::sync::Arc::ptr_eq(cached, native_fields) => {
                            fields.bind(py).clone()
                        }
                        _ => {
                            let fields = PyList::new(py, row.fields())?;
                            self.fields = Some((
                                std::sync::Arc::clone(native_fields),
                                fields.clone().unbind(),
                            ));
                            fields
                        }
                    },
                    None => {
                        self.fields = None;
                        PyList::new(py, row.fields())?
                    }
                };
                crate::row::row_values_to_py(
                    py,
                    &fields,
                    row.values(),
                    converters.as_ref(),
                    binary_as_bytes,
                    &mut self.nested_fields,
                )
                .map(Some)
            }
            Some(Err(error)) => Err(crate::errors::spark_error_to_py_exception(error)),
            None => Ok(None),
        }
    }
}

/// `next`, the next row of an iterator of rows, of its error as the error of the wait of the call
/// when it is the exception of Python code of the client that is a `BaseException` only, as the
/// `KeyboardInterrupt` of a handler of the logs at a response, which goes out of PySpark's
/// generator before the row and the checks of its schema, and which the wait converts, as the
/// `except KeyboardInterrupt` of PySpark's execution, into the interrupt of the execution (see
/// `crate::interrupt::interruptible`). The other errors of a row are raised once the schema of the
/// result that came before them is checked.
#[allow(clippy::type_complexity)]
fn out_of_the_wait(
    next: Option<spark_connect_core::error::Result<spark_connect::row::Row>>,
) -> Result<
    Option<spark_connect_core::error::Result<spark_connect::row::Row>>,
    Box<spark_connect_core::error::SparkError>,
> {
    match next {
        Some(Err(error)) if crate::errors::is_base_exception_only(&error) => Err(Box::new(error)),
        next => Ok(next),
    }
}

impl PyLocalRowIterator {
    /// The iterator, for a call that runs or closes it. The fetch of a batch releases the GIL
    /// with the iterator borrowed, so that a second thread can call it meanwhile: its state is
    /// never shared, as the borrow is exclusive, and the second call raises the `ValueError` of
    /// a generator that is already executing, as PySpark's generator does, rather than the
    /// `RuntimeError` of the borrow.
    fn not_executing<'py>(slf: &Bound<'py, Self>) -> PyResult<PyRefMut<'py, Self>> {
        slf.try_borrow_mut()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("generator already executing"))
    }

    /// End the iteration, releasing its execution without the GIL.
    fn finish(&mut self, py: Python<'_>) {
        let iterator = self.iterator.take();
        self.prepared = None;
        self.finished = true;
        self.fields = None;
        self.nested_fields.clear();
        crate::gc::detach_without_parts(py, move || drop(iterator));
    }
}

#[pymethods]
impl PyLocalRowIterator {
    /// Make this object an iterator by returning self.
    fn __iter__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }

    /// Close the iterator, as a generator does: the rows that are left are not fetched, so the
    /// execution of the query is released, and the iteration ends. Closing it again does nothing.
    fn close(slf: &Bound<'_, Self>) -> PyResult<()> {
        Self::not_executing(slf)?.finish(slf.py());
        Ok(())
    }

    /// Fetch the next row from the iterator.
    ///
    /// The first call builds the plan, then reads the binary result setting, then starts the
    /// execution, as PySpark's generator does. The Config RPC and the fetch of each batch release
    /// the GIL; the rows of a fetched batch are converted with it held.
    fn __next__(slf: &Bound<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        let py = slf.py();
        let mut this = Self::not_executing(slf)?;
        if this.finished {
            return Ok(None);
        }
        match this.next_python_row(py) {
            Ok(Some(row)) => Ok(Some(row)),
            result => {
                this.finish(py);
                result
            }
        }
    }

    /// As the generator of PySpark's `toLocalIterator`, which holds its DataFrame, a Python object
    /// that Python's garbage collector sees: the DataFrame is reported, and the parts of the plan
    /// of which the rows hold a clone, such as the functions of its UDFs (see [`crate::gc`]). So a
    /// reference cycle through the rows and a function of the plan, such as a bound method of an
    /// object that keeps the rows, is freed. The iterator has no `__dict__`: it is tracked when it
    /// is allocated, and its value is written before any allocation that could start a
    /// collection, so no [`crate::gc::GcReady`] is needed.
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        visit.call(&self.frame)?;
        self.parts.visit(&visit)
    }

    /// Break a reference cycle that the garbage collector found unreachable: the iterator drops
    /// its clone of the plan, and its reference to its parts.
    fn __clear__(&mut self) {
        self.parts.release();
        self.prepared = None;
        self.dataframe.clear_plan();
    }
}

/// The metrics and observed metrics of an execution, as the lists of PySpark's client: of the
/// `PlanMetrics` that `SparkConnectClient._build_metrics` makes, and of the `PlanObservedMetrics`
/// that `_execute_and_fetch_as_iterator` makes of the `Expression.Literal` messages that the
/// server sent, which their `pairs` converts when it is read.
pub(crate) fn execution_metrics<'py>(
    py: Python<'py>,
    info: &spark_connect::session::ExecutionInfo,
) -> PyResult<(
    Bound<'py, pyo3::types::PyList>,
    Bound<'py, pyo3::types::PyList>,
)> {
    let metrics_mod = py.import("pyspark.sql.metrics")?;
    let plan_metrics_cls = metrics_mod.getattr("PlanMetrics")?;
    let metric_value_cls = metrics_mod.getattr("MetricValue")?;
    let plan_metrics = pyo3::types::PyList::empty(py);
    if let Some(m) = &info.metrics {
        for x in &m.metrics {
            let values = pyo3::types::PyList::empty(py);
            for (k, v) in &x.execution_metrics {
                values.append(metric_value_cls.call1((k, v.value, &v.metric_type))?)?;
            }
            plan_metrics.append(plan_metrics_cls.call1((&x.name, x.plan_id, x.parent, values))?)?;
        }
    }
    let observed_cls = py
        .import("pyspark.sql.connect.client.core")?
        .getattr("PlanObservedMetrics")?;
    // The metrics as PySpark's client keeps them, the `Expression.Literal` messages that the
    // server sent, of `pyspark.sql.connect.proto`, which `PlanObservedMetrics.pairs` converts
    // when it is read, and so raises the `UNSUPPORTED_LITERAL` of a literal that PySpark does not
    // convert there. They are parsed of any depth, as the transport has read their response
    // within the limit of upb's parser (see `spark_connect_proto::ClientCodec`).
    let literal_cls = py
        .import("pyspark.sql.connect.proto")?
        .getattr("Expression")?
        .getattr("Literal")?;
    let observed = pyo3::types::PyList::empty(py);
    for o in &info.observed_metrics {
        let values = o
            .values
            .iter()
            .map(|literal| {
                use prost::Message;
                crate::pbmsg::parse_deep(&literal_cls, &literal.encode_to_vec())
            })
            .collect::<PyResult<Vec<_>>>()?;
        observed.append(observed_cls.call1((&o.name, values, o.keys.clone()))?)?;
    }
    Ok((plan_metrics, observed))
}

/// Whether `a` and `b` are the information of one execution, of which `executionInfo` gives one
/// object: of the same operation, which the core makes for each execution, with the same metrics
/// and observed metrics. An observed metric of a NaN, which is not equal to itself, compares by
/// its encoding.
fn same_execution(
    a: &spark_connect::session::ExecutionInfo,
    b: &spark_connect::session::ExecutionInfo,
) -> bool {
    use prost::Message;
    a.operation_id == b.operation_id
        && a.metrics == b.metrics
        && a.observed_metrics.len() == b.observed_metrics.len()
        // Of literals of up to `spark_connect_proto::MAX_KEPT_DEPTH` levels.
        && (a.observed_metrics.is_empty()
            || crate::on_grown_stack(|| {
                a.observed_metrics
                    .iter()
                    .zip(&b.observed_metrics)
                    .all(|(x, y)| x == y || x.encode_to_vec() == y.encode_to_vec())
            }))
}

/// Convert an observed-metric literal to a Python value, as the metrics of an Observation.
/// Mirrors `LiteralExpression._to_value`: the kinds it decodes map to the same Python values,
/// and for any other kind (which the reference rejects with `UNSUPPORTED_LITERAL`, and which
/// `check_observed_metrics` refuses before) this returns an `_UnsupportedLiteral` marker.
/// `PlanObservedMetrics` keeps the literals themselves (see [`execution_metrics`]).
///
/// `data_type` is the literal's type when the enclosing literal carries it; a struct's
/// field names come from it, else from the literal's own `data_type`, else from the
/// deprecated `struct_type`.
pub(crate) fn literal_to_py(
    py: Python<'_>,
    literal: &spark_connect_proto::expression::Literal,
    data_type: Option<&spark_connect_proto::DataType>,
) -> PyResult<Py<PyAny>> {
    // A level of a literal of many levels, of the observed metrics of a response, on a stack with
    // room.
    crate::pbmsg::grow(|| literal_to_py_level(py, literal, data_type))
}

/// [`literal_to_py`] of one level of a literal.
fn literal_to_py_level(
    py: Python<'_>,
    literal: &spark_connect_proto::expression::Literal,
    data_type: Option<&spark_connect_proto::DataType>,
) -> PyResult<Py<PyAny>> {
    use pyo3::IntoPyObjectExt;
    use spark_connect_proto::data_type::Kind;
    use spark_connect_proto::expression::literal::LiteralType;
    let data_type = data_type.or(literal.data_type.as_ref());
    let datetime = || py.import("datetime");
    Ok(match &literal.literal_type {
        None | Some(LiteralType::Null(_)) => py.None(),
        Some(LiteralType::Boolean(b)) => b.into_py_any(py)?,
        Some(LiteralType::Byte(v))
        | Some(LiteralType::Short(v))
        | Some(LiteralType::Integer(v)) => v.into_py_any(py)?,
        Some(LiteralType::Long(v)) => v.into_py_any(py)?,
        Some(LiteralType::Float(v)) => (*v as f64).into_py_any(py)?,
        Some(LiteralType::Double(v)) => v.into_py_any(py)?,
        Some(LiteralType::String(s)) => s.into_py_any(py)?,
        Some(LiteralType::Binary(b)) => pyo3::types::PyBytes::new(py, b).into_any().unbind(),
        Some(LiteralType::Decimal(d)) => py
            .import("decimal")?
            .getattr("Decimal")?
            .call1((&d.value,))?
            .unbind(),
        // DateType.fromInternal: days since the epoch (ordinal 719163).
        Some(LiteralType::Date(days)) => datetime()?
            .getattr("date")?
            .call_method1("fromordinal", (*days as i64 + 719_163,))?
            .unbind(),
        // TimeType.fromInternal: nanoseconds of the day, truncated to microseconds.
        Some(LiteralType::Time(t)) => {
            let nano = t.nano;
            datetime()?
                .getattr("time")?
                .call1((
                    nano.div_euclid(3_600_000_000_000),
                    nano.rem_euclid(3_600_000_000_000) / 60_000_000_000,
                    nano.rem_euclid(60_000_000_000) / 1_000_000_000,
                    nano.rem_euclid(1_000_000_000) / 1_000,
                ))?
                .unbind()
        }
        // TimestampType.fromInternal: a naive local datetime, built from whole seconds so
        // the microseconds are exact.
        Some(LiteralType::Timestamp(ts)) => {
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("microsecond", ts.rem_euclid(1_000_000))?;
            datetime()?
                .getattr("datetime")?
                .call_method1("fromtimestamp", (ts.div_euclid(1_000_000),))?
                .call_method("replace", (), Some(&kwargs))?
                .unbind()
        }
        // TimestampNTZType.fromInternal: the UTC wall clock, without a zone.
        Some(LiteralType::TimestampNtz(ts)) => {
            let dt = datetime()?;
            let utc = dt.getattr("timezone")?.getattr("utc")?;
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("microsecond", ts.rem_euclid(1_000_000))?;
            kwargs.set_item("tzinfo", py.None())?;
            dt.getattr("datetime")?
                .call_method1("fromtimestamp", (ts.div_euclid(1_000_000), utc))?
                .call_method("replace", (), Some(&kwargs))?
                .unbind()
        }
        // DayTimeIntervalType.fromInternal: timedelta(microseconds=v).
        Some(LiteralType::DayTimeInterval(micros)) => {
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("microseconds", micros)?;
            datetime()?
                .getattr("timedelta")?
                .call((), Some(&kwargs))?
                .unbind()
        }
        Some(LiteralType::Array(a)) => {
            let element_type = match data_type.and_then(|t| t.kind.as_ref()) {
                Some(Kind::Array(t)) => t.element_type.as_deref(),
                _ => None,
            };
            let items = a
                .elements
                .iter()
                .map(|e| literal_to_py(py, e, element_type))
                .collect::<PyResult<Vec<_>>>()?;
            items.into_py_any(py)?
        }
        Some(LiteralType::Map(m)) => {
            let (key_type, value_type) = match data_type.and_then(|t| t.kind.as_ref()) {
                Some(Kind::Map(t)) => (t.key_type.as_deref(), t.value_type.as_deref()),
                _ => (None, None),
            };
            let dict = pyo3::types::PyDict::new(py);
            for (k, v) in m.keys.iter().zip(&m.values) {
                dict.set_item(
                    literal_to_py(py, k, key_type)?,
                    literal_to_py(py, v, value_type)?,
                )?;
            }
            dict.into_any().unbind()
        }
        Some(LiteralType::Struct(s)) => {
            #[allow(deprecated)]
            let struct_type = match data_type.and_then(|t| t.kind.as_ref()) {
                Some(Kind::Struct(t)) => Some(t),
                _ => match s.struct_type.as_ref().and_then(|t| t.kind.as_ref()) {
                    Some(Kind::Struct(t)) => Some(t),
                    _ => None,
                },
            };
            let fields = struct_type.map(|t| t.fields.as_slice()).unwrap_or_default();
            let names = (0..s.elements.len())
                .map(|i| match fields.get(i) {
                    Some(f) => f.name.clone(),
                    None => format!("_{}", i + 1),
                })
                .collect::<Vec<_>>();
            let values = s
                .elements
                .iter()
                .enumerate()
                .map(|(i, e)| {
                    literal_to_py(py, e, fields.get(i).and_then(|f| f.data_type.as_ref()))
                })
                .collect::<PyResult<Vec<_>>>()?;
            py.import("pyspark.sql.types")?
                .getattr("_create_row")?
                .call1((names, values))?
                .unbind()
        }
        Some(_) => unsupported_literal_marker(py, literal)?,
    })
}

/// The `_UnsupportedLiteral` marker of `literal`, a literal of a kind that `literal_to_py` does
/// not convert, of its text: the value of such a metric of an Observation, whose metrics
/// `check_observed_metrics` refuses before they are converted, raising PySpark's
/// `UNSUPPORTED_LITERAL`. `PlanObservedMetrics` keeps the literals themselves, which its `pairs`
/// converts with PySpark's `LiteralExpression._to_value` (see [`execution_metrics`]).
fn unsupported_literal_marker(
    py: Python<'_>,
    literal: &spark_connect_proto::expression::Literal,
) -> PyResult<Py<PyAny>> {
    Ok(py
        .import("pyspark.sql.connect.client.core")?
        .getattr("_UnsupportedLiteral")?
        .call1((literal_text(py, literal),))?
        .unbind())
}

/// The first literal of `literal`, itself or an element, whose kind `literal_to_py` does not
/// convert, in the order in which PySpark's `LiteralExpression._to_value` converts them: the
/// literal that PySpark's `UNSUPPORTED_LITERAL` names.
pub(crate) fn unsupported_literal(
    literal: &spark_connect_proto::expression::Literal,
) -> Option<&spark_connect_proto::expression::Literal> {
    use spark_connect_proto::expression::literal::LiteralType;
    match &literal.literal_type {
        None
        | Some(LiteralType::Null(_))
        | Some(LiteralType::Boolean(_))
        | Some(LiteralType::Byte(_))
        | Some(LiteralType::Short(_))
        | Some(LiteralType::Integer(_))
        | Some(LiteralType::Long(_))
        | Some(LiteralType::Float(_))
        | Some(LiteralType::Double(_))
        | Some(LiteralType::String(_))
        | Some(LiteralType::Binary(_))
        | Some(LiteralType::Decimal(_))
        | Some(LiteralType::Date(_))
        | Some(LiteralType::Time(_))
        | Some(LiteralType::Timestamp(_))
        | Some(LiteralType::TimestampNtz(_))
        | Some(LiteralType::DayTimeInterval(_)) => None,
        Some(LiteralType::Array(array)) => array.elements.iter().find_map(unsupported_literal),
        // A key before its value, as a dict comprehension evaluates them.
        Some(LiteralType::Map(map)) => map.keys.iter().zip(&map.values).find_map(|(key, value)| {
            unsupported_literal(key).or_else(|| unsupported_literal(value))
        }),
        Some(LiteralType::Struct(fields)) => fields.elements.iter().find_map(unsupported_literal),
        Some(_) => Some(literal),
    }
}

/// The text of `literal` as PySpark's `str` of its protobuf message gives it, which its
/// `UNSUPPORTED_LITERAL` shows; the Rust debug text of the message without `protobuf`.
pub(crate) fn literal_text(
    py: Python<'_>,
    literal: &spark_connect_proto::expression::Literal,
) -> String {
    use prost::Message;
    let text = || -> PyResult<String> {
        let literal_cls = py
            .import("pyspark.sql.connect.proto")?
            .getattr("Expression")?
            .getattr("Literal")?;
        crate::pbmsg::parse_deep(&literal_cls, &literal.encode_to_vec())?
            .str()?
            .extract()
    };
    text().unwrap_or_else(|_| format!("{literal:?}"))
}

#[cfg(test)]
mod execution_info_tests {
    use super::same_execution;
    use spark_connect::session::ExecutionInfo;
    use spark_connect_proto::execute_plan_response::{Metrics, ObservedMetrics};
    use spark_connect_proto::expression::literal::LiteralType;
    use spark_connect_proto::expression::Literal;

    fn info(operation_id: &str, value: f64) -> ExecutionInfo {
        ExecutionInfo {
            metrics: Some(Metrics::default()),
            observed_metrics: vec![ObservedMetrics {
                name: "metric".to_string(),
                values: vec![Literal {
                    literal_type: Some(LiteralType::Double(value)),
                    ..Default::default()
                }],
                keys: vec!["value".to_string()],
                ..Default::default()
            }],
            operation_id: Some(operation_id.to_string()),
        }
    }

    #[test]
    fn the_information_of_one_execution_is_the_same_with_a_nan_observed_too() {
        assert!(same_execution(&info("a", 1.0), &info("a", 1.0)));
        assert!(same_execution(&info("a", f64::NAN), &info("a", f64::NAN)));
    }

    #[test]
    fn another_operation_or_other_metrics_are_of_another_execution() {
        assert!(!same_execution(&info("a", 1.0), &info("b", 1.0)));
        assert!(!same_execution(&info("a", 1.0), &info("a", 2.0)));
        assert!(!same_execution(&info("a", 1.0), &ExecutionInfo::default()));
        let mut without_metrics = info("a", 1.0);
        without_metrics.metrics = None;
        assert!(!same_execution(&info("a", 1.0), &without_metrics));
    }
}

#[cfg(test)]
mod writer_state_tests {
    use super::{writer_parts, WriterState};
    use pyo3::prelude::*;
    use pyo3::types::{PyList, PyTuple};
    use spark_connect::expression::{Expression, ExpressionNode};
    use spark_connect::gc::PartRef;
    use spark_connect::udf::{CommonInlineUserDefinedFunctionExpression, PythonUDFPayload};
    use std::sync::atomic::Ordering;

    /// The expression of a UDF of `object`, as the binding makes it.
    fn udf_of(object: Py<PyAny>) -> Expression {
        let pickler = crate::udf_command::udf_pickler(object);
        ExpressionNode::CommonInlineUserDefinedFunction(Box::new(
            CommonInlineUserDefinedFunctionExpression::new(
                "f".to_string(),
                true,
                vec![Expression::column("x")],
                PythonUDFPayload::pickled_at_build(
                    spark_connect::types::DataType::Long,
                    100,
                    pickler,
                    "3.11".to_string(),
                ),
            ),
        ))
        .into()
    }

    fn held(state: &Option<Expression>) -> Vec<PartRef<'_>> {
        state.iter().map(PartRef::Expression).collect()
    }

    /// The number of the referents of `parts` that the collector sees, once no call without the
    /// GIL is counted.
    fn referents(py: Python<'_>, parts: &Py<super::WriterParts>) -> usize {
        crate::gc::wait_for_no_counted_call(py);
        let referents = py
            .import("gc")
            .unwrap()
            .call_method1("get_referents", (parts,))
            .unwrap();
        referents.cast::<PyList>().unwrap().len()
    }

    #[test]
    fn the_parts_of_an_earlier_state_take_no_place_of_those_of_a_later_one() {
        let _alone = crate::gc::tests_of_the_state_alone();
        Python::initialize();
        Python::attach(|py| {
            let object = |py| PyTuple::new(py, [py.None()]).unwrap().into_any().unbind();
            let state = std::sync::Arc::new(WriterState::of_parts(Some(udf_of(object(py))), held));
            let parts = writer_parts(&state).unwrap();
            assert_eq!(referents(py, &parts), 1);
            // The refresh of a shallow copy on another thread, which copied the state before a
            // change of this thread, and gives its parts after the change gave its own: those of
            // the later state stay.
            let (generation, copy) = (state.generation.load(Ordering::SeqCst), state.get());
            let stale = crate::gc::Parts::of(py, held(&copy));
            drop(copy);
            state.set(Some(udf_of(object(py))));
            assert_eq!(state.generation.load(Ordering::SeqCst), generation + 1);
            assert_eq!(state.parts.lock().unwrap().0, generation + 1);
            state.give_parts(generation, stale);
            assert_eq!(state.parts.lock().unwrap().0, generation + 1);
            assert_eq!(referents(py, &parts), 1);
            // While a change has not given the state its parts yet, those of the earlier state are
            // not reported.
            state.generation.fetch_add(1, Ordering::SeqCst);
            assert_eq!(referents(py, &parts), 0);
            state.refresh_parts();
            assert_eq!(referents(py, &parts), 1);
        });
    }
}
