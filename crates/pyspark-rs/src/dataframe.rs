//! PyO3 wrapper for spark_connect::dataframe::DataFrame.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt, PyList, PyTuple};
use spark_connect::dataframe::{AsOfJoinOptions, DataFrame, LocalRowIterator, PreparedExecution};
use spark_connect::plan::{AsOfJoinDirection, AsOfJoinType, JoinType};
use spark_connect::udf::{eval_type, CommonInlineUserDefinedFunctionExpression, PythonUDFPayload};

use crate::column::PyColumn;
use crate::errors::{not_expected_type, ResultExt};
use crate::group::PyGroupedData;
use crate::interrupt::interruptible;
use crate::options::{set_option_skip_none, set_options_skip_none, DefaultedOption};
use crate::row::{FieldConverters, NestedFieldNames};
use crate::stat_arguments::proto_int;
use crate::streaming::{proto_bool, ProtoBool, PyDataStreamWriter};
use crate::types::PyDataType;

/// Python wrapper for a Spark DataFrame.
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
    /// The holders of the plan of this DataFrame that `__traverse__` answers for: this DataFrame,
    /// and the grouped data, the writers and the other objects of it that keep it and a clone of
    /// its plan, each with a share (see [`crate::gc::Holders`]).
    plan_holders: crate::gc::Holders,
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

impl PyDataFrame {
    /// The names of the columns of the schema of this DataFrame, as Rust callers need them. Not
    /// a method of the Python class, as PySpark's DataFrame has none of this name: so
    /// `df.column_names` is the column of that name, of `__getattr__`.
    pub(crate) fn column_names(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        if self.schema_object.is_some() {
            return self.columns(py)?.bind(py).extract();
        }
        // The schema is cached, so the names come from it without another request.
        let session = self.dataframe.spark_session();
        let schema = interruptible(py, &session, || self.dataframe.schema())?;
        check_schema_conversion(py, Some(&schema))?;
        interruptible(py, &session, || self.dataframe.columns())
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
    /// by `pyspark.sql.connect.client._to_pandas`.
    pub(crate) fn to_pandas(
        &self,
        py: Python<'_>,
        struct_handling_mode: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let session = self.dataframe.spark_session();
        let execution = interruptible(py, &session, || {
            self.dataframe.prepare_execution().map_err(Box::new)
        })?;
        let settings = interruptible(py, &session, || {
            session.conf().get_many(&[
                "spark.sql.session.timeZone",
                "spark.sql.execution.pandas.structHandlingMode",
                "spark.sql.execution.arrow.pyspark.selfDestruct.enabled",
            ])
        })?;
        let (batches, schema) = interruptible(py, &session, || {
            execution
                .collect_record_batches_checked(check_schema_at_receipt)
                .map_err(Box::new)
        })?;
        check_schema_conversion(py, schema.as_ref())?;
        let mut table = crate::pyarrow_export::record_batches_to_pyarrow(py, batches)?;
        // As SparkConnectClient._execute_and_fetch with self_destruct, which copies each column
        // of each batch into its own allocation before the conversion, with its errors.
        if settings[2].as_deref() == Some("true") {
            table = py
                .import("pyspark.sql.connect.client")?
                .call_method1("_self_destruct_table", (table,))?
                .unbind();
        }
        let schema = match &schema {
            Some(schema) => crate::types::data_type_to_py(py, schema)?,
            None => py.None(),
        };
        let (metrics, observed) = match self.dataframe.execution_info() {
            Ok(info) => execution_metrics(py, &info)?,
            Err(_) => (
                pyo3::types::PyList::empty(py),
                pyo3::types::PyList::empty(py),
            ),
        };
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("timezone", &settings[0])?;
        match struct_handling_mode {
            Some(mode) => kwargs.set_item("struct_handling_mode", mode.str()?)?,
            None => kwargs.set_item("struct_handling_mode", &settings[1])?,
        }
        kwargs.set_item("self_destruct", &settings[2])?;
        let pdf = py.import("pyspark.sql.connect.client")?.call_method(
            "_to_pandas",
            (table, schema, metrics, observed),
            Some(&kwargs),
        )?;
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
        interruptible(py, &self.dataframe.spark_session(), || {
            self.dataframe
                .show_string(n, truncate, vertical)
                .map_err(Box::new)
        })
    }

    pub fn new(dataframe: DataFrame) -> Self {
        PyDataFrame {
            dataframe,
            support_repr_html: std::sync::atomic::AtomicBool::new(false),
            gc_ready: crate::gc::GcReady::new(),
            plan_holders: crate::gc::Holders::new(),
            is_streaming: std::sync::OnceLock::new(),
            semantic_hash: std::sync::OnceLock::new(),
            is_empty: std::sync::OnceLock::new(),
            input_files: std::sync::OnceLock::new(),
            schema_object: None,
            execution_info_object: std::sync::Mutex::new(None),
        }
    }

    /// A share of the holders of the plan of the DataFrame `frame`, for an object that keeps
    /// `frame` and a clone of its plan, of which `shares_plan` tells whether it is a clone of that
    /// plan: else, or when `frame` is borrowed mutably, a share that is not counted, so that
    /// nothing of the plan is reported (see [`crate::gc::Holders`]).
    pub(crate) fn plan_share(
        frame: &Py<PyDataFrame>,
        shares_plan: impl FnOnce(&DataFrame) -> bool,
    ) -> crate::gc::Share {
        Python::attach(|py| match frame.try_borrow(py) {
            Ok(frame) if shares_plan(&frame.dataframe) => frame.plan_holders.share(),
            _ => crate::gc::Share::none(),
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
    fn of_same_schema(&self, mut result: PyDataFrame) -> PyDataFrame {
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
    /// has no schema yet, even when this one has one. The request is that of a DataFrame of the
    /// same relation, which has no schema cached, so that it fails as PySpark's does.
    fn fresh_schema_request(&self, py: Python<'_>) -> PyResult<()> {
        let session = self.dataframe.spark_session();
        let relation = self.dataframe.to_relation().to_pyerr()?;
        let fresh = DataFrame::from_sent_relation(session.clone(), relation);
        interruptible(py, &session, || fresh.schema())?;
        Ok(())
    }

    /// The number of rows and the truncation of the eager evaluation, as PySpark's `int` of
    /// each setting, if it is enabled.
    fn eager_eval(&self, py: Python<'_>) -> PyResult<Option<(i32, i32)>> {
        let session = self.dataframe.spark_session();
        let values = interruptible(py, &session, || session.conf().get_many(&EAGER_EVAL))?;
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
    /// relation of `plan.plan(session.client)`, which the client sends as it is.
    #[new]
    fn py_new(
        plan: &Bound<'_, PyAny>,
        session: &Bound<'_, crate::session::PySparkSession>,
    ) -> PyResult<Self> {
        use prost::Message;
        let relation = plan
            .call_method1("plan", (session.getattr("client")?,))?
            .call_method0("SerializeToString")?;
        let relation = spark_connect_proto::Relation::decode(
            relation.cast::<pyo3::types::PyBytes>()?.as_bytes(),
        )
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        // The plan, as PySpark Connect's DataFrame keeps it, keeps the models whose cached
        // objects on the server the relation refers to (`__source_instance__`).
        let owner: std::sync::Arc<dyn std::any::Any + Send + Sync> =
            std::sync::Arc::new(plan.clone().unbind());
        Ok(PyDataFrame::new(
            DataFrame::from_relation(session.borrow().session.clone(), relation, Some(owner))
                .to_pyerr()?,
        ))
    }

    /// As a DataFrame of PySpark Connect, a Python object whose plan Python's garbage collector
    /// sees: the Python objects that only this DataFrame keeps alive, the plans of `pyspark.ml`
    /// that `DataFrame(plan, session)` keeps, are reported to the collector. So a reference cycle
    /// through this DataFrame is freed, as PySpark frees it: a `pyspark.ml` model whose summary
    /// keeps `model.transform(dataset)`, whose plan keeps the model, is released then. So are the
    /// functions of its UDFs, which a request pickles when it builds the plan, as the plan of
    /// PySpark keeps them: a function may keep the DataFrame, as a bound method of an object that
    /// keeps it does. So are the Python Observations of its `observe`, which its plan holds, as
    /// PySpark's plan holds them, and which may keep it too. The objects of the plan that its
    /// grouped data, its writers and its other
    /// objects that keep it share are reported here too, as these report this DataFrame, as
    /// PySpark's report the DataFrame or its plan (see [`crate::gc::Holders`]).
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        if !self.gc_ready.get() {
            return Ok(());
        }
        let holders = self.plan_holders.count();
        for owner in self.dataframe.exclusive_relation_owners(holders) {
            if let Some(object) = owner.downcast_ref::<Py<PyAny>>() {
                visit.call(object)?;
            }
        }
        // The Python Observations that the plan holds, as PySpark's plan holds them, which may
        // keep this DataFrame.
        for owner in self.dataframe.exclusive_observation_owners(holders) {
            if let Some(object) = owner.downcast_ref::<Py<PyAny>>() {
                visit.call(object)?;
            }
        }
        crate::udf_command::visit_pickler_objects(
            self.dataframe.exclusive_pickler_objects(holders),
            &visit,
        )?;
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
        Ok(())
    }

    /// Break a reference cycle that the garbage collector found unreachable: the DataFrame drops
    /// its plan, and so the Python objects that `__traverse__` reported. The shares of the plan
    /// that its objects hold are not counted any more, as their clones are of the dropped plan.
    fn __clear__(&mut self) {
        if self.gc_ready.get() {
            self.plan_holders = crate::gc::Holders::new();
            self.dataframe.clear_plan();
            self.schema_object = None;
            let kept = self
                .execution_info_object
                .get_mut()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .take();
            drop(kept);
        }
    }

    /// The bytes of the relation of this DataFrame as it is sent, for the plan of `_plan`.
    fn _relation_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let relation = self.dataframe.to_relation().to_pyerr()?;
        Ok(pyo3::types::PyBytes::new(
            py,
            &crate::encode_on_grown_stack(relation),
        ))
    }

    /// The bytes of the placeholder of the relation of this DataFrame, which the plan of `_plan`
    /// sends in its place: shallow, whatever the depth of the DataFrame, and replaced by the
    /// core with that relation when the plan reaches it.
    fn _relation_placeholder<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
        use prost::Message;
        pyo3::types::PyBytes::new(py, &self.dataframe.placeholder_relation().encode_to_vec())
    }

    /// A DataFrame pickles, as PySpark Connect's, of its session, which gives back the active
    /// session of its ID, of its plan, here the bytes of its relation, and of the
    /// `_support_repr_html` and the `_cached_schema` of its state. The session unpickles first: so
    /// on a Python worker of the server, which has no active session, a pickled DataFrame raises
    /// `NO_ACTIVE_SESSION`, as PySpark's does. The unpickled DataFrame has copies of the
    /// Observations of this one (see `DataFrame::from_sent_relation`).
    ///
    /// PySpark's pickle keeps the objects of the plan, which it does not build, and its unpickle
    /// builds the plan, as pickle looks the `__setstate__` of the new DataFrame up through its
    /// `__getattr__`, which reads its columns. So a plan that does not build, of a value that
    /// PySpark converts when it builds the plan, pickles of its error instead of its relation,
    /// which the unpickle raises ([`crate::column::described_error`]).
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        use pyo3::types::PyBytes;
        let py = slf.py();
        let this = slf.borrow();
        let session = this.spark_session(py)?;
        let (relation, build_error) = match crate::origin::pickling(|| this.dataframe.to_relation())
        {
            Ok(relation) => (
                PyBytes::new(py, &crate::encode_on_grown_stack(relation)),
                None,
            ),
            Err(error) => match crate::column::described_error(py, &error)? {
                Some(description) => (PyBytes::new(py, b""), Some(description)),
                None => return Err(crate::errors::spark_error_to_py_exception(error)),
            },
        };
        // The schema that this DataFrame cached, of the types of PySpark, when they hold it: the
        // `StructType` that it keeps as an object, of `to` or of a map function, as it is now, as
        // PySpark's state carries the object.
        let cached_schema = match &this.schema_object {
            Some(schema)
                if schema
                    .bind(py)
                    .is_instance_of::<crate::types::PyStructType>() =>
            {
                Some(schema.clone_ref(py))
            }
            _ => this
                .dataframe
                .known_schema()
                .filter(|schema| check_schema_conversion(py, Some(schema)).is_ok())
                .map(|schema| crate::types::data_type_to_py(py, &schema))
                .transpose()?,
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
            ))
    }

    /// The reduce of a pickle at `protocol`, as PySpark Connect's DataFrame pickles its plan before
    /// its session, of the objects of `pyspark.ml` that the plan keeps (see
    /// `ml::object::pickle_block`, of the plans of its subqueries and of the Python plans of
    /// PySpark's own classes of `pyspark.ml` too): of a plan that keeps a model first (a transform
    /// of a model, a DataFrame of a method of a model or of the summary of an evaluation), the
    /// error of the lock of the model's `RemoteModelRef` at the protocol, before the session; of a
    /// DataFrame of a method of the training summary of a fit, the reduce of `__reduce__`, whose
    /// local function a plain pickle refuses first, as PySpark's refuses the session of the
    /// summary's `_summary_dataset`, and then a lock, as the model of the summary
    /// (`_remote_model_obj`) keeps one, which cloudpickle, of the function, reaches next; else the
    /// reduce of `__reduce__`.
    fn __reduce_ex__<'py>(
        slf: &Bound<'py, Self>,
        protocol: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let dataframe = slf.borrow().dataframe.clone();
        let block = crate::ml::object::pickle_block(py, &dataframe)?;
        let reduced = match block {
            Some(spark_connect::ml::MlPickleBlock::ModelLock) => {
                return Err(crate::ml::object::lock_pickle_error(py, protocol));
            }
            Some(spark_connect::ml::MlPickleBlock::SummarySession(_)) => {
                slf.call_method0(pyo3::intern!(py, "__reduce__"))?
            }
            None => return slf.call_method0(pyo3::intern!(py, "__reduce__")),
        };
        let reduced = reduced.cast_into::<PyTuple>()?;
        let state = PyDict::new(py);
        match reduced.get_item(2).ok() {
            Some(given) if given.is_instance_of::<PyDict>() => state.update(given.cast()?)?,
            Some(given) if !given.is_none() => return Ok(reduced.into_any()),
            _ => {}
        }
        state.set_item("_model_lock", py.import("threading")?.call_method0("Lock")?)?;
        let mut items: Vec<Bound<'py, PyAny>> = reduced.iter().take(2).collect();
        items.push(state.into_any());
        items.extend(reduced.iter().skip(3));
        Ok(PyTuple::new(py, items)?.into_any())
    }

    /// The shallow copy of a DataFrame, as PySpark's `copy.copy` of its `__reduce__`: the same
    /// plan, and so the same Observations, as PySpark's copy shares its plan, with the
    /// `_support_repr_html` and the schema that its state carries, the one that it keeps as an
    /// object, of `to` or of a map function, as the same object, and an `executionInfo` of its
    /// own, None until it executes, as that of PySpark's `DataFrame(plan, session)`.
    fn __copy__(&self, py: Python<'_>) -> PyResult<PyDataFrame> {
        self.fresh_schema_request(py)?;
        let mut copy = self.copy_of(self.dataframe.shallow_copy());
        copy.schema_object = self
            .schema_object
            .as_ref()
            .map(|schema| schema.clone_ref(py));
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
    /// the session: so do these (`deep_copied_picklers`), of which the plan of a copy in the
    /// session of this DataFrame is made, which pickles them at each request, as PySpark's copy
    /// does; a copy in another session is of the relation (`deep_copy_with_pickler_copies`).
    fn __deepcopy__(&self, py: Python<'_>, memo: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        // As PySpark's: `copy.deepcopy` looks up `__deepcopy__` through `__getattr__`, whose check
        // reads the columns, so the schema is requested unless it is cached. On a stopped session,
        // that request raises the SparkConnectException of its closed client, before the session
        // is copied, which raises a PySparkRuntimeError.
        let session = self.dataframe.spark_session();
        interruptible(py, &session, || self.dataframe.schema())?;
        // As PySpark's copy of the plan, which copies the objects of `pyspark.ml` that it keeps
        // (see `__reduce_ex__`): the TypeError of the lock of a model, after a copy of the
        // `_summary_dataset` of the training summary of a DataFrame of a method of the summary, of
        // its requests of its schema.
        if let Some(block) = crate::ml::object::pickle_block(py, &self.dataframe)? {
            if let spark_connect::ml::MlPickleBlock::SummarySession(dataset) = block {
                let dataset = Py::new(py, PyDataFrame::new(dataset))?;
                py.import("copy")?
                    .call_method1(pyo3::intern!(py, "deepcopy"), (dataset, memo))?;
            }
            let protocol = pyo3::types::PyInt::new(py, 4).into_any();
            return Err(crate::ml::object::lock_pickle_error(py, &protocol));
        }
        let observations = self
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
        let copies = crate::udf_command::deep_copied_picklers(py, self.dataframe.picklers(), memo)?;
        let session = py
            .import("copy")?
            .call_method1(
                pyo3::intern!(py, "deepcopy"),
                (self.spark_session(py)?, memo),
            )?
            .extract::<PyRef<'_, crate::session::PySparkSession>>()?
            .session
            .clone();
        let copy = crate::on_grown_stack(|| {
            self.dataframe
                .deep_copy_with_pickler_copies(session, observations, &copies)
        })
        .to_pyerr()?;
        // As PySpark's reconstruction of the copy, which requests its schema, as the copy has none,
        // before it sets the deep copy of the state, of the schema that it keeps as an object too,
        // which the deep copy of the plan shares in `memo`.
        let mut copy = self.copy_of(copy);
        copy.fresh_schema_request(py)?;
        if let Some(schema) = &self.schema_object {
            copy.schema_object = Some(
                py.import("copy")?
                    .call_method1(pyo3::intern!(py, "deepcopy"), (schema.bind(py), memo))?
                    .unbind(),
            );
        }
        Ok(copy)
    }

    /// The DataFrame of `relation`, the bytes of a relation of `levels` levels of `session`, as a
    /// pickled DataFrame unpickles, with the `_support_repr_html` and the `_cached_schema` of its
    /// state. As PySpark's unpickle, whose lookup of the `__setstate__` of the new DataFrame reads
    /// its columns, it requests the schema of the relation, whose error it raises, and then the
    /// DataFrame has the cached schema of the pickled one, none when it had none, so that its
    /// `schema` requests it again then. Without `request_schema`, of an object that PySpark
    /// pickles of the plan of the DataFrame, as a DataStreamWriter, the schema is not requested.
    #[staticmethod]
    #[pyo3(signature = (
        session, relation, levels=1, support_repr_html=false, cached_schema=None, request_schema=true
    ))]
    fn _of_relation(
        py: Python<'_>,
        session: &Bound<'_, crate::session::PySparkSession>,
        relation: &[u8],
        levels: usize,
        support_repr_html: bool,
        cached_schema: Option<&Bound<'_, PyAny>>,
        request_schema: bool,
    ) -> PyResult<PyDataFrame> {
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
            spark_connect_proto::Relation::decode(relation)
                .map(|relation| DataFrame::from_sent_relation_levels(session, relation, levels))
        })
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?
        .to_pyerr()?;
        if request_schema {
            let schema = interruptible(py, &dataframe.spark_session(), || dataframe.schema())?;
            check_schema_conversion(py, Some(&schema))?;
        }
        let dataframe = PyDataFrame::new(dataframe.with_known_schema(known_schema));
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

    /// Select columns (accepts Column objects or string names).
    #[pyo3(signature = (*cols))]
    fn select(&self, _py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        let columns = to_column_list(cols)?;
        Ok(PyDataFrame::new(self.dataframe.select(columns)))
    }

    /// Select using SQL expressions, as PySpark's `DataFrame.selectExpr`: strings, or lists of
    /// them.
    #[pyo3(signature = (*expr))]
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

    /// Filter rows by a Column condition or a SQL-string condition.
    fn filter(&self, condition: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        Ok(self.of_same_schema(PyDataFrame::new(
            self.dataframe.filter(cond_column(condition)?),
        )))
    }

    /// Alias for filter (reference pyspark name is `where`).
    #[pyo3(name = "where")]
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

    /// Add or replace a column, of a Column, as PySpark's `DataFrame.withColumn`, which checks
    /// the Column first. Its `WithColumns` plan asserts a str name, and sets it when it builds
    /// the plan, after the Column.
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

    /// Rename a column, as PySpark's, which renames of the dict `{existing: new}`, so an
    /// `existing` that is not hashable raises the `TypeError` of the dict.
    fn withColumnRenamed(
        &self,
        existing: &Bound<'_, PyAny>,
        new: &Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        let renames = PyDict::new(existing.py());
        renames.set_item(existing, new)?;
        self.with_columns_renamed(renames.as_any())
    }

    /// Add or replace multiple columns from a {name: Column} mapping, as PySpark's
    /// `DataFrame.withColumns`, whose `WithColumns` plan asserts, as the call builds it, that the
    /// mapping is not empty, and of str names and Columns.
    #[pyo3(name = "withColumns", signature = (*colsMap))]
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

    /// Specify a relation hint (e.g. "broadcast"), with optional parameters. Reference
    /// hint params may be ints (e.g. `df.hint("rebalance", 3)`), not just strings, so
    /// accept any value and stringify it; the core parses a numeric string back into a
    /// long literal, matching reference semantics.
    #[pyo3(signature = (name, *parameters))]
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

    /// Drop columns, of Columns or column names, as PySpark's `DataFrame.drop`.
    #[pyo3(signature = (*cols))]
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

    /// Limit the number of rows, as PySpark's `limit`, which does not check `num`: its plan sets
    /// `num` in the proto's int32, whose errors ([`proto_int`]) PySpark raises once it sends the
    /// plan, so the DataFrame raises them at each request. The server checks the rest.
    fn limit(&self, num: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let py = num.py();
        let value = proto_int(num, "spark.connect.Limit.limit");
        let df = converted_plan(py, &self.dataframe, value, |num| self.dataframe.limit(num))?;
        Ok(self.of_same_schema(PyDataFrame::new(df)))
    }

    /// Skip the first `num` rows, as PySpark's `offset`, which sends `num` as `limit` does, but
    /// does not copy the `_cached_schema` of its input, also when the count fails the plan.
    fn offset(&self, num: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let py = num.py();
        let df = match proto_int(num, "spark.connect.Offset.offset") {
            Ok(num) => self.dataframe.offset(num),
            Err(err) => self.dataframe.unbuildable(unbuildable_error(py, err)?),
        };
        Ok(PyDataFrame::new(df))
    }

    /// Remove duplicate rows.
    fn distinct(&self) -> PyDataFrame {
        self.of_same_schema(PyDataFrame::new(self.dataframe.distinct()))
    }

    /// Remove duplicate rows, optionally on specific columns.
    #[pyo3(signature = (subset=None))]
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

    /// Sort rows. `ascending` (a bool or list of bools, via kwargs) mirrors reference
    /// pyspark: it wraps each sort column with asc()/desc().
    #[pyo3(signature = (*cols, **kwargs))]
    fn sort(
        &self,
        _py: Python<'_>,
        cols: Vec<Bound<'_, PyAny>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        let exprs = self.sort_columns(_py, cols, kwargs.as_ref())?;
        Ok(self.of_same_schema(PyDataFrame::new(self.dataframe.sort(exprs))))
    }

    /// Alias for sort.
    #[pyo3(signature = (*cols, **kwargs))]
    fn orderBy(
        &self,
        _py: Python<'_>,
        cols: Vec<Bound<'_, PyAny>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        let exprs = self.sort_columns(_py, cols, kwargs.as_ref())?;
        Ok(self.of_same_schema(PyDataFrame::new(self.dataframe.order_by(exprs))))
    }

    /// Join with another DataFrame.
    ///
    /// `on` is a Column condition, a column name, or a list of column names (a "using"
    /// join) or of Column conditions (joined on their conjunction), as the connect client's
    /// `plan.Join` takes it. Mirrors `pyspark.sql.DataFrame.join(other, on=None, how=None)`
    /// (default inner), including its errors.
    #[pyo3(signature = (other, on=None, how=None))]
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

    /// Cross join.
    fn crossJoin(&self, other: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let other = self.same_session(other)?;
        Ok(PyDataFrame::new(
            self.dataframe.cross_join(&other.dataframe),
        ))
    }

    /// Lateral (correlated) join. `on` is an optional Column condition; `how` is inner by
    /// default and takes the join types of a lateral join. Mirrors `DataFrame.lateralJoin`.
    #[pyo3(signature = (other, on=None, how=None))]
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

    /// Transpose (swap rows/columns). `indexColumn` optionally names the header
    /// column. Mirrors `DataFrame.transpose(indexColumn=None)`.
    #[pyo3(signature = (indexColumn=None))]
    #[allow(non_snake_case)]
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

    /// Repartition into `numPartitions` by a range partitioning on the given
    /// columns. Mirrors `DataFrame.repartitionByRange(numPartitions, *cols)`.
    #[pyo3(signature = (numPartitions, *cols))]
    #[allow(non_snake_case)]
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

    /// Repartition into `numPartitions` using a column's value directly as the shuffle
    /// partition id. Mirrors `DataFrame.repartitionById(numPartitions, partitionIdCol)`.
    #[pyo3(signature = (numPartitions, partitionIdCol))]
    #[allow(non_snake_case)]
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
        // The count is the proto's int32 of the plan, converted after the column, as in
        // `repartitionByRange`.
        let value = proto_int(
            numPartitions,
            "spark.connect.RepartitionByExpression.num_partitions",
        );
        let df = converted_plan(
            numPartitions.py(),
            &self.dataframe,
            value,
            |num_partitions| self.dataframe.repartition_by_id(num_partitions, col),
        )?;
        Ok(self.of_same_schema(PyDataFrame::new(df)))
    }

    /// Append a distributed monotonically-increasing index column, as PySpark's
    /// `DataFrame.zipWithIndex(indexColName="index")`: `self.select(F.col("*"),
    /// F._invoke_function("distributed_sequence_id").alias(indexColName))`, of the origins of
    /// `col` and `alias`. The name is that of the alias of a Column ([`PyColumn::alias`]), which
    /// the plan sets when it is built: a str or bytes of UTF-8, and another value fails each
    /// request of the DataFrame, not the call.
    #[pyo3(
        signature = (indexColName=DefaultedOption::Omitted),
        text_signature = "($self, indexColName='index')"
    )]
    #[allow(non_snake_case)]
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

    /// Zip this DataFrame with another column-wise (side-by-side). Mirrors
    /// `DataFrame.zip(other)`.
    fn zip(&self, other: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let other = self.same_session(other)?;
        Ok(PyDataFrame::new(
            self.dataframe.zip(&other.dataframe).to_pyerr()?,
        ))
    }

    /// Union with another DataFrame.
    fn union(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let other = self.same_session(other)?;
        let dataframe = self.dataframe.union(&other.dataframe);
        set_operation_result(py, self, &other, dataframe)
    }

    /// Union by name, optionally filling columns missing on one side with null. The flag is
    /// converted as PySpark's set operation sets it: None is a `TypeError`.
    #[pyo3(
        signature = (other, allowMissingColumns=ProtoBool::default()),
        text_signature = "($self, other, allowMissingColumns=False)"
    )]
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

    /// Intersect with another DataFrame.
    fn intersect(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let other = self.same_session(other)?;
        let dataframe = self.dataframe.intersect(&other.dataframe);
        set_operation_result(py, self, &other, dataframe)
    }

    /// Intersect keeping duplicates (reference pyspark `intersectAll`).
    #[pyo3(name = "intersectAll")]
    fn intersect_all(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let other = self.same_session(other)?;
        let dataframe = self.dataframe.intersect_all(&other.dataframe);
        set_operation_result(py, self, &other, dataframe)
    }

    /// Subtract another DataFrame.
    fn subtract(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let other = self.same_session(other)?;
        let dataframe = self.dataframe.subtract(&other.dataframe);
        set_operation_result(py, self, &other, dataframe)
    }

    /// Repartition. Mirrors reference `repartition(numPartitions, *cols)` where the
    /// first argument may instead be a Column/str - `df.repartition("country")` /
    /// `df.repartition(col("country"))` repartition by that column at the default
    /// partition count (no explicit number).
    #[pyo3(signature = (numPartitions, *cols))]
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
            let n = proto_int(
                &numPartitions,
                "spark.connect.RepartitionByExpression.num_partitions",
            );
            let df = converted_plan(py, &self.dataframe, n, |n| {
                self.dataframe.repartition_by_expressions(n, exprs)
            })?;
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

    /// Coalesce into `numPartitions` partitions, as PySpark's `coalesce`: any value that is
    /// `> 0`, which is then the proto's int32 of the plan, as in `repartitionByRange`.
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

    /// Convert to DataFrame with new column names.
    #[pyo3(signature = (*cols))]
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

    /// Alias this DataFrame, as PySpark's, whose `SubqueryAlias` plan sets the alias when it
    /// builds the plan, as protobuf's string field takes it: bytes of UTF-8 too, and anything
    /// else fails the requests of the DataFrame.
    fn alias(&self, alias: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let mut args = crate::plan_args::PlanArgs::new();
        let alias = args.str(alias)?;
        Ok(self.of_same_schema(PyDataFrame::new(args.plan(self.dataframe.alias(&alias)))))
    }

    /// Group by columns for aggregation.
    #[pyo3(signature = (*cols))]
    fn groupBy(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyGroupedData> {
        let this = slf.borrow();
        let columns = this.grouping_columns(py, cols, "cols")?;
        Ok(PyGroupedData::new(
            this.dataframe.group_by(columns),
            slf.clone().unbind(),
        ))
    }

    /// Rollup multi-dimensional aggregation.
    #[pyo3(signature = (*cols))]
    fn rollup(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyGroupedData> {
        let this = slf.borrow();
        Ok(PyGroupedData::new(
            this.dataframe
                .rollup(this.grouping_columns(py, cols, "cols")?),
            slf.clone().unbind(),
        ))
    }

    /// Cube multi-dimensional aggregation.
    #[pyo3(signature = (*cols))]
    fn cube(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyGroupedData> {
        let this = slf.borrow();
        Ok(PyGroupedData::new(
            this.dataframe
                .cube(this.grouping_columns(py, cols, "cols")?),
            slf.clone().unbind(),
        ))
    }

    /// Grouping sets aggregation: `groupingSets([[a, b], [a], []], "a", "b")`.
    ///
    /// As PySpark's: `*cols` are the grouping columns, the columns of the result besides the
    /// aggregations, of Columns, names and ordinals, as are those of the sets.
    #[pyo3(name = "groupingSets", signature = (groupingSets, *cols))]
    fn grouping_sets(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        groupingSets: &Bound<'_, PyAny>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyGroupedData> {
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
        Ok(PyGroupedData::new(
            this.dataframe.grouping_sets_with_columns(sets, columns),
            slf.clone().unbind(),
        ))
    }

    /// Drop duplicates within the event-time watermark.
    #[pyo3(name = "dropDuplicatesWithinWatermark", signature = (subset=None))]
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

    /// `DataFrame.toJSON`, as PySpark Connect's: a DataFrame of each row as a JSON object
    /// string, in a column `value`, of `self.select(F.to_json(F.struct(F.col("*")))
    /// .alias("value"))`, so its expressions have the origins of `col` and `alias`.
    #[pyo3(name = "toJSON")]
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

    /// Whether this DataFrame is semantically equal to `other`.
    #[pyo3(name = "sameSemantics")]
    fn same_semantics(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        // As PySpark's: a DataFrame, else NOT_EXPECTED_TYPE, of the same session.
        if !other.is_instance_of::<PyDataFrame>() {
            return Err(not_expected_type(other, "other", "DataFrame"));
        }
        let other = self.same_session(other)?.dataframe.clone();
        interruptible(py, &self.dataframe.spark_session(), || {
            self.dataframe.same_semantics(&other)
        })
    }

    /// The server-side hash of this DataFrame's logical plan, asked once, as PySpark's.
    #[pyo3(name = "semanticHash")]
    fn semantic_hash(&self, py: Python<'_>) -> PyResult<i32> {
        cached(&self.semantic_hash, || {
            interruptible(py, &self.dataframe.spark_session(), || {
                self.dataframe.semantic_hash().map_err(Box::new)
            })
        })
    }

    /// Sort within each partition.
    #[pyo3(name = "sortWithinPartitions", signature = (*cols, **kwargs))]
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

    /// Except all (keeps duplicates).
    #[pyo3(name = "exceptAll")]
    fn except_all(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let other = self.same_session(other)?;
        let dataframe = self.dataframe.except_all(&other.dataframe);
        set_operation_result(py, self, &other, dataframe)
    }

    /// Rename multiple columns from an {old: new} mapping.
    #[pyo3(name = "withColumnsRenamed")]
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

    /// The columns of this DataFrame whose names match a regex, as a Column. Mirrors
    /// `DataFrame.colRegex`.
    #[pyo3(name = "colRegex")]
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

    /// Basic statistics for the given columns (count/mean/stddev/min/max), of column names or
    /// a list of them, as PySpark's `DataFrame.describe`.
    #[pyo3(signature = (*cols))]
    fn describe(&self, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        let mut args = crate::plan_args::PlanArgs::new();
        let cols = crate::stat_arguments::describe_columns(&mut args, &cols)?;
        Ok(PyDataFrame::new(args.plan(self.dataframe.describe(cols))))
    }

    /// Summary statistics (percentiles etc.), of strings, as PySpark's `DataFrame.summary`.
    #[pyo3(signature = (*statistics))]
    fn summary(&self, statistics: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        let mut args = crate::plan_args::PlanArgs::new();
        let statistics = crate::stat_arguments::summary_statistics(&mut args, &statistics)?;
        Ok(PyDataFrame::new(
            args.plan(self.dataframe.summary(statistics)),
        ))
    }

    /// Persist with the default MEMORY_AND_DISK storage level.
    fn cache<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<PyRef<'py, Self>> {
        // As PySpark's, which returns the DataFrame itself.
        let df = slf.dataframe.clone();
        interruptible(py, &df.spark_session(), || df.cache().map_err(Box::new))?;
        Ok(slf)
    }

    /// Mark the DataFrame as non-persistent. As PySpark's `unpersist(blocking=False)`, which sets
    /// `blocking` in the bool field of its request unless it is None, as
    /// [`crate::streaming::proto_bool`] converts it: None sends no `blocking`.
    #[pyo3(
        signature = (blocking=ProtoBool::default()),
        text_signature = "($self, blocking=False)"
    )]
    fn unpersist<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        blocking: ProtoBool,
    ) -> PyResult<PyRef<'py, Self>> {
        let blocking = blocking.value_unless_none(py)?;
        // As PySpark's, which returns the DataFrame itself.
        let df = slf.dataframe.clone();
        interruptible(py, &df.spark_session(), || {
            df.unpersist_opt(blocking).map_err(Box::new)
        })?;
        Ok(slf)
    }

    /// Print the plan with Python's `print`, as PySpark's `DataFrame.explain`, which prints
    /// `_explain_string(extended, mode)`.
    #[pyo3(signature = (extended=None, mode=None))]
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
        interruptible(py, &self.dataframe.spark_session(), || {
            self.dataframe
                .explain_string(&explain_mode)
                .map_err(Box::new)
        })
    }

    /// Register this DataFrame as a temporary view, of a name that PySpark's `CreateView` sets in
    /// its string field as it is, as [`crate::str_argument`] converts it, as the other views do.
    #[pyo3(name = "createOrReplaceTempView")]
    fn create_or_replace_temp_view(&self, py: Python<'_>, name: &Bound<'_, PyAny>) -> PyResult<()> {
        let name = crate::str_argument(name)?;
        interruptible(py, &self.dataframe.spark_session(), || {
            self.dataframe
                .create_or_replace_temp_view(&name)
                .map_err(Box::new)
        })
    }

    /// Drop rows containing nulls, as PySpark's `DataFrame.dropna`.
    #[pyo3(
        signature = (how=None, thresh=None, subset=None),
        text_signature = "($self, how='any', thresh=None, subset=None)"
    )]
    fn dropna(
        &self,
        how: Option<&Bound<'_, PyAny>>,
        thresh: Option<&Bound<'_, PyAny>>,
        subset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        crate::stat_arguments::dropna(&self.dataframe, how, thresh, subset)
    }

    /// Fill null values, as PySpark's `DataFrame.fillna`: `value` is a scalar, over
    /// `subset`, or a {column: value} mapping.
    #[pyo3(signature = (value, subset=None))]
    fn fillna(
        &self,
        value: &Bound<'_, PyAny>,
        subset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        crate::stat_arguments::fillna(&self.dataframe, value, subset)
    }

    /// Statistical functions (`df.stat`), of this DataFrame, their `df`, as PySpark's.
    #[getter]
    #[pyo3(name = "stat")]
    fn stat(slf: &Bound<'_, Self>) -> crate::stat::PyStatFunctions {
        crate::stat::PyStatFunctions::new(slf.borrow().dataframe.stat(), slf.clone().unbind())
    }

    /// `df.crosstab(col1, col2)` - pyspark exposes it on DataFrame as well as
    /// `df.stat.crosstab`.
    fn crosstab(&self, col1: &Bound<'_, PyAny>, col2: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        crate::stat_arguments::crosstab(&self.dataframe.stat(), col1, col2)
    }

    /// `df.freqItems(cols, support=None)` (also on df.stat).
    #[pyo3(name = "freqItems", signature = (cols, support=None))]
    fn freq_items(
        &self,
        cols: &Bound<'_, PyAny>,
        support: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        crate::stat_arguments::freq_items(&self.dataframe.stat(), cols, support)
    }

    /// `df.approxQuantile(col, probabilities, relativeError)` (also on df.stat): the
    /// quantiles of a column, or a list of them of each of a list or tuple of columns.
    #[pyo3(name = "approxQuantile")]
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

    /// Observe metrics, as PySpark's `DataFrame.observe(observation, *exprs)`: the Columns
    /// `exprs`, which an Observation receives, or which are observed by a name.
    #[pyo3(signature = (observation, *exprs))]
    fn observe(
        &self,
        py: Python<'_>,
        observation: &Bound<'_, PyAny>,
        exprs: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
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
        if let Ok(observation) = observation.cast::<crate::observation::PyObservation>() {
            let observed = crate::observation::observe(py, self, observation, columns)?;
            return Ok(self.of_same_schema(observed));
        }
        if observation.is_instance_of::<pyo3::types::PyString>() {
            // Its `CollectMetrics` plan sets the name when it builds the plan, before the metrics.
            let mut args = crate::plan_args::PlanArgs::new();
            let name = args.str(observation)?;
            let args = args.before(&mut columns);
            return Ok(self.of_same_schema(PyDataFrame::new(
                args.plan(self.dataframe.observe(&name, columns)),
            )));
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

    /// Register as a temporary view (session-scoped).
    #[pyo3(name = "createTempView")]
    fn create_temp_view(&self, py: Python<'_>, name: &Bound<'_, PyAny>) -> PyResult<()> {
        let name = crate::str_argument(name)?;
        interruptible(py, &self.dataframe.spark_session(), || {
            self.dataframe.create_temp_view(&name).map_err(Box::new)
        })
    }

    /// Register as a global temporary view.
    #[pyo3(name = "createGlobalTempView")]
    fn create_global_temp_view(&self, py: Python<'_>, name: &Bound<'_, PyAny>) -> PyResult<()> {
        let name = crate::str_argument(name)?;
        interruptible(py, &self.dataframe.spark_session(), || {
            self.dataframe
                .create_global_temp_view(&name)
                .map_err(Box::new)
        })
    }

    /// Register as (or replace) a global temporary view.
    #[pyo3(name = "createOrReplaceGlobalTempView")]
    fn create_or_replace_global_temp_view(
        &self,
        py: Python<'_>,
        name: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let name = crate::str_argument(name)?;
        interruptible(py, &self.dataframe.spark_session(), || {
            self.dataframe
                .create_or_replace_global_temp_view(&name)
                .map_err(Box::new)
        })
    }

    /// Deprecated alias for createOrReplaceTempView, as PySpark's, which warns that it is
    /// deprecated and then replaces the view as `createOrReplaceTempView` does.
    #[pyo3(name = "registerTempTable")]
    fn register_temp_table(&self, py: Python<'_>, name: &Bound<'_, PyAny>) -> PyResult<()> {
        py.import("warnings")?.call_method1(
            "warn",
            (
                "Deprecated in 2.0, use createOrReplaceTempView instead.",
                py.get_type::<pyo3::exceptions::PyFutureWarning>(),
            ),
        )?;
        self.create_or_replace_temp_view(py, name)
    }

    /// Reliable checkpoint (materialize + truncate lineage), at once if `eager`, as PySpark's.
    #[pyo3(signature = (eager=DefaultedOption::Omitted), text_signature = "($self, eager=True)")]
    fn checkpoint(&self, py: Python<'_>, eager: DefaultedOption) -> PyResult<PyDataFrame> {
        let eager = checkpoint_eager(py, &eager)?;
        Ok(PyDataFrame::new(interruptible(
            py,
            &self.dataframe.spark_session(),
            || self.dataframe.checkpoint_with(eager).map_err(Box::new),
        )?))
    }

    /// Local checkpoint, at once if `eager`, kept at `storageLevel`, else at the server's
    /// default, as PySpark's.
    #[pyo3(
        name = "localCheckpoint",
        signature = (eager=DefaultedOption::Omitted, storageLevel=None),
        text_signature = "($self, eager=True, storageLevel=None)"
    )]
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
        Ok(PyDataFrame::new(interruptible(
            py,
            &self.dataframe.spark_session(),
            || {
                self.dataframe
                    .local_checkpoint_with(eager, level)
                    .map_err(Box::new)
            },
        )?))
    }

    /// Pearson correlation between two columns; PySpark's `method` is only "pearson".
    #[pyo3(signature = (col1, col2, method=None))]
    fn corr(
        &self,
        py: Python<'_>,
        col1: &Bound<'_, PyAny>,
        col2: &Bound<'_, PyAny>,
        method: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<f64> {
        crate::stat_arguments::corr(py, &self.dataframe.stat(), col1, col2, method)
    }

    /// Sample covariance between two columns.
    fn cov(
        &self,
        py: Python<'_>,
        col1: &Bound<'_, PyAny>,
        col2: &Bound<'_, PyAny>,
    ) -> PyResult<f64> {
        crate::stat_arguments::cov(py, &self.dataframe.stat(), col1, col2)
    }

    /// Stratified sample without replacement per key. Without a seed, the client draws one
    /// from Python's `random`, as PySpark Connect's `sampleBy` does.
    #[pyo3(name = "sampleBy", signature = (col, fractions, seed=None))]
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

    /// Replace values, as PySpark's `DataFrame.replace`: `to_replace` is a {old: new} dict, or
    /// a value or a list or tuple of them with `value`, a value or a list or tuple of them.
    #[pyo3(signature = (to_replace, value=ReplaceValue::Omitted, subset=None))]
    fn replace(
        &self,
        to_replace: &Bound<'_, PyAny>,
        value: ReplaceValue,
        subset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        replace_values(&self.dataframe, to_replace, value, subset)
    }

    /// Reconcile to a target schema (StructType or DDL string). As PySpark's, a schema that is
    /// not a DataType is kept as it is, and the DataFrame fails at each action, which PySpark's
    /// client raises when it builds the plan; a DDL string is not parsed. As PySpark's, the
    /// schema is the cached schema of the DataFrame whatever it is, as the object that it is, so a
    /// DataType that is not a struct, which the core does not cache, is kept as it is too, while
    /// its plan is sent; and as PySpark's `ToSchema` converts the type object when it builds the
    /// plan, a `StructType` changed in place after the call is the schema of the DataFrame, and
    /// the schema that its requests send, as it is then.
    fn to(&self, schema: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        if schema.is_none() {
            // As PySpark's `assert schema is not None`.
            return Err(pyo3::exceptions::PyAssertionError::new_err(()));
        }
        // The type that PySpark's conversion of the plan fails at, at each action: the schema
        // when it is not a DataType, such as a str, or the first StructField that is a type in
        // it, which that conversion (`pyspark_types_to_proto_types`) does not take either.
        let mut unsendable = None;
        if !schema.is_instance_of::<pyo3::types::PyString>() {
            if let Ok(dt) = crate::types::py_to_data_type(schema) {
                match crate::types::unsendable_field_repr(schema, &dt)? {
                    None => {
                        let reader = crate::udf_command::data_type_pickler(schema.clone().unbind());
                        let mut result =
                            PyDataFrame::new(self.dataframe.to_read_at_build(dt, reader));
                        result.schema_object = Some(schema.clone().unbind());
                        return Ok(result);
                    }
                    Some(field) => unsendable = Some(field),
                }
            }
        }
        let operation = match unsendable {
            Some(field) => format!("data type {field}"),
            None => crate::plan_args::data_type_operation(schema)?,
        };
        let error = spark_connect_core::error::SparkError::classed(
            spark_connect_core::error::SparkErrorKind::ValueError,
            "UNSUPPORTED_OPERATION",
            &[("operation", &operation)],
        );
        let mut result = PyDataFrame::new(self.dataframe.unbuildable(error));
        result.schema_object = Some(schema.clone().unbind());
        Ok(result)
    }

    /// Set the metadata of a column, as PySpark's: a dict of any JSON values, which it sends as
    /// `json.dumps(metadata)`, and `NOT_EXPECTED_TYPE` for one that is not a dict.
    #[pyo3(name = "withMetadata")]
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

    /// Define an event-time watermark, as PySpark's: `NOT_EXPECTED_TYPE` for an empty event time
    /// or delay, or one that is not a str.
    #[pyo3(name = "withWatermark")]
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

    /// Randomly split into multiple DataFrames by the given weights.
    /// As PySpark's: `VALUE_NOT_POSITIVE` for a negative weight and for weights of a sum that is
    /// not positive, of their `str`; without a seed, the client chooses one of every split; a
    /// seed goes through `int()`, so a float or a str is accepted.
    #[pyo3(name = "randomSplit", signature = (weights, seed=None))]
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

    /// The list of files that make up this DataFrame, asked once, as PySpark's.
    #[pyo3(name = "inputFiles")]
    fn input_files(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        cached(&self.input_files, || {
            interruptible(py, &self.dataframe.spark_session(), || {
                self.dataframe.input_files().map_err(Box::new)
            })
        })
    }

    /// Whether the server answers that this DataFrame is local (a local relation).
    #[pyo3(name = "isLocal")]
    fn is_local(&self, py: Python<'_>) -> PyResult<bool> {
        interruptible(py, &self.dataframe.spark_session(), || {
            self.dataframe.is_local_on_server().map_err(Box::new)
        })
    }

    /// Whether this (cached) DataFrame is cached.
    #[pyo3(name = "is_cached")]
    #[getter]
    fn is_cached(&self, py: Python<'_>) -> PyResult<bool> {
        interruptible(py, &self.dataframe.spark_session(), || {
            self.dataframe.is_cached().map_err(Box::new)
        })
    }

    /// Whether this DataFrame contains one or more sources that continuously return data, asked
    /// once, as PySpark's.
    #[pyo3(name = "isStreaming")]
    #[getter]
    fn is_streaming(&self, py: Python<'_>) -> PyResult<bool> {
        cached(&self.is_streaming, || {
            interruptible(py, &self.dataframe.spark_session(), || {
                self.dataframe.is_streaming().map_err(Box::new)
            })
        })
    }

    /// The storage level currently in use for this DataFrame.
    #[pyo3(name = "storageLevel")]
    #[getter]
    fn storage_level(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let sl = interruptible(py, &self.dataframe.spark_session(), || {
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

    /// The SparkSession that created this DataFrame.
    #[pyo3(name = "sparkSession")]
    #[getter]
    fn spark_session(&self, py: Python<'_>) -> PyResult<Py<crate::session::PySparkSession>> {
        crate::session::session_object(py, &self.dataframe.spark_session())
    }

    /// The SparkSession of this DataFrame, as PySpark Connect's `DataFrame._session`, which the
    /// modules of PySpark read, such as `pyspark.ml`, `pyspark.sql.plot` and `pyspark.pandas`.
    #[getter]
    fn _session(&self, py: Python<'_>) -> PyResult<Py<crate::session::PySparkSession>> {
        self.spark_session(py)
    }

    /// The EXISTS subquery of this DataFrame, as a Column. Mirrors `DataFrame.exists()`.
    fn exists(&self) -> PyColumn {
        PyColumn::new(self.dataframe.exists())
    }

    /// The scalar subquery of this DataFrame, as a Column. Mirrors `DataFrame.scalar()`.
    fn scalar(&self) -> PyColumn {
        PyColumn::new(self.dataframe.scalar())
    }

    /// Return a `TableArg` wrapping this DataFrame, for use as a table-valued function
    /// argument. Mirrors `DataFrame.asTable()`.
    #[pyo3(name = "asTable")]
    fn as_table(&self) -> crate::tablearg::PyTableArg {
        crate::tablearg::PyTableArg::new(spark_connect::table_arg::TableArg::new(
            self.dataframe.clone(),
        ))
    }

    /// Deprecated alias for dropDuplicates.
    #[pyo3(name = "drop_duplicates", signature = (subset=None))]
    fn drop_duplicates(&self, subset: Option<&Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        self.dropDuplicates(subset)
    }

    /// Alias for groupBy.
    #[pyo3(signature = (*cols))]
    fn groupby(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyGroupedData> {
        Self::groupBy(slf, py, cols)
    }

    /// Apply a function to this DataFrame and return its result:
    /// `df.transform(func, *args, **kwargs)` == `func(df, *args, **kwargs)`.
    #[pyo3(signature = (func, *args, **kwargs))]
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

    /// Unpivot (wide-to-long), as PySpark's `DataFrame.unpivot`: `ids` and `values` are each a
    /// column or a column name, or a list or tuple of them, and `values=None` unpivots all
    /// non-id columns.
    #[pyo3(signature = (ids, values, variableColumnName, valueColumnName))]
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

    /// The same function as `unpivot`, as PySpark's `DataFrame.melt`.
    #[pyo3(signature = (ids, values, variableColumnName, valueColumnName))]
    fn melt(
        &self,
        ids: &Bound<'_, PyAny>,
        values: Option<&Bound<'_, PyAny>>,
        variableColumnName: &Bound<'_, PyAny>,
        valueColumnName: &Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        self.unpivot(ids, values, variableColumnName, valueColumnName)
    }

    /// Persist the DataFrame at `storageLevel`, as PySpark's `DataFrame.persist()`:
    /// MEMORY_AND_DISK_DESER when it is omitted, and no storage level for None, so that the
    /// server caches at its `spark.sql.defaultCacheStorageLevel`.
    #[pyo3(signature = (storageLevel=crate::storagelevel::PersistLevel::Omitted))]
    fn persist<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        storageLevel: crate::storagelevel::PersistLevel,
    ) -> PyResult<PyRef<'py, Self>> {
        let level = storageLevel.proto(py)?;
        // As PySpark's, which returns the DataFrame itself.
        let df = slf.dataframe.clone();
        interruptible(py, &df.spark_session(), || {
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

    /// Class attributes and all column names, matching Python Connect.
    fn __dir__(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        // Inspect the class rather than the instance to avoid calling __dir__ again.
        let mut attrs = py.get_type::<Self>().dir()?.extract::<Vec<String>>()?;
        attrs.extend(self.column_names(py)?);
        attrs.sort();
        attrs.dedup();
        Ok(attrs)
    }

    /// Bracket completion keeps column order, duplicates and non-identifiers.
    fn _ipython_key_completions_(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        self.column_names(py)
    }

    /// Aggregate over the whole DataFrame (shorthand for `groupBy().agg(...)`).
    ///
    /// Mirrors `pyspark.sql.DataFrame.agg(*exprs)`.
    #[pyo3(signature = (*exprs))]
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

    /// Collect all rows into memory.
    fn collect(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        // Release the GIL across the blocking RPC so other Python threads run.
        let (rows, schema) = interruptible(py, &self.dataframe.spark_session(), || {
            self.dataframe
                .collect_with_schema_checked(check_schema_at_receipt)
                .map_err(Box::new)
        })?;
        rows_to_py(py, &self.dataframe, rows, schema.as_ref())
    }

    /// Return an iterator over rows without materializing the entire result in memory.
    ///
    /// Mirrors `pyspark.sql.DataFrame.toLocalIterator(prefetchPartitions=False)`, a generator:
    /// the execution starts with the first row, and its errors are raised by the iteration.
    /// Yields Row objects lazily as batches are fetched from the server. As PySpark Connect's,
    /// which ignores `prefetchPartitions`, a batch is fetched when the iteration reaches it, so
    /// the progress handlers of the execution are called by `next()`, on the iterating thread.
    #[pyo3(name = "toLocalIterator")]
    #[pyo3(
        signature = (prefetchPartitions=None),
        text_signature = "($self, prefetchPartitions=False)"
    )]
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
    /// (`pyspark.sql.connect.client._table_as_received`).
    fn _to_table(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let (batches, schema) = interruptible(py, &self.dataframe.spark_session(), || {
            self.dataframe
                .collect_record_batches_with_schema_checked(check_schema_at_receipt)
                .map_err(Box::new)
        })?;
        if batches.is_empty() {
            return Err(pyo3::exceptions::PyAssertionError::new_err(()));
        }
        let table = crate::pyarrow_export::record_batches_to_pyarrow(py, batches)?;
        let table = py
            .import("pyspark.sql.connect.client")?
            .call_method1("_table_as_received", (table,))?
            .unbind();
        let schema = match &schema {
            Some(schema) => crate::types::data_type_to_py(py, schema)?,
            None => py.None(),
        };
        Ok((table, schema))
    }

    /// Collect the DataFrame into a `pyarrow.Table`.
    ///
    /// Mirrors PySpark Connect's `toArrow()`: the Arrow schema of the `schema` of the DataFrame
    /// first, `to_arrow_schema(self.schema, error_on_duplicated_field_names_in_struct=True,
    /// timezone="UTC")`, then the table of the result, cast to it. The schema is that of the
    /// analysis of the plan, which the DataFrame keeps, so `columns` asks nothing afterwards; a
    /// type that PySpark does not convert to Arrow, and duplicate names of the fields of a
    /// nested struct, raise before the query runs, so before any error of its execution.
    ///
    /// The result is decoded from the server's Arrow IPC stream and handed to `pyarrow` through
    /// arrow-rs's reader conversion, sharing compatible decoded Arrow value buffers. Bitmap
    /// alignment and schemas requiring the IPC compatibility fallback can still copy.
    #[pyo3(name = "toArrow")]
    fn to_arrow(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("error_on_duplicated_field_names_in_struct", true)?;
        kwargs.set_item("timezone", "UTC")?;
        let arrow_schema = py
            .import("pyspark.sql.pandas.types")?
            .getattr("to_arrow_schema")?
            .call((self.schema(py)?,), Some(&kwargs))?;
        let (batches, _) = interruptible(py, &self.dataframe.spark_session(), || {
            self.dataframe
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

    /// Collect the DataFrame into a pandas DataFrame.
    ///
    /// Mirrors `pyspark.sql.DataFrame.toPandas()`, as `SparkConnectClient.to_pandas` converts
    /// the Arrow result: each column by its Spark type, in the session time zone.
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

    /// Get the count of rows.
    fn count(&self, py: Python<'_>) -> PyResult<i64> {
        interruptible(py, &self.dataframe.spark_session(), || {
            self.dataframe.count().map_err(Box::new)
        })
    }

    /// Print the first `n` rows as the server formats them. Mirrors `DataFrame.show`, which
    /// prints the table of `_show_string` with Python's `print`.
    #[pyo3(signature = (*args, **kwargs), text_signature = "($self, n=20, truncate=True, vertical=False)")]
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

    /// Execution info from the most recent action, as a `pyspark.sql.metrics.ExecutionInfo`
    /// (plan metrics, observed-metric flows, and the ExecutePlan `operation_id`). Mirrors
    /// `DataFrame.executionInfo`; `None` before any action ran. As PySpark's, which returns the
    /// `_execution_info` that the last execution of the DataFrame set, it is one object until
    /// another execution replaces it: the one that the first access after an execution makes.
    #[getter]
    #[pyo3(name = "executionInfo")]
    fn execution_info(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let Ok(info) = self.dataframe.execution_info() else {
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

    /// Convert to a pandas-on-Spark DataFrame, as PySpark's `DataFrame.pandas_api`: of the
    /// `InternalFrame` of this DataFrame, of the index of the columns of `index_col`
    /// (`pyspark.pandas.namespace._get_index_map`), of the vendored `pyspark.pandas`.
    #[pyo3(name = "pandas_api")]
    #[pyo3(signature = (index_col=None))]
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

    /// The plot accessor of the DataFrame, as PySpark's `DataFrame.plot`: the
    /// `PySparkPlotAccessor` of `pyspark.sql.plot`, of this DataFrame.
    #[getter]
    fn plot(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(py
            .import("pyspark.sql.plot")?
            .getattr("PySparkPlotAccessor")?
            .call1((slf,))?
            .unbind())
    }

    /// The schema of this DataFrame as a `StructType` (a property, mirroring
    /// `DataFrame.schema`) — materialized into the proper Python type classes so
    /// `df.schema.fields`, `df.schema["c"]`, `df.schema.fieldNames()` all work.
    #[getter]
    fn schema(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // As PySpark's `copy.deepcopy` of the schema that `to` keeps, when it is not a DataType.
        if let Some(schema) = &self.schema_object {
            return Ok(py
                .import("copy")?
                .call_method1("deepcopy", (schema.bind(py),))?
                .unbind());
        }
        let schema = interruptible(py, &self.dataframe.spark_session(), || {
            self.dataframe.schema()
        })?;
        check_schema_conversion(py, Some(&schema))?;
        crate::types::data_type_to_py(py, &schema)
    }

    /// Get the first row.
    fn first(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let (row, schema) = interruptible(py, &self.dataframe.spark_session(), || {
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

    /// The first row, or None, or with `n`, a list of the first `n` rows, as PySpark's `head`,
    /// whose `head(n)` is `take(n)`.
    #[pyo3(signature = (n=None))]
    fn head(&self, py: Python<'_>, n: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        use pyo3::IntoPyObjectExt;
        match n {
            None => self.first(py)?.into_py_any(py),
            Some(n) => self.take(py, n)?.into_py_any(py),
        }
    }

    /// Get the first n rows, as PySpark's `DataFrame.take`, which is `limit(num).collect()`: `num`
    /// is converted as `limit` converts it, and the server checks the rest.
    fn take(&self, py: Python<'_>, num: &Bound<'_, PyAny>) -> PyResult<Vec<Py<PyAny>>> {
        let num = proto_int(num, "spark.connect.Limit.limit")?;
        let (rows, schema) = interruptible(py, &self.dataframe.spark_session(), || {
            self.dataframe
                .limit(num)
                .collect_with_schema_checked(check_schema_at_receipt)
                .map_err(Box::new)
        })?;
        rows_to_py(py, &self.dataframe, rows, schema.as_ref())
    }

    /// Column names (a property, mirroring `DataFrame.columns`). A schema of `to` that is not a
    /// DataType gives the names of its fields as it is, as PySpark's `[field.name for field in
    /// self._schema.fields]`: an AttributeError of it falls back to a Column, as in PySpark.
    #[getter]
    fn columns(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if let Some(schema) = &self.schema_object {
            let names = PyList::empty(py);
            for field in schema.bind(py).getattr("fields")?.try_iter()? {
                names.append(field?.getattr("name")?)?;
            }
            return Ok(names.into_any().unbind());
        }
        Ok(PyList::new(py, self.column_names(py)?)?.into_any().unbind())
    }

    /// Get the last `num` rows as a list of Rows (eager, like pyspark `tail`), of `num` as the
    /// proto's int32 of the plan, converted as `limit` converts it.
    fn tail(&self, py: Python<'_>, num: &Bound<'_, PyAny>) -> PyResult<Vec<Py<PyAny>>> {
        let num = proto_int(num, "spark.connect.Tail.limit")?;
        let (rows, schema) = interruptible(py, &self.dataframe.spark_session(), || {
            self.dataframe
                .tail(num)
                .collect_with_schema_checked(check_schema_at_receipt)
                .map_err(Box::new)
        })?;
        rows_to_py(py, &self.dataframe, rows, schema.as_ref())
    }

    /// Sample a fraction of rows, as PySpark's `DataFrame.sample(withReplacement=None,
    /// fraction=None, seed=None)`, which takes `sample(fraction[, seed])` positionally as
    /// well: a bool and a float, a float, or a float and the seed.
    #[pyo3(signature = (withReplacement=None, fraction=None, seed=None))]
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

    /// Check if the DataFrame is empty, asked once, as PySpark's.
    fn isEmpty(&self, py: Python<'_>) -> PyResult<bool> {
        cached(&self.is_empty, || {
            let empty = interruptible(py, &self.dataframe.spark_session(), || {
                self.dataframe.is_empty().map_err(Box::new)
            })?;
            // As PySpark's `select().take(1)`, whose `collect` reads the binary setting once the
            // rows are fetched, though they have no columns.
            read_binary_as_bytes(py, &self.dataframe)?;
            Ok(empty)
        })
    }

    /// Print the schema through Python stdout, honoring the maximum nesting depth.
    #[pyo3(signature = (level=None))]
    fn printSchema(&self, py: Python<'_>, level: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        // Reference's `if level:` selects the default for false values before
        // resolving the schema; positive fractional depths behave as ceil(level).
        let level = match level {
            Some(level) if level.is_truthy()? => Some(level),
            _ => None,
        };
        // A schema of `to` that is not a DataType is printed as PySpark's `print(self.schema
        // .treeString(level))`, of a copy of it, whose `treeString` raises its own AttributeError.
        if let Some(schema) = &self.schema_object {
            let schema = py
                .import("copy")?
                .call_method1("deepcopy", (schema.bind(py),))?;
            let tree = match level {
                Some(level) => schema.call_method1("treeString", (level,))?,
                None => schema.call_method0("treeString")?,
            };
            py.import("builtins")?.getattr("print")?.call1((tree,))?;
            return Ok(());
        }
        let schema = interruptible(py, &self.dataframe.spark_session(), || {
            self.dataframe.schema()
        })?;
        check_schema_conversion(py, Some(&schema))?;
        let depth = level
            .map(crate::types::normalize_tree_depth)
            .transpose()?
            .unwrap_or(i32::MAX);
        let tree = schema.tree_string_with_depth(depth).to_pyerr()?;
        py.import("builtins")?.getattr("print")?.call1((tree,))?;
        Ok(())
    }

    /// Get the schema as a list of (name, type) tuples. A schema of `to` that is not a DataType
    /// gives them as it is, as PySpark's `dtypes` reads its fields.
    #[getter]
    fn dtypes(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if let Some(schema) = &self.schema_object {
            return Ok(object_dtypes(py, schema.bind(py))?.into_any().unbind());
        }
        let schema = interruptible(py, &self.dataframe.spark_session(), || {
            self.dataframe.schema()
        })?;
        check_schema_conversion(py, Some(&schema))?;
        let spark_connect::types::DataType::Struct { fields } = schema else {
            unreachable!("schema() validates struct responses")
        };
        // As PySpark, a user-defined type is named by its simpleString(), such as `vector`,
        // which only its Python class knows.
        let pairs = fields
            .into_iter()
            .map(|field| {
                let name = crate::types::simple_string_with_udts(py, &field.data_type)?;
                Ok((field.name, name))
            })
            .collect::<PyResult<Vec<(String, String)>>>()?;
        Ok(PyList::new(py, pairs)?.into_any().unbind())
    }

    /// Additional set operation: unionAll (alias for union).
    fn unionAll(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let other = self.same_session(other)?;
        let dataframe = self.dataframe.union(&other.dataframe);
        set_operation_result(py, self, &other, dataframe)
    }

    /// The V1 writer interface (`df.write`).
    #[getter]
    fn write(slf: &Bound<'_, Self>) -> PyDataFrameWriter {
        let this = slf.borrow();
        PyDataFrameWriter::new(this.dataframe.write(), slf.clone().unbind())
    }

    /// The V2 writer interface (`df.writeTo("table")`), of the table as [`written_table`] takes
    /// it.
    #[pyo3(name = "writeTo")]
    fn write_to(slf: &Bound<'_, Self>, table: &Bound<'_, PyAny>) -> PyDataFrameWriterV2 {
        let (name, unsent_table) = written_table(table);
        let frame = slf.clone().unbind();
        PyDataFrameWriterV2 {
            inner: std::sync::Arc::new(WriterState::of_frame(
                slf.borrow().dataframe.write_to(&name),
                &frame,
                spark_connect::readwriter::DataFrameWriterV2::shares_plan_with,
            )),
            unsent_table,
            frame,
            unsent_provider: std::sync::Arc::new(std::sync::Mutex::new(None)),
            unsent_clustering: std::sync::Arc::new(std::sync::Mutex::new(None)),
            unsent_partition: std::sync::Arc::new(std::sync::Mutex::new(None)),
            unsent_options: std::sync::Arc::new(std::sync::Mutex::new(Default::default())),
            unsent_properties: std::sync::Arc::new(std::sync::Mutex::new(Default::default())),
            gc_ready: crate::gc::GcReady::new(),
        }
    }

    /// The streaming writer interface (`df.writeStream`).
    #[getter]
    #[pyo3(name = "writeStream")]
    fn write_stream(slf: &Bound<'_, Self>) -> PyDataStreamWriter {
        let writer = slf.borrow().dataframe.write_stream();
        PyDataStreamWriter::new(writer, slf.clone().unbind())
    }

    /// Null-handling functions (`df.na.drop()/fill()/replace()`), of this DataFrame, their
    /// `df`, as PySpark's.
    #[getter]
    fn na(slf: &Bound<'_, Self>) -> PyDataFrameNaFunctions {
        PyDataFrameNaFunctions::new(slf.borrow().dataframe.clone(), slf.clone().unbind())
    }

    /// Merge-into (`df.mergeInto("table", cond)`), returning a V2 merge builder, of the table as
    /// [`written_table`] takes it, and of the condition as PySpark's builder keeps it, which the
    /// merge builds ([`merge_condition`]). The builder keeps this DataFrame, as PySpark's keeps it
    /// in the callback of its execution.
    #[pyo3(name = "mergeInto")]
    fn merge_into(
        slf: &Bound<'_, Self>,
        table: &Bound<'_, PyAny>,
        condition: &Bound<'_, PyAny>,
    ) -> PyResult<PyMergeIntoWriter> {
        let (name, unsent_table) = written_table(table);
        let frame = slf.clone().unbind();
        Ok(PyMergeIntoWriter {
            inner: std::sync::Arc::new(WriterState::of_frame(
                slf.borrow()
                    .dataframe
                    .merge_into(&name, merge_condition(condition)?),
                &frame,
                spark_connect::MergeIntoWriter::shares_plan_with,
            )),
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
                    // `self._schema`: the schema that it keeps as a Python object, as it is.
                    let schema = match &self.schema_object {
                        Some(schema) => schema.clone_ref(py),
                        None => self.schema(py)?,
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
    /// `truncate`, and `DataFrame[name: type, ...]` otherwise. Like PySpark, it reads the
    /// three settings, in one request, on every call.
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        // As PySpark's: once `_repr_html_` was called, the string of the schema.
        if !self
            .support_repr_html
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            if let Some((n, truncate)) = self.eager_eval(py)? {
                return interruptible(py, &self.dataframe.spark_session(), || {
                    self.dataframe
                        .show_string(n, truncate, false)
                        .map_err(Box::new)
                });
            }
        }
        if let Some(schema) = &self.schema_object {
            let pairs = match object_dtypes(py, schema.bind(py)) {
                Ok(pairs) => pairs,
                // As PySpark's: an AttributeError of `dtypes` falls back to a Column, which the
                // join of the representation cannot iterate.
                Err(err) if err.is_instance_of::<pyo3::exceptions::PyAttributeError>(py) => {
                    return Err(crate::column::raise_pyspark(
                        py,
                        "PySparkTypeError",
                        "NOT_ITERABLE",
                        &[("objectName", "Column")],
                    ));
                }
                Err(err) => return Err(err),
            };
            let fields = pairs
                .iter()
                .map(|pair| {
                    let name = pair.get_item(0)?.str()?.to_string();
                    let dtype = pair.get_item(1)?.str()?.to_string();
                    Ok(format!("{name}: {dtype}"))
                })
                .collect::<PyResult<Vec<_>>>()?
                .join(", ");
            return Ok(format!("DataFrame[{fields}]"));
        }
        let fields = self
            .dtypes(py)?
            .bind(py)
            .extract::<Vec<(String, String)>>()?
            .into_iter()
            .map(|(name, dtype)| format!("{name}: {dtype}"))
            .collect::<Vec<_>>()
            .join(", ");
        Ok(format!("DataFrame[{fields}]"))
    }

    /// The HTML table of the DataFrame, which notebooks show, as PySpark Connect's
    /// `_repr_html_`: with `spark.sql.repl.eagerEval.enabled`, the table of the first
    /// `maxNumRows` rows, values truncated to `truncate` characters, which the server formats
    /// (`HtmlString`); else None. Once called, `__repr__` is the string of the schema.
    fn _repr_html_(&self, py: Python<'_>) -> PyResult<Option<String>> {
        self.support_repr_html
            .store(true, std::sync::atomic::Ordering::Relaxed);
        match self.eager_eval(py)? {
            Some((n, truncate)) => interruptible(py, &self.dataframe.spark_session(), || {
                self.dataframe.html_string(n, truncate).map_err(Box::new)
            })
            .map(Some),
            None => Ok(None),
        }
    }

    /// `DataFrame.mapInPandas` — map an iterator of pandas DataFrames, in a stage of the
    /// resource profile `profile`, if any, as PySpark's.
    #[pyo3(
        name = "mapInPandas",
        signature = (func, schema, barrier = ProtoBool::default(), profile = None),
        text_signature = "($self, func, schema, barrier=False, profile=None)"
    )]
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

    /// `DataFrame.mapInArrow` — map an iterator of Arrow batches, in a stage of the
    /// resource profile `profile`, if any, as PySpark's.
    #[pyo3(
        name = "mapInArrow",
        signature = (func, schema, barrier = ProtoBool::default(), profile = None),
        text_signature = "($self, func, schema, barrier=False, profile=None)"
    )]
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

    /// `DataFrame.foreach`, as PySpark Connect's: a UDF over a struct of the columns of each
    /// row calls `f` with the row. The UDF is Python, as it runs on the server
    /// (`pyspark.sql._foreach`).
    fn foreach(slf: &Bound<'_, Self>, f: &Bound<'_, PyAny>) -> PyResult<()> {
        slf.py()
            .import("pyspark.sql._foreach")?
            .call_method1("foreach", (slf, f))?;
        Ok(())
    }

    /// `DataFrame.foreachPartition`, as PySpark Connect's: a `mapInArrow` function calls `f`
    /// with an iterator of the rows of each partition.
    #[pyo3(name = "foreachPartition")]
    fn foreach_partition(slf: &Bound<'_, Self>, f: &Bound<'_, PyAny>) -> PyResult<()> {
        slf.py()
            .import("pyspark.sql._foreach")?
            .call_method1("foreachPartition", (slf, f))?;
        Ok(())
    }

    /// `DataFrame.nearestByJoin(other, rankingExpression, numResults, mode, direction, *,
    /// joinType="inner")`, as PySpark's, which checks its arguments as the Scala
    /// `Dataset.validateNearestByJoinArgs` does: an `AnalysisException` of
    /// `NEAREST_BY_JOIN.NUM_RESULTS_OUT_OF_RANGE`, `UNSUPPORTED_JOIN_TYPE`, `UNSUPPORTED_MODE` or
    /// `UNSUPPORTED_DIRECTION`. As PySpark, it compares `joinType.lower().replace("_", "")`,
    /// `mode.lower()` and `direction.lower()` to the values that it takes, so a value that has no
    /// `lower` raises its `AttributeError`, bytes as the join type the `TypeError` of `replace`,
    /// and another value, such as bytes as the mode or a str of a lone surrogate, is named as it
    /// is in the error.
    #[pyo3(
        name = "nearestByJoin",
        signature = (other, rankingExpression, numResults, mode, direction, *, joinType = DefaultedOption::Omitted),
        text_signature = "($self, other, rankingExpression, numResults, mode, direction, *, joinType='inner')"
    )]
    #[allow(clippy::too_many_arguments)]
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

    /// `DataFrame.metadataColumn`.
    #[pyo3(name = "metadataColumn")]
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

/// Python wrapper for `DataFrameNaFunctions` (`df.na`).
#[pyclass(
    name = "DataFrameNaFunctions",
    module = "pyspark.sql.connect.dataframe",
    weakref,
    dict
)]
pub struct PyDataFrameNaFunctions {
    /// The share of the holders of the plan of `frame`, of which `df` is a clone, dropped before
    /// it (see [`crate::gc::Holders`]).
    frame_share: crate::gc::Share,
    df: spark_connect::dataframe::DataFrame,
    /// The DataFrame of `df.na`, as PySpark's `DataFrameNaFunctions.df`.
    frame: Py<PyDataFrame>,
    /// Set once the value is written, as `__traverse__` reads `frame`, which is not valid while
    /// the object is created (see [`crate::gc::GcReady`]).
    gc_ready: crate::gc::GcReady,
}

impl PyDataFrameNaFunctions {
    fn new(df: spark_connect::dataframe::DataFrame, frame: Py<PyDataFrame>) -> Self {
        let frame_share =
            PyDataFrame::plan_share(&frame, |dataframe| df.shares_plan_with(dataframe));
        PyDataFrameNaFunctions {
            frame_share,
            df,
            frame,
            gc_ready: crate::gc::GcReady::new(),
        }
    }
}

#[pymethods]
impl PyDataFrameNaFunctions {
    /// The shallow copy of the null functions, as PySpark's `copy.copy`: of the same DataFrame,
    /// with the instance attributes of these.
    fn __copy__(slf: &Bound<'_, Self>) -> PyResult<Py<Self>> {
        let py = slf.py();
        let this = slf.borrow();
        let copy = Py::new(
            py,
            PyDataFrameNaFunctions::new(this.df.clone(), this.frame.clone_ref(py)),
        )?;
        crate::reduce::copy_instance_dict(slf.as_any(), copy.bind(py).as_any())?;
        Ok(copy)
    }

    /// As PySpark's null functions, a Python object that holds its DataFrame, which Python's
    /// garbage collector sees: the DataFrame is reported, which reports the objects of the plan
    /// that the two share (see [`crate::gc::Holders`]).
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        if !self.gc_ready.get() {
            return Ok(());
        }
        visit.call(&self.frame)
    }

    /// Break a reference cycle that the garbage collector found unreachable: the null functions
    /// drop their clone of the plan, after their share of it.
    fn __clear__(&mut self) {
        if self.gc_ready.get() {
            self.frame_share = crate::gc::Share::none();
            self.df.clear_plan();
        }
    }

    /// The reduce of the null functions, as PySpark's: `na` of their DataFrame, which unpickles
    /// with it, and their instance attributes.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let frame = slf.borrow().frame.clone_ref(py).into_any().into_bound(py);
        let reduced = crate::reduce::getattr_reduce(py, frame, "na")?;
        crate::reduce::reduce_with_instance_dict(slf.as_any(), reduced.into_any())
    }

    /// The DataFrame of these functions, as PySpark's `DataFrameNaFunctions.df`: `df.na.df` is
    /// `df`.
    #[getter(df)]
    fn frame(&self, py: Python<'_>) -> Py<PyDataFrame> {
        self.frame.clone_ref(py)
    }

    /// Drop rows containing nulls, as PySpark's `DataFrameNaFunctions.drop`, which is
    /// `DataFrame.dropna`.
    #[pyo3(
        signature = (how=None, thresh=None, subset=None),
        text_signature = "($self, how='any', thresh=None, subset=None)"
    )]
    fn drop(
        &self,
        how: Option<&Bound<'_, PyAny>>,
        thresh: Option<&Bound<'_, PyAny>>,
        subset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        crate::stat_arguments::dropna(&self.df, how, thresh, subset)
    }

    /// Fill nulls, as PySpark's `DataFrameNaFunctions.fill`, which is `DataFrame.fillna`.
    #[pyo3(signature = (value, subset=None))]
    fn fill(
        &self,
        value: &Bound<'_, PyAny>,
        subset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        crate::stat_arguments::fillna(&self.df, value, subset)
    }

    /// Replace values, as PySpark's `DataFrameNaFunctions.replace`, which is
    /// `DataFrame.replace`.
    #[pyo3(signature = (to_replace, value=ReplaceValue::Omitted, subset=None))]
    fn replace(
        &self,
        to_replace: &Bound<'_, PyAny>,
        value: ReplaceValue,
        subset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        replace_values(&self.df, to_replace, value, subset)
    }
}

/// The `value` argument of `DataFrame.replace`, where PySpark tells an omitted value apart
/// from None: `replace(to_replace)` requires a dict `to_replace`, while
/// `replace(to_replace, None)` replaces with nulls.
pub enum ReplaceValue {
    Omitted,
    Given(Py<PyAny>),
}

impl<'a, 'py> FromPyObject<'a, 'py> for ReplaceValue {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
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

    // The replacements as a dict, as PySpark builds it, so that equal values are one key.
    let replacements = if is_dict {
        if !value.is_none() {
            py.import("warnings")?.call_method1(
                "warn",
                ("to_replace is a dict and value is not None. value will be ignored.",),
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

/// Python wrapper for `MergeIntoWriter` (`df.mergeInto(...)`).
///
/// Mirrors PySpark's mutable writer: `withSchemaEvolution` and every clause action
/// (`whenMatched().updateAll()`, ...) change this object and return it, and `merge`
/// leaves it usable. The core writer is a consuming builder, so the wrapped value is
/// replaced in place and `merge` runs on a clone.
#[pyclass(
    name = "MergeIntoWriter",
    module = "pyspark.sql.connect.merge",
    weakref,
    dict
)]
pub struct PyMergeIntoWriter {
    inner: std::sync::Arc<WriterState<spark_connect::MergeIntoWriter>>,
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
/// `to_plan` when an action adds the clause: none, a Column, or another object.
enum ClauseCondition {
    None,
    Column(spark_connect::column::Column),
    Object(Py<PyAny>),
}

impl ClauseCondition {
    fn of(condition: Option<&Bound<'_, PyAny>>) -> Self {
        match condition.filter(|condition| !condition.is_none()) {
            None => ClauseCondition::None,
            Some(condition) => match condition.extract::<PyRef<'_, PyColumn>>() {
                Ok(column) => ClauseCondition::Column(column.column.clone()),
                Err(_) => ClauseCondition::Object(condition.clone().unbind()),
            },
        }
    }

    fn clone_ref(&self, py: Python<'_>) -> Self {
        match self {
            ClauseCondition::None => ClauseCondition::None,
            ClauseCondition::Column(column) => ClauseCondition::Column(column.clone()),
            ClauseCondition::Object(object) => ClauseCondition::Object(object.clone_ref(py)),
        }
    }

    /// Report the condition to Python's garbage collector, as PySpark's clause holds it: an
    /// object as it is, and the functions of the UDFs that a Column alone keeps alive, as the
    /// Column reports them.
    fn visit(&self, visit: &pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        match self {
            ClauseCondition::None => Ok(()),
            ClauseCondition::Column(column) => crate::udf_command::visit_pickler_objects(
                column.expression().exclusive_pickler_objects(),
                visit,
            ),
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
/// serializer of one too deep to be sent, and the clause is not added. The merge builds them again
/// when it runs. The assignments keep their order, in which the merge sends them.
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
            value.try_to_proto().to_pyerr()?;
            built.push((key, value));
        }
    }
    let condition = match condition {
        ClauseCondition::None => None,
        ClauseCondition::Column(column) => Some(column.clone()),
        ClauseCondition::Object(object) => {
            let object = object.bind(py);
            object.getattr("to_plan")?;
            return Err(not_column(object, "condition"));
        }
    };
    if let Some(condition) = &condition {
        condition.try_to_proto().to_pyerr()?;
    }
    Ok((condition, built))
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

    /// A clause of the condition `condition`, which PySpark keeps as it is given: its action builds
    /// it ([`merge_clause`]).
    #[pyo3(name = "whenMatched", signature = (condition=None))]
    fn when_matched(slf: &Bound<'_, Self>, condition: Option<&Bound<'_, PyAny>>) -> PyWhenMatched {
        PyWhenMatched::new(slf.clone().unbind(), ClauseCondition::of(condition))
    }

    #[pyo3(name = "whenNotMatched", signature = (condition=None))]
    fn when_not_matched(
        slf: &Bound<'_, Self>,
        condition: Option<&Bound<'_, PyAny>>,
    ) -> PyWhenNotMatched {
        PyWhenNotMatched::new(slf.clone().unbind(), ClauseCondition::of(condition))
    }

    #[pyo3(name = "whenNotMatchedBySource", signature = (condition=None))]
    fn when_not_matched_by_source(
        slf: &Bound<'_, Self>,
        condition: Option<&Bound<'_, PyAny>>,
    ) -> PyWhenNotMatchedBySource {
        PyWhenNotMatchedBySource::new(slf.clone().unbind(), ClauseCondition::of(condition))
    }

    #[pyo3(name = "withSchemaEvolution")]
    fn with_schema_evolution<'py>(slf: &Bound<'py, Self>) -> Bound<'py, Self> {
        slf.borrow()
            .schema_evolution
            .store(true, std::sync::atomic::Ordering::SeqCst);

        slf.clone()
    }

    fn merge(&self, py: Python<'_>) -> PyResult<()> {
        check_written_table(py, &self.unsent_table)?;
        let mut writer = self.inner.get();
        if self
            .schema_evolution
            .load(std::sync::atomic::Ordering::SeqCst)
        {
            writer = writer.with_schema_evolution();
        }
        interruptible(py, &writer.spark_session(), || {
            writer.merge().map_err(Box::new)
        })
    }

    /// The DataFrame and the table that the merge keeps, Python objects, which Python's garbage
    /// collector sees, and the functions of the UDFs of its condition and its clauses that it
    /// alone keeps alive, as PySpark's merge builder, a Python object, holds the DataFrame and the
    /// Columns: so a reference cycle through them is collected, as PySpark collects it. The
    /// DataFrame reports the objects of the plan that the two share (see [`crate::gc::Holders`]).
    /// A state that a copy shares, or that another thread holds, is not visited, so that nothing
    /// is counted twice.
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        if !self.gc_ready.get() {
            return Ok(());
        }
        visit.call(&self.frame)?;
        if let Some(table) = &self.unsent_table {
            visit.call(table)?;
        }
        if std::sync::Arc::strong_count(&self.inner) == 1 {
            if let Some(writer) = self.inner.try_lock() {
                crate::udf_command::visit_pickler_objects(
                    writer.exclusive_pickler_objects(),
                    &visit,
                )?;
            }
        }
        Ok(())
    }

    /// Drop this builder's references to its table and to its state, of which it keeps a cleared
    /// copy; a copy of the builder keeps the state.
    fn __clear__(&mut self) {
        if self.gc_ready.get() {
            self.unsent_table = None;
            self.inner = std::sync::Arc::new(
                self.inner
                    .cleared(spark_connect::MergeIntoWriter::clear_plan),
            );
        }
    }
}

/// `whenMatched(...)` clause builder. Holds its parent writer, like PySpark's
/// `MergeIntoWriter.WhenMatched`; each action may be called more than once.
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
    fn update_all(&self, py: Python<'_>) -> PyResult<Py<PyMergeIntoWriter>> {
        let (cond, _) = merge_clause(py, &self.condition, None)?;
        merge_clause_action(py, &self.writer, |w| w.when_matched(cond).update_all())
    }

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

    fn delete(&self, py: Python<'_>) -> PyResult<Py<PyMergeIntoWriter>> {
        let (cond, _) = merge_clause(py, &self.condition, None)?;
        merge_clause_action(py, &self.writer, |w| w.when_matched(cond).delete())
    }
}

/// `whenNotMatched(...)` clause builder. Holds its parent writer, like PySpark's
/// `MergeIntoWriter.WhenNotMatched`.
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
    fn insert_all(&self, py: Python<'_>) -> PyResult<Py<PyMergeIntoWriter>> {
        let (cond, _) = merge_clause(py, &self.condition, None)?;
        merge_clause_action(py, &self.writer, |w| w.when_not_matched(cond).insert_all())
    }

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

/// `whenNotMatchedBySource(...)` clause builder. Holds its parent writer, like
/// PySpark's `MergeIntoWriter.WhenNotMatchedBySource`.
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
    fn update_all(&self, py: Python<'_>) -> PyResult<Py<PyMergeIntoWriter>> {
        let (cond, _) = merge_clause(py, &self.condition, None)?;
        merge_clause_action(py, &self.writer, |w| {
            w.when_not_matched_by_source(cond).update_all()
        })
    }

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

    fn delete(&self, py: Python<'_>) -> PyResult<Py<PyMergeIntoWriter>> {
        let (cond, _) = merge_clause(py, &self.condition, None)?;
        merge_clause_action(py, &self.writer, |w| {
            w.when_not_matched_by_source(cond).delete()
        })
    }
}

/// Python wrapper for the V1 `DataFrameWriter` (`df.write`).
#[pyclass(
    name = "DataFrameWriter",
    module = "pyspark.sql.connect.readwriter",
    weakref,
    dict
)]
pub struct PyDataFrameWriter {
    inner: std::sync::Arc<WriterState<spark_connect::readwriter::DataFrameWriter>>,
    /// The DataFrame that is written, as the Python object of `df.write`, as PySpark's `_df`.
    frame: Py<PyDataFrame>,

    /// The `numBuckets` of the last `bucketBy` when it is a positive count that the proto
    /// refuses, which PySpark keeps and fails each write with, as its `WriteOperation.command`
    /// sets it in the proto; the core writer then has no bucketing.
    refused_bucket_count: std::sync::Arc<std::sync::Mutex<Option<Py<PyAny>>>>,
    /// The bucket columns of that `bucketBy`, which PySpark keeps with the count: a pickle
    /// gives them back with it, so that PySpark's writer of the state on a Python worker of the
    /// server, which replays `bucketBy`, takes them.
    refused_bucket_cols: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    /// The format, the partitioning columns and the save mode that PySpark keeps as they are,
    /// and whose `WriteOperation.command` converts, or resolves, at each write ([`UnsentArgs`]).
    unsent: std::sync::Arc<std::sync::Mutex<UnsentArgs>>,
    /// Set once the value is written, as `__traverse__` reads `unsent`, which is not valid while
    /// the writer object is created (see [`crate::gc::GcReady`]).
    gc_ready: crate::gc::GcReady,
}

/// The arguments of a writer that PySpark keeps as they are, and that its `WriteOperation.command`
/// converts at the write, which fails with the first one that does not convert. Each one is
/// replaced by the next call that sets the same argument, as PySpark's attribute is.
#[derive(Default)]
pub(crate) struct UnsentArgs {
    /// The format, when it is not a str of UTF-8 (`format`, and the `format` of a write).
    format: Option<Py<PyAny>>,
    /// The first partitioning column that is not a str of UTF-8.
    partition: Option<Py<PyAny>>,
    /// The first clustering column that is not a str of UTF-8, which PySpark's command extends
    /// after the partitioning columns.
    clustering: Option<Py<PyAny>>,
    /// The save mode, when it is not a str.
    mode: Option<Py<PyAny>>,
    /// The first sorting column that is not a str of UTF-8, which the command sends after the
    /// format.
    sort: Option<Py<PyAny>>,
    /// The first bucketing column that is not a str of UTF-8, which the command sends after the
    /// clustering columns.
    bucket: Option<Py<PyAny>>,
    /// The options that the core writer does not take, in the order of PySpark's dict: a key that
    /// is not a str, or a key or value that UTF-8 cannot encode
    /// ([`crate::options::UnencodableOptions`]).
    options: crate::options::UnencodableOptions,
}

impl PyDataFrameWriter {
    pub(crate) fn new(
        writer: spark_connect::readwriter::DataFrameWriter,
        frame: Py<PyDataFrame>,
    ) -> Self {
        PyDataFrameWriter {
            inner: std::sync::Arc::new(WriterState::of_frame(
                writer,
                &frame,
                spark_connect::readwriter::DataFrameWriter::shares_plan_with,
            )),
            refused_bucket_count: std::sync::Arc::new(std::sync::Mutex::new(None)),
            refused_bucket_cols: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            unsent: std::sync::Arc::new(std::sync::Mutex::new(UnsentArgs::default())),
            frame,
            gc_ready: crate::gc::GcReady::new(),
        }
    }

    /// Set the option `key` to `value` on the writer, as PySpark's dict takes it: the core writer
    /// takes what encodes, and the others are kept for the write ([`UnsentArgs::options`]).
    fn put_option(&self, key: &crate::options::OptionKey, value: &crate::options::OptionText) {
        let mut unsent = self.unsent();
        self.inner
            .update(|w| crate::options::put_writer_option(w, &mut unsent.options, key, value));
    }

    fn unsent(&self) -> std::sync::MutexGuard<'_, UnsentArgs> {
        self.unsent
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The writer `w` with the format `source` of PySpark's `format`, which sets it at once when it
    /// is a str of UTF-8 (bytes too), and else keeps the object for the write. None unsets it.
    fn with_format(
        &self,
        w: spark_connect::readwriter::DataFrameWriter,
        source: Option<&Bound<'_, PyAny>>,
    ) -> spark_connect::readwriter::DataFrameWriter {
        let mut unsent = self.unsent();
        match source.map(crate::str_argument) {
            Some(Ok(text)) => {
                unsent.format = None;
                w.format(&text)
            }
            Some(Err(_)) => {
                unsent.format = source.map(|source| source.clone().unbind());
                w.clear_format()
            }
            None => {
                unsent.format = None;
                w.clear_format()
            }
        }
    }

    /// The writer `w` of a format method of PySpark, which sets the format `format` at once, over
    /// a format that was kept.
    fn with_fixed_format(
        &self,
        w: spark_connect::readwriter::DataFrameWriter,
        format: &str,
    ) -> spark_connect::readwriter::DataFrameWriter {
        self.unsent().format = None;
        w.format(format)
    }

    /// The writer `w` with the save mode `mode` of PySpark's `mode`: a str sets it at once, and any
    /// other object is kept for the write, which resolves it ([`crate::readwriter::unsent_save_mode`]).
    /// None leaves the mode as it is.
    fn with_mode(
        &self,
        w: spark_connect::readwriter::DataFrameWriter,
        mode: Option<&Bound<'_, PyAny>>,
    ) -> spark_connect::readwriter::DataFrameWriter {
        let Some(mode) = mode else {
            return w;
        };
        match mode.extract::<String>() {
            Ok(text) => {
                self.unsent().mode = None;
                w.mode(&text)
            }
            Err(_) => {
                self.unsent().mode = Some(mode.clone().unbind());
                w
            }
        }
    }

    /// The writer `w` with the partitioning columns `cols` of PySpark's `partitionBy`, which replace
    /// those of an earlier call. A column that is not a str of UTF-8 is kept for the write.
    fn with_partition_by(
        &self,
        w: spark_connect::readwriter::DataFrameWriter,
        cols: &[Bound<'_, PyAny>],
    ) -> PyResult<spark_connect::readwriter::DataFrameWriter> {
        let (names, unsent) = crate::readwriter::writer_partition_columns(cols)?;
        self.unsent().partition = unsent;
        Ok(w.partition_by(names))
    }

    /// Fail a write whose format or partitioning columns were kept, as PySpark's
    /// `WriteOperation.command` does after its input plan and before its bucketing. Each one fails
    /// again at each write, as it does not convert.
    fn check_unsent_source_and_partition(
        &self,
        py: Python<'_>,
        w: spark_connect::readwriter::DataFrameWriter,
    ) -> PyResult<spark_connect::readwriter::DataFrameWriter> {
        let (format, sort, partition, clustering, bucket) = {
            let unsent = self.unsent();
            (
                unsent.format.as_ref().map(|f| f.clone_ref(py)),
                unsent.sort.as_ref().map(|c| c.clone_ref(py)),
                unsent.partition.as_ref().map(|p| p.clone_ref(py)),
                unsent.clustering.as_ref().map(|c| c.clone_ref(py)),
                unsent.bucket.as_ref().map(|c| c.clone_ref(py)),
            )
        };
        // The order of the command: the format, the sorting, the partitioning and the clustering
        // columns, the bucketing columns, and then the options.
        for value in [format, sort, partition, clustering, bucket]
            .into_iter()
            .flatten()
        {
            crate::str_argument(value.bind(py))?;
        }
        if let Some(error) = self.unsent().options.error() {
            return Err(crate::errors::spark_error_to_py_exception(error));
        }
        Ok(w)
    }

    /// The writer `w` with the kept save mode resolved, after its path, as PySpark's
    /// `WriteOperation.command` does last ([`crate::readwriter::unsent_save_mode`]).
    fn check_unsent_mode(
        &self,
        py: Python<'_>,
        w: spark_connect::readwriter::DataFrameWriter,
    ) -> PyResult<spark_connect::readwriter::DataFrameWriter> {
        let mode = self.unsent().mode.as_ref().map(|m| m.clone_ref(py));
        match mode {
            Some(mode) => Ok(w.mode(&crate::readwriter::unsent_save_mode(mode.bind(py))?)),
            None => Ok(w),
        }
    }

    fn refused_bucket_count(&self) -> std::sync::MutexGuard<'_, Option<Py<PyAny>>> {
        self.refused_bucket_count
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn refused_bucket_cols(&self) -> std::sync::MutexGuard<'_, Vec<String>> {
        self.refused_bucket_cols
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Fail a write of a bucket count that the proto refuses, as PySpark's
    /// `WriteOperation.command`, which sets the count after the input plan and before the path,
    /// the table name and the save mode, so before any request; each write fails again.
    fn check_bucket_count(&self, py: Python<'_>) -> PyResult<()> {
        let refused = self
            .refused_bucket_count()
            .as_ref()
            .map(|n| n.clone_ref(py));
        match refused {
            Some(num_buckets) => {
                crate::readwriter::check_refused_bucket_count(num_buckets.bind(py))
            }
            None => Ok(()),
        }
    }
}

impl PyDataFrameWriter {
    /// Apply the shared `format` / `mode` / `partitionBy` / `**options` arguments that reference
    /// `DataFrameWriter.save` and `.saveAsTable` accept inline, mirroring
    /// `.format().mode().partitionBy().options()` chaining. None of them fails here, as PySpark
    /// sets each as it is and its command fails with it at the write.
    fn apply_write_common(
        &self,
        mut w: spark_connect::readwriter::DataFrameWriter,
        format: Option<&Bound<'_, PyAny>>,
        mode: Option<&Bound<'_, PyAny>>,
        partitionBy: Option<&Bound<'_, PyAny>>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<spark_connect::readwriter::DataFrameWriter> {
        crate::options::check_keyword_keys(options)?;
        if let Some(f) = format {
            w = self.with_format(w, Some(f));
        }
        w = self.with_mode(w, mode);
        if let Some(pb) = partitionBy {
            w = self.with_partition_by(w, std::slice::from_ref(pb))?;
        }
        if let Some(options) = options {
            let mut unsent = self.unsent();
            for (key, value) in options.iter() {
                let Some(value) = crate::options::reader_option_text(&value)? else {
                    continue;
                };
                let key = crate::options::option_key(&key)?;
                w = crate::options::put_writer_option(w, &mut unsent.options, &key, &value);
            }
        }
        Ok(w)
    }
}

#[pymethods]
impl PyDataFrameWriter {
    /// The shallow copy of a writer, as PySpark's `copy.copy`: the copy shares the state of
    /// this writer, as its `_write` is the same object, so that a setter of either one is seen
    /// by both, and the same DataFrame.
    fn __copy__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let this = slf.borrow();
        let copy = Bound::new(
            py,
            PyDataFrameWriter {
                inner: std::sync::Arc::clone(&this.inner),
                refused_bucket_count: std::sync::Arc::clone(&this.refused_bucket_count),
                refused_bucket_cols: std::sync::Arc::clone(&this.refused_bucket_cols),
                unsent: std::sync::Arc::clone(&this.unsent),
                frame: this.frame.clone_ref(py),
                gc_ready: crate::gc::GcReady::new(),
            },
        )?;
        copy_instance_dict(slf.as_any(), copy.as_any())?;
        Ok(copy)
    }

    /// The deep copy of a writer, as PySpark's, which copies its session before its DataFrame (see
    /// [`deep_copy_writer`]).
    fn __deepcopy__<'py>(
        slf: &Bound<'py, Self>,
        memo: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let frame = slf.borrow().frame.clone_ref(slf.py());
        deep_copy_writer(slf.as_any(), &frame, memo)
    }

    /// The reduce of a writer, as PySpark's pickle of its DataFrame and its state, which
    /// `_of_state` gives back.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let this = slf.borrow();
        use crate::reduce::{optional_object, save_mode_name, state_reduce, str_dict};

        let writer = this.inner.get();
        let mode = match writer.get_unsupported_mode() {
            Some(mode) => Some(mode.to_string()),
            None => writer
                .get_save_mode()
                .map(|mode| save_mode_name(mode).to_string()),
        };
        let refused = this
            .refused_bucket_count()
            .as_ref()
            .map(|n| n.clone_ref(py));
        // The core writer has no bucketing of a refused count, whose columns are kept aside.
        let bucket_cols = match refused {
            Some(_) => this.refused_bucket_cols().clone(),
            None => writer.get_bucket_cols().to_vec(),
        };
        let (kept_format, kept_partition, kept_clustering, kept_mode, kept_sort, kept_bucket) = {
            let unsent = this.unsent();
            let kept = |object: &Option<Py<PyAny>>| object.as_ref().map(|o| o.clone_ref(py));
            (
                kept(&unsent.format),
                kept(&unsent.partition),
                kept(&unsent.clustering),
                kept(&unsent.mode),
                kept(&unsent.sort),
                kept(&unsent.bucket),
            )
        };
        let kept_options = this.unsent().options.to_list(py)?;
        let args = PyTuple::new(
            py,
            [
                this.frame.clone_ref(py).into_any().into_bound(py),
                writer.get_format().into_pyobject(py)?.into_any(),
                mode.into_pyobject(py)?.into_any(),
                str_dict(py, writer.get_options())?.into_any(),
                PyList::new(py, writer.get_partition_cols())?.into_any(),
                writer.get_num_buckets().into_pyobject(py)?.into_any(),
                PyList::new(py, bucket_cols)?.into_any(),
                PyList::new(py, writer.get_sort_cols())?.into_any(),
                PyList::new(py, writer.get_cluster_cols())?.into_any(),
                optional_object(py, refused.as_ref()),
                optional_object(py, kept_format.as_ref()),
                optional_object(py, kept_partition.as_ref()),
                optional_object(py, kept_clustering.as_ref()),
                optional_object(py, kept_mode.as_ref()),
                optional_object(py, kept_sort.as_ref()),
                optional_object(py, kept_bucket.as_ref()),
                kept_options.into_any(),
            ],
        )?;
        reduce_with_instance_dict(slf, state_reduce(py, "writer_reduce", args)?)
    }

    /// A writer of its state, as `__reduce__` gives it: of the DataFrame `frame`, the format,
    /// the save mode, if one is set (or the last mode that is not one), the options, the
    /// partitioning, the buckets (their number, None if there is none, and their columns), the
    /// sorting and the clustering columns, and the number of buckets that the proto refuses.
    #[staticmethod]
    #[pyo3(
        signature = (
            frame,
            format,
            mode,
            options,
            partition_cols,
            num_buckets,
            bucket_cols,
            sort_cols,
            cluster_cols,
            refused,
            kept_format,
            kept_partition,
            kept_clustering,
            kept_mode,
            kept_sort,
            kept_bucket,
            kept_options
        )
    )]
    fn _of_state(
        py: Python<'_>,
        frame: Py<PyDataFrame>,
        format: Option<String>,
        mode: Option<String>,
        options: std::collections::HashMap<String, String>,
        partition_cols: Vec<String>,
        num_buckets: Option<i32>,
        bucket_cols: Vec<String>,
        sort_cols: Vec<String>,
        cluster_cols: Vec<String>,
        refused: Option<Py<PyAny>>,
        kept_format: Option<Py<PyAny>>,
        kept_partition: Option<Py<PyAny>>,
        kept_clustering: Option<Py<PyAny>>,
        kept_mode: Option<Py<PyAny>>,
        kept_sort: Option<Py<PyAny>>,
        kept_bucket: Option<Py<PyAny>>,
        kept_options: Vec<(Py<PyAny>, Py<PyAny>)>,
    ) -> PyResult<PyDataFrameWriter> {
        let mut writer = frame.borrow(py).dataframe.write();
        if let Some(format) = format {
            writer = writer.format(&format);
        }
        if let Some(mode) = mode {
            writer = writer.mode(&mode);
        }
        writer = writer.options(options);
        if !partition_cols.is_empty() {
            writer = writer.partition_by(partition_cols);
        }
        let refused_cols = match refused {
            Some(_) => bucket_cols.clone(),
            None => Vec::new(),
        };
        if let Some(num_buckets) = num_buckets {
            writer = writer.bucket_by(num_buckets, bucket_cols);
        }
        if !sort_cols.is_empty() {
            writer = writer.sort_by(sort_cols);
        }
        if !cluster_cols.is_empty() {
            writer = writer.cluster_by(cluster_cols);
        }
        let restored = PyDataFrameWriter::new(writer, frame);
        *restored.refused_bucket_count() = refused;
        *restored.refused_bucket_cols() = refused_cols;
        *restored.unsent() = UnsentArgs {
            format: kept_format,
            partition: kept_partition,
            clustering: kept_clustering,
            mode: kept_mode,
            sort: kept_sort,
            bucket: kept_bucket,
            ..UnsentArgs::default()
        };
        // The options that the core writer does not take are set in their order, as PySpark's dict.
        for (key, value) in &kept_options {
            let key = crate::options::option_key(key.bind(py))?;
            if let Some(value) = crate::options::reader_option_text(value.bind(py))? {
                restored.put_option(&key, &value);
            }
        }
        Ok(restored)
    }

    /// Set the schema, if any, and the options of `options` that are not None, through its own
    /// `schema` and `option`, as PySpark's `OptionUtils._set_opts`, which pandas-on-Spark calls.
    #[pyo3(signature = (schema=None, **options))]
    fn _set_opts(
        slf: &Bound<'_, Self>,
        schema: Option<Bound<'_, PyAny>>,
        options: Option<Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<()> {
        crate::options::set_opts(slf.as_any(), schema.as_ref(), options.as_ref())
    }

    /// The save mode, as PySpark's: `None` keeps the default, "error".
    fn mode<'py>(
        slf: &Bound<'py, Self>,
        saveMode: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        let writer = slf.borrow();
        writer.inner.update(|w| writer.with_mode(w, saveMode));
        Ok(slf.clone())
    }

    /// The format of the write. As PySpark's, None unsets it, so that the server writes its
    /// default source.
    fn format<'py>(
        slf: &Bound<'py, Self>,
        source: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        let writer = slf.borrow();
        writer.inner.update(|w| writer.with_format(w, source));
        Ok(slf.clone())
    }

    fn option<'py>(
        slf: &Bound<'py, Self>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        // None -> option left unset, before the key is looked at, as PySpark's returns at once;
        // bools -> "true"/"false" (reference `to_str`). A key or a value that the core writer does
        // not take fails the write, as PySpark's command, which encodes the dict at the write.
        if value.is_none() {
            return Ok(slf.clone());
        }
        let key = crate::options::option_key(key)?;
        if let Some(value) = crate::options::reader_option_text(value)? {
            slf.borrow().put_option(&key, &value);
        }
        Ok(slf.clone())
    }

    // Mirrors reference `DataFrameWriter.options(**options)`: keyword args; None values
    // skipped, booleans lowercased.
    #[pyo3(signature = (**options))]
    fn options<'py>(
        slf: &Bound<'py, Self>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        crate::options::check_keyword_keys(options)?;
        let writer = slf.borrow();
        for (key, value) in options.into_iter().flat_map(|options| options.iter()) {
            let Some(value) = crate::options::reader_option_text(&value)? else {
                continue;
            };
            let key = crate::options::option_key(&key)?;
            writer.put_option(&key, &value);
        }
        Ok(slf.clone())
    }

    #[pyo3(name = "partitionBy", signature = (*cols))]
    fn partition_by<'py>(
        slf: &Bound<'py, Self>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        let writer = slf.borrow();
        let (cols, unsent) = crate::readwriter::writer_partition_columns(&cols)?;
        writer.unsent().partition = unsent;
        writer.inner.update(|w| w.partition_by(cols));
        Ok(slf.clone())
    }

    #[pyo3(name = "bucketBy", signature = (numBuckets, col, *cols))]
    fn bucket_by<'py>(
        slf: &Bound<'py, Self>,
        numBuckets: &Bound<'_, PyAny>,
        col: &Bound<'_, PyAny>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        use crate::readwriter::BucketCount;

        let num_buckets = crate::readwriter::bucket_count(numBuckets)?;
        let (columns, unsent) = crate::readwriter::writer_columns_kept_of(col, &cols)?;
        // PySpark sends the bucketing columns only with a positive count, so none is kept
        // without one.
        slf.borrow().unsent().bucket = match num_buckets {
            BucketCount::Unsent => None,
            _ => unsent,
        };
        // As PySpark's, the last call wins: a count that is not positive, or one that the
        // proto refuses, leaves no bucketing of an earlier call on the writer.
        let writer = slf.borrow();
        let (refused, refused_cols) = match num_buckets {
            BucketCount::Count(count) => {
                writer.inner.update(|w| w.bucket_by(count, columns));
                (None, Vec::new())
            }
            BucketCount::Unsent => {
                writer.inner.update(|w| w.clear_bucket_by());
                (None, Vec::new())
            }
            BucketCount::Refused(count) => {
                writer.inner.update(|w| w.clear_bucket_by());
                (Some(count.unbind()), columns)
            }
        };
        *writer.refused_bucket_count() = refused;
        *writer.refused_bucket_cols() = refused_cols;
        Ok(slf.clone())
    }

    #[pyo3(name = "sortBy", signature = (col, *cols))]
    fn sort_by<'py>(
        slf: &Bound<'py, Self>,
        col: &Bound<'_, PyAny>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        // A column that UTF-8 cannot encode is kept for the write, as PySpark's command sends it.
        let (columns, unsent) = crate::readwriter::writer_columns_kept_of(col, &cols)?;
        let writer = slf.borrow();
        writer.unsent().sort = unsent;
        writer.inner.update(|w| w.sort_by(columns));
        Ok(slf.clone())
    }

    /// The clustering columns, as PySpark's: column names, or a list or tuple of them, at
    /// least one.
    #[pyo3(name = "clusterBy", signature = (*cols))]
    fn cluster_by<'py>(
        slf: &Bound<'py, Self>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        // As PySpark's, which asserts the count after unpacking, and keeps the columns as they are:
        // a column that does not convert fails each write, after the partitioning columns.
        let items = crate::readwriter::writer_column_items(&cols)?;
        if items.is_empty() {
            return Err(pyo3::exceptions::PyAssertionError::new_err(
                "clusterBy needs one or more clustering columns.",
            ));
        }
        let (names, unsent) = crate::readwriter::writer_columns_kept(&items);
        let writer = slf.borrow();
        writer.unsent().clustering = unsent;
        writer.inner.update(|w| w.cluster_by(names));
        Ok(slf.clone())
    }

    /// Write to an external JDBC table. Connection details go through `properties`, options of
    /// any type (None skipped, bools lowercased), as PySpark's.
    #[pyo3(signature = (url, table, mode=None, properties=None))]
    fn jdbc(
        &self,
        py: Python<'_>,
        url: &Bound<'_, PyAny>,
        table: &Bound<'_, PyAny>,
        mode: Option<&Bound<'_, PyAny>>,
        properties: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        // Same order as PySpark: format, mode, properties, then url/dbtable (which
        // win over same-named properties), all kept on the writer, then save.
        let mut w = self.with_fixed_format(self.inner.get(), "jdbc");
        w = self.with_mode(w, mode);
        if let Some(props) = properties {
            // As PySpark's `options(**properties)`, whose unpacking raises for a key that is not
            // a str before any property is set, with the format and the mode kept on the writer.
            if props
                .keys()
                .iter()
                .any(|key| !key.is_instance_of::<pyo3::types::PyString>())
            {
                self.inner.set(w);
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "keywords must be strings",
                ));
            }
            for (key, value) in props.iter() {
                if let Some(value) = crate::coerce_option_value(&value)? {
                    w = w.option(&crate::str_argument(&key)?, &value);
                }
            }
        }
        // As PySpark's `options(url=url, dbtable=table)`, which skips a None.
        w = set_option_skip_none(w, "url", Some(url))?;
        w = set_option_skip_none(w, "dbtable", Some(table))?;
        self.inner.set(w.clone());
        let w = self.check_unsent_source_and_partition(py, w)?;
        self.check_bucket_count(py)?;
        let w = self.check_unsent_mode(py, w)?;
        interruptible(py, &w.spark_session(), || w.save(None).map_err(Box::new))
    }

    #[pyo3(signature = (path=None, format=None, mode=None, partitionBy=None, **options))]
    #[allow(non_snake_case)]
    fn save(
        &self,
        py: Python<'_>,
        path: Option<&Bound<'_, PyAny>>,
        format: Option<&Bound<'_, PyAny>>,
        mode: Option<&Bound<'_, PyAny>>,
        partitionBy: Option<&Bound<'_, PyAny>>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let w = self.apply_write_common(self.inner.get(), format, mode, partitionBy, options)?;
        // PySpark sets the inline format/mode/partitionBy/options on the writer itself, before its
        // command sets the source and the partitioning columns, the bucket count, the path, and
        // last the save mode.
        self.inner.set(w.clone());
        let w = self.check_unsent_source_and_partition(py, w)?;
        self.check_bucket_count(py)?;
        let path = path.map(crate::readwriter::writer_path).transpose()?;
        let w = self.check_unsent_mode(py, w)?;
        interruptible(py, &w.spark_session(), || {
            w.save(path.as_deref()).map_err(Box::new)
        })
    }

    #[pyo3(name = "saveAsTable", signature = (name, format=None, mode=None, partitionBy=None, **options))]
    #[allow(non_snake_case)]
    fn save_as_table(
        &self,
        py: Python<'_>,
        name: &Bound<'_, PyAny>,
        format: Option<&Bound<'_, PyAny>>,
        mode: Option<&Bound<'_, PyAny>>,
        partitionBy: Option<&Bound<'_, PyAny>>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let w = self.apply_write_common(self.inner.get(), format, mode, partitionBy, options)?;
        self.inner.set(w.clone());
        let w = self.check_unsent_source_and_partition(py, w)?;
        self.check_bucket_count(py)?;
        let name = crate::readwriter::writer_table_name(name)?;
        let w = self.check_unsent_mode(py, w)?;
        interruptible(py, &w.spark_session(), || {
            match &name {
                Some(name) => w.save_as_table(name),
                None => w.save(None),
            }
            .map_err(Box::new)
        })
    }

    /// Insert into an existing table. `overwrite` maps to overwrite/append mode,
    /// mirroring reference `DataFrameWriter.insertInto(tableName, overwrite=None)`, of its
    /// truth when it is not None (`"overwrite" if overwrite else "append"`).
    #[pyo3(name = "insertInto", signature = (tableName, overwrite=None))]
    fn insert_into(
        &self,
        py: Python<'_>,
        tableName: &Bound<'_, PyAny>,
        overwrite: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let mut w = self.inner.get();
        if let Some(ow) = overwrite {
            let ow = ow.is_truthy()?;
            self.unsent().mode = None;
            w = w.mode(if ow { "overwrite" } else { "append" });
        }
        // PySpark sets the overwrite/append mode on the writer itself.
        self.inner.set(w.clone());
        let w = self.check_unsent_source_and_partition(py, w)?;
        self.check_bucket_count(py)?;
        let tableName = crate::readwriter::writer_table_name(tableName)?;
        let w = self.check_unsent_mode(py, w)?;
        interruptible(py, &w.spark_session(), || {
            match &tableName {
                Some(name) => w.insert_into(name),
                None => w.save(None),
            }
            .map_err(Box::new)
        })
    }

    /// Write as PARQUET - full pyspark signature; each named option is
    /// applied when provided.
    #[pyo3(signature = (path, mode=None, partitionBy=None, compression=None, **unexpected), text_signature = "($self, path, mode=None, partitionBy=None, compression=None)")]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn parquet(
        &self,
        py: Python<'_>,
        path: &Bound<'_, PyAny>,
        mode: Option<&Bound<'_, PyAny>>,
        partitionBy: Option<&Bound<'_, PyAny>>,
        compression: Option<&Bound<'_, PyAny>>,
        unexpected: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        crate::options::check_unexpected_keywords("DataFrameWriter.parquet", unexpected)?;
        let mut w = self.inner.get();
        w = self.with_mode(w, mode);
        if let Some(pb) = partitionBy {
            w = self.with_partition_by(w, std::slice::from_ref(pb))?;
        }
        w = set_option_skip_none(w, "compression", compression)?;
        // PySpark sets the mode/options/format on the writer itself, then saves.
        self.inner.set(self.with_fixed_format(w.clone(), "parquet"));
        let w = self.check_unsent_source_and_partition(py, w)?;
        self.check_bucket_count(py)?;
        let path = crate::readwriter::writer_path(path)?;
        let w = self.check_unsent_mode(py, w)?;
        interruptible(py, &w.spark_session(), || {
            w.parquet(&path).map_err(Box::new)
        })
    }

    /// Write as JSON - full pyspark signature; each named option is
    /// applied when provided.
    #[pyo3(signature = (path, mode=None, compression=None, dateFormat=None, timestampFormat=None, lineSep=None, encoding=None, ignoreNullFields=None, **unexpected), text_signature = "($self, path, mode=None, compression=None, dateFormat=None, timestampFormat=None, lineSep=None, encoding=None, ignoreNullFields=None)")]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn json(
        &self,
        py: Python<'_>,
        path: &Bound<'_, PyAny>,
        mode: Option<&Bound<'_, PyAny>>,
        compression: Option<&Bound<'_, PyAny>>,
        dateFormat: Option<&Bound<'_, PyAny>>,
        timestampFormat: Option<&Bound<'_, PyAny>>,
        lineSep: Option<&Bound<'_, PyAny>>,
        encoding: Option<&Bound<'_, PyAny>>,
        ignoreNullFields: Option<&Bound<'_, PyAny>>,
        unexpected: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        crate::options::check_unexpected_keywords("DataFrameWriter.json", unexpected)?;
        let mut w = self.inner.get();
        w = self.with_mode(w, mode);
        w = set_options_skip_none(
            w,
            &[
                ("compression", compression),
                ("dateFormat", dateFormat),
                ("timestampFormat", timestampFormat),
                ("lineSep", lineSep),
                ("encoding", encoding),
                ("ignoreNullFields", ignoreNullFields),
            ],
        )?;
        // PySpark sets the mode/options/format on the writer itself, then saves.
        self.inner.set(self.with_fixed_format(w.clone(), "json"));
        let w = self.check_unsent_source_and_partition(py, w)?;
        self.check_bucket_count(py)?;
        let path = crate::readwriter::writer_path(path)?;
        let w = self.check_unsent_mode(py, w)?;
        interruptible(py, &w.spark_session(), || w.json(&path).map_err(Box::new))
    }

    /// Write as CSV - full pyspark signature; each named option is
    /// applied when provided.
    #[pyo3(signature = (path, mode=None, compression=None, sep=None, quote=None, escape=None, header=None, nullValue=None, escapeQuotes=None, quoteAll=None, dateFormat=None, timestampFormat=None, ignoreLeadingWhiteSpace=None, ignoreTrailingWhiteSpace=None, charToEscapeQuoteEscaping=None, encoding=None, emptyValue=None, lineSep=None, **unexpected), text_signature = "($self, path, mode=None, compression=None, sep=None, quote=None, escape=None, header=None, nullValue=None, escapeQuotes=None, quoteAll=None, dateFormat=None, timestampFormat=None, ignoreLeadingWhiteSpace=None, ignoreTrailingWhiteSpace=None, charToEscapeQuoteEscaping=None, encoding=None, emptyValue=None, lineSep=None)")]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn csv(
        &self,
        py: Python<'_>,
        path: &Bound<'_, PyAny>,
        mode: Option<&Bound<'_, PyAny>>,
        compression: Option<&Bound<'_, PyAny>>,
        sep: Option<&Bound<'_, PyAny>>,
        quote: Option<&Bound<'_, PyAny>>,
        escape: Option<&Bound<'_, PyAny>>,
        header: Option<&Bound<'_, PyAny>>,
        nullValue: Option<&Bound<'_, PyAny>>,
        escapeQuotes: Option<&Bound<'_, PyAny>>,
        quoteAll: Option<&Bound<'_, PyAny>>,
        dateFormat: Option<&Bound<'_, PyAny>>,
        timestampFormat: Option<&Bound<'_, PyAny>>,
        ignoreLeadingWhiteSpace: Option<&Bound<'_, PyAny>>,
        ignoreTrailingWhiteSpace: Option<&Bound<'_, PyAny>>,
        charToEscapeQuoteEscaping: Option<&Bound<'_, PyAny>>,
        encoding: Option<&Bound<'_, PyAny>>,
        emptyValue: Option<&Bound<'_, PyAny>>,
        lineSep: Option<&Bound<'_, PyAny>>,
        unexpected: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        crate::options::check_unexpected_keywords("DataFrameWriter.csv", unexpected)?;
        let mut w = self.inner.get();
        w = self.with_mode(w, mode);
        w = set_options_skip_none(
            w,
            &[
                ("compression", compression),
                ("sep", sep),
                ("quote", quote),
                ("escape", escape),
                ("header", header),
                ("nullValue", nullValue),
                ("escapeQuotes", escapeQuotes),
                ("quoteAll", quoteAll),
                ("dateFormat", dateFormat),
                ("timestampFormat", timestampFormat),
                ("ignoreLeadingWhiteSpace", ignoreLeadingWhiteSpace),
                ("ignoreTrailingWhiteSpace", ignoreTrailingWhiteSpace),
                ("charToEscapeQuoteEscaping", charToEscapeQuoteEscaping),
                ("encoding", encoding),
                ("emptyValue", emptyValue),
                ("lineSep", lineSep),
            ],
        )?;
        // PySpark sets the mode/options/format on the writer itself, then saves.
        self.inner.set(self.with_fixed_format(w.clone(), "csv"));
        let w = self.check_unsent_source_and_partition(py, w)?;
        self.check_bucket_count(py)?;
        let path = crate::readwriter::writer_path(path)?;
        let w = self.check_unsent_mode(py, w)?;
        interruptible(py, &w.spark_session(), || w.csv(&path).map_err(Box::new))
    }

    /// Write as ORC - full pyspark signature; each named option is
    /// applied when provided.
    #[pyo3(signature = (path, mode=None, partitionBy=None, compression=None, **unexpected), text_signature = "($self, path, mode=None, partitionBy=None, compression=None)")]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn orc(
        &self,
        py: Python<'_>,
        path: &Bound<'_, PyAny>,
        mode: Option<&Bound<'_, PyAny>>,
        partitionBy: Option<&Bound<'_, PyAny>>,
        compression: Option<&Bound<'_, PyAny>>,
        unexpected: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        crate::options::check_unexpected_keywords("DataFrameWriter.orc", unexpected)?;
        let mut w = self.inner.get();
        w = self.with_mode(w, mode);
        if let Some(pb) = partitionBy {
            w = self.with_partition_by(w, std::slice::from_ref(pb))?;
        }
        w = set_option_skip_none(w, "compression", compression)?;
        // PySpark sets the mode/options/format on the writer itself, then saves.
        self.inner.set(self.with_fixed_format(w.clone(), "orc"));
        let w = self.check_unsent_source_and_partition(py, w)?;
        self.check_bucket_count(py)?;
        let path = crate::readwriter::writer_path(path)?;
        let w = self.check_unsent_mode(py, w)?;
        interruptible(py, &w.spark_session(), || w.orc(&path).map_err(Box::new))
    }

    /// Write as TEXT - full pyspark signature; each named option is
    /// applied when provided.
    #[pyo3(signature = (path, compression=None, lineSep=None, **unexpected), text_signature = "($self, path, compression=None, lineSep=None)")]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn text(
        &self,
        py: Python<'_>,
        path: &Bound<'_, PyAny>,
        compression: Option<&Bound<'_, PyAny>>,
        lineSep: Option<&Bound<'_, PyAny>>,
        unexpected: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        crate::options::check_unexpected_keywords("DataFrameWriter.text", unexpected)?;
        let mut w = self.inner.get();
        w = set_options_skip_none(w, &[("compression", compression), ("lineSep", lineSep)])?;
        // PySpark sets the mode/options/format on the writer itself, then saves.
        self.inner.set(self.with_fixed_format(w.clone(), "text"));
        let w = self.check_unsent_source_and_partition(py, w)?;
        self.check_bucket_count(py)?;
        let path = crate::readwriter::writer_path(path)?;
        let w = self.check_unsent_mode(py, w)?;
        interruptible(py, &w.spark_session(), || w.text(&path).map_err(Box::new))
    }

    /// Write as XML - full pyspark signature; each named option is
    /// applied when provided.
    #[pyo3(signature = (path, rowTag=None, mode=None, attributePrefix=None, valueTag=None, rootTag=None, declaration=None, arrayElementName=None, nullValue=None, dateFormat=None, timestampFormat=None, compression=None, encoding=None, validateName=None, **unexpected), text_signature = "($self, path, rowTag=None, mode=None, attributePrefix=None, valueTag=None, rootTag=None, declaration=None, arrayElementName=None, nullValue=None, dateFormat=None, timestampFormat=None, compression=None, encoding=None, validateName=None)")]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn xml(
        &self,
        py: Python<'_>,
        path: &Bound<'_, PyAny>,
        rowTag: Option<&Bound<'_, PyAny>>,
        mode: Option<&Bound<'_, PyAny>>,
        attributePrefix: Option<&Bound<'_, PyAny>>,
        valueTag: Option<&Bound<'_, PyAny>>,
        rootTag: Option<&Bound<'_, PyAny>>,
        declaration: Option<&Bound<'_, PyAny>>,
        arrayElementName: Option<&Bound<'_, PyAny>>,
        nullValue: Option<&Bound<'_, PyAny>>,
        dateFormat: Option<&Bound<'_, PyAny>>,
        timestampFormat: Option<&Bound<'_, PyAny>>,
        compression: Option<&Bound<'_, PyAny>>,
        encoding: Option<&Bound<'_, PyAny>>,
        validateName: Option<&Bound<'_, PyAny>>,
        unexpected: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        crate::options::check_unexpected_keywords("DataFrameWriter.xml", unexpected)?;
        let mut w = self.inner.get();
        w = set_option_skip_none(w, "rowTag", rowTag)?;
        w = self.with_mode(w, mode);
        w = set_options_skip_none(
            w,
            &[
                ("attributePrefix", attributePrefix),
                ("valueTag", valueTag),
                ("rootTag", rootTag),
                ("declaration", declaration),
                ("arrayElementName", arrayElementName),
                ("nullValue", nullValue),
                ("dateFormat", dateFormat),
                ("timestampFormat", timestampFormat),
                ("compression", compression),
                ("encoding", encoding),
                ("validateName", validateName),
            ],
        )?;
        // PySpark sets the mode/options/format on the writer itself, then saves.
        self.inner.set(self.with_fixed_format(w.clone(), "xml"));
        let w = self.check_unsent_source_and_partition(py, w)?;
        self.check_bucket_count(py)?;
        let path = crate::readwriter::writer_path(path)?;
        let w = self.check_unsent_mode(py, w)?;
        interruptible(py, &w.spark_session(), || w.xml(&path).map_err(Box::new))
    }

    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        // The DataFrame, which reports the objects of the plan that the writer shares with it, as
        // PySpark's writer holds the plan (see `crate::gc::Holders`). The kept arguments, which
        // Python's garbage collector sees, as `exclusive_relation_owners` reports a DataFrame's
        // owners: only when this writer holds them alone, so that a copy that shares them does not
        // count one reference twice. A lock held elsewhere is skipped. Nothing of a writer object
        // that is being created.
        if !self.gc_ready.get() {
            return Ok(());
        }
        visit.call(&self.frame)?;
        if std::sync::Arc::strong_count(&self.unsent) != 1 {
            return Ok(());
        }
        if let Ok(unsent) = self.unsent.try_lock() {
            for object in [
                &unsent.format,
                &unsent.partition,
                &unsent.clustering,
                &unsent.mode,
            ]
            .into_iter()
            .flatten()
            {
                visit.call(object)?;
            }
        }
        Ok(())
    }

    /// Drop this writer's references to the kept arguments and to its state, of which it keeps a
    /// cleared copy; a copy's are kept.
    fn __clear__(&mut self) {
        self.unsent = std::sync::Arc::new(std::sync::Mutex::new(UnsentArgs::default()));
        self.inner = std::sync::Arc::new(
            self.inner
                .cleared(spark_connect::readwriter::DataFrameWriter::clear_plan),
        );
    }
}

/// Python wrapper for the V2 `DataFrameWriter` (`df.writeTo`).
#[pyclass(
    name = "DataFrameWriterV2",
    module = "pyspark.sql.connect.readwriter",
    weakref,
    dict
)]
pub struct PyDataFrameWriterV2 {
    inner: std::sync::Arc<WriterState<spark_connect::readwriter::DataFrameWriterV2>>,
    /// The DataFrame that is written, as the Python object of `df.writeTo`, as PySpark's `_df`.
    frame: Py<PyDataFrame>,

    /// The table of `writeTo` that its string field does not take, which Python's garbage
    /// collector sees ([`written_table`]).
    unsent_table: Option<Py<PyAny>>,
    /// The provider of `using` when it is not a str of UTF-8, which each write fails with after
    /// its table, as PySpark's command sets it there. A lock, as the state of the writer, since
    /// the writer's methods run on several threads ([`WriterState`]).
    unsent_provider: std::sync::Arc<std::sync::Mutex<Option<Py<PyAny>>>>,
    /// The first clustering column of `clusterBy` that is not a str of UTF-8, which each write
    /// fails with after its provider, as PySpark's command extends the clustering columns there.
    unsent_clustering: std::sync::Arc<std::sync::Mutex<Option<Py<PyAny>>>>,
    /// The first partitioning column of `partitionedBy` that is not a str of UTF-8, which each
    /// write fails with after its provider.
    unsent_partition: std::sync::Arc<std::sync::Mutex<Option<Py<PyAny>>>>,
    /// The options that the core writer does not take, in the order of PySpark's dict, which each
    /// write fails with after its partitioning and clustering columns.
    unsent_options: std::sync::Arc<std::sync::Mutex<crate::options::UnencodableOptions>>,
    /// The table properties that the core writer does not take, which each write fails with after
    /// its options.
    unsent_properties: std::sync::Arc<std::sync::Mutex<crate::options::UnencodableOptions>>,
    /// Set once the value is written, as `__traverse__` reads `unsent_provider` and
    /// `unsent_clustering`, which are not valid while the writer object is created (see
    /// [`crate::gc::GcReady`]).
    gc_ready: crate::gc::GcReady,
}

impl PyDataFrameWriterV2 {
    fn unsent_provider(&self) -> std::sync::MutexGuard<'_, Option<Py<PyAny>>> {
        self.unsent_provider
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn unsent_clustering(&self) -> std::sync::MutexGuard<'_, Option<Py<PyAny>>> {
        self.unsent_clustering
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn unsent_partition(&self) -> std::sync::MutexGuard<'_, Option<Py<PyAny>>> {
        self.unsent_partition
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn unsent_options(&self) -> std::sync::MutexGuard<'_, crate::options::UnencodableOptions> {
        self.unsent_options
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn unsent_properties(&self) -> std::sync::MutexGuard<'_, crate::options::UnencodableOptions> {
        self.unsent_properties
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Set the option `key` to `value` on the writer, as PySpark's dict takes it
    /// ([`put_writer_option`]).
    fn put_option(&self, key: &crate::options::OptionKey, value: &crate::options::OptionText) {
        let mut unsent = self.unsent_options();
        self.inner
            .update(|w| crate::options::put_writer_option(w, &mut unsent, key, value));
    }

    /// Set the table property `key` to `value`, or unset it for None, as PySpark's
    /// `tableProperty`: the core writer takes what encodes, and the others are kept for the write.
    fn put_property(
        &self,
        key: &crate::options::OptionKey,
        value: Option<&crate::options::OptionKey>,
    ) {
        let mut unsent = self.unsent_properties();
        self.inner.update(
            |w| match key.core_text().and_then(|name| name.text.as_deref()) {
                Some(name) => match value {
                    Some(value) => match value.core_text().and_then(|v| v.text.as_deref()) {
                        Some(text) => {
                            unsent.remove(key);
                            w.table_property(name, text)
                        }
                        None => {
                            unsent.set(key, Some(value.clone()));
                            w.unset_table_property(name)
                        }
                    },
                    None => {
                        unsent.remove(key);
                        w.unset_table_property(name)
                    }
                },
                None => {
                    unsent.set(key, value.cloned());
                    w
                }
            },
        );
    }

    /// The error of a write whose partitioning column is not text, which PySpark's command sets
    /// after the provider.
    fn check_unsent_partition(&self, py: Python<'_>) -> PyResult<()> {
        let column = self.unsent_partition().as_ref().map(|c| c.clone_ref(py));
        match column {
            Some(column) => crate::str_argument(column.bind(py)).map(drop),
            None => Ok(()),
        }
    }

    /// The errors of a write, in the order of PySpark's command: the provider, the partitioning
    /// and the clustering columns, the options, and then the table properties.
    fn check_unsent_all(&self, py: Python<'_>) -> PyResult<()> {
        self.check_unsent_provider(py)?;
        self.check_unsent_partition(py)?;
        self.check_unsent_clustering(py)?;
        if let Some(error) = self.unsent_options().error() {
            return Err(crate::errors::spark_error_to_py_exception(error));
        }
        if let Some(error) = self.unsent_properties().error() {
            return Err(crate::errors::spark_error_to_py_exception(error));
        }
        Ok(())
    }

    /// The error of a write whose clustering column is not text, which PySpark's command extends
    /// after the provider and the partitioning columns.
    fn check_unsent_clustering(&self, py: Python<'_>) -> PyResult<()> {
        let column = self.unsent_clustering().as_ref().map(|c| c.clone_ref(py));
        match column {
            Some(column) => crate::str_argument(column.bind(py)).map(drop),
            None => Ok(()),
        }
    }

    /// The error of a write whose provider is not text, which PySpark's command sets after the
    /// table ([`check_written_table`]).
    fn check_unsent_provider(&self, py: Python<'_>) -> PyResult<()> {
        let provider = self.unsent_provider().as_ref().map(|p| p.clone_ref(py));
        match provider {
            Some(provider) => crate::str_argument(provider.bind(py)).map(drop),
            None => Ok(()),
        }
    }
}

#[pymethods]
impl PyDataFrameWriterV2 {
    /// The shallow copy of a V2 writer, as PySpark's `copy.copy`: the copy shares the state of
    /// this writer, as its `_write` is the same object, and the same DataFrame.
    fn __copy__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let this = slf.borrow();
        let copy = Bound::new(
            py,
            PyDataFrameWriterV2 {
                inner: std::sync::Arc::clone(&this.inner),
                unsent_table: this.unsent_table.as_ref().map(|table| table.clone_ref(py)),
                unsent_provider: std::sync::Arc::clone(&this.unsent_provider),
                unsent_clustering: std::sync::Arc::clone(&this.unsent_clustering),
                unsent_partition: std::sync::Arc::clone(&this.unsent_partition),
                unsent_options: std::sync::Arc::clone(&this.unsent_options),
                unsent_properties: std::sync::Arc::clone(&this.unsent_properties),
                frame: this.frame.clone_ref(py),
                gc_ready: crate::gc::GcReady::new(),
            },
        )?;
        copy_instance_dict(slf.as_any(), copy.as_any())?;
        Ok(copy)
    }

    /// The deep copy of a writer, as PySpark's, which copies its session before its DataFrame (see
    /// [`deep_copy_writer`]).
    fn __deepcopy__<'py>(
        slf: &Bound<'py, Self>,
        memo: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let frame = slf.borrow().frame.clone_ref(slf.py());
        deep_copy_writer(slf.as_any(), &frame, memo)
    }

    /// The reduce of a V2 writer, as PySpark's pickle of its DataFrame and its state, which
    /// `_of_state` gives back.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let this = slf.borrow();
        use crate::reduce::{column_list, optional_object, state_reduce, str_dict};

        let writer = this.inner.get();
        let (kept_provider, kept_clustering, kept_partition) = {
            let kept = |object: &std::sync::Mutex<Option<Py<PyAny>>>| {
                object
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .as_ref()
                    .map(|o| o.clone_ref(py))
            };
            (
                kept(&this.unsent_provider),
                kept(&this.unsent_clustering),
                kept(&this.unsent_partition),
            )
        };
        let kept_options = this.unsent_options().to_list(py)?;
        let kept_properties = this.unsent_properties().to_list(py)?;
        let args = PyTuple::new(
            py,
            [
                this.frame.clone_ref(py).into_any().into_bound(py),
                writer.get_table_name().into_pyobject(py)?.into_any(),
                writer.get_provider().into_pyobject(py)?.into_any(),
                str_dict(py, writer.get_options())?.into_any(),
                str_dict(py, writer.get_table_properties())?.into_any(),
                column_list(py, writer.get_partition_cols())?.into_any(),
                PyList::new(py, writer.get_cluster_cols())?.into_any(),
                optional_object(py, this.unsent_table.as_ref()),
                optional_object(py, kept_provider.as_ref()),
                optional_object(py, kept_clustering.as_ref()),
                optional_object(py, kept_partition.as_ref()),
                kept_options.into_any(),
                kept_properties.into_any(),
            ],
        )?;
        reduce_with_instance_dict(slf, state_reduce(py, "writer_v2_reduce", args)?)
    }

    /// A V2 writer of its state, as `__reduce__` gives it: of the DataFrame `frame`, the table,
    /// the provider, the options, the table properties, the partitioning and the clustering.
    #[staticmethod]
    fn _of_state(
        py: Python<'_>,
        frame: Py<PyDataFrame>,
        table: &str,
        provider: Option<String>,
        options: std::collections::HashMap<String, String>,
        table_properties: std::collections::HashMap<String, String>,
        partition_cols: &Bound<'_, PyAny>,
        cluster_cols: Vec<String>,
        unsent_table: Option<Py<PyAny>>,
        kept_provider: Option<Py<PyAny>>,
        kept_clustering: Option<Py<PyAny>>,
        kept_partition: Option<Py<PyAny>>,
        kept_options: Vec<(Py<PyAny>, Py<PyAny>)>,
        kept_properties: Vec<(Py<PyAny>, Option<Py<PyAny>>)>,
    ) -> PyResult<PyDataFrameWriterV2> {
        let mut writer = frame.borrow(py).dataframe.write_to(table);
        if let Some(provider) = provider {
            writer = writer.using(&provider);
        }
        writer = writer.options(options);
        for (property, value) in &table_properties {
            writer = writer.table_property(property, value);
        }
        let partition_cols = crate::reduce::columns_of(partition_cols)?;
        if !partition_cols.is_empty() {
            writer = writer.partition_by(partition_cols);
        }
        if !cluster_cols.is_empty() {
            writer = writer.cluster_by(cluster_cols);
        }
        let restored = PyDataFrameWriterV2 {
            inner: std::sync::Arc::new(WriterState::of_frame(
                writer,
                &frame,
                spark_connect::readwriter::DataFrameWriterV2::shares_plan_with,
            )),
            unsent_table,
            unsent_provider: std::sync::Arc::new(std::sync::Mutex::new(kept_provider)),
            unsent_clustering: std::sync::Arc::new(std::sync::Mutex::new(kept_clustering)),
            unsent_partition: std::sync::Arc::new(std::sync::Mutex::new(kept_partition)),
            unsent_options: std::sync::Arc::new(std::sync::Mutex::new(Default::default())),
            unsent_properties: std::sync::Arc::new(std::sync::Mutex::new(Default::default())),
            frame,
            gc_ready: crate::gc::GcReady::new(),
        };
        // The options and the table properties that the core writer does not take are set in
        // their order, as PySpark's dicts, which keep them.
        for (key, value) in &kept_options {
            let key = crate::options::option_key(key.bind(py))?;
            if let Some(value) = crate::options::reader_option_text(value.bind(py))? {
                restored.put_option(&key, &value);
            }
        }
        for (key, value) in &kept_properties {
            let key = crate::options::option_key(key.bind(py))?;
            let value = match value {
                Some(value) => Some(crate::options::option_object(value.bind(py))?),
                None => None,
            };
            restored.put_property(&key, value.as_ref());
        }
        Ok(restored)
    }

    /// Set the schema, if any, and the options of `options` that are not None, through its own
    /// `schema` and `option`, as PySpark's `OptionUtils._set_opts`, which pandas-on-Spark calls.
    #[pyo3(signature = (schema=None, **options))]
    fn _set_opts(
        slf: &Bound<'_, Self>,
        schema: Option<Bound<'_, PyAny>>,
        options: Option<Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<()> {
        crate::options::set_opts(slf.as_any(), schema.as_ref(), options.as_ref())
    }

    /// Set the provider of the write, as PySpark's `using`: a str of UTF-8 (bytes too) at once, else
    /// kept for each write to fail with. None unsets it.
    fn using<'py>(
        slf: &Bound<'py, Self>,
        provider: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        let writer = slf.borrow();
        match provider.map(crate::str_argument) {
            Some(Ok(text)) => {
                *writer.unsent_provider() = None;
                writer.inner.update(|w| w.using(&text));
            }
            Some(Err(_)) => {
                *writer.unsent_provider() = provider.map(|p| p.clone().unbind());
                writer.inner.update(|w| w.clear_provider());
            }
            None => {
                *writer.unsent_provider() = None;
                writer.inner.update(|w| w.clear_provider());
            }
        }
        Ok(slf.clone())
    }

    // None -> option left unset, before the key is looked at, as PySpark's returns at once;
    // bools -> "true"/"false" (reference `to_str`), same as the other writer option bindings.
    fn option<'py>(
        slf: &Bound<'py, Self>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        if value.is_none() {
            return Ok(slf.clone());
        }
        let key = crate::options::option_key(key)?;
        if let Some(value) = crate::options::reader_option_text(value)? {
            slf.borrow().put_option(&key, &value);
        }
        Ok(slf.clone())
    }

    // Mirrors reference `DataFrameWriterV2.options(**options)`; None skipped, bools lowercased.
    #[pyo3(signature = (**options))]
    fn options<'py>(
        slf: &Bound<'py, Self>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        crate::options::check_keyword_keys(options)?;
        let writer = slf.borrow();
        for (key, value) in options.into_iter().flat_map(|options| options.iter()) {
            let Some(value) = crate::options::reader_option_text(&value)? else {
                continue;
            };
            let key = crate::options::option_key(&key)?;
            writer.put_option(&key, &value);
        }
        Ok(slf.clone())
    }

    /// Set the table property `property` to `value`, or leave it unset for None, as PySpark's. A
    /// value that is not a str or bytes of UTF-8 is kept, and fails each write, as PySpark's
    /// command sends the map at the write.
    #[pyo3(name = "tableProperty")]
    fn table_property<'py>(
        slf: &Bound<'py, Self>,
        property: &Bound<'_, PyAny>,
        value: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        let key = crate::options::option_key(property)?;
        let value = match value {
            Some(value) if !value.is_none() => Some(crate::options::option_object(value)?),
            _ => None,
        };
        slf.borrow().put_property(&key, value.as_ref());
        Ok(slf.clone())
    }

    /// Partition by the columns, each a Column or a column name, as PySpark's `F._to_col` of
    /// each, else `NOT_EXPECTED_TYPE` of `col`, also for a list. A column name that UTF-8 cannot
    /// encode is kept for the write.
    #[pyo3(name = "partitionedBy", signature = (col, *cols))]
    fn partitioned_by<'py>(
        slf: &Bound<'py, Self>,
        col: Bound<'_, PyAny>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        let writer = slf.borrow();
        let mut columns = Vec::new();
        let mut unsent = None;
        for item in std::iter::once(&col).chain(&cols) {
            if item.is_instance_of::<pyo3::types::PyString>() && item.extract::<String>().is_err() {
                unsent.get_or_insert_with(|| item.clone().unbind());
                continue;
            }
            columns.push(column_or_name(item)?);
        }
        *writer.unsent_partition() = unsent;
        writer.inner.update(|w| w.partition_by(columns));
        Ok(slf.clone())
    }

    /// The clustering columns of `clusterBy(col, *cols)`, which PySpark keeps as `[col, *cols]`:
    /// each column is set in the string field of the command at each write, as
    /// [`crate::str_argument`] converts it.
    #[pyo3(name = "clusterBy", signature = (col, *cols))]
    fn cluster_by<'py>(
        slf: &Bound<'py, Self>,
        col: &Bound<'_, PyAny>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        let items: Vec<Bound<'_, PyAny>> = std::iter::once(col.clone()).chain(cols).collect();
        let (names, unsent) = crate::readwriter::writer_columns_kept(&items);
        let writer = slf.borrow();
        *writer.unsent_clustering() = unsent;
        writer.inner.update(|w| w.cluster_by(names));
        Ok(slf.clone())
    }

    fn create(&self, py: Python<'_>) -> PyResult<()> {
        check_written_table(py, &self.unsent_table)?;
        self.check_unsent_all(py)?;
        let writer = self.inner.get();
        interruptible(py, &writer.spark_session(), || {
            writer.create().map_err(Box::new)
        })
    }

    fn replace(&self, py: Python<'_>) -> PyResult<()> {
        check_written_table(py, &self.unsent_table)?;
        self.check_unsent_all(py)?;
        let writer = self.inner.get();
        interruptible(py, &writer.spark_session(), || {
            writer.replace().map_err(Box::new)
        })
    }

    #[pyo3(name = "createOrReplace")]
    fn create_or_replace(&self, py: Python<'_>) -> PyResult<()> {
        check_written_table(py, &self.unsent_table)?;
        self.check_unsent_all(py)?;
        let writer = self.inner.get();
        interruptible(py, &writer.spark_session(), || {
            writer.create_or_replace().map_err(Box::new)
        })
    }

    fn append(&self, py: Python<'_>) -> PyResult<()> {
        check_written_table(py, &self.unsent_table)?;
        self.check_unsent_all(py)?;
        let writer = self.inner.get();
        interruptible(py, &writer.spark_session(), || {
            writer.append().map_err(Box::new)
        })
    }

    /// Overwrite the rows of `condition`, a Column or a column name, as PySpark's `F._to_col`,
    /// else `NOT_EXPECTED_TYPE` of `col`, before the command sets the table.
    fn overwrite(&self, py: Python<'_>, condition: &Bound<'_, PyAny>) -> PyResult<()> {
        let (writer, condition) = (self.inner.get(), column_or_name(condition)?);
        check_written_table(py, &self.unsent_table)?;
        self.check_unsent_all(py)?;
        interruptible(py, &writer.spark_session(), || {
            writer.overwrite(condition).map_err(Box::new)
        })
    }

    #[pyo3(name = "overwritePartitions")]
    fn overwrite_partitions(&self, py: Python<'_>) -> PyResult<()> {
        check_written_table(py, &self.unsent_table)?;
        self.check_unsent_all(py)?;
        let writer = self.inner.get();
        interruptible(py, &writer.spark_session(), || {
            writer.overwrite_partitions().map_err(Box::new)
        })
    }

    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        // Nothing of a writer object that is being created.
        if !self.gc_ready.get() {
            return Ok(());
        }
        // The DataFrame, which reports the objects of the plan that the writer shares with it, as
        // PySpark's writer holds the plan (see `crate::gc::Holders`).
        visit.call(&self.frame)?;
        if let Some(table) = &self.unsent_table {
            visit.call(table)?;
        }
        // The kept arguments and the state that a copy shares are reported by no owner, as the
        // writer's are, and a lock held elsewhere is skipped: of the state, the functions of the
        // UDFs of the partitioning columns that it alone keeps alive.
        for kept in [&self.unsent_provider, &self.unsent_clustering] {
            if std::sync::Arc::strong_count(kept) != 1 {
                continue;
            }
            if let Ok(kept) = kept.try_lock() {
                if let Some(object) = &*kept {
                    visit.call(object)?;
                }
            }
        }
        if std::sync::Arc::strong_count(&self.inner) == 1 {
            if let Some(writer) = self.inner.try_lock() {
                crate::udf_command::visit_pickler_objects(
                    writer.exclusive_pickler_objects(),
                    &visit,
                )?;
            }
        }
        Ok(())
    }

    /// Drop this writer's references to the kept arguments and to its state, of which it keeps a
    /// cleared copy; a copy's are kept.
    fn __clear__(&mut self) {
        self.unsent_table = None;
        self.unsent_provider = std::sync::Arc::new(std::sync::Mutex::new(None));
        self.unsent_clustering = std::sync::Arc::new(std::sync::Mutex::new(None));
        self.inner = std::sync::Arc::new(
            self.inner
                .cleared(spark_connect::readwriter::DataFrameWriterV2::clear_plan),
        );
    }
}

/// The state of a Python writer (`df.write`, `df.writeTo`, `df.mergeInto`, `df.writeStream`).
/// PySpark's writers are plain Python objects, so their methods may run on several threads at
/// once; a method that releases the GIL for its request must not hold a Rust borrow of the
/// object (that would make the calls of the other threads fail with "Already borrowed"). So
/// the lock is held only to copy or replace the state, and a request runs on its own copy.
pub(crate) struct WriterState<T> {
    /// The share of the holders of the plan of the DataFrame of the writers that hold this
    /// state, which keep the DataFrame, of which the state holds a clone, dropped before it: the
    /// builder steps keep the plan of the state (see [`crate::gc::Holders`]). It is only held.
    _share: crate::gc::Share,
    state: std::sync::Mutex<T>,
}

impl<T: Clone> WriterState<T> {
    pub(crate) fn new(state: T) -> Self {
        WriterState {
            _share: crate::gc::Share::none(),
            state: std::sync::Mutex::new(state),
        }
    }

    /// The state of a writer that keeps the DataFrame `frame`, of which `shares_plan` tells
    /// whether the state holds a clone of the plan, with a share of the holders of the plan.
    pub(crate) fn of_frame(
        state: T,
        frame: &Py<PyDataFrame>,
        shares_plan: impl FnOnce(&T, &DataFrame) -> bool,
    ) -> Self {
        let share = PyDataFrame::plan_share(frame, |dataframe| shares_plan(&state, dataframe));
        WriterState {
            _share: share,
            state: std::sync::Mutex::new(state),
        }
    }

    /// A state of no share, of a copy of this one that `clear` cleared, for the `__clear__` of a
    /// writer, whose copies keep this one.
    pub(crate) fn cleared(&self, clear: impl FnOnce(&mut T)) -> Self {
        let mut state = self.get();
        clear(&mut state);
        WriterState::new(state)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, T> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The state, unless another thread holds the lock, for a `__traverse__`, which must not
    /// wait for it.
    pub(crate) fn try_lock(&self) -> Option<std::sync::MutexGuard<'_, T>> {
        self.state.try_lock().ok()
    }

    /// A copy of the state, to run a request or a builder step on.
    pub(crate) fn get(&self) -> T {
        self.lock().clone()
    }

    pub(crate) fn set(&self, state: T) {
        *self.lock() = state;
    }

    /// Replace the state by `f` of it, as the builder methods do (`f` only builds, no Python).
    pub(crate) fn update(&self, f: impl FnOnce(T) -> T) {
        let mut state = self.lock();
        *state = f(state.clone());
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

/// Resolve the return type of a map or group function (a `DataType` wrapper or a DDL string)
/// into a Rust DataType: a DDL string as the server of `session` parses it, as PySpark's
/// `_parse_ddl`. Another object raises as [`check_return_type_argument`].
pub(crate) fn resolve_datatype(
    py: Python<'_>,
    session: &spark_connect::session::SparkSession,
    schema: &Bound<'_, PyAny>,
) -> PyResult<spark_connect::types::DataType> {
    // The live conversion, not the base snapshot: an edit in place of a StructType's fields or
    // metadata after it was built is kept, as PySpark does.
    if schema.extract::<PyRef<PyDataType>>().is_ok() {
        return crate::types::py_to_data_type(schema);
    }
    if schema.is_instance_of::<pyo3::types::PyString>() {
        return crate::types::schema_data_type(py, session, schema);
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
/// read, after the reference client's `warning`.
pub(crate) fn infer_group_map_eval_type(
    py: Python<'_>,
    func: &Bound<'_, PyAny>,
    infer: &str,
    warning: &str,
    default: i32,
) -> PyResult<i32> {
    match py
        .import("pyspark.sql.pandas.typehints")?
        .call_method1(infer, (func,))
    {
        Ok(inferred) if inferred.is_none() => Ok(default),
        Ok(inferred) => inferred.extract(),
        Err(err) if err.is_instance_of::<pyo3::exceptions::PyException>(py) => {
            let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
            py.import("warnings")?
                .call_method1("warn", (warning, category))?;
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
    return_type: spark_connect::types::DataType,
    eval_type_id: i32,
    input_cols: &[String],
) -> PyResult<CommonInlineUserDefinedFunctionExpression> {
    let name = udf_name(func)?;
    // As PySpark's `MapPartitions`, `GroupMap` and the other map nodes do when they build the
    // UDF, the return type is checked for the eval type: a map or group function returns a
    // struct, so another type raises `INVALID_RETURN_TYPE_FOR_PANDAS_UDF` here, not at the action.
    py.import("pyspark.sql.udf")?
        .getattr("UserDefinedFunction")?
        .call_method1(
            "_check_return_type",
            (
                crate::types::data_type_to_py(py, &return_type)?,
                eval_type_id,
            ),
        )?;
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
    let schema = crate::types::data_type_to_py(py, schema)?;
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

/// The result of a set operation of `this` and `other`, `dataframe`, which keeps the schema that
/// PySpark's `_merge_cached_schema` keeps: `this.schema` when the cached schemas of both are
/// equal, else none. The core keeps that of two cached structs; of a schema that a DataFrame
/// keeps as a Python object, the schema of `to` that is not a DataType or the `StructType` of a
/// map function, the objects are compared as they are now, as PySpark compares them with `==`,
/// and the result keeps a deep copy of that of `this`, as `this.schema` gives.
fn set_operation_result(
    py: Python<'_>,
    this: &PyDataFrame,
    other: &PyDataFrame,
    dataframe: DataFrame,
) -> PyResult<PyDataFrame> {
    if this.schema_object.is_none() && other.schema_object.is_none() {
        return Ok(PyDataFrame::new(dataframe));
    }
    let cached = |df: &PyDataFrame| -> PyResult<Option<Py<PyAny>>> {
        match &df.schema_object {
            Some(schema) => Ok(Some(schema.clone_ref(py))),
            None => df
                .dataframe
                .known_schema()
                .filter(|schema| check_schema_conversion(py, Some(schema)).is_ok())
                .map(|schema| crate::types::data_type_to_py(py, &schema))
                .transpose(),
        }
    };
    if let (Some(left), Some(right)) = (cached(this)?, cached(other)?) {
        if left.bind(py).eq(right.bind(py))? {
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

/// The reduce of a writer with the instance dictionary of `writer` as its state, as the reduce of
/// a plain Python object carries its `__dict__`: `copy.deepcopy` and a pickle keep the attributes
/// that were set on the writer.
fn reduce_with_instance_dict<'py>(
    writer: &Bound<'py, PyAny>,
    reduced: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let (creator, args): (Bound<'py, PyAny>, Bound<'py, PyAny>) = reduced.extract()?;
    let state = writer.getattr("__dict__")?;
    Ok(PyTuple::new(writer.py(), [creator, args, state])?.into_any())
}

/// The deep copy of a writer, as PySpark's, which copies the session of the writer before its
/// DataFrame: a stopped session raises the PySparkRuntimeError of its copy, and the DataFrame is
/// not copied. The copy is that of the reduce of the writer, `copy._reconstruct` with `memo`.
fn deep_copy_writer<'py>(
    writer: &Bound<'py, PyAny>,
    frame: &Py<PyDataFrame>,
    memo: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = writer.py();
    let session = frame.bind(py).borrow().spark_session(py)?;
    let copy = py.import("copy")?;
    copy.call_method1("deepcopy", (session, memo))?;
    let reduced = writer.call_method0("__reduce__")?;
    let mut args = vec![writer.clone(), memo.clone()];
    args.extend(reduced.cast::<PyTuple>()?.iter());
    copy.call_method1("_reconstruct", PyTuple::new(py, args)?)
}

/// Sets the instance dictionary of `target` to a copy of the one of `source`, as `copy.copy` of a
/// plain Python object does.
fn copy_instance_dict(source: &Bound<'_, PyAny>, target: &Bound<'_, PyAny>) -> PyResult<()> {
    target
        .getattr("__dict__")?
        .call_method1("update", (source.getattr("__dict__")?,))?;
    Ok(())
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
    let value = interruptible(py, &session, || {
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
    /// The share of the holders of the plan of `frame`, of which `dataframe` is a clone, dropped
    /// before it (see [`crate::gc::Holders`]).
    frame_share: crate::gc::Share,
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
        let frame = frame.clone().unbind();
        let frame_share =
            PyDataFrame::plan_share(&frame, |of_frame| dataframe.shares_plan_with(of_frame));
        PyLocalRowIterator {
            frame_share,
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
                let prepared = interruptible(py, &session, || {
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
            None => interruptible(py, &session, || {
                Ok::<_, Box<spark_connect_core::error::SparkError>>(self.next_row())
            })?,
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
        py.detach(move || drop(iterator));
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
    /// that Python's garbage collector sees: the DataFrame is reported, which reports the objects
    /// of the plan that the two share, such as the functions of its UDFs (see
    /// [`crate::gc::Holders`]). So a reference cycle through the rows and a function of the plan,
    /// such as a bound method of an object that keeps the rows, is freed. The iterator has no
    /// `__dict__`: it is tracked when it is allocated, and its value is written before any
    /// allocation that could start a collection, so no [`crate::gc::GcReady`] is needed.
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        visit.call(&self.frame)
    }

    /// Break a reference cycle that the garbage collector found unreachable: the iterator drops
    /// its clone of the plan, after its share of it.
    fn __clear__(&mut self) {
        self.frame_share = crate::gc::Share::none();
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
        .import("pyspark.sql.connect.client")?
        .getattr("PlanObservedMetrics")?;
    // The metrics as PySpark's client keeps them, the `Expression.Literal` messages that the
    // server sent, of `pyspark.sql.connect.proto`, which `PlanObservedMetrics.pairs` converts
    // when it is read, and so raises the `UNSUPPORTED_LITERAL` of a literal that PySpark does not
    // convert there.
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
                let encoded = pyo3::types::PyBytes::new(py, &literal.encode_to_vec());
                literal_cls.call_method1("FromString", (encoded,))
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
        && a.observed_metrics
            .iter()
            .zip(&b.observed_metrics)
            .all(|(x, y)| x == y || x.encode_to_vec() == y.encode_to_vec())
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
        .import("pyspark.sql.connect.client")?
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
        py.import("pyspark.sql.connect.proto")?
            .getattr("Expression")?
            .getattr("Literal")?
            .call_method1(
                "FromString",
                (pyo3::types::PyBytes::new(py, &literal.encode_to_vec()),),
            )?
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
