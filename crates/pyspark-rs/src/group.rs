//! PyO3 wrapper for spark_connect::group::GroupedData.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use spark_connect::group::{CoGroupedData as RustCoGroupedData, GroupedData as RustGroupedData};
use spark_connect::udf::eval_type;

use crate::column::PyColumn;
use crate::dataframe::{
    build_map_udf, check_return_type_argument, infer_group_map_eval_type, map_udf_of,
    resolve_datatype, validate_map_function, PyDataFrame,
};
use crate::errors::ResultExt;

use crate::interrupt::interruptible;
use crate::options::DefaultedOption;

/// A schema of `applyInPandasWithState` as PySpark sends it: a `StructType` as its JSON, and
/// anything else as it is, which protobuf converts as a str when the plan is built. The `Err` is
/// the error of a value that it does not take, which the plan raises then (see
/// [`crate::dataframe::unbuildable_str_error`]).
fn sent_schema(
    py: Python<'_>,
    schema: &Bound<'_, PyAny>,
) -> PyResult<Result<String, spark_connect_core::error::SparkError>> {
    if let Ok(st) = schema.extract::<PyRef<'_, crate::types::PyStructType>>() {
        return Ok(Ok(spark_connect::types::DataType::Struct {
            fields: st.current_fields(py)?,
        }
        .json()));
    }
    if let Some(error) = crate::dataframe::unbuildable_str_error(schema)? {
        return Ok(Err(error));
    }
    Ok(Ok(crate::options::str_argument(schema)?))
}

/// As PySpark's `_check_same_session` of the DataFrames of grouped data: `SESSION_NOT_SAME`
/// unless `other` is of the session of `this`.
fn check_same_session(
    py: Python<'_>,
    this: &RustGroupedData,
    other: &RustGroupedData,
) -> PyResult<()> {
    if this.spark_session().session_id() == other.spark_session().session_id() {
        Ok(())
    } else {
        Err(crate::column::raise_pyspark(
            py,
            "SessionNotSameException",
            "SESSION_NOT_SAME",
            &[],
        ))
    }
}

/// The shallow copy of the user-set attributes of `source`, its `__dict__`, onto `copy`, as
/// `copy.copy` of a plain Python object carries them: the same values, not copies of them.
pub(crate) fn copy_instance_state(
    copy: &Bound<'_, PyAny>,
    source: &Bound<'_, PyAny>,
) -> PyResult<()> {
    copy.getattr("__dict__")?
        .call_method1("update", (source.getattr("__dict__")?,))?;
    Ok(())
}

/// The state of `owner` in its reduce, as a plain Python object pickles it: its `__dict__`, the
/// user-set attributes, or None if it has none.
pub(crate) fn instance_state<'py>(owner: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let dict = owner.getattr("__dict__")?;
    if dict.len()? == 0 {
        return Ok(owner.py().None().into_bound(owner.py()));
    }
    Ok(dict)
}

/// The reduce `reduce` of `owner` with its state as its third element, which `copy` and `pickle`
/// restore into the `__dict__` of the object they rebuild, as they do for a plain object. A
/// reduce of no state is left as it is.
pub(crate) fn reduce_with_state<'py>(
    reduce: &Bound<'py, PyAny>,
    owner: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let state = instance_state(owner)?;
    if state.is_none() {
        return Ok(reduce.clone());
    }
    let mut items: Vec<Bound<'py, PyAny>> = reduce.cast::<PyTuple>()?.iter().collect();
    items.push(state);
    Ok(PyTuple::new(owner.py(), items)?.into_any())
}

/// Python wrapper for Spark GroupedData.
#[pyclass(
    name = "GroupedData",
    module = "pyspark.sql.connect.group",
    weakref,
    dict
)]
pub struct PyGroupedData {
    pub(crate) grouped_data: RustGroupedData,
    /// The DataFrame that is grouped, as the Python object that `groupBy` was called on, as
    /// PySpark's `GroupedData._df` is.
    frame: Py<PyDataFrame>,
}

impl PyGroupedData {
    pub fn new(grouped_data: RustGroupedData, frame: Py<PyDataFrame>) -> Self {
        PyGroupedData {
            grouped_data,
            frame,
        }
    }
}

/// The aggregates of the dict form `agg({column: function})`, one `function(column)` per entry
/// in order, as PySpark builds them (`F._invoke_function(function, F.col(column))`). The column
/// raises the errors of `F.col` (an int the `AttributeError` of its `endswith`), and then a
/// function that is not a str the `AssertionError` of `_invoke_function`, whose
/// `UnresolvedFunction` sets the name when it builds the plan, before the column.
pub(crate) fn dict_aggregates(
    dict: &Bound<'_, pyo3::types::PyDict>,
) -> PyResult<Vec<spark_connect::expression::Expression>> {
    let mut agg_exprs = Vec::with_capacity(dict.len());
    for (column, function) in dict.iter() {
        // As PySpark's `agg` of a dict, `F._invoke_function(func, F.col(name))`: the column is of
        // `col`, of its origin, and the function is of none.
        let c = {
            let _origin = crate::origin::with_origin("col")?;
            crate::plan_args::col(&column)?
        };
        if !function.is_instance_of::<pyo3::types::PyString>() {
            return Err(pyo3::exceptions::PyAssertionError::new_err(()));
        }
        let mut args = crate::plan_args::PlanArgs::new();
        let function = args.str(&function)?;
        // Build func(col) as an unresolved function resolved server-side, so aggregate names
        // (max/min/sum/avg/count/...) all work regardless of whether they are in the generated
        // dispatch.
        agg_exprs.push(
            args.expression(spark_connect::expression::Expression::function(
                function,
                vec![c.expression().clone()],
            )),
        );
    }
    Ok(agg_exprs)
}

impl PyGroupedData {
    /// `sum`, `avg`, `mean`, `min` and `max`, as PySpark's `_numeric_agg`: of column names,
    /// which it asserts, every numeric column without any, and `NOT_NUMERIC_COLUMNS` for a
    /// column that is not numeric.
    fn numeric_agg(
        &self,
        py: Python<'_>,
        function: &str,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let mut names = Vec::with_capacity(cols.len());
        for col in &cols {
            let Ok(name) = col.cast::<pyo3::types::PyString>() else {
                return Err(pyo3::exceptions::PyAssertionError::new_err(()));
            };
            names.push(name.to_cow().ok().map(|name| name.into_owned()));
        }
        let grouped = &self.grouped_data;
        if names.iter().any(Option::is_none) {
            // A name that UTF-8 cannot encode is no column of the schema, which the server
            // sends: PySpark reads the schema, and raises `NOT_NUMERIC_COLUMNS` of the `str` of
            // the list of the names that are not of numeric columns, in their order.
            let session = grouped.spark_session();
            let mut invalid = Vec::new();
            for (col, name) in cols.iter().zip(&names) {
                let numeric = match name {
                    None => {
                        interruptible(py, &session, || grouped.dataframe().schema())?;
                        false
                    }
                    Some(name) => interruptible(py, &session, || grouped.is_numeric_column(name))?,
                };
                if !numeric {
                    invalid.push(col.clone());
                }
            }
            let invalid = PyList::new(py, invalid)?.str()?;
            return Err(crate::errors::raise_pyspark_of(
                py,
                "PySparkTypeError",
                "NOT_NUMERIC_COLUMNS",
                &[("invalid_columns", invalid.as_any())],
            ));
        }
        let names = interruptible(py, &grouped.spark_session(), || {
            grouped.numeric_agg_columns(names.into_iter().flatten())
        })?;
        // As PySpark's `F._invoke_function(function, F.col(c))` of each column: the column is of
        // `col`, of its origin, and the function is of none.
        let mut expressions = Vec::with_capacity(names.len());
        for name in &names {
            let column = {
                let _origin = crate::origin::with_origin("col")?;
                spark_connect::col(name)
            };
            expressions.push(spark_connect::expression::Expression::function(
                function,
                vec![column.expression().clone()],
            ));
        }
        Ok(PyDataFrame::new(grouped.agg(expressions)))
    }
}

#[pymethods]
impl PyGroupedData {
    /// The DataFrame that is grouped, as PySpark's `GroupedData._df`.
    #[getter(_df)]
    fn df(&self, py: Python<'_>) -> Py<PyDataFrame> {
        self.frame.clone_ref(py)
    }

    /// The pivot values of the grouped data, as PySpark's `_pivot_values`: the literal Columns of
    /// the values that `pivot` was given, None if it was given none.
    #[getter(_pivot_values)]
    fn pivot_values<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyList>>> {
        let values = self.grouped_data.pivot_value_columns();
        if values.is_empty() {
            return Ok(None);
        }
        Ok(Some(crate::reduce::column_list(py, &values)?))
    }

    /// The shallow copy of grouped data, as PySpark's `copy.copy`: of the DataFrame that it
    /// groups, which the copy holds too, and of the same grouping.
    fn __copy__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let this = slf.borrow();
        let copy = Bound::new(
            py,
            PyGroupedData {
                grouped_data: this.grouped_data.clone(),
                frame: this.frame.clone_ref(py),
            },
        )?;
        copy_instance_state(copy.as_any(), slf.as_any())?;
        Ok(copy)
    }

    /// The reduce of grouped data, as PySpark's pickle of its state: its DataFrame, the
    /// grouping columns, the type of the grouping, the pivot column and values, and the
    /// grouping sets, which `_of_state` gives back.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        use crate::reduce::{column_list, group_type_name, state_reduce};

        let py = slf.py();
        let this = slf.borrow();
        let group = &this.grouped_data;
        let pivot_col = match group.pivot_column() {
            Some(column) => Bound::new(py, PyColumn::new(column))?.into_any(),
            None => py.None().into_bound(py),
        };
        let sets = group
            .grouping_set_columns()
            .iter()
            .map(|set| column_list(py, set))
            .collect::<PyResult<Vec<_>>>()?;
        let args = PyTuple::new(
            py,
            [
                this.frame.clone_ref(py).into_any().into_bound(py),
                group_type_name(group.group_type())
                    .into_pyobject(py)?
                    .into_any(),
                column_list(py, group.grouping_columns())?.into_any(),
                pivot_col,
                column_list(py, &group.pivot_value_columns())?.into_any(),
                PyList::new(py, sets)?.into_any(),
            ],
        )?;
        let reduce = state_reduce(py, "grouped_reduce", args)?;
        reduce_with_state(&reduce, slf.as_any())
    }

    /// Grouped data of its state, as `__reduce__` gives it: of the DataFrame `frame`, the
    /// grouping columns `cols`, the type of the grouping `group_type`, the pivot column
    /// `pivot_col` (None if there is none), the pivot values `pivot_values`, and the grouping
    /// sets `grouping_sets`, each a list of columns.
    #[staticmethod]
    fn _of_state(
        py: Python<'_>,
        frame: Py<PyDataFrame>,
        group_type: &str,
        cols: &Bound<'_, PyAny>,
        pivot_col: &Bound<'_, PyAny>,
        pivot_values: &Bound<'_, PyAny>,
        grouping_sets: &Bound<'_, PyAny>,
    ) -> PyResult<PyGroupedData> {
        use crate::reduce::{columns_of, group_type_of};

        let pivot_col = match pivot_col.is_none() {
            true => None,
            false => Some(pivot_col.extract::<PyRef<'_, PyColumn>>()?.column.clone()),
        };
        let sets = grouping_sets
            .try_iter()?
            .map(|set| columns_of(&set?))
            .collect::<PyResult<Vec<_>>>()?;
        let dataframe = frame.borrow(py).dataframe.clone();
        let grouped = RustGroupedData::from_parts(
            dataframe,
            columns_of(cols)?,
            group_type_of(group_type)?,
            pivot_col,
            columns_of(pivot_values)?,
            sets,
        );
        Ok(PyGroupedData::new(grouped, frame))
    }

    /// As PySpark Connect's: the grouping expressions, the columns of the DataFrame and their
    /// types, and the type of the grouping, such as
    /// `GroupedData[grouping expressions: [k], value: [k: int, v: double], type: GroupBy]`.
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        use spark_connect::plan::AggregateGroupType;

        let grouping: Vec<String> = self
            .grouped_data
            .grouping_columns()
            .iter()
            .map(|column| column.expression().render())
            .collect();
        let dataframe = self.grouped_data.dataframe().clone();
        let dtypes = interruptible(py, &dataframe.spark_session(), || dataframe.dtypes())?;
        let values: Vec<String> = dtypes
            .iter()
            .map(|(name, data_type)| format!("{name}: {data_type}"))
            .collect();
        let kind = match self.grouped_data.group_type() {
            AggregateGroupType::GroupBy => "GroupBy",
            AggregateGroupType::Rollup => "RollUp",
            AggregateGroupType::Cube => "Cube",
            AggregateGroupType::GroupingSets => "GroupingSets",
            AggregateGroupType::Pivot => "Pivot",
        };
        Ok(format!(
            "GroupedData[grouping expressions: [{}], value: [{}], type: {kind}]",
            grouping.join(", "),
            values.join(", ")
        ))
    }

    /// Aggregate with expressions (accepts Column objects), or the dict form
    /// `agg({"col": "aggfunc"})` (PySpark parity), e.g. `agg({"age": "max"})`.
    #[pyo3(signature = (*exprs))]
    fn agg(&self, _py: Python<'_>, exprs: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        // As PySpark's, which asserts that it has expressions.
        if exprs.is_empty() {
            return Err(pyo3::exceptions::PyAssertionError::new_err(
                "exprs should not be empty",
            ));
        }
        // Dict form: {column_name: aggregate_function_name} -> func(col(name)) per entry.
        if exprs.len() == 1 {
            if let Ok(dict) = exprs[0].cast::<pyo3::types::PyDict>() {
                let agg_exprs = dict_aggregates(dict)?;
                let df = self.grouped_data.agg(agg_exprs);
                return Ok(PyDataFrame::new(df));
            }
        }
        // As PySpark's, which asserts that the expressions are Columns.
        let mut agg_exprs = vec![];
        for col in exprs {
            let Ok(col) = col.extract::<PyRef<'_, crate::column::PyColumn>>() else {
                return Err(pyo3::exceptions::PyAssertionError::new_err(
                    "all exprs should be Column",
                ));
            };
            agg_exprs.push(col.column.expression().clone());
        }
        let df = self.grouped_data.agg(agg_exprs);
        Ok(PyDataFrame::new(df))
    }

    /// Count rows in each group, as PySpark's `agg(F._invoke_function("count", F.lit(1))
    /// .alias("count"))`: the alias is of `alias`, of its origin, and the count is of none.
    fn count(&self) -> PyResult<PyDataFrame> {
        use spark_connect::expression::{Expression, LiteralExpression};
        let count = spark_connect::functions::count(spark_connect::Column::new(
            Expression::literal(LiteralExpression::int(1)),
        ));
        let counted = {
            let _origin = crate::origin::with_origin("alias")?;
            count.alias("count")
        };
        let df = self.grouped_data.agg(vec![counted.expression().clone()]);
        Ok(PyDataFrame::new(df))
    }

    /// Pivot on a column; `values=None` auto-discovers the distinct pivot values. As PySpark's,
    /// only the grouping of `groupBy` pivots, else `UNSUPPORTED_OPERATION`, and then the column
    /// is a str and the values a list of bools, floats, ints and strs, else `NOT_EXPECTED_TYPE`.
    #[pyo3(signature = (pivot_col, values=None))]
    fn pivot(
        &self,
        py: Python<'_>,
        pivot_col: &Bound<'_, PyAny>,
        values: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyGroupedData> {
        use crate::stat_arguments::not_expected_type;
        use pyo3::types::{PyBool, PyFloat, PyInt, PyList, PyString};
        self.grouped_data.check_pivot().to_pyerr()?;
        let Ok(name) = pivot_col.cast::<PyString>() else {
            return Err(not_expected_type(pivot_col, "pivot_col", "str"));
        };
        // The `F.lit` of each value, whose str PySpark sets when it builds the plan: of each, the
        // error of protobuf, if any (`values_args`).
        let mut values_args = Vec::new();
        let vals = match values {
            None => None,
            Some(values) => {
                let Ok(values) = values.cast::<PyList>() else {
                    return Err(not_expected_type(values, "values", "list"));
                };
                let mut out = Vec::with_capacity(values.len());
                for value in values.iter() {
                    if !(value.is_instance_of::<PyBool>()
                        || value.is_instance_of::<PyFloat>()
                        || value.is_instance_of::<PyInt>()
                        || value.is_instance_of::<PyString>())
                    {
                        return Err(not_expected_type(
                            &value,
                            "value",
                            "bool, float, int or str",
                        ));
                    }
                    let mut args = crate::plan_args::PlanArgs::new();
                    out.push(match value.is_instance_of::<PyString>() {
                        true => spark_connect::row::Value::String(args.str(&value)?),
                        false => crate::session::py_to_value(&value)?,
                    });
                    values_args.push(args);
                }
                Some(out)
            }
        };
        // As PySpark's `self._df[pivot_col]`: a column of the DataFrame that is grouped, by its
        // plan ID.
        let pcol = crate::plan_args::df_col(self.grouped_data.dataframe(), name.as_any(), false)?;
        let grouped = self.grouped_data.pivot(pcol, vals).to_pyerr()?;
        let grouped = if values_args.iter().all(|args| args.error().is_none()) {
            grouped
        } else {
            let values = grouped
                .pivot_value_columns()
                .into_iter()
                .zip(values_args)
                .map(|(value, args)| args.column(value))
                .collect();
            RustGroupedData::from_parts(
                grouped.dataframe().clone(),
                grouped.grouping_columns().to_vec(),
                grouped.group_type(),
                grouped.pivot_column(),
                values,
                Vec::new(),
            )
        };
        Ok(PyGroupedData::new(grouped, self.frame.clone_ref(py)))
    }

    /// Sum values in each group.
    #[pyo3(signature = (*cols))]
    fn sum(&self, py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        self.numeric_agg(py, "sum", cols)
    }

    /// Average values in each group.
    #[pyo3(signature = (*cols))]
    fn avg(&self, py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        self.numeric_agg(py, "avg", cols)
    }

    /// Minimum values in each group.
    #[pyo3(signature = (*cols))]
    fn min(&self, py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        self.numeric_agg(py, "min", cols)
    }

    /// Maximum values in each group.
    #[pyo3(signature = (*cols))]
    fn max(&self, py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        self.numeric_agg(py, "max", cols)
    }

    /// Mean (alias for avg).
    #[pyo3(signature = (*cols))]
    fn mean(&self, py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        self.numeric_agg(py, "avg", cols)
    }

    /// `GroupedData.applyInPandas`.
    #[pyo3(name = "applyInPandas")]
    fn apply_in_pandas(
        &self,
        py: Python<'_>,
        func: Bound<'_, PyAny>,
        schema: Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        let eval_type_id = infer_group_map_eval_type(
            py,
            &func,
            "infer_group_pandas_eval_type_from_func",
            "Cannot infer the eval type from type hints.",
            eval_type::SQL_GROUPED_MAP_PANDAS_UDF,
        )?;
        validate_map_function(py, &func, eval_type_id)?;
        // As PySpark's: a DDL string is parsed before the columns of the input are read.
        let session = self.grouped_data.spark_session();
        let return_type = resolve_datatype(py, &session, &schema)?;
        let cols = interruptible(py, &session, || self.grouped_data.input_columns())?;
        let udf = map_udf_of(py, &func, &schema, return_type, eval_type_id, &cols)?;
        Ok(PyDataFrame::new(self.grouped_data.apply_in_pandas(udf)))
    }

    /// Deprecated grouped-map apply: `apply(udf)` where `udf` is a
    /// `SQL_GROUPED_MAP_PANDAS_UDF` pandas UDF. Delegates to `applyInPandas` with the
    /// UDF's wrapped function and return type, as PySpark Connect's: anything else, such as a
    /// Column, a function or a UDF of another eval type, raises `PySparkTypeError`
    /// `INVALID_UDF_EVAL_TYPE`, and it warns to prefer `applyInPandas`.
    fn apply(&self, py: Python<'_>, udf: Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        // A Column has any attribute, of its field of that name, so it is checked first.
        let grouped_map = !udf.is_instance_of::<crate::column::PyColumn>()
            && udf.hasattr("func")?
            && udf
                .getattr("evalType")?
                .eq(eval_type::SQL_GROUPED_MAP_PANDAS_UDF)?;
        if !grouped_map {
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkTypeError",
                "INVALID_UDF_EVAL_TYPE",
                &[("eval_type", "SQL_GROUPED_MAP_PANDAS_UDF")],
            ));
        }
        let warnings = py.import("warnings")?;
        warnings.call_method1(
            "warn",
            (
                "It is preferred to use 'applyInPandas' over this API. This API will be deprecated in the \
                 future releases. See SPARK-28264 for more details.",
                py.get_type::<pyo3::exceptions::PyUserWarning>(),
            ),
        )?;
        let func = udf.getattr("func")?;
        let schema = udf.getattr("returnType")?;
        self.apply_in_pandas(py, func, schema)
    }

    /// `GroupedData.applyInArrow`.
    #[pyo3(name = "applyInArrow")]
    fn apply_in_arrow(
        &self,
        py: Python<'_>,
        func: Bound<'_, PyAny>,
        schema: Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        // The reference client's warning, with its trailing space.
        let eval_type_id = infer_group_map_eval_type(
            py,
            &func,
            "infer_group_arrow_eval_type_from_func",
            "Cannot infer the eval type from type hints. ",
            eval_type::SQL_GROUPED_MAP_ARROW_UDF,
        )?;
        validate_map_function(py, &func, eval_type_id)?;
        // As PySpark's: a DDL string is parsed before the columns of the input are read.
        let session = self.grouped_data.spark_session();
        let return_type = resolve_datatype(py, &session, &schema)?;
        let cols = interruptible(py, &session, || self.grouped_data.input_columns())?;
        let udf = map_udf_of(py, &func, &schema, return_type, eval_type_id, &cols)?;
        Ok(PyDataFrame::new(self.grouped_data.apply_in_arrow(udf)))
    }

    /// `GroupedData.cogroup`, as PySpark's `PandasCogroupedOps`, which checks
    /// `gd1._df._check_same_session(gd2._df)` first: an object that is not grouped data raises the
    /// `AttributeError` of its `_df`, or `SESSION_NOT_SAME` of a `_df` that is not a DataFrame of
    /// the session, such as the Column of a DataFrame's `_df`.
    #[pyo3(name = "cogroup")]
    fn cogroup(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<PyCoGroupedData> {
        let other = match other.cast::<PyGroupedData>() {
            Ok(other) => other,
            Err(err) => {
                let df = other.getattr("_df")?;
                let session = slf.borrow().grouped_data.spark_session();
                let same_session = df.extract::<PyRef<'_, PyDataFrame>>().is_ok_and(|df| {
                    df.dataframe.spark_session().session_id() == session.session_id()
                });
                if !same_session {
                    return Err(crate::column::raise_pyspark(
                        py,
                        "SessionNotSameException",
                        "SESSION_NOT_SAME",
                        &[],
                    ));
                }
                return Err(err.into());
            }
        };
        let this = slf.borrow();
        let that = other.borrow();
        check_same_session(py, &this.grouped_data, &that.grouped_data)?;
        Ok(PyCoGroupedData {
            inner: this.grouped_data.cogroup(&that.grouped_data),
            left: slf.clone().unbind(),
            right: other.clone().unbind(),
        })
    }

    /// `GroupedData.applyInPandasWithState` (stateful streaming), as PySpark's: the output
    /// schema is the return type of the function, whose DDL string is parsed once the columns of
    /// the input are read, and both schemas are sent as they are given, a `StructType` as its
    /// JSON, which the server parses. The plan sets the schemas, the output mode and the timeout
    /// in its string fields, in this order, after it pickles the function, when it is built: a
    /// value that protobuf does not take as a str (bytes of UTF-8 it does) fails each request of
    /// the DataFrame, as PySpark's plan fails to build, and not this call.
    #[pyo3(name = "applyInPandasWithState")]
    #[pyo3(signature = (func, outputStructType, stateStructType, outputMode, timeoutConf))]
    fn apply_in_pandas_with_state(
        &self,
        py: Python<'_>,
        func: Bound<'_, PyAny>,
        outputStructType: Bound<'_, PyAny>,
        stateStructType: Bound<'_, PyAny>,
        outputMode: Bound<'_, PyAny>,
        timeoutConf: Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        validate_map_function(py, &func, eval_type::SQL_GROUPED_MAP_PANDAS_UDF_WITH_STATE)?;
        check_return_type_argument(&outputStructType)?;
        let output_schema = sent_schema(py, &outputStructType)?;
        let state_schema = sent_schema(py, &stateStructType)?;
        let session = self.grouped_data.spark_session();
        let cols = interruptible(py, &session, || self.grouped_data.input_columns())?;
        let udf = build_map_udf(
            py,
            &session,
            &func,
            &outputStructType,
            eval_type::SQL_GROUPED_MAP_PANDAS_UDF_WITH_STATE,
            &cols,
        )?;
        // The plan sets the output schema, the state schema, the output mode and the timeout.
        let mut args = crate::plan_args::PlanArgs::new();
        let mut schema = |schema: Result<String, spark_connect_core::error::SparkError>| {
            schema.unwrap_or_else(|error| {
                args.record(error);
                String::new()
            })
        };
        let output_schema = schema(output_schema);
        let state_schema = schema(state_schema);
        let output_mode = args.str(&outputMode)?;
        let timeout_conf = args.str(&timeoutConf)?;
        let df = self.grouped_data.apply_in_pandas_with_state_with_schemas(
            udf,
            &output_schema,
            &state_schema,
            &output_mode,
            &timeout_conf,
        );
        Ok(PyDataFrame::new(args.plan(df)))
    }

    /// `GroupedData.transformWithStateInPandas` — arbitrary stateful processing with a
    /// user `StatefulProcessor` (pandas output). Mirrors the Connect API: the processor is
    /// wrapped by `TransformWithStateInPandasUdfUtils` into a UDF (cloudpickled + run on the
    /// server), and the `GroupMap` proto carries the transform-with-state info, as
    /// [`transform_with_state_of`] builds it.
    #[pyo3(name = "transformWithStateInPandas")]
    #[pyo3(
        signature = (statefulProcessor, outputStructType, outputMode, timeMode, initialState=None, eventTimeColumnName=DefaultedOption::Omitted),
        text_signature = "($self, statefulProcessor, outputStructType, outputMode, timeMode, initialState=None, eventTimeColumnName='')"
    )]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn transform_with_state_in_pandas(
        &self,
        py: Python<'_>,
        statefulProcessor: Bound<'_, PyAny>,
        outputStructType: Bound<'_, PyAny>,
        outputMode: Bound<'_, PyAny>,
        timeMode: Bound<'_, PyAny>,
        initialState: Option<PyRef<'_, PyGroupedData>>,
        eventTimeColumnName: DefaultedOption,
    ) -> PyResult<PyDataFrame> {
        transform_with_state_of(
            py,
            &self.grouped_data,
            &statefulProcessor,
            &outputStructType,
            [&outputMode, &timeMode],
            initialState.as_deref(),
            &eventTimeColumnName,
            true,
        )
    }

    /// `GroupedData.transformWithState` (row output). Like `transformWithStateInPandas` but
    /// with the row-oriented eval type, of PySpark's `TransformWithStateInPySpark` plan, which
    /// sends the output schema as the pandas one does ([`transform_with_state_of`]).
    #[pyo3(name = "transformWithState")]
    #[pyo3(
        signature = (statefulProcessor, outputStructType, outputMode, timeMode, initialState=None, eventTimeColumnName=DefaultedOption::Omitted),
        text_signature = "($self, statefulProcessor, outputStructType, outputMode, timeMode, initialState=None, eventTimeColumnName='')"
    )]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn transform_with_state(
        &self,
        py: Python<'_>,
        statefulProcessor: Bound<'_, PyAny>,
        outputStructType: Bound<'_, PyAny>,
        outputMode: Bound<'_, PyAny>,
        timeMode: Bound<'_, PyAny>,
        initialState: Option<PyRef<'_, PyGroupedData>>,
        eventTimeColumnName: DefaultedOption,
    ) -> PyResult<PyDataFrame> {
        transform_with_state_of(
            py,
            &self.grouped_data,
            &statefulProcessor,
            &outputStructType,
            [&outputMode, &timeMode],
            initialState.as_deref(),
            &eventTimeColumnName,
            false,
        )
    }
}

/// The DataFrame of `transformWithStateInPandas` (`pandas`) or `transformWithState` of `grouped`,
/// as PySpark's `BaseTransformWithStateInPySpark` plan: its output schema is `outputStructType`,
/// a DDL string as an `UnparsedDataType`, which the server parses, or the type, and its event-time
/// column is always set, `""` when it is omitted. The plan sets its values in its string fields
/// when it is built: the output mode once it built the grouping, and the time mode and the
/// event-time column once it built the initial state, before it pickles the function. A value that
/// protobuf does not take as a str (bytes of UTF-8 it does) fails each request of the DataFrame
/// there, as PySpark's plan fails to build, and not this call: the last grouping expression
/// carries its error, or the input when there is none.
#[allow(clippy::too_many_arguments)]
fn transform_with_state_of(
    py: Python<'_>,
    grouped: &RustGroupedData,
    processor: &Bound<'_, PyAny>,
    output_struct_type: &Bound<'_, PyAny>,
    [output_mode, time_mode]: [&Bound<'_, PyAny>; 2],
    initial_state: Option<&PyGroupedData>,
    event_time_column_name: &DefaultedOption,
    pandas: bool,
) -> PyResult<PyDataFrame> {
    if let Some(state) = initial_state {
        check_same_session(py, grouped, &state.grouped_data)?;
    }
    let (func, et) = twus_func(py, processor, time_mode, initial_state.is_some(), pandas)?;
    let session = grouped.spark_session();
    let cols = interruptible(py, &session, || grouped.input_columns())?;
    let udf = build_map_udf(py, &session, &func, output_struct_type, et, &cols)?;
    // As PySpark's, which sends a DDL string as an `UnparsedDataType`, for the server to parse.
    let out = if output_struct_type.is_instance_of::<pyo3::types::PyString>() {
        spark_connect::types::DataType::Unparsed {
            data_type_string: output_struct_type.extract()?,
        }
    } else {
        crate::types::py_to_data_type(output_struct_type)?
    };
    let event_time_column_name = match event_time_column_name {
        DefaultedOption::Omitted => pyo3::types::PyString::new(py, "").into_any(),
        DefaultedOption::Given(name) => name.bind(py).clone(),
    };
    let mut mode_args = crate::plan_args::PlanArgs::new();
    let output_mode = mode_args.str(output_mode)?;
    let mut info_args = crate::plan_args::PlanArgs::new();
    let time_mode = info_args.str(time_mode)?;
    let event_time_column_name = info_args.str(&event_time_column_name)?;
    let (grouped, initial_state) = match initial_state {
        Some(state) => (
            failing_after_grouping(grouped, mode_args),
            Some(failing_after_grouping(&state.grouped_data, info_args)),
        ),
        None => {
            let mut args = mode_args;
            if let Some(error) = info_args.error() {
                args.record(error.clone());
            }
            (failing_after_grouping(grouped, args), None)
        }
    };
    let df = match pandas {
        true => grouped.transform_with_state_in_pandas(
            udf,
            out,
            &output_mode,
            &time_mode,
            Some(&event_time_column_name),
            initial_state.as_ref(),
        ),
        false => grouped.transform_with_state(
            udf,
            out,
            &output_mode,
            &time_mode,
            Some(&event_time_column_name),
            initial_state.as_ref(),
        ),
    };
    Ok(PyDataFrame::new(df))
}

/// `grouped`, whose plan raises the error of `args`, if any, once it builds the grouping
/// expressions of `grouped`, as PySpark's plan sets a value of `args` then: the last grouping
/// column carries it, or else, of no grouping columns, the input.
fn failing_after_grouping(
    grouped: &RustGroupedData,
    args: crate::plan_args::PlanArgs,
) -> RustGroupedData {
    let Some(error) = args.error().cloned() else {
        return grouped.clone();
    };
    let mut columns = grouped.grouping_columns().to_vec();
    let dataframe = match columns.last_mut() {
        Some(last) => {
            *last = spark_connect::column::Column::new(
                spark_connect::expression::Expression::unbuildable(
                    last.expression().clone(),
                    None,
                    Some(error),
                    None,
                ),
            );
            grouped.dataframe().clone()
        }
        None => grouped.dataframe().unbuildable(error),
    };
    RustGroupedData::from_parts(
        dataframe,
        columns,
        grouped.group_type(),
        grouped.pivot_column(),
        grouped.pivot_value_columns(),
        grouped.grouping_set_columns().to_vec(),
    )
}

/// Wrap a user `StatefulProcessor` into the transform-with-state UDF callable, returning it
/// with the matching `PythonEvalType`. Mirrors the Connect client's
/// `TransformWithStateInPandasUdfUtils` usage (pandas vs row output; with/without init state).
fn twus_func<'py>(
    py: Python<'py>,
    processor: &Bound<'py, PyAny>,
    time_mode: &Bound<'py, PyAny>,
    has_init_state: bool,
    pandas: bool,
) -> PyResult<(Bound<'py, PyAny>, i32)> {
    let util = py
        .import("pyspark.sql.streaming.stateful_processor_util")?
        .getattr("TransformWithStateInPandasUdfUtils")?
        .call1((processor.clone(), time_mode))?;
    let (attr, et) = match (has_init_state, pandas) {
        (false, true) => (
            "transformWithStateUDF",
            eval_type::SQL_TRANSFORM_WITH_STATE_PANDAS_UDF,
        ),
        (true, true) => (
            "transformWithStateWithInitStateUDF",
            eval_type::SQL_TRANSFORM_WITH_STATE_PANDAS_INIT_STATE_UDF,
        ),
        (false, false) => (
            "transformWithStateUDF",
            eval_type::SQL_TRANSFORM_WITH_STATE_PYTHON_ROW_UDF,
        ),
        (true, false) => (
            "transformWithStateWithInitStateUDF",
            eval_type::SQL_TRANSFORM_WITH_STATE_PYTHON_ROW_INIT_STATE_UDF,
        ),
    };
    Ok((util.getattr(attr)?, et))
}

/// Python wrapper for cogrouped pandas/arrow ops, of the module of PySpark Connect's class of the
/// objects of `cogroup`, `pyspark.sql.connect.group.PandasCogroupedOps`.
#[pyclass(
    name = "PandasCogroupedOps",
    module = "pyspark.sql.connect.group",
    weakref,
    dict
)]
pub struct PyCoGroupedData {
    pub(crate) inner: RustCoGroupedData,
    /// The grouped data on the left and on the right, as the Python objects of `cogroup`.
    left: Py<PyGroupedData>,
    right: Py<PyGroupedData>,
}

#[pymethods]
impl PyCoGroupedData {
    /// The shallow copy of cogrouped ops, as PySpark's `copy.copy`: of the same two groups, with
    /// the instance attributes of these.
    fn __copy__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let this = slf.borrow();
        let copy = Bound::new(
            py,
            PyCoGroupedData {
                inner: this.inner.clone(),
                left: this.left.clone_ref(py),
                right: this.right.clone_ref(py),
            },
        )?;
        copy_instance_state(copy.as_any(), slf.as_any())?;
        Ok(copy)
    }

    /// The reduce of cogrouped ops, as PySpark's pickle of its two grouped data, which
    /// `_of_state` gives back, and of its instance attributes.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let this = slf.borrow();
        let args = PyTuple::new(
            py,
            [
                this.left.clone_ref(py).into_any().into_bound(py),
                this.right.clone_ref(py).into_any().into_bound(py),
            ],
        )?;
        let reduce = crate::reduce::state_reduce(py, "cogrouped_reduce", args)?;
        reduce_with_state(&reduce, slf.as_any())
    }

    /// The cogrouped ops of two grouped data, as `__reduce__` gives them back.
    #[staticmethod]
    fn _of_state(
        py: Python<'_>,
        left: Py<PyGroupedData>,
        right: Py<PyGroupedData>,
    ) -> PyResult<PyCoGroupedData> {
        let inner = {
            let this = left.borrow(py);
            let that = right.borrow(py);
            this.grouped_data.cogroup(&that.grouped_data)
        };
        Ok(PyCoGroupedData { inner, left, right })
    }

    /// `PandasCogroupedOps.applyInPandas`.
    #[pyo3(name = "applyInPandas")]
    fn apply_in_pandas(
        &self,
        py: Python<'_>,
        func: Bound<'_, PyAny>,
        schema: Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        validate_map_function(py, &func, eval_type::SQL_COGROUPED_MAP_PANDAS_UDF)?;
        // As PySpark's: a DDL string is parsed, and the function binds no columns, as it takes
        // the whole cogroups, so the inputs are not analyzed.
        let return_type = resolve_datatype(py, &self.inner.spark_session(), &schema)?;
        let udf = map_udf_of(
            py,
            &func,
            &schema,
            return_type,
            eval_type::SQL_COGROUPED_MAP_PANDAS_UDF,
            &[],
        )?;
        Ok(PyDataFrame::new(self.inner.apply_in_pandas(udf)))
    }

    /// `PandasCogroupedOps.applyInArrow`.
    #[pyo3(name = "applyInArrow")]
    fn apply_in_arrow(
        &self,
        py: Python<'_>,
        func: Bound<'_, PyAny>,
        schema: Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        validate_map_function(py, &func, eval_type::SQL_COGROUPED_MAP_ARROW_UDF)?;
        // As PySpark's: a DDL string is parsed, and the function binds no columns, as it takes
        // the whole cogroups, so the inputs are not analyzed.
        let return_type = resolve_datatype(py, &self.inner.spark_session(), &schema)?;
        let udf = map_udf_of(
            py,
            &func,
            &schema,
            return_type,
            eval_type::SQL_COGROUPED_MAP_ARROW_UDF,
            &[],
        )?;
        Ok(PyDataFrame::new(self.inner.apply_in_arrow(udf)))
    }
}
