//! PyO3 wrapper for spark_connect::group::GroupedData.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};

use spark_connect::column::Column;
use spark_connect::group::GroupedData as RustGroupedData;
use spark_connect::plan::AggregateGroupType;
use spark_connect::udf::eval_type;

use crate::column::PyColumn;
use crate::dataframe::{
    build_map_udf, check_return_type_argument, infer_group_map_eval_type, map_result_of,
    map_udf_of, return_type_of, validate_map_function, PyDataFrame,
};
use crate::interrupt::interruptible_of_values;
use crate::options::DefaultedOption;
use crate::state::attr;

/// A schema of `applyInPandasWithState` as PySpark sends it: a `StructType` as its `json()`, of
/// the methods of a Python subclass, and anything else as it is, which protobuf converts as a
/// str when the plan is built. The `Err` is the error of a value that it does not take, which the
/// plan raises then (see [`crate::dataframe::unbuildable_str_error`]).
fn sent_schema(
    py: Python<'_>,
    schema: &Bound<'_, PyAny>,
) -> PyResult<Result<String, spark_connect_core::error::SparkError>> {
    if schema.is_instance_of::<crate::types::PyStructType>() {
        return Ok(Ok(schema
            .call_method0(pyo3::intern!(py, "json"))?
            .extract()?));
    }
    if let Some(error) = crate::dataframe::unbuildable_str_error(schema)? {
        return Ok(Err(error));
    }
    Ok(Ok(crate::options::str_argument(schema)?))
}

/// The bare `AssertionError` of an `assert` of PySpark's grouped data and of the plans that they
/// make, of the state that they read.
fn assertion() -> PyErr {
    pyo3::exceptions::PyAssertionError::new_err(())
}

/// The type of the grouping of `value`, the `_group_type` of grouped data, of the names of
/// PySpark's groupings, which its `__init__` and its `Aggregate` plan assert.
fn group_type_of(value: &Bound<'_, PyAny>) -> Option<AggregateGroupType> {
    let name = value.cast::<PyString>().ok()?;
    Some(match &*name.to_cow().ok()? {
        "groupby" => AggregateGroupType::GroupBy,
        "rollup" => AggregateGroupType::Rollup,
        "cube" => AggregateGroupType::Cube,
        "pivot" => AggregateGroupType::Pivot,
        "grouping_sets" => AggregateGroupType::GroupingSets,
        _ => return None,
    })
}

/// Whether `value` is a list of Columns, as PySpark's `GroupedData.__init__` and its plans of a
/// grouping assert of its grouping columns.
fn is_column_list(value: &Bound<'_, PyAny>) -> bool {
    value
        .cast::<PyList>()
        .is_ok_and(|list| list.iter().all(|item| item.is_instance_of::<PyColumn>()))
}

/// The columns of `value`, the grouping columns of grouped data, or the pivot values or a set of
/// the grouping sets, as the plan of a grouping takes them: Columns, else the bare
/// `AssertionError` of PySpark's assertion of them. Of a list, or of any iterable where `list` is
/// false, as PySpark iterates a grouping set.
fn columns_of(value: &Bound<'_, PyAny>, list: bool) -> PyResult<Vec<Column>> {
    if list && !value.is_instance_of::<PyList>() {
        return Err(assertion());
    }
    value
        .try_iter()?
        .map(|item| match item?.cast::<PyColumn>() {
            Ok(column) => Ok(crate::column::column_of(column)?),
            Err(_) => Err(assertion()),
        })
        .collect()
}

/// The DataFrame of `value`, the `_df` of grouped data, of which PySpark's methods read
/// `self._df._plan` and `self._df._session`: a DataFrame of this client, of a subclass too, is
/// itself; another object, such as a proxy of a DataFrame, is `DataFrame(plan, session)` of its
/// `_plan` and its `_session`, which raise PySpark's `AttributeError` of an object without them.
fn frame_of(value: Bound<'_, PyAny>) -> PyResult<Bound<'_, PyDataFrame>> {
    let value = match value.cast_into::<PyDataFrame>() {
        Ok(frame) => return Ok(frame),
        Err(err) => err.into_inner(),
    };
    let py = value.py();
    let plan = attr(&value, pyo3::intern!(py, "_plan"))?;
    let session = attr(&value, pyo3::intern!(py, "_session"))?;
    Ok(py
        .get_type::<PyDataFrame>()
        .call1((plan, session))?
        .cast_into::<PyDataFrame>()?)
}

/// The grouped data of the core of `dataframe` and of `grouping_cols`, the `_grouping_cols` of
/// grouped data, as PySpark's plans of a map of the groups, which read no other state, take them:
/// of the plan of `dataframe`, as PySpark's read `self._df._plan` (see
/// [`crate::dataframe::PlanFrame::plan`]), in `session` when it is given, the session of the plan
/// of the other grouped data of a plan of both, which gives the plan of both its plan IDs there.
fn grouping_of(
    dataframe: &Bound<'_, PyDataFrame>,
    grouping_cols: &Bound<'_, PyAny>,
    session: Option<&spark_connect::session::SparkSession>,
) -> PyResult<RustGroupedData> {
    let mut plan = crate::dataframe::PlanFrame::of(dataframe)?.plan().clone();
    if let Some(session) = session {
        if !plan.is_numbered_in(session) {
            plan = plan.of_plan_in(session.clone());
        }
    }
    Ok(RustGroupedData::from_parts(
        plan,
        columns_of(grouping_cols, true)?,
        AggregateGroupType::GroupBy,
        None,
        Vec::new(),
        Vec::new(),
    ))
}

/// The grouped data of the core of `dataframe` and of the state of `grouped`, as PySpark's `agg`
/// and `_numeric_agg` read it for their `Aggregate` plan, after its `_df`: its `_group_type`,
/// `_grouping_cols`, `_pivot_col`, `_pivot_values` and `_grouping_sets`, in this order, with the
/// assertions of the plan. The pivot is of a grouping of `pivot` alone and the grouping sets of
/// one of `grouping_sets`, as the plan builds them.
fn aggregation_of(
    grouped: &Bound<'_, PyAny>,
    dataframe: spark_connect::dataframe::DataFrame,
) -> PyResult<RustGroupedData> {
    let py = grouped.py();
    let group_type = attr(grouped, pyo3::intern!(py, "_group_type"))?;
    let grouping_cols = attr(grouped, pyo3::intern!(py, "_grouping_cols"))?;
    let pivot_col = attr(grouped, pyo3::intern!(py, "_pivot_col"))?;
    let pivot_values = attr(grouped, pyo3::intern!(py, "_pivot_values"))?;
    let grouping_sets = attr(grouped, pyo3::intern!(py, "_grouping_sets"))?;
    let group_type = group_type_of(&group_type).ok_or_else(assertion)?;
    let grouping_cols = columns_of(&grouping_cols, true)?;
    let (pivot_col, pivot_values, grouping_sets) = match group_type {
        AggregateGroupType::Pivot => {
            let Ok(pivot_col) = pivot_col.cast::<PyColumn>() else {
                return Err(assertion());
            };
            let pivot_values = match pivot_values.is_none() {
                true => Vec::new(),
                false => columns_of(&pivot_values, true)?,
            };
            let pivot_col = crate::column::column_of(pivot_col)?;
            (Some(pivot_col), pivot_values, Vec::new())
        }
        AggregateGroupType::GroupingSets => {
            let sets = match grouping_sets.is_none() {
                true => Vec::new(),
                false => {
                    if !grouping_sets.is_instance_of::<PyList>() {
                        return Err(assertion());
                    }
                    grouping_sets
                        .try_iter()?
                        .map(|set| columns_of(&set?, false))
                        .collect::<PyResult<Vec<_>>>()?
                }
            };
            (None, Vec::new(), sets)
        }
        _ => {
            if !(pivot_col.is_none() && pivot_values.is_none() && grouping_sets.is_none()) {
                return Err(assertion());
            }
            (None, Vec::new(), Vec::new())
        }
    };
    Ok(RustGroupedData::from_parts(
        dataframe,
        grouping_cols,
        group_type,
        pivot_col,
        pivot_values,
        grouping_sets,
    ))
}

/// As PySpark's `this._df._check_same_session(other._df)` of grouped data `this` and `other`, as
/// `cogroup` and the `initialState` of `transformWithState` and `transformWithStateInPandas`
/// check it: of a DataFrame of this client, `SESSION_NOT_SAME` unless the `_df` of `other` is a
/// DataFrame of its session; of another object, its method of the name, of the `AttributeError`
/// of an object without it, which PySpark raises before it reads the `_df` of `other`. The
/// `AttributeError` of an object without a `_df` is PySpark's too.
fn check_same_session(this: &Bound<'_, PyAny>, other: &Bound<'_, PyAny>) -> PyResult<()> {
    let py = this.py();
    let frame = attr(this, pyo3::intern!(py, "_df"))?;
    match frame.cast::<PyDataFrame>() {
        Ok(frame) => {
            let other = attr(other, pyo3::intern!(py, "_df"))?;
            crate::dataframe::PlanFrame::of(frame)?.same_session(&other)?;
        }
        Err(_) => {
            let check = frame.getattr(pyo3::intern!(py, "_check_same_session"))?;
            check.call1((attr(other, pyo3::intern!(py, "_df"))?,))?;
        }
    }
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

// Python wrapper for Spark GroupedData. As PySpark's plain class, it keeps its state in its
// `__dict__`, its `_df`, `_group_type`, `_grouping_cols`, `_pivot_col`, `_pivot_values` and
// `_grouping_sets`, which its methods read when they make their DataFrames, and it is subclassed
// and constructed of PySpark's arguments (see `crate::state`).
#[doc = connect_doc!(GroupedData)]
#[pyclass(
    name = "GroupedData",
    module = "pyspark.sql.connect.group",
    subclass,
    weakref,
    dict
)]
pub struct PyGroupedData {}

impl PyGroupedData {
    /// The grouped data of `df` of the grouping `group_type`, as PySpark's `groupBy`, `rollup`,
    /// `cube` and `groupingSets` make them: `GroupedData(df=self, group_type=...,
    /// grouping_cols=..., grouping_sets=...)`, of the DataFrame object and of lists of the Columns
    /// of `grouping_cols` and of each set of `grouping_sets`, which its `__init__` keeps.
    pub(crate) fn of_grouping<'py>(
        df: &Bound<'py, PyAny>,
        group_type: &str,
        grouping_cols: &[Column],
        grouping_sets: Option<&[Vec<Column>]>,
    ) -> PyResult<Bound<'py, PyGroupedData>> {
        let py = df.py();
        let grouping_sets = grouping_sets
            .map(|sets| {
                let sets = sets
                    .iter()
                    .map(|set| crate::reduce::column_list(py, set))
                    .collect::<PyResult<Vec<_>>>()?;
                PyList::new(py, sets)
            })
            .transpose()?;
        let grouping_cols = crate::reduce::column_list(py, grouping_cols)?;
        let grouped = Bound::new(py, PyGroupedData {})?;
        Self::__init__(
            &grouped,
            df,
            PyString::new(py, group_type).as_any(),
            grouping_cols.as_any(),
            None,
            None,
            grouping_sets.as_ref().map(|sets| sets.as_any()),
        )?;
        Ok(grouped)
    }

    /// `sum`, `avg`, `mean`, `min` and `max`, as PySpark's `_numeric_agg`: of column names,
    /// which it asserts, every numeric column without any, and `NOT_NUMERIC_COLUMNS` for a
    /// column that is not numeric, of the columns of `self._df.schema`, the load of its
    /// `_cached_schema_serialized` or a copy of its `_schema` (see `Frame::schema_read`), and of
    /// the state of the grouped data, which its `Aggregate` reads then.
    fn numeric_agg(
        slf: &Bound<'_, Self>,
        function: &str,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<crate::dataframe::NewDataFrame> {
        let py = slf.py();
        let mut names = Vec::with_capacity(cols.len());
        for col in &cols {
            let Ok(name) = col.cast::<PyString>() else {
                return Err(assertion());
            };
            names.push(name.to_cow().ok().map(|name| name.into_owned()));
        }
        // As PySpark's `self._df.schema`, which raises the errors of its conversion of the schema
        // of the server, such as the `TypeError` of a user-defined type whose `__init__` takes
        // arguments.
        let frame = frame_of(attr(slf.as_any(), pyo3::intern!(py, "_df"))?)?;
        let (dataframe, frame_session) = {
            let frame = crate::dataframe::PlanFrame::of(&frame)?;
            let schema = frame.schema_read(py)?;
            (
                frame.plan().with_known_schema(Some(schema)),
                frame.session_object().clone().unbind(),
            )
        };
        let grouped = aggregation_of(slf.as_any(), dataframe)?;
        let session = grouped.spark_session();
        if names.iter().any(Option::is_none) {
            // A name that UTF-8 cannot encode is no column of the schema, which the server
            // sends: PySpark reads the schema, and raises `NOT_NUMERIC_COLUMNS` of the `str` of
            // the list of the names that are not of numeric columns, in their order.
            let mut invalid = Vec::new();
            for (col, name) in cols.iter().zip(&names) {
                let numeric = match name {
                    None => {
                        interruptible_of_values(py, &session, grouped.held_parts(), || {
                            grouped.dataframe().schema()
                        })?;
                        false
                    }
                    Some(name) => {
                        interruptible_of_values(py, &session, grouped.held_parts(), || {
                            grouped.is_numeric_column(name)
                        })?
                    }
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
        let names = interruptible_of_values(py, &session, grouped.held_parts(), || {
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
        // As PySpark's `DataFrame(plan.Aggregate(...), session=self._df._session)`.
        Ok(PyDataFrame::new(grouped.agg(expressions)).with_session(frame_session))
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

#[pymethods]
impl PyGroupedData {
    /// Grouped data of no state, as the `object.__new__` of PySpark's plain class: of any
    /// arguments, which `__init__` takes, so that a subclass of another `__init__` is built as it
    /// is in PySpark. The class shows the signature of `__init__`, as `inspect.signature` of
    /// PySpark's class does.
    #[new]
    #[pyo3(
        signature = (*_args, **_kwargs),
        text_signature = "(df, group_type, grouping_cols, pivot_col=None, pivot_values=None, grouping_sets=None)"
    )]
    fn py_new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        PyGroupedData {}
    }

    /// `GroupedData(df, group_type, grouping_cols, pivot_col=None, pivot_values=None,
    /// grouping_sets=None)`, as PySpark Connect's `__init__`, which asserts its arguments and
    /// sets its state in its order: the DataFrame, the type of the grouping and the list of the
    /// grouping columns as they are, the pivot column and the `F.lit` of each pivot value of a
    /// pivot, and the grouping sets of grouping sets, else None.
    #[pyo3(signature = (df, group_type, grouping_cols, pivot_col=None, pivot_values=None, grouping_sets=None))]
    fn __init__(
        slf: &Bound<'_, Self>,
        df: &Bound<'_, PyAny>,
        group_type: &Bound<'_, PyAny>,
        grouping_cols: &Bound<'_, PyAny>,
        pivot_col: Option<&Bound<'_, PyAny>>,
        pivot_values: Option<&Bound<'_, PyAny>>,
        grouping_sets: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        use crate::state::assign;
        let py = slf.py();
        let this = slf.as_any();
        let none = py.None().into_bound(py);
        if !df.is_instance_of::<PyDataFrame>() {
            return Err(assertion());
        }
        assign(this, &[(pyo3::intern!(py, "_df"), df)])?;
        let kind = group_type_of(group_type).ok_or_else(assertion)?;
        assign(this, &[(pyo3::intern!(py, "_group_type"), group_type)])?;
        if !is_column_list(grouping_cols) {
            return Err(assertion());
        }
        assign(
            this,
            &[
                (pyo3::intern!(py, "_grouping_cols"), grouping_cols),
                (pyo3::intern!(py, "_pivot_col"), &none),
                (pyo3::intern!(py, "_pivot_values"), &none),
            ],
        )?;
        if kind == AggregateGroupType::Pivot {
            let pivot_col = pivot_col
                .filter(|column| column.is_instance_of::<PyColumn>())
                .ok_or_else(assertion)?;
            assign(this, &[(pyo3::intern!(py, "_pivot_col"), pivot_col)])?;
            if let Some(values) = pivot_values {
                let Ok(values) = values.cast::<PyList>() else {
                    return Err(assertion());
                };
                // As PySpark's `[F.lit(v) for v in pivot_values]`.
                let lit = py
                    .import(pyo3::intern!(py, "pyspark.sql.connect.functions.builtin"))?
                    .getattr(pyo3::intern!(py, "lit"))?;
                let values = values
                    .iter()
                    .map(|value| lit.call1((value,)))
                    .collect::<PyResult<Vec<_>>>()?;
                let values = PyList::new(py, values)?;
                assign(
                    this,
                    &[(pyo3::intern!(py, "_pivot_values"), values.as_any())],
                )?;
            }
        }
        assign(this, &[(pyo3::intern!(py, "_grouping_sets"), &none)])?;
        if kind == AggregateGroupType::GroupingSets {
            let grouping_sets = grouping_sets.unwrap_or(&none);
            if !(grouping_sets.is_none() || grouping_sets.is_instance_of::<PyList>()) {
                return Err(assertion());
            }
            assign(
                this,
                &[(pyo3::intern!(py, "_grouping_sets"), grouping_sets)],
            )?;
        }
        Ok(())
    }

    /// The state of the grouped data, their `__dict__`, as `object.__getstate__` gives that of
    /// PySpark's plain object, which a copy and a pickle keep (see `state::getstate`).
    fn __getstate__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::state::getstate(slf.as_any())
    }

    /// The reduce of the grouped data, as PySpark's plain object's: `copyreg.__newobj__` of their
    /// class, of their `__dict__`, so a copy shares the DataFrame and the lists of the state, a
    /// deep copy copies them, and a pickle reaches the DataFrame first (see `state::reduce_ex`).
    fn __reduce_ex__<'py>(slf: &Bound<'py, Self>, protocol: i32) -> PyResult<Bound<'py, PyAny>> {
        crate::state::reduce_ex(slf.as_any(), protocol)
    }

    /// As PySpark Connect's: the grouping expressions, of the `str` of the `_expr` of each
    /// grouping column, the columns of the DataFrame and their types, of its `dtypes`, and the
    /// type of the grouping, such as
    /// `GroupedData[grouping expressions: [k], value: [k: int, v: double], type: GroupBy]`.
    fn __repr__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyString>> {
        let py = slf.py();
        let separator = PyString::new(py, ", ");
        let grouping = attr(slf.as_any(), pyo3::intern!(py, "_grouping_cols"))?
            .try_iter()?
            .map(|column| column?.getattr(pyo3::intern!(py, "_expr"))?.str())
            .collect::<PyResult<Vec<_>>>()?;
        let grouping = separator.call_method1(pyo3::intern!(py, "join"), (grouping,))?;
        let pair = PyString::new(py, "%s: %s");
        let values = attr(slf.as_any(), pyo3::intern!(py, "_df"))?
            .getattr(pyo3::intern!(py, "dtypes"))?
            .try_iter()?
            .map(|dtype| pair.rem(dtype?))
            .collect::<PyResult<Vec<_>>>()?;
        let values = separator.call_method1(pyo3::intern!(py, "join"), (values,))?;
        let group_type = attr(slf.as_any(), pyo3::intern!(py, "_group_type"))?;
        let mut kind = "Pivot";
        for (name, shown) in [
            ("groupby", "GroupBy"),
            ("rollup", "RollUp"),
            ("cube", "Cube"),
            ("grouping_sets", "GroupingSets"),
        ] {
            if group_type.eq(name)? {
                kind = shown;
                break;
            }
        }
        Ok(PyString::new(
            py,
            "GroupedData[grouping expressions: [{}], value: [{}], type: {}]",
        )
        .call_method1(pyo3::intern!(py, "format"), (grouping, values, kind))?
        .cast_into::<PyString>()?)
    }

    // Aggregate with expressions (accepts Column objects), or the dict form
    // `agg({"col": "aggfunc"})` (PySpark parity), e.g. `agg({"age": "max"})`, of the state that
    // PySpark's `Aggregate` reads then.
    #[pyo3(signature = (*exprs))]
    #[doc = connect_doc!(GroupedData.agg)]
    fn agg(
        slf: &Bound<'_, Self>,
        exprs: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<crate::dataframe::NewDataFrame> {
        // As PySpark's, which asserts that it has expressions.
        if exprs.is_empty() {
            return Err(pyo3::exceptions::PyAssertionError::new_err(
                "exprs should not be empty",
            ));
        }
        let mut agg_exprs = vec![];
        match exprs[0].cast::<pyo3::types::PyDict>() {
            // Dict form: {column_name: aggregate_function_name} -> func(col(name)) per entry.
            Ok(dict) if exprs.len() == 1 => agg_exprs = dict_aggregates(dict)?,
            // As PySpark's, which asserts that the expressions are Columns.
            _ => {
                for col in &exprs {
                    let Ok(col) = col.cast::<PyColumn>() else {
                        return Err(pyo3::exceptions::PyAssertionError::new_err(
                            "all exprs should be Column",
                        ));
                    };
                    agg_exprs.push(crate::column::column_of(col)?.expression().clone());
                }
            }
        }
        let py = slf.py();
        let frame = frame_of(attr(slf.as_any(), pyo3::intern!(py, "_df"))?)?;
        let (dataframe, session) = {
            let frame = crate::dataframe::PlanFrame::of(&frame)?;
            (
                frame.plan().clone(),
                frame.session_object().clone().unbind(),
            )
        };
        let grouped = aggregation_of(slf.as_any(), dataframe)?;
        // As PySpark's `DataFrame(plan.Aggregate(...), session=self._df._session)`.
        Ok(PyDataFrame::new(grouped.agg(agg_exprs)).with_session(session))
    }

    // Count rows in each group, as PySpark's `self.agg(F._invoke_function("count", F.lit(1))
    // .alias("count"))`, a call of the `agg` of the object, of a subclass too: the alias is of
    // `alias`, of its origin, and the count is of none.
    #[doc = connect_doc!(GroupedData.count)]
    fn count<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        use spark_connect::expression::{Expression, LiteralExpression};
        let py = slf.py();
        let count = spark_connect::functions::count(spark_connect::Column::new(
            Expression::literal(LiteralExpression::int(1)),
        ));
        let counted = {
            let _origin = crate::origin::with_origin("alias")?;
            count.alias("count")
        };
        let counted = crate::column::column_object(py, counted)?;
        slf.call_method1(pyo3::intern!(py, "agg"), (counted,))
    }

    // Pivot on a column; `values=None` auto-discovers the distinct pivot values. As PySpark's,
    // only the grouping of `groupBy` pivots, else `UNSUPPORTED_OPERATION`, of the `_group_type`
    // of the grouped data, and then the column is a str and the values a list of bools, floats,
    // ints and strs, else `NOT_EXPECTED_TYPE`. The grouped data that it makes are
    // `GroupedData(df=self._df, group_type="pivot", grouping_cols=self._grouping_cols,
    // pivot_col=self._df[pivot_col], pivot_values=values)`, which share the DataFrame and the list
    // of the grouping columns, as PySpark's.
    #[pyo3(signature = (pivot_col, values=None))]
    #[doc = connect_doc!(GroupedData.pivot)]
    fn pivot<'py>(
        slf: &Bound<'py, Self>,
        pivot_col: &Bound<'py, PyAny>,
        values: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyGroupedData>> {
        use crate::stat_arguments::not_expected_type;
        use pyo3::types::{PyBool, PyFloat, PyInt};
        let py = slf.py();
        let group_type = attr(slf.as_any(), pyo3::intern!(py, "_group_type"))?;
        if !group_type.eq("groupby")? {
            let operation = match group_type.eq("pivot")? {
                true => PyString::new(py, "Repeated PIVOT operation"),
                false => PyString::new(py, "PIVOT after ")
                    .add(group_type.call_method0(pyo3::intern!(py, "upper"))?.str()?)?
                    .cast_into::<PyString>()?,
            };
            return Err(crate::errors::raise_pyspark_of(
                py,
                "PySparkNotImplementedError",
                "UNSUPPORTED_OPERATION",
                &[("operation", operation.as_any())],
            ));
        }
        if !pivot_col.is_instance_of::<PyString>() {
            return Err(not_expected_type(pivot_col, "pivot_col", "str"));
        }
        if let Some(values) = values {
            let Ok(values) = values.cast::<PyList>() else {
                return Err(not_expected_type(values, "values", "list"));
            };
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
            }
        }
        let df = attr(slf.as_any(), pyo3::intern!(py, "_df"))?;
        let grouping_cols = attr(slf.as_any(), pyo3::intern!(py, "_grouping_cols"))?;
        let pivot_col = df.get_item(pivot_col)?;
        let grouped = Bound::new(py, PyGroupedData {})?;
        Self::__init__(
            &grouped,
            &df,
            PyString::new(py, "pivot").as_any(),
            &grouping_cols,
            Some(&pivot_col),
            values,
            None,
        )?;
        Ok(grouped)
    }

    // Sum values in each group.
    #[pyo3(signature = (*cols))]
    #[doc = connect_doc!(GroupedData.sum)]
    fn sum(
        slf: &Bound<'_, Self>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<crate::dataframe::NewDataFrame> {
        Self::numeric_agg(slf, "sum", cols)
    }

    // Average values in each group.
    #[pyo3(signature = (*cols))]
    #[doc = connect_doc!(GroupedData.avg)]
    fn avg(
        slf: &Bound<'_, Self>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<crate::dataframe::NewDataFrame> {
        Self::numeric_agg(slf, "avg", cols)
    }

    // Minimum values in each group.
    #[pyo3(signature = (*cols))]
    #[doc = connect_doc!(GroupedData.min)]
    fn min(
        slf: &Bound<'_, Self>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<crate::dataframe::NewDataFrame> {
        Self::numeric_agg(slf, "min", cols)
    }

    // Maximum values in each group.
    #[pyo3(signature = (*cols))]
    #[doc = connect_doc!(GroupedData.max)]
    fn max(
        slf: &Bound<'_, Self>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<crate::dataframe::NewDataFrame> {
        Self::numeric_agg(slf, "max", cols)
    }

    // Mean (alias for avg).
    #[pyo3(signature = (*cols))]
    #[doc = connect_doc!(GroupedData.mean)]
    fn mean(
        slf: &Bound<'_, Self>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<crate::dataframe::NewDataFrame> {
        Self::numeric_agg(slf, "avg", cols)
    }

    // `GroupedData.applyInPandas`, of the `_df` and the `_grouping_cols` that PySpark's
    // `GroupMap` reads.
    #[pyo3(name = "applyInPandas")]
    #[doc = connect_doc!(GroupedData.applyInPandas)]
    fn apply_in_pandas(
        slf: &Bound<'_, Self>,
        func: Bound<'_, PyAny>,
        schema: Bound<'_, PyAny>,
    ) -> PyResult<crate::dataframe::NewDataFrame> {
        let py = slf.py();
        let eval_type_id = infer_group_map_eval_type(
            py,
            &func,
            "infer_group_pandas_eval_type_from_func",
            "Cannot infer the eval type from type hints.",
            308,
            eval_type::SQL_GROUPED_MAP_PANDAS_UDF,
        )?;
        validate_map_function(py, &func, eval_type_id)?;
        // As PySpark's: a DDL string is parsed before the columns of the input are read.
        let frame = frame_of(attr(slf.as_any(), pyo3::intern!(py, "_df"))?)?;
        let session = attr(frame.as_any(), pyo3::intern!(py, "_session"))?;
        let return_type = return_type_of(py, &session, &schema)?;
        let grouped = grouping_of(
            &frame,
            &attr(slf.as_any(), pyo3::intern!(py, "_grouping_cols"))?,
            None,
        )?;
        let cols = crate::dataframe::column_names_of(&frame, py)?;
        let udf = map_udf_of(py, &func, &schema, return_type, eval_type_id, &cols)?;
        map_result_of(&schema, &session, grouped.apply_in_pandas(udf))
    }

    // Deprecated grouped-map apply: `apply(udf)` where `udf` is a
    // `SQL_GROUPED_MAP_PANDAS_UDF` pandas UDF. As PySpark Connect's, it calls the
    // `applyInPandas` of the object, of a subclass too, with the UDF's wrapped function and
    // return type: anything else, such as a Column, a function or a UDF of another eval type,
    // raises `PySparkTypeError` `INVALID_UDF_EVAL_TYPE`, and it warns to prefer `applyInPandas`.
    #[doc = connect_doc!(GroupedData.apply)]
    fn apply<'py>(slf: &Bound<'py, Self>, udf: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
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
        // Of the line of PySpark's `warnings.warn` in `pyspark.sql.connect.group`.
        crate::errors::warn_as_pyspark(
            py,
            "pyspark.sql.connect.group",
            285,
            "It is preferred to use 'applyInPandas' over this API. This API will be deprecated in the \
             future releases. See SPARK-28264 for more details.",
            &py.get_type::<pyo3::exceptions::PyUserWarning>(),
        )?;
        let func = udf.getattr("func")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item(pyo3::intern!(py, "schema"), udf.getattr("returnType")?)?;
        slf.call_method(pyo3::intern!(py, "applyInPandas"), (func,), Some(&kwargs))
    }

    // `GroupedData.applyInArrow`, of the `_df` and the `_grouping_cols` that PySpark's
    // `GroupMap` reads.
    #[pyo3(name = "applyInArrow")]
    #[doc = connect_doc!(GroupedData.applyInArrow)]
    fn apply_in_arrow(
        slf: &Bound<'_, Self>,
        func: Bound<'_, PyAny>,
        schema: Bound<'_, PyAny>,
    ) -> PyResult<crate::dataframe::NewDataFrame> {
        let py = slf.py();
        // The reference client's warning, with its trailing space.
        let eval_type_id = infer_group_map_eval_type(
            py,
            &func,
            "infer_group_arrow_eval_type_from_func",
            "Cannot infer the eval type from type hints. ",
            495,
            eval_type::SQL_GROUPED_MAP_ARROW_UDF,
        )?;
        validate_map_function(py, &func, eval_type_id)?;
        // As PySpark's: a DDL string is parsed before the columns of the input are read.
        let frame = frame_of(attr(slf.as_any(), pyo3::intern!(py, "_df"))?)?;
        let session = attr(frame.as_any(), pyo3::intern!(py, "_session"))?;
        let return_type = return_type_of(py, &session, &schema)?;
        let grouped = grouping_of(
            &frame,
            &attr(slf.as_any(), pyo3::intern!(py, "_grouping_cols"))?,
            None,
        )?;
        let cols = crate::dataframe::column_names_of(&frame, py)?;
        let udf = map_udf_of(py, &func, &schema, return_type, eval_type_id, &cols)?;
        map_result_of(&schema, &session, grouped.apply_in_arrow(udf))
    }

    // `GroupedData.cogroup`, as PySpark's `PandasCogroupedOps(self, other)`, which checks
    // `gd1._df._check_same_session(gd2._df)` first: an object that is not grouped data raises the
    // `AttributeError` of its `_df`, or `SESSION_NOT_SAME` of a `_df` that is not a DataFrame of
    // the session, such as the Column of a DataFrame's `_df`.
    #[pyo3(name = "cogroup")]
    #[doc = connect_doc!(GroupedData.cogroup)]
    fn cogroup<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyCoGroupedData>> {
        let cogrouped = Bound::new(slf.py(), PyCoGroupedData {})?;
        PyCoGroupedData::__init__(&cogrouped, slf.as_any(), other)?;
        Ok(cogrouped)
    }

    // `GroupedData.applyInPandasWithState` (stateful streaming), as PySpark's: the output
    // schema is the return type of the function, whose DDL string is parsed once the columns of
    // the input are read, and both schemas are sent as they are given, a `StructType` as its
    // JSON, which the server parses. The plan sets the schemas, the output mode and the timeout
    // in its string fields, in this order, after it pickles the function, when it is built: a
    // value that protobuf does not take as a str (bytes of UTF-8 it does) fails each request of
    // the DataFrame, as PySpark's plan fails to build, and not this call. The plan is of the
    // `_df` and the `_grouping_cols` that PySpark's `ApplyInPandasWithState` reads.
    #[pyo3(name = "applyInPandasWithState")]
    #[pyo3(signature = (func, outputStructType, stateStructType, outputMode, timeoutConf))]
    #[doc = connect_doc!(GroupedData.applyInPandasWithState)]
    fn apply_in_pandas_with_state(
        slf: &Bound<'_, Self>,
        func: Bound<'_, PyAny>,
        outputStructType: Bound<'_, PyAny>,
        stateStructType: Bound<'_, PyAny>,
        outputMode: Bound<'_, PyAny>,
        timeoutConf: Bound<'_, PyAny>,
    ) -> PyResult<crate::dataframe::NewDataFrame> {
        let py = slf.py();
        validate_map_function(py, &func, eval_type::SQL_GROUPED_MAP_PANDAS_UDF_WITH_STATE)?;
        check_return_type_argument(&outputStructType)?;
        let output_schema = sent_schema(py, &outputStructType)?;
        let state_schema = sent_schema(py, &stateStructType)?;
        let frame = frame_of(attr(slf.as_any(), pyo3::intern!(py, "_df"))?)?;
        let grouped = grouping_of(
            &frame,
            &attr(slf.as_any(), pyo3::intern!(py, "_grouping_cols"))?,
            None,
        )?;
        let session = grouped.spark_session();
        let cols = crate::dataframe::column_names_of(&frame, py)?;
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
        let df = grouped.apply_in_pandas_with_state_with_schemas(
            udf,
            &output_schema,
            &state_schema,
            &output_mode,
            &timeout_conf,
        );
        // As PySpark's `DataFrame(plan, session=self._df._session)`.
        let session = attr(frame.as_any(), pyo3::intern!(py, "_session"))?;
        Ok(PyDataFrame::new(args.plan(df)).with_session(session.unbind()))
    }

    // `GroupedData.transformWithStateInPandas` — arbitrary stateful processing with a
    // user `StatefulProcessor` (pandas output). Mirrors the Connect API: the processor is
    // wrapped by `TransformWithStateInPandasUdfUtils` into a UDF (cloudpickled + run on the
    // server), and the `GroupMap` proto carries the transform-with-state info, as
    // [`transform_with_state_of`] builds it.
    #[pyo3(name = "transformWithStateInPandas")]
    #[pyo3(
        signature = (statefulProcessor, outputStructType, outputMode, timeMode, initialState=None, eventTimeColumnName=DefaultedOption::Omitted),
        text_signature = "($self, statefulProcessor, outputStructType, outputMode, timeMode, initialState=None, eventTimeColumnName='')"
    )]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    #[doc = connect_doc!(GroupedData.transformWithStateInPandas)]
    fn transform_with_state_in_pandas(
        slf: &Bound<'_, Self>,
        statefulProcessor: Bound<'_, PyAny>,
        outputStructType: Bound<'_, PyAny>,
        outputMode: Bound<'_, PyAny>,
        timeMode: Bound<'_, PyAny>,
        initialState: Option<Bound<'_, PyAny>>,
        eventTimeColumnName: DefaultedOption,
    ) -> PyResult<crate::dataframe::NewDataFrame> {
        transform_with_state_of(
            slf,
            &statefulProcessor,
            &outputStructType,
            [&outputMode, &timeMode],
            initialState.as_ref(),
            &eventTimeColumnName,
            true,
        )
    }

    // `GroupedData.transformWithState` (row output). Like `transformWithStateInPandas` but
    // with the row-oriented eval type, of PySpark's `TransformWithStateInPySpark` plan, which
    // sends the output schema as the pandas one does ([`transform_with_state_of`]).
    #[pyo3(name = "transformWithState")]
    #[pyo3(
        signature = (statefulProcessor, outputStructType, outputMode, timeMode, initialState=None, eventTimeColumnName=DefaultedOption::Omitted),
        text_signature = "($self, statefulProcessor, outputStructType, outputMode, timeMode, initialState=None, eventTimeColumnName='')"
    )]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    #[doc = connect_doc!(GroupedData.transformWithState)]
    fn transform_with_state(
        slf: &Bound<'_, Self>,
        statefulProcessor: Bound<'_, PyAny>,
        outputStructType: Bound<'_, PyAny>,
        outputMode: Bound<'_, PyAny>,
        timeMode: Bound<'_, PyAny>,
        initialState: Option<Bound<'_, PyAny>>,
        eventTimeColumnName: DefaultedOption,
    ) -> PyResult<crate::dataframe::NewDataFrame> {
        transform_with_state_of(
            slf,
            &statefulProcessor,
            &outputStructType,
            [&outputMode, &timeMode],
            initialState.as_ref(),
            &eventTimeColumnName,
            false,
        )
    }
}

/// The DataFrame of `transformWithStateInPandas` (`pandas`) or `transformWithState` of `this`,
/// as PySpark's `BaseTransformWithStateInPySpark` plan: its output schema is `outputStructType`,
/// a DDL string as an `UnparsedDataType`, which the server parses, or the type, and its event-time
/// column is always set, `""` when it is omitted. As PySpark's, the initial state, if any, is
/// checked with `self._df._check_same_session(initialState._df)` (see [`check_same_session`]),
/// then the output type with the `NOT_EXPECTED_TYPE` of `UserDefinedFunction`, then the `_df` and
/// the `_grouping_cols` of the initial state are read, of any object, and then the `_df`, the
/// `_grouping_cols` and the columns of the input. The plan sets its values in its string fields
/// when it is built: the output mode once it built the grouping, and the time mode and the
/// event-time column once it built the initial state, before it pickles the function. A value
/// that protobuf does not take as a str (bytes of UTF-8 it does) fails each request of the
/// DataFrame there, as PySpark's plan fails to build, and not this call: the last grouping
/// expression carries its error, or the input when there is none.
#[allow(clippy::too_many_arguments)]
fn transform_with_state_of(
    this: &Bound<'_, PyGroupedData>,
    processor: &Bound<'_, PyAny>,
    output_struct_type: &Bound<'_, PyAny>,
    [output_mode, time_mode]: [&Bound<'_, PyAny>; 2],
    initial_state: Option<&Bound<'_, PyAny>>,
    event_time_column_name: &DefaultedOption,
    pandas: bool,
) -> PyResult<crate::dataframe::NewDataFrame> {
    let py = this.py();
    let (func, et) = twus_func(py, processor, time_mode, initial_state.is_some(), pandas)?;
    if let Some(state) = initial_state {
        check_same_session(this.as_any(), state)?;
    }
    check_return_type_argument(output_struct_type)?;
    // As PySpark's `initialState._df._plan` and `initialState._grouping_cols`, once it made the
    // function.
    let initial_state = initial_state
        .map(|state| {
            let frame = frame_of(attr(state, pyo3::intern!(py, "_df"))?)?;
            grouping_of(
                &frame,
                &attr(state, pyo3::intern!(py, "_grouping_cols"))?,
                None,
            )
        })
        .transpose()?;
    let frame = frame_of(attr(this.as_any(), pyo3::intern!(py, "_df"))?)?;
    let grouped = grouping_of(
        &frame,
        &attr(this.as_any(), pyo3::intern!(py, "_grouping_cols"))?,
        initial_state
            .as_ref()
            .map(RustGroupedData::spark_session)
            .as_ref(),
    )?;
    let session = grouped.spark_session();
    let cols = crate::dataframe::column_names_of(&frame, py)?;
    let udf = build_map_udf(py, &session, &func, output_struct_type, et, &cols)?;
    // As PySpark's, which sends a DDL string as an `UnparsedDataType`, for the server to parse. A
    // struct whose fields hold an element that is not a `StructField` stands as the types that
    // stand for its elements, as the output type of the function does (see `map_udf_of`): the
    // plan sends the output type that the function's pickler converts as the request builds it,
    // of PySpark's reads of the elements and their errors, as PySpark's plan converts its output
    // schema then (see `transform_with_state_output_schema`).
    let out = if output_struct_type.is_instance_of::<pyo3::types::PyString>() {
        spark_connect::types::DataType::Unparsed {
            data_type_string: output_struct_type.extract()?,
        }
    } else {
        let (lowered, maybe_foreign) = crate::types::lower_noting_subclasses(
            output_struct_type,
            crate::types::MetadataOf::Strict,
        );
        match crate::types::foreign_of_lowering(output_struct_type, maybe_foreign)? {
            true => crate::types::lenient_type(output_struct_type)?,
            false => lowered?,
        }
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
            failing_after_grouping(&grouped, mode_args),
            Some(failing_after_grouping(&state, info_args)),
        ),
        None => {
            let mut args = mode_args;
            if let Some(error) = info_args.error() {
                args.record(error.clone());
            }
            (failing_after_grouping(&grouped, args), None)
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
    // As PySpark's `DataFrame(plan, session=self._df._session)`.
    let session = attr(frame.as_any(), pyo3::intern!(py, "_session"))?;
    Ok(PyDataFrame::new(df).with_session(session.unbind()))
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

// Python wrapper for cogrouped pandas/arrow ops, of the module of PySpark Connect's class of the
// objects of `cogroup`, `pyspark.sql.connect.group.PandasCogroupedOps`. As PySpark's plain class,
// it keeps its state in its `__dict__`, its two grouped data, `_gd1` and `_gd2`, which its methods
// read when they make their DataFrames, and it is subclassed and constructed of PySpark's
// arguments (see `crate::state`).
#[doc = connect_doc!(PandasCogroupedOps)]
#[pyclass(
    name = "PandasCogroupedOps",
    module = "pyspark.sql.connect.group",
    subclass,
    weakref,
    dict
)]
pub struct PyCoGroupedData {}

impl PyCoGroupedData {
    /// The DataFrame of a map of the cogroups of `slf` of `udf`, of the function `apply` of the
    /// cogrouped data of the core, as PySpark's `CoGroupMap` reads them: the `_df` and the
    /// `_grouping_cols` of `self._gd1`, then those of `self._gd2`. The function is of the eval
    /// type `eval_type_id` and of the return type of `schema`, a DDL string of which is parsed of
    /// the session of `self._gd1._df` first, and binds no columns, as it takes the whole
    /// cogroups, so the inputs are not analyzed.
    fn map_cogroups(
        slf: &Bound<'_, Self>,
        func: &Bound<'_, PyAny>,
        schema: &Bound<'_, PyAny>,
        eval_type_id: i32,
        apply: impl FnOnce(
            &spark_connect::group::CoGroupedData,
            spark_connect::udf::CommonInlineUserDefinedFunctionExpression,
        ) -> spark_connect::dataframe::DataFrame,
    ) -> PyResult<crate::dataframe::NewDataFrame> {
        let py = slf.py();
        validate_map_function(py, func, eval_type_id)?;
        let left = attr(slf.as_any(), pyo3::intern!(py, "_gd1"))?;
        let left_frame = frame_of(attr(&left, pyo3::intern!(py, "_df"))?)?;
        let session = attr(left_frame.as_any(), pyo3::intern!(py, "_session"))?;
        let return_type = return_type_of(py, &session, schema)?;
        let udf = map_udf_of(py, func, schema, return_type, eval_type_id, &[])?;
        let left = grouping_of(
            &left_frame,
            &attr(&left, pyo3::intern!(py, "_grouping_cols"))?,
            None,
        )?;
        let right = attr(slf.as_any(), pyo3::intern!(py, "_gd2"))?;
        let right_frame = frame_of(attr(&right, pyo3::intern!(py, "_df"))?)?;
        let right = grouping_of(
            &right_frame,
            &attr(&right, pyo3::intern!(py, "_grouping_cols"))?,
            Some(&left.spark_session()),
        )?;
        map_result_of(schema, &session, apply(&left.cogroup(&right), udf))
    }
}

#[pymethods]
impl PyCoGroupedData {
    /// Cogrouped ops of no state, as the `object.__new__` of PySpark's plain class: of any
    /// arguments, which `__init__` takes, so that a subclass of another `__init__` is built as it
    /// is in PySpark. The class shows the signature of `__init__`, as `inspect.signature` of
    /// PySpark's class does.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs), text_signature = "(gd1, gd2)")]
    fn py_new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        PyCoGroupedData {}
    }

    /// `PandasCogroupedOps(gd1, gd2)`, as PySpark Connect's `__init__`, which checks
    /// `gd1._df._check_same_session(gd2._df)` (see [`check_same_session`]), and then keeps the
    /// two grouped data as they are, as its `_gd1` and `_gd2`.
    fn __init__(
        slf: &Bound<'_, Self>,
        gd1: &Bound<'_, PyAny>,
        gd2: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let py = slf.py();
        check_same_session(gd1, gd2)?;
        crate::state::assign(
            slf.as_any(),
            &[
                (pyo3::intern!(py, "_gd1"), gd1),
                (pyo3::intern!(py, "_gd2"), gd2),
            ],
        )
    }

    /// The state of the cogrouped ops, their `__dict__`, as `object.__getstate__` gives that of
    /// PySpark's plain object, which a copy and a pickle keep (see `state::getstate`).
    fn __getstate__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::state::getstate(slf.as_any())
    }

    /// The reduce of the cogrouped ops, as PySpark's plain object's: `copyreg.__newobj__` of
    /// their class, of their `__dict__` (see `state::reduce_ex`).
    fn __reduce_ex__<'py>(slf: &Bound<'py, Self>, protocol: i32) -> PyResult<Bound<'py, PyAny>> {
        crate::state::reduce_ex(slf.as_any(), protocol)
    }

    // `PandasCogroupedOps.applyInPandas`.
    #[pyo3(name = "applyInPandas")]
    #[doc = connect_doc!(PandasCogroupedOps.applyInPandas)]
    fn apply_in_pandas(
        slf: &Bound<'_, Self>,
        func: Bound<'_, PyAny>,
        schema: Bound<'_, PyAny>,
    ) -> PyResult<crate::dataframe::NewDataFrame> {
        Self::map_cogroups(
            slf,
            &func,
            &schema,
            eval_type::SQL_COGROUPED_MAP_PANDAS_UDF,
            |cogrouped, udf| cogrouped.apply_in_pandas(udf),
        )
    }

    // `PandasCogroupedOps.applyInArrow`.
    #[pyo3(name = "applyInArrow")]
    #[doc = connect_doc!(PandasCogroupedOps.applyInArrow)]
    fn apply_in_arrow(
        slf: &Bound<'_, Self>,
        func: Bound<'_, PyAny>,
        schema: Bound<'_, PyAny>,
    ) -> PyResult<crate::dataframe::NewDataFrame> {
        Self::map_cogroups(
            slf,
            &func,
            &schema,
            eval_type::SQL_COGROUPED_MAP_ARROW_UDF,
            |cogrouped, udf| cogrouped.apply_in_arrow(udf),
        )
    }
}
