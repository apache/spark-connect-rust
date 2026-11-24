//! PyO3 wrapper for spark_connect::table_arg::TableArg (`df.asTable()`).

use pyo3::prelude::*;
use pyo3::types::PyDict;
use spark_connect::dataframe::DataFrame;
use spark_connect::table_arg::TableArg;

use crate::errors::ResultExt;
use crate::window::window_columns;

// A table argument for a table-valued function, mirroring
// `pyspark.sql.connect.table_arg.TableArg`. Built via `df.asTable()`, then optionally
// `partitionBy`/`orderBy`/`withSinglePartition` (each returns a new TableArg). As PySpark's plain
// class, it keeps its subquery expression in its `__dict__`, its `_subquery_expr`, which its
// methods read, and it is subclassed and constructed of PySpark's `TableArg(subquery_expr)`.
#[doc = connect_doc!(TableArg)]
#[pyclass(
    name = "TableArg",
    module = "pyspark.sql.connect.table_arg",
    subclass,
    weakref,
    dict
)]
pub struct PyTableArg {
    /// The ID of the session of the DataFrame of the table argument, which its pickle keeps: it
    /// unpickles of the active session of that ID (see `__reduce__`). None of a table argument of
    /// PySpark's `TableArg(subquery_expr)`, which pickles of the ID of the active session.
    session_id: Option<String>,
}

/// A new table argument of the class, of `inner`, a table argument of a DataFrame of the session
/// of `session_id`: of the `_ColumnExpression` of its subquery as its `_subquery_expr`, as
/// PySpark's methods give one of `TableArg(subquery_expr)`.
pub(crate) fn table_arg_object(
    py: Python<'_>,
    inner: TableArg,
    session_id: Option<String>,
) -> PyResult<Bound<'_, PyTableArg>> {
    let expression = crate::column::expression_object(py, inner.to_column())?;
    let table_arg = Bound::new(py, PyTableArg { session_id })?;
    crate::state::assign(
        table_arg.as_any(),
        &[(pyo3::intern!(py, "_subquery_expr"), expression.as_any())],
    )?;
    Ok(table_arg)
}

impl PyTableArg {
    /// The table argument of the core of the `_subquery_expr` of `slf`, as PySpark's methods read
    /// `self._subquery_expr`: of the instance dict, with its errors. Of an object that is not the
    /// subquery of a table argument of this client, the `AttributeError`s of the fields of
    /// PySpark's `SubqueryExpression` that the methods read first (`_partition_spec` of
    /// `_is_partitioned`), and else a `TypeError` of the object, whose plan this client does not
    /// have.
    fn inner_of(slf: &Bound<'_, Self>) -> PyResult<TableArg> {
        let py = slf.py();
        let expression = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_subquery_expr"))?;
        if let Ok(expression) = expression.cast::<crate::column::PyColumnExpression>() {
            if let Some(inner) = TableArg::of_column(expression.borrow().column()) {
                return Ok(inner);
            }
        }
        expression.getattr(pyo3::intern!(py, "_partition_spec"))?;
        expression.getattr(pyo3::intern!(py, "_with_single_partition"))?;
        let type_name = expression.get_type().name()?.to_string();
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "'{type_name}' object is not the subquery expression of a table argument of this \
             client, whose plan the drop-in builds"
        )))
    }

    /// A table argument of `inner`, of the session of `slf`, as the methods of PySpark's table
    /// argument give a new one of the same plan.
    fn of_same_session<'py>(slf: &Bound<'py, Self>, inner: TableArg) -> PyResult<Bound<'py, Self>> {
        table_arg_object(slf.py(), inner, slf.borrow().session_id.clone())
    }

    /// The ID of the session of the pickle of `slf`: of the DataFrame of the table argument, or of
    /// the active session, of a table argument of PySpark's `TableArg(subquery_expr)`.
    fn pickled_session_id(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        if let Some(session_id) = slf.borrow().session_id.clone() {
            return Ok(session_id);
        }
        match crate::session::active_session(py) {
            Some(session) => Ok(session.bind(py).borrow().session.session_id().to_string()),
            None => Err(crate::errors::raise_pyspark(
                py,
                "PySparkRuntimeError",
                "NO_ACTIVE_SESSION",
                &[],
            )),
        }
    }
}

#[pymethods]
impl PyTableArg {
    /// A table argument of no subquery yet, as the `object.__new__` of PySpark's plain class: of
    /// any arguments, which `__init__` takes, so that a subclass of another `__init__` is built as
    /// it is in PySpark. The class shows the signature of `__init__`, as `inspect.signature` of
    /// PySpark's class does.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs), text_signature = "(subquery_expr)")]
    fn py_new(
        _args: &Bound<'_, pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> Self {
        PyTableArg { session_id: None }
    }

    /// `TableArg(subquery_expr)`, as PySpark Connect's `__init__`, which keeps its argument as it
    /// is, as its `_subquery_expr`, which its methods read.
    fn __init__(slf: &Bound<'_, Self>, subquery_expr: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        crate::state::assign(
            slf.as_any(),
            &[(pyo3::intern!(py, "_subquery_expr"), subquery_expr)],
        )
    }

    // Partition the table argument by the given columns.
    #[pyo3(signature = (*cols))]
    #[allow(non_snake_case)]
    #[doc = connect_doc!(TableArg.partitionBy)]
    fn partitionBy<'py>(
        slf: &Bound<'py, Self>,
        cols: Vec<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        let inner = Self::inner_of(slf)?;
        // PySpark checks the partitioning before it converts the columns.
        inner.check_partition_by().to_pyerr()?;
        let columns = window_columns(cols)?;
        Self::of_same_session(slf, inner.partition_by(columns).to_pyerr()?)
    }

    // Order the (partitioned) table argument by the given columns.
    #[pyo3(signature = (*cols))]
    #[allow(non_snake_case)]
    #[doc = connect_doc!(TableArg.orderBy)]
    fn orderBy<'py>(
        slf: &Bound<'py, Self>,
        cols: Vec<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        let inner = Self::inner_of(slf)?;
        // PySpark checks the partitioning before it converts the columns, as
        // `[F._sort_col(c)._expr for c in _to_cols(cols)]`, of the `asc()` of a Column of a
        // subclass too, as the window's `orderBy`.
        inner.check_order_by().to_pyerr()?;
        let columns = crate::window::sort_expressions_of(slf.py(), cols)?
            .iter()
            .map(crate::column::expression_value)
            .collect::<PyResult<Vec<_>>>()?;
        Self::of_same_session(slf, inner.order_by(columns).to_pyerr()?)
    }

    // The `spark.connect.Expression` of the table argument, as PySpark Connect's
    // `TableArg.to_plan`: that of its subquery expression, `self._subquery_expr.to_plan(session)`.
    fn to_plan<'py>(
        slf: &Bound<'py, Self>,
        session: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        crate::state::attr(slf.as_any(), pyo3::intern!(py, "_subquery_expr"))?
            .call_method1(pyo3::intern!(py, "to_plan"), (session,))
    }

    /// Whether the table argument is partitioned, by `partitionBy` or `withSinglePartition`, as
    /// PySpark Connect's `TableArg._is_partitioned`.
    fn _is_partitioned(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(Self::inner_of(slf)?.is_partitioned())
    }

    // Mark the table argument as requiring a single partition.
    #[allow(non_snake_case)]
    #[doc = connect_doc!(TableArg.withSinglePartition)]
    fn withSinglePartition<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        let inner = Self::inner_of(slf)?;
        Self::of_same_session(slf, inner.with_single_partition().to_pyerr()?)
    }

    /// The shallow copy of the table argument, as PySpark's `copy.copy` of a plain object: an
    /// object of its class of the same entries of its `__dict__`, so the copy has the subquery
    /// expression of the table argument, and of the same session.
    fn __copy__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let copy = copy_of_class(slf)?;
        let state = crate::group::instance_state(slf.as_any())?;
        if !state.is_none() {
            copy.getattr(pyo3::intern!(py, "__dict__"))?
                .call_method1(pyo3::intern!(py, "update"), (state,))?;
        }
        Ok(copy)
    }

    /// The deep copy of the table argument, as PySpark's `copy.deepcopy` of a plain object: an
    /// object of its class, of the same session, and then of the deep copies of the entries of its
    /// `__dict__`, with the `memo` of the copy, which has the copy before them, so its
    /// `_subquery_expr` is the copy of the subquery expression, whose plan PySpark copies with the
    /// function of each UDF of it (see `PyColumnExpression::__deepcopy__`).
    fn __deepcopy__<'py>(
        slf: &Bound<'py, Self>,
        memo: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let copy = copy_of_class(slf)?;
        memo.set_item(slf.as_ptr() as usize, &copy)?;
        let state = crate::group::instance_state(slf.as_any())?;
        if !state.is_none() {
            let state = py.import("copy")?.call_method1("deepcopy", (state, memo))?;
            copy.getattr(pyo3::intern!(py, "__dict__"))?
                .call_method1(pyo3::intern!(py, "update"), (state,))?;
        }
        Ok(copy)
    }

    /// The reduce of a pickle, as PySpark's pickle of a table argument, which pickles its subquery
    /// expression, its plan first, and so the objects of `pyspark.ml` that the plan keeps (of a
    /// transform of a model, a DataFrame of a method of a model or of a summary), as the pickle of
    /// a Column of a subquery fails (see `PyColumn::__reduce_ex__`). Else that of `__reduce__`.
    fn __reduce_ex__<'py>(
        slf: &Bound<'py, Self>,
        protocol: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = protocol;
        let py = slf.py();
        let column = Self::inner_of(slf)?.to_column();
        let reach = crate::ml::object::expression_pickle_reach(py, column.expression())?;
        let reduced = slf.call_method0(pyo3::intern!(py, "__reduce__"))?;
        crate::ml::object::reach_reduce(py, &reach, reduced)
    }

    /// A table argument pickles, as PySpark's, which pickles the objects of its subquery
    /// expression, its plan, its partitioning, its ordering and its single partition: of the ID of
    /// its session and the relation of its plan, as a DataFrame pickles them, of the Columns of
    /// its partitioning and its ordering and of its single partition, of which
    /// `pyspark.sql.connect._column_reduce.table_arg_of` gives back the table argument, as
    /// `_of_relation` and the calls that made it, with no request, as PySpark's unpickle sends
    /// none. The relation is of holes in place of the commands of its UDFs, and the pickle
    /// carries their functions, which the requests of the unpickled table argument pickle, as
    /// PySpark's pickle carries the functions of its plan ([`crate::udf_command::PickleHoles`]).
    /// A plan that does not build pickles of its error, which the requests of the unpickled table
    /// argument raise ([`crate::column::described_error`]), as PySpark's raise it when they build
    /// the plan; of a plan of the Python client whose build raised, of that plan, which the
    /// unpickle builds again, as PySpark's table argument builds the plan objects that it carries. It keeps the ID of its session rather than the session, which pickles of a local
    /// function, as PySpark's table argument holds no session: so a plain pickle of it succeeds
    /// but for a function that it does not pickle. A user-set attribute is the state, as of a
    /// plain object.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        use pyo3::types::PyBytes;
        use spark_connect::expression::ExpressionNode;
        let py = slf.py();
        let inner = Self::inner_of(slf)?;
        let session_id = Self::pickled_session_id(slf)?;
        let (holes, copies) = crate::udf_command::PickleHoles::of(py, inner.picklers());
        let built = if copies.is_empty() {
            crate::origin::pickling(|| inner.relation_with_levels())
        } else {
            let marked = inner.with_pickler_copies(&copies);
            crate::origin::pickling(|| marked.relation_with_levels())
        };
        let (relation, levels, build_error, holes) = match built {
            Ok((relation, levels)) => {
                let relation = crate::encode_on_grown_stack(relation);
                let holes = holes.holes_of(py, &relation)?;
                (PyBytes::new(py, &relation), levels, None, holes)
            }
            // Of the table argument of the DataFrame of the plan of a kept error itself, or of
            // one over it.
            Err(error) => {
                match crate::column::described_error_of(py, &error, !inner.is_unbuilt_with(&error))?
                {
                    Some(description) => (
                        PyBytes::new(py, b""),
                        1,
                        Some(description),
                        pyo3::types::PyList::empty(py),
                    ),
                    None => return Err(crate::errors::spark_error_to_py_exception(error)),
                }
            }
        };
        let subquery = inner.to_column();
        let ExpressionNode::Subquery(subquery) = subquery.expression().node() else {
            unreachable!("a table argument is a subquery expression");
        };
        let columns = |expressions: &[spark_connect::expression::Expression]| {
            expressions
                .iter()
                .map(|expression| {
                    crate::column::column_object(
                        py,
                        spark_connect::column::Column::new(expression.clone()),
                    )
                })
                .collect::<PyResult<Vec<_>>>()
        };
        let reduce_module = py
            .import("pyspark.sql.connect.dataframe")?
            .getattr("_column_reduce")?;
        // The function that fills the holes on a Python worker of the server, of holes only, so
        // that only the pickle of a table argument of holes carries its code.
        let fill = if holes.is_empty() {
            None
        } else {
            Some(reduce_module.getattr("fill_holes")?)
        };
        let args = (
            session_id,
            relation,
            levels,
            columns(&subquery.partition_spec)?,
            columns(&subquery.order_spec)?,
            subquery.with_single_partition == Some(true),
            build_error,
            holes,
            fill,
        )
            .into_pyobject(py)?;
        let table_arg_of = reduce_module.getattr("table_arg_of")?;
        let reduce = pyo3::types::PyTuple::new(py, [table_arg_of, args.into_any()])?.into_any();
        let reduce =
            crate::column::subclass_reduce(slf.as_any(), &py.get_type::<PyTableArg>(), reduce)?;
        // The state of the other attributes, as the subquery expression is the reduce's.
        crate::column::reduce_with_state_but(
            &reduce,
            slf.as_any(),
            pyo3::intern!(py, "_subquery_expr"),
        )
    }

    /// Give `target`, a table argument of a subclass that an unpickle makes, the session of
    /// `source`, the table argument that the pickle gave back, which it keeps natively, as the
    /// pickle of an object of a subclass unpickles (`_column_reduce.instance_of`).
    #[staticmethod]
    fn _take_session_of(target: &Bound<'_, Self>, source: &Bound<'_, Self>) {
        let session_id = source.borrow().session_id.clone();
        target.borrow_mut().session_id = session_id;
    }

    /// The table argument of `relation`, the bytes of the relation of `levels` levels of the plan
    /// of a DataFrame of `session`, as a pickled table argument unpickles
    /// (`_column_reduce.table_arg_of`): the `asTable()` of the DataFrame of the relation, of its
    /// plan ID, as `DataFrame._of_relation` gives it, with no request, of `holes`, whose
    /// functions each request pickles (see `udf_command::sent_holes`). Of `build_error`, the
    /// description of the error of a plan that did not build, each request of a plan of it
    /// raises the error.
    #[staticmethod]
    #[pyo3(signature = (session, relation, levels=1, build_error=None, holes=None))]
    fn _of_relation<'py>(
        session: &Bound<'py, crate::session::PySparkSession>,
        relation: &[u8],
        levels: usize,
        build_error: Option<&Bound<'py, PyAny>>,
        holes: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyTableArg>> {
        let holes = crate::udf_command::sent_holes(holes)?;
        let python_session = session;
        let session = python_session.borrow().session.clone();
        let session_id = session.session_id().to_string();
        let dataframe = match build_error {
            // The plan of the Python client whose build failed, built again, as PySpark's table
            // argument builds the plan objects that it carries.
            Some(description) => match crate::column::described_plan(description)? {
                Some((plan, derived)) => {
                    crate::dataframe::rebuilt_python_plan(&plan, python_session, derived)?.0
                }
                None => {
                    let error = crate::column::error_of_description(description)?;
                    DataFrame::from_sent_relation(session, Default::default()).unbuildable(error)
                }
            },
            None => {
                use prost::Message;
                // Decoded, and then built or dropped, on a stack that holds its levels.
                crate::on_grown_stack(|| {
                    spark_connect_proto::Relation::decode(relation).map(|relation| {
                        DataFrame::from_sent_relation_with_holes(session, relation, levels, holes)
                    })
                })
                .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?
                .to_pyerr()?
            }
        };
        table_arg_object(
            python_session.py(),
            TableArg::new(dataframe),
            Some(session_id),
        )
    }
}

/// A table argument of the class of `slf` and of its session, of no subquery yet, as the
/// `copyreg.__newobj__` of a copy of PySpark's plain object makes one, of `__new__` of its class.
fn copy_of_class<'py>(slf: &Bound<'py, PyTableArg>) -> PyResult<Bound<'py, PyAny>> {
    let py = slf.py();
    let class = slf.get_type();
    let copy = class.call_method1(pyo3::intern!(py, "__new__"), (&class,))?;
    if let Ok(copy) = copy.cast::<PyTableArg>() {
        copy.borrow_mut().session_id = slf.borrow().session_id.clone();
    }
    Ok(copy)
}
