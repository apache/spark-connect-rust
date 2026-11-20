//! PyO3 wrapper for spark_connect::table_arg::TableArg (`df.asTable()`).

use pyo3::prelude::*;
use pyo3::types::PyDict;
use spark_connect::dataframe::DataFrame;
use spark_connect::table_arg::TableArg;

use crate::errors::ResultExt;
use crate::group::copy_instance_state;
use crate::window::window_columns;

// A table argument for a table-valued function, mirroring
// `pyspark.sql.connect.table_arg.TableArg`. Built via `df.asTable()`, then optionally
// `partitionBy`/`orderBy`/`withSinglePartition` (each returns a new TableArg).
#[doc = connect_doc!(TableArg)]
#[pyclass(
    name = "TableArg",
    module = "pyspark.sql.connect.table_arg",
    weakref,
    dict
)]
pub struct PyTableArg {
    pub(crate) inner: TableArg,
    /// The ID of the session of the DataFrame of the table argument, which its pickle keeps: it
    /// unpickles of the active session of that ID (see `__reduce__`).
    session_id: String,
    /// Set once the value is written, as `__traverse__` reads `inner`, which is not valid while
    /// the object is created (see [`crate::gc::GcReady`]).
    gc_ready: crate::gc::GcReady,
}

impl PyTableArg {
    /// The table argument `inner` of a DataFrame of the session of `session_id`.
    pub fn new(inner: TableArg, session_id: String) -> Self {
        PyTableArg {
            inner,
            session_id,
            gc_ready: crate::gc::GcReady::new(),
        }
    }

    /// The table argument `inner`, of the session of this one, as the methods of PySpark's table
    /// argument give a new one of the same plan.
    fn of_same_session(&self, inner: TableArg) -> Self {
        PyTableArg::new(inner, self.session_id.clone())
    }
}

#[pymethods]
impl PyTableArg {
    // Partition the table argument by the given columns.
    #[pyo3(signature = (*cols))]
    #[allow(non_snake_case)]
    #[doc = connect_doc!(TableArg.partitionBy)]
    fn partitionBy(&self, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyTableArg> {
        // PySpark checks the partitioning before it converts the columns.
        self.inner.check_partition_by().to_pyerr()?;
        let columns = window_columns(cols)?;
        Ok(self.of_same_session(self.inner.clone().partition_by(columns).to_pyerr()?))
    }

    // Order the (partitioned) table argument by the given columns.
    #[pyo3(signature = (*cols))]
    #[allow(non_snake_case)]
    #[doc = connect_doc!(TableArg.orderBy)]
    fn orderBy(&self, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyTableArg> {
        // PySpark checks the partitioning before it converts the columns.
        self.inner.check_order_by().to_pyerr()?;
        let columns = window_columns(cols)?;
        Ok(self.of_same_session(self.inner.clone().order_by(columns).to_pyerr()?))
    }

    // The `spark.connect.Expression` of the table argument, as PySpark Connect's
    // `TableArg.to_plan`: that of its subquery expression, of `pyspark.sql.connect.proto`, of
    // its plan ID, its partitioning, its ordering and its single partition, built of the session
    // of the client `session` (see `errors::of_client_session`).
    fn to_plan(&self, py: Python<'_>, session: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        crate::errors::of_client_session(session, || {
            crate::column::expression_message(py, &self.inner.to_column())
        })
    }

    /// The subquery expression of the table argument, as PySpark Connect's
    /// `TableArg._subquery_expr`: an expression of `pyspark.sql.connect.expressions`, as
    /// `Column._expr` is, whose `to_plan` is that of the table argument, and whose `repr` is that
    /// of PySpark's `SubqueryExpression`, of the plan ID of its plan.
    #[getter]
    fn _subquery_expr(&self) -> crate::column::PyColumnExpression {
        crate::column::PyColumnExpression::new(self.inner.to_column())
    }

    /// Whether the table argument is partitioned, by `partitionBy` or `withSinglePartition`, as
    /// PySpark Connect's `TableArg._is_partitioned`.
    fn _is_partitioned(&self) -> bool {
        self.inner.is_partitioned()
    }

    // Mark the table argument as requiring a single partition.
    #[allow(non_snake_case)]
    #[doc = connect_doc!(TableArg.withSinglePartition)]
    fn withSinglePartition(&self) -> PyResult<PyTableArg> {
        Ok(self.of_same_session(self.inner.clone().with_single_partition().to_pyerr()?))
    }

    /// The shallow copy of the table argument, as PySpark's `copy.copy`: of the same subquery,
    /// and of the same user-set attributes.
    fn __copy__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        let copy = {
            let this = slf.borrow();
            this.of_same_session(this.inner.clone())
        };
        let copy = Bound::new(slf.py(), copy)?;
        copy_instance_state(copy.as_any(), slf.as_any())?;
        Ok(copy)
    }

    /// The deep copy of the table argument, as PySpark's `copy.deepcopy`: of its subquery, whose
    /// plan PySpark copies with the function of each UDF of it, with `memo` (see the deep copy of
    /// a Column), and of its user-set attributes, each deep-copied with the `memo` of the copy,
    /// which has the copy before them, as `copy` reconstructs a plain object.
    fn __deepcopy__<'py>(
        slf: &Bound<'py, Self>,
        memo: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let (inner, copy) = {
            let this = slf.borrow();
            (this.inner.clone(), this.of_same_session(this.inner.clone()))
        };
        let copy = Bound::new(py, copy)?;
        memo.set_item(slf.as_ptr() as usize, &copy)?;
        let copies = crate::udf_command::deep_copied_picklers(py, inner.picklers(), memo)?;
        if !copies.is_empty() {
            copy.borrow_mut().inner = inner.with_pickler_copies(&copies);
        }
        let deepcopy = py.import("copy")?.getattr("deepcopy")?;
        let state = slf.getattr("__dict__")?;
        for (name, value) in state.cast::<PyDict>()?.iter() {
            let value = deepcopy.call1((value, memo))?;
            copy.getattr("__dict__")?.set_item(name, value)?;
        }
        Ok(copy)
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
        let this = slf.borrow();
        let (holes, copies) = crate::udf_command::PickleHoles::of(py, this.inner.picklers());
        let built = if copies.is_empty() {
            crate::origin::pickling(|| this.inner.relation_with_levels())
        } else {
            let marked = this.inner.with_pickler_copies(&copies);
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
            Err(error) => match crate::column::described_error_of(
                py,
                &error,
                !this.inner.is_unbuilt_with(&error),
            )? {
                Some(description) => (
                    PyBytes::new(py, b""),
                    1,
                    Some(description),
                    pyo3::types::PyList::empty(py),
                ),
                None => return Err(crate::errors::spark_error_to_py_exception(error)),
            },
        };
        let subquery = this.inner.to_column();
        let ExpressionNode::Subquery(subquery) = subquery.expression().node() else {
            unreachable!("a table argument is a subquery expression");
        };
        let columns = |expressions: &[spark_connect::expression::Expression]| {
            expressions
                .iter()
                .map(|expression| {
                    Py::new(
                        py,
                        crate::column::PyColumn::new(spark_connect::column::Column::new(
                            expression.clone(),
                        )),
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
            this.session_id.clone(),
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
        crate::group::reduce_with_state(&reduce, slf.as_any())
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
    fn _of_relation(
        session: &Bound<'_, crate::session::PySparkSession>,
        relation: &[u8],
        levels: usize,
        build_error: Option<&Bound<'_, PyAny>>,
        holes: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyTableArg> {
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
        Ok(PyTableArg::new(TableArg::new(dataframe), session_id))
    }

    /// As PySpark's table argument, a Python object that holds the plan of its DataFrame and its
    /// partitioning and ordering columns, which Python's garbage collector sees: the functions of
    /// the UDFs that the table argument alone keeps alive are reported, those of the plan once its
    /// DataFrame is gone, so that a reference cycle through them is collected, as PySpark collects
    /// it.
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        if !self.gc_ready.get() {
            return Ok(());
        }
        crate::udf_command::visit_pickler_objects(self.inner.exclusive_pickler_objects(), &visit)?;
        // The plans of the Python client of the relations of its plan, which it alone holds, as
        // PySpark's table argument holds the plan objects of its DataFrame.
        crate::udf_command::visit_owners(self.inner.exclusive_owners(), &visit)
    }

    /// Break a reference cycle that the garbage collector found unreachable: the table argument
    /// drops its plan and its expressions, and so the objects that `__traverse__` reported.
    fn __clear__(&mut self) {
        if self.gc_ready.get() {
            self.inner.clear_plan();
        }
    }
}
