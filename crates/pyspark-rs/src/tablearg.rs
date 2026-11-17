//! PyO3 wrapper for spark_connect::table_arg::TableArg (`df.asTable()`).

use pyo3::prelude::*;
use pyo3::types::PyDict;
use spark_connect::table_arg::TableArg;

use crate::errors::ResultExt;
use crate::group::copy_instance_state;
use crate::window::window_columns;

/// A table argument for a table-valued function, mirroring
/// `pyspark.sql.connect.table_arg.TableArg`. Built via `df.asTable()`, then optionally
/// `partitionBy`/`orderBy`/`withSinglePartition` (each returns a new TableArg).
#[pyclass(
    name = "TableArg",
    module = "pyspark.sql.connect.table_arg",
    weakref,
    dict
)]
pub struct PyTableArg {
    pub(crate) inner: TableArg,
}

impl PyTableArg {
    pub fn new(inner: TableArg) -> Self {
        PyTableArg { inner }
    }
}

#[pymethods]
impl PyTableArg {
    /// Partition the table argument by the given columns.
    #[pyo3(signature = (*cols))]
    #[allow(non_snake_case)]
    fn partitionBy(&self, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyTableArg> {
        // PySpark checks the partitioning before it converts the columns.
        self.inner.check_partition_by().to_pyerr()?;
        let columns = window_columns(cols)?;
        Ok(PyTableArg {
            inner: self.inner.clone().partition_by(columns).to_pyerr()?,
        })
    }

    /// Order the (partitioned) table argument by the given columns.
    #[pyo3(signature = (*cols))]
    #[allow(non_snake_case)]
    fn orderBy(&self, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyTableArg> {
        // PySpark checks the partitioning before it converts the columns.
        self.inner.check_order_by().to_pyerr()?;
        let columns = window_columns(cols)?;
        Ok(PyTableArg {
            inner: self.inner.clone().order_by(columns).to_pyerr()?,
        })
    }

    /// The `spark.connect.Expression` of the table argument, as PySpark Connect's
    /// `TableArg.to_plan`: that of its subquery expression, of `pyspark.sql.connect.proto`, of
    /// its plan ID, its partitioning, its ordering and its single partition.
    fn to_plan(&self, py: Python<'_>, session: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let _ = session;
        crate::column::expression_message(py, &self.inner.to_column())
    }

    /// Mark the table argument as requiring a single partition.
    #[allow(non_snake_case)]
    fn withSinglePartition(&self) -> PyResult<PyTableArg> {
        Ok(PyTableArg {
            inner: self.inner.clone().with_single_partition().to_pyerr()?,
        })
    }

    /// The shallow copy of the table argument, as PySpark's `copy.copy`: of the same subquery,
    /// and of the same user-set attributes.
    fn __copy__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        let copy = Bound::new(slf.py(), PyTableArg::new(slf.borrow().inner.clone()))?;
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
        let inner = slf.borrow().inner.clone();
        let copy = Bound::new(py, PyTableArg::new(inner.clone()))?;
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
}
