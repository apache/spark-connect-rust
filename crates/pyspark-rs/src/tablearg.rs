//! PyO3 wrapper for spark_connect::table_arg::TableArg (`df.asTable()`).

use pyo3::prelude::*;
use spark_connect::table_arg::TableArg;

use crate::dataframe::to_column_list;
use crate::errors::ResultExt;

/// A table argument for a table-valued function, mirroring
/// `pyspark.sql.connect.table_arg.TableArg`. Built via `df.asTable()`, then optionally
/// `partitionBy`/`orderBy`/`withSinglePartition` (each returns a new TableArg).
#[pyclass(name = "TableArg")]
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
        let columns = to_column_list(cols)?;
        Ok(PyTableArg {
            inner: self.inner.clone().partition_by(columns).to_pyerr()?,
        })
    }

    /// Order the (partitioned) table argument by the given columns.
    #[pyo3(signature = (*cols))]
    #[allow(non_snake_case)]
    fn orderBy(&self, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyTableArg> {
        let columns = to_column_list(cols)?;
        Ok(PyTableArg {
            inner: self.inner.clone().order_by(columns).to_pyerr()?,
        })
    }

    /// Mark the table argument as requiring a single partition.
    #[allow(non_snake_case)]
    fn withSinglePartition(&self) -> PyResult<PyTableArg> {
        Ok(PyTableArg {
            inner: self.inner.clone().with_single_partition().to_pyerr()?,
        })
    }

    fn __repr__(&self) -> String {
        "TableArg(...)".to_string()
    }
}
