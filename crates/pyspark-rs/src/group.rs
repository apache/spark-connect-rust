//! PyO3 wrapper for spark_connect::group::GroupedData.

use pyo3::prelude::*;
use spark_connect::group::GroupedData as RustGroupedData;

use crate::dataframe::PyDataFrame;
use crate::functions::to_column;

/// Python wrapper for Spark GroupedData.
#[pyclass(name = "GroupedData")]
pub struct PyGroupedData {
    pub(crate) grouped_data: RustGroupedData,
}

impl PyGroupedData {
    pub fn new(grouped_data: RustGroupedData) -> Self {
        PyGroupedData { grouped_data }
    }
}

#[pymethods]
impl PyGroupedData {
    /// Aggregate with expressions (accepts Column objects).
    #[pyo3(signature = (*cols))]
    fn agg(&self, _py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        let mut exprs = vec![];
        for col in cols {
            let rust_col = to_column(&col)?;
            exprs.push(rust_col.expression().clone());
        }
        let df = self.grouped_data.agg(exprs);
        Ok(PyDataFrame::new(df))
    }

    /// Count rows in each group.
    fn count(&self) -> PyResult<PyDataFrame> {
        let df = self.grouped_data.count();
        Ok(PyDataFrame::new(df))
    }

    /// Sum values in each group.
    #[pyo3(signature = (*cols))]
    fn sum(&self, _py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        let col_names: Vec<String> = cols
            .iter()
            .map(|c| c.extract::<String>())
            .collect::<PyResult<Vec<_>>>()?;
        let col_refs: Vec<&str> = col_names.iter().map(|s| s.as_str()).collect();
        let df = self.grouped_data.sum(col_refs);
        Ok(PyDataFrame::new(df))
    }

    /// Average values in each group.
    #[pyo3(signature = (*cols))]
    fn avg(&self, _py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        let col_names: Vec<String> = cols
            .iter()
            .map(|c| c.extract::<String>())
            .collect::<PyResult<Vec<_>>>()?;
        let col_refs: Vec<&str> = col_names.iter().map(|s| s.as_str()).collect();
        let df = self.grouped_data.avg(col_refs);
        Ok(PyDataFrame::new(df))
    }

    /// Minimum values in each group.
    #[pyo3(signature = (*cols))]
    fn min(&self, _py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        let col_names: Vec<String> = cols
            .iter()
            .map(|c| c.extract::<String>())
            .collect::<PyResult<Vec<_>>>()?;
        let col_refs: Vec<&str> = col_names.iter().map(|s| s.as_str()).collect();
        let df = self.grouped_data.min(col_refs);
        Ok(PyDataFrame::new(df))
    }

    /// Maximum values in each group.
    #[pyo3(signature = (*cols))]
    fn max(&self, _py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        let col_names: Vec<String> = cols
            .iter()
            .map(|c| c.extract::<String>())
            .collect::<PyResult<Vec<_>>>()?;
        let col_refs: Vec<&str> = col_names.iter().map(|s| s.as_str()).collect();
        let df = self.grouped_data.max(col_refs);
        Ok(PyDataFrame::new(df))
    }

    /// Mean (alias for avg).
    #[pyo3(signature = (*cols))]
    fn mean(&self, _py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        let col_names: Vec<String> = cols
            .iter()
            .map(|c| c.extract::<String>())
            .collect::<PyResult<Vec<_>>>()?;
        let col_refs: Vec<&str> = col_names.iter().map(|s| s.as_str()).collect();
        let df = self.grouped_data.mean(col_refs);
        Ok(PyDataFrame::new(df))
    }
}
