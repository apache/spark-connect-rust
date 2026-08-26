//! PyO3 wrapper for `spark_connect::tvf::TableValuedFunction` (`spark.tvf`).

use pyo3::prelude::*;
use spark_connect::tvf::TableValuedFunction;

use crate::dataframe::PyDataFrame;
use crate::errors::ResultExt;
use crate::functions::to_column;

/// Python wrapper for the table-valued-function namespace (`spark.tvf`).
#[pyclass(name = "TableValuedFunction")]
pub struct PyTableValuedFunction {
    inner: TableValuedFunction,
}

impl PyTableValuedFunction {
    pub fn new(inner: TableValuedFunction) -> Self {
        PyTableValuedFunction { inner }
    }
}

fn cols(items: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<spark_connect::column::Column>> {
    items.iter().map(to_column).collect()
}

#[pymethods]
impl PyTableValuedFunction {
    #[pyo3(signature = (start, end=None, step=1, numPartitions=None))]
    #[allow(non_snake_case)]
    fn range(
        &self,
        py: Python<'_>,
        start: i64,
        end: Option<i64>,
        step: i64,
        numPartitions: Option<i32>,
    ) -> PyResult<PyDataFrame> {
        let df = py
            .detach(|| self.inner.range(start, end, step, numPartitions))
            .to_pyerr()?;
        Ok(PyDataFrame::new(df))
    }

    fn explode(&self, py: Python<'_>, collection: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let c = to_column(collection)?;
        Ok(PyDataFrame::new(
            py.detach(|| self.inner.explode(&c)).to_pyerr()?,
        ))
    }

    #[pyo3(name = "explode_outer")]
    fn explode_outer(
        &self,
        py: Python<'_>,
        collection: &Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        let c = to_column(collection)?;
        Ok(PyDataFrame::new(
            py.detach(|| self.inner.explode_outer(&c)).to_pyerr()?,
        ))
    }

    fn inline(&self, py: Python<'_>, input: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let c = to_column(input)?;
        Ok(PyDataFrame::new(
            py.detach(|| self.inner.inline(&c)).to_pyerr()?,
        ))
    }

    #[pyo3(name = "inline_outer")]
    fn inline_outer(&self, py: Python<'_>, input: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let c = to_column(input)?;
        Ok(PyDataFrame::new(
            py.detach(|| self.inner.inline_outer(&c)).to_pyerr()?,
        ))
    }

    #[pyo3(name = "json_tuple")]
    fn json_tuple(
        &self,
        py: Python<'_>,
        input: &Bound<'_, PyAny>,
        fields: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let inp = to_column(input)?;
        let fs = cols(fields)?;
        Ok(PyDataFrame::new(
            py.detach(|| self.inner.json_tuple(&inp, fs)).to_pyerr()?,
        ))
    }

    fn posexplode(&self, py: Python<'_>, collection: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let c = to_column(collection)?;
        Ok(PyDataFrame::new(
            py.detach(|| self.inner.posexplode(&c)).to_pyerr()?,
        ))
    }

    #[pyo3(name = "posexplode_outer")]
    fn posexplode_outer(
        &self,
        py: Python<'_>,
        collection: &Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        let c = to_column(collection)?;
        Ok(PyDataFrame::new(
            py.detach(|| self.inner.posexplode_outer(&c)).to_pyerr()?,
        ))
    }

    fn stack(
        &self,
        py: Python<'_>,
        n: &Bound<'_, PyAny>,
        fields: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let nc = to_column(n)?;
        let fs = cols(fields)?;
        Ok(PyDataFrame::new(
            py.detach(|| self.inner.stack(&nc, fs)).to_pyerr()?,
        ))
    }

    fn collations(&self, py: Python<'_>) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(
            py.detach(|| self.inner.collations()).to_pyerr()?,
        ))
    }

    #[pyo3(name = "sql_keywords")]
    fn sql_keywords(&self, py: Python<'_>) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(
            py.detach(|| self.inner.sql_keywords()).to_pyerr()?,
        ))
    }

    #[pyo3(name = "variant_explode")]
    fn variant_explode(&self, py: Python<'_>, input: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let c = to_column(input)?;
        Ok(PyDataFrame::new(
            py.detach(|| self.inner.variant_explode(&c)).to_pyerr()?,
        ))
    }

    #[pyo3(name = "variant_explode_outer")]
    fn variant_explode_outer(
        &self,
        py: Python<'_>,
        input: &Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        let c = to_column(input)?;
        Ok(PyDataFrame::new(
            py.detach(|| self.inner.variant_explode_outer(&c))
                .to_pyerr()?,
        ))
    }

    #[pyo3(name = "python_worker_logs")]
    fn python_worker_logs(&self, py: Python<'_>) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(
            py.detach(|| self.inner.python_worker_logs()).to_pyerr()?,
        ))
    }
}
