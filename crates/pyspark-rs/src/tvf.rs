//! PyO3 wrapper for `spark_connect::tvf::TableValuedFunction` (`spark.tvf`).

use pyo3::prelude::*;
use spark_connect::tvf::TableValuedFunction;

use crate::dataframe::column_or_name;
use crate::dataframe::PyDataFrame;
use crate::errors::ResultExt;

/// Python wrapper for the table-valued-function namespace (`spark.tvf`).
#[pyclass(
    name = "TableValuedFunction",
    module = "pyspark.sql.tvf",
    weakref,
    dict
)]
pub struct PyTableValuedFunction {
    inner: TableValuedFunction,
}

impl PyTableValuedFunction {
    pub fn new(inner: TableValuedFunction) -> Self {
        PyTableValuedFunction { inner }
    }
}

/// The arguments of a table-valued function, as PySpark's `_fn` takes them: Columns or column
/// names, as `F._to_col` does, else `NOT_EXPECTED_TYPE` of `col`.
fn cols(items: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<spark_connect::column::Column>> {
    items.iter().map(column_or_name).collect()
}

#[pymethods]
impl PyTableValuedFunction {
    /// Each argument is taken by its `int()`, as PySpark's `spark.tvf.range` takes them, which
    /// is `spark.range`.
    #[pyo3(signature = (start, end=None, step=crate::session::RangeStep::Omitted, numPartitions=None))]
    #[allow(non_snake_case)]
    fn range(
        &self,
        py: Python<'_>,
        start: &Bound<'_, PyAny>,
        end: Option<&Bound<'_, PyAny>>,
        step: crate::session::RangeStep,
        numPartitions: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let (start, end, step, num_partitions) =
            crate::session::range_arguments(start, end, step, numPartitions)?;
        let df = py
            .detach(|| {
                self.inner
                    .range(start, Some(end), step, num_partitions)
                    .map_err(Box::new)
            })
            .to_pyerr()?;
        Ok(PyDataFrame::new(df))
    }

    fn explode(&self, py: Python<'_>, collection: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let c = column_or_name(collection)?;
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
        let c = column_or_name(collection)?;
        Ok(PyDataFrame::new(
            py.detach(|| self.inner.explode_outer(&c)).to_pyerr()?,
        ))
    }

    fn inline(&self, py: Python<'_>, input: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let c = column_or_name(input)?;
        Ok(PyDataFrame::new(
            py.detach(|| self.inner.inline(&c)).to_pyerr()?,
        ))
    }

    #[pyo3(name = "inline_outer")]
    fn inline_outer(&self, py: Python<'_>, input: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let c = column_or_name(input)?;
        Ok(PyDataFrame::new(
            py.detach(|| self.inner.inline_outer(&c)).to_pyerr()?,
        ))
    }

    #[pyo3(name = "json_tuple", signature = (input, *fields))]
    fn json_tuple(
        &self,
        py: Python<'_>,
        input: &Bound<'_, PyAny>,
        fields: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        // As PySpark's, which asks for fields before it takes the arguments.
        if fields.is_empty() {
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkValueError",
                "CANNOT_BE_EMPTY",
                &[("item", "field")],
            ));
        }
        let inp = column_or_name(input)?;
        let fs = cols(fields)?;
        Ok(PyDataFrame::new(
            py.detach(|| self.inner.json_tuple(&inp, fs)).to_pyerr()?,
        ))
    }

    fn posexplode(&self, py: Python<'_>, collection: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let c = column_or_name(collection)?;
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
        let c = column_or_name(collection)?;
        Ok(PyDataFrame::new(
            py.detach(|| self.inner.posexplode_outer(&c)).to_pyerr()?,
        ))
    }

    #[pyo3(signature = (n, *fields))]
    fn stack(
        &self,
        py: Python<'_>,
        n: &Bound<'_, PyAny>,
        fields: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let nc = column_or_name(n)?;
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
        let c = column_or_name(input)?;
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
        let c = column_or_name(input)?;
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
