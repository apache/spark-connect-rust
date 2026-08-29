//! PyO3 wrapper for spark_connect::group::StatFunctions (`df.stat`).

use pyo3::prelude::*;
use spark_connect::group::StatFunctions;

use crate::dataframe::PyDataFrame;
use crate::errors::ResultExt;

/// Python wrapper for DataFrame statistical functions.
#[pyclass(name = "DataFrameStatFunctions")]
pub struct PyStatFunctions {
    stat: StatFunctions,
}

impl PyStatFunctions {
    pub fn new(stat: StatFunctions) -> Self {
        PyStatFunctions { stat }
    }
}

#[pymethods]
impl PyStatFunctions {
    /// Pearson correlation between two numeric columns.
    ///
    /// `corr`/`cov` execute a plan (server round-trip via `scalar()`), so release
    /// the GIL across the call - otherwise a Python thread is blocked for the whole
    /// RPC. (`crosstab`/`freq_items`/`approx_quantile` return lazy DataFrames and do
    /// no I/O here, so they need no detach.)
    fn corr(&self, py: Python<'_>, col1: &str, col2: &str) -> PyResult<f64> {
        py.detach(|| self.stat.corr(col1, col2)).to_pyerr()
    }

    /// Sample covariance between two numeric columns.
    fn cov(&self, py: Python<'_>, col1: &str, col2: &str) -> PyResult<f64> {
        py.detach(|| self.stat.cov(col1, col2)).to_pyerr()
    }

    /// Contingency table (cross-tabulation) of two columns.
    fn crosstab(&self, col1: &str, col2: &str) -> PyDataFrame {
        PyDataFrame::new(self.stat.crosstab(col1, col2))
    }

    /// Frequent items for the given columns.
    #[pyo3(name = "freqItems", signature = (cols, support=0.01))]
    fn freq_items(&self, cols: Vec<String>, support: f64) -> PyDataFrame {
        let refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        PyDataFrame::new(self.stat.freq_items(refs, support))
    }

    /// Stratified sample without replacement. Mirrors `DataFrameStatFunctions.sampleBy(
    /// col, fractions, seed)`; `fractions` maps stratum values to sampling fractions.
    #[pyo3(name = "sampleBy", signature = (col, fractions, seed=None))]
    fn sample_by(
        &self,
        col: &str,
        fractions: &Bound<'_, pyo3::types::PyDict>,
        seed: Option<i64>,
    ) -> PyResult<PyDataFrame> {
        let mut fr = Vec::with_capacity(fractions.len());
        for (k, v) in fractions.iter() {
            let key = crate::functions::to_column(&k)?.expression().clone();
            fr.push((key, v.extract::<f64>()?));
        }
        Ok(PyDataFrame::new(self.stat.sample_by(col, fr, seed)))
    }

    /// Approximate quantiles of a column at the given probabilities.
    ///
    /// Mirrors reference `DataFrameStatFunctions.approxQuantile(col, probabilities,
    /// relativeError)`, which returns a `list[float]` (the quantiles) - not a
    /// DataFrame. The server returns a single row whose one column is the
    /// `array<double>` of quantiles; collect it (releasing the GIL for the RPC) and
    /// return that list.
    #[pyo3(name = "approxQuantile")]
    fn approx_quantile(
        &self,
        py: Python<'_>,
        col: &str,
        probabilities: Vec<f64>,
        relative_error: f64,
    ) -> PyResult<Vec<f64>> {
        use spark_connect::row::Value;
        let df = self
            .stat
            .approx_quantile(vec![col], probabilities, relative_error);
        let rows = py.detach(|| df.collect()).to_pyerr()?;
        // The server returns a single row, col 0 = array-of-arrays (one inner array of
        // quantiles per input column); for a single column we return that inner array.
        let quantiles = match rows.first().and_then(|r| r.get(0)) {
            Some(Value::List(outer)) => match outer.first() {
                Some(Value::List(inner)) => inner.iter().filter_map(|v| v.as_f64()).collect(),
                _ => outer.iter().filter_map(|v| v.as_f64()).collect(),
            },
            _ => Vec::new(),
        };
        Ok(quantiles)
    }
}
