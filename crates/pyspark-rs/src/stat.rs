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

    /// Approximate quantiles of a column at the given probabilities.
    #[pyo3(name = "approxQuantile")]
    fn approx_quantile(
        &self,
        col: &str,
        probabilities: Vec<f64>,
        relative_error: f64,
    ) -> PyDataFrame {
        PyDataFrame::new(
            self.stat
                .approx_quantile(vec![col], probabilities, relative_error),
        )
    }
}
