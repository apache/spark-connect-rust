//! PyO3 wrapper for spark_connect::group::StatFunctions (`df.stat`).

use pyo3::prelude::*;
use spark_connect::group::StatFunctions;

use crate::dataframe::PyDataFrame;
use crate::errors::ResultExt;

/// Python wrapper for DataFrame statistical functions.
#[pyclass(name = "DataFrameStatFunctions", module = "pyspark.sql.dataframe")]
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
    /// Pearson correlation. `method` is accepted for signature parity with reference
    /// pyspark, which currently supports only "pearson".
    #[pyo3(signature = (col1, col2, method=None))]
    fn corr(&self, py: Python<'_>, col1: &str, col2: &str, method: Option<&str>) -> PyResult<f64> {
        if let Some(m) = method {
            if m != "pearson" {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Currently only the calculation of the Pearson Correlation coefficient is supported, got {m}"
                )));
            }
        }
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
        col: &Bound<'_, PyAny>,
        fractions: &Bound<'_, PyAny>,
        seed: Option<i64>,
    ) -> PyResult<PyDataFrame> {
        let (col, fractions) = sample_by_arguments(col, fractions)?;
        Ok(PyDataFrame::new(self.stat.sample_by(col, fractions, seed)))
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
        relativeError: f64,
    ) -> PyResult<Vec<f64>> {
        use spark_connect::row::Value;
        let df = self
            .stat
            .approx_quantile(vec![col], probabilities, relativeError);
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

/// The column and the fractions of `sampleBy`, as PySpark Connect's `DataFrame.sampleBy` takes
/// them: a column name or a Column, and a dict of the fraction of each stratum, a float, an
/// int or a string.
pub(crate) fn sample_by_arguments(
    col: &Bound<'_, PyAny>,
    fractions: &Bound<'_, PyAny>,
) -> PyResult<(
    spark_connect::column::Column,
    Vec<(spark_connect::expression::Expression, f64)>,
)> {
    use crate::column::{raise_pyspark, PyColumn};
    use pyo3::types::{PyDict, PyFloat, PyInt, PyString};

    let py = col.py();
    let type_name =
        |obj: &Bound<'_, PyAny>| -> PyResult<String> { Ok(obj.get_type().name()?.to_string()) };
    let col = if let Ok(name) = col.cast::<PyString>() {
        spark_connect::functions::col(&name.to_cow()?)
    } else if let Ok(column) = col.extract::<PyRef<'_, PyColumn>>() {
        column.column.clone()
    } else {
        return Err(raise_pyspark(
            py,
            "PySparkTypeError",
            "NOT_EXPECTED_TYPE",
            &[
                ("expected_type", "Column or str"),
                ("arg_name", "col"),
                ("arg_type", &type_name(col)?),
            ],
        ));
    };
    let Ok(fractions) = fractions.cast::<PyDict>() else {
        return Err(raise_pyspark(
            py,
            "PySparkTypeError",
            "NOT_EXPECTED_TYPE",
            &[
                ("expected_type", "dict"),
                ("arg_name", "fractions"),
                ("arg_type", &type_name(fractions)?),
            ],
        ));
    };
    let mut strata = Vec::with_capacity(fractions.len());
    for (key, fraction) in fractions.iter() {
        if !(key.is_instance_of::<PyFloat>()
            || key.is_instance_of::<PyInt>()
            || key.is_instance_of::<PyString>())
        {
            return Err(raise_pyspark(
                py,
                "PySparkTypeError",
                "DISALLOWED_TYPE_FOR_CONTAINER",
                &[
                    ("arg_name", "fractions"),
                    ("arg_type", "dict"),
                    ("allowed_types", "float, int, str"),
                    ("item_type", &type_name(&key)?),
                ],
            ));
        }
        let stratum = crate::functions::to_column(&key)?.expression().clone();
        // As PySpark's float(v).
        let fraction: f64 = py.get_type::<PyFloat>().call1((fraction,))?.extract()?;
        strata.push((stratum, fraction));
    }
    Ok((col, strata))
}
