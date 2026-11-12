//! PyO3 wrapper for spark_connect::group::StatFunctions (`df.stat`).

use pyo3::prelude::*;
use spark_connect::group::StatFunctions;

use crate::dataframe::{python_random_seed, PyDataFrame};
use crate::errors::ResultExt;
use crate::session::value_out_of_range;

/// Python wrapper for DataFrame statistical functions.
#[pyclass(
    name = "DataFrameStatFunctions",
    module = "pyspark.sql.dataframe",
    weakref,
    dict
)]
pub struct PyStatFunctions {
    stat: StatFunctions,
    /// The DataFrame of `df.stat`, as PySpark's `DataFrameStatFunctions.df`.
    frame: Py<PyDataFrame>,
}

impl PyStatFunctions {
    pub fn new(stat: StatFunctions, frame: Py<PyDataFrame>) -> Self {
        PyStatFunctions { stat, frame }
    }
}

#[pymethods]
impl PyStatFunctions {
    /// The DataFrame of these functions, as PySpark's `DataFrameStatFunctions.df`: `df.stat.df`
    /// is `df`.
    #[getter(df)]
    fn frame(&self, py: Python<'_>) -> Py<PyDataFrame> {
        self.frame.clone_ref(py)
    }

    /// Pearson correlation between two numeric columns, as PySpark's, whose `method` is only
    /// "pearson".
    ///
    /// `corr`/`cov` execute a plan (server round-trip via `scalar()`), so they release
    /// the GIL across the call - otherwise a Python thread is blocked for the whole
    /// RPC.
    #[pyo3(signature = (col1, col2, method=None))]
    fn corr(
        &self,
        py: Python<'_>,
        col1: &Bound<'_, PyAny>,
        col2: &Bound<'_, PyAny>,
        method: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<f64> {
        crate::stat_arguments::corr(py, &self.stat, col1, col2, method)
    }

    /// Sample covariance between two numeric columns.
    fn cov(
        &self,
        py: Python<'_>,
        col1: &Bound<'_, PyAny>,
        col2: &Bound<'_, PyAny>,
    ) -> PyResult<f64> {
        crate::stat_arguments::cov(py, &self.stat, col1, col2)
    }

    /// Contingency table (cross-tabulation) of two columns.
    fn crosstab(&self, col1: &Bound<'_, PyAny>, col2: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        crate::stat_arguments::crosstab(&self.stat, col1, col2)
    }

    /// Frequent items for the given columns.
    #[pyo3(name = "freqItems", signature = (cols, support=None))]
    fn freq_items(
        &self,
        cols: &Bound<'_, PyAny>,
        support: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        crate::stat_arguments::freq_items(&self.stat, cols, support)
    }

    /// Stratified sample without replacement. Mirrors `DataFrameStatFunctions.sampleBy(
    /// col, fractions, seed)`; `fractions` maps stratum values to sampling fractions.
    #[pyo3(name = "sampleBy", signature = (col, fractions, seed=None))]
    fn sample_by(
        &self,
        col: &Bound<'_, PyAny>,
        fractions: &Bound<'_, PyAny>,
        seed: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let py = col.py();
        let (col, fractions) = sample_by_arguments(col, fractions)?;
        let seed = sample_by_seed(py, seed)?;
        Ok(PyDataFrame::new(
            self.stat.sample_by(col, fractions, Some(seed)).to_pyerr()?,
        ))
    }

    /// Approximate quantiles of a column, or of each of a list or tuple of columns, at the
    /// given probabilities, as PySpark's `DataFrameStatFunctions.approxQuantile`.
    #[pyo3(name = "approxQuantile")]
    fn approx_quantile(
        &self,
        py: Python<'_>,
        col: &Bound<'_, PyAny>,
        probabilities: &Bound<'_, PyAny>,
        relativeError: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        crate::stat_arguments::approx_quantile(py, &self.stat, col, probabilities, relativeError)
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
        // As PySpark's `F._to_col`, by `col`, of its origin.
        let _origin = crate::origin::with_origin("col")?;
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

/// The seed of `sampleBy`, drawn from Python's `random` when there is none. As PySpark's
/// `StatSampleBy` asserts an int, any other type is a bare `AssertionError`. A bool, which is an
/// int, is refused by the proto, and an int out of the range of a long is a `ValueError`.
pub(crate) fn sample_by_seed(py: Python<'_>, seed: Option<&Bound<'_, PyAny>>) -> PyResult<i64> {
    use pyo3::types::{PyBool, PyInt};

    let Some(seed) = seed else {
        return python_random_seed(py);
    };
    if !seed.is_instance_of::<PyInt>() {
        return Err(pyo3::exceptions::PyAssertionError::new_err(()));
    }
    if seed.is_instance_of::<PyBool>() {
        return Err(crate::stat_arguments::bool_not_proto_int(
            "spark.connect.StatSampleBy.seed",
        ));
    }
    seed.extract::<i64>().map_err(|_| value_out_of_range(seed))
}
