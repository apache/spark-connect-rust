//! PyO3 wrapper of PySpark Connect's `DataFrameStatFunctions` (`df.stat`).

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::dataframe::python_random_seed;

// Python wrapper for DataFrame statistical functions. As PySpark's plain class, it keeps its state
// in its `__dict__`, its `df`, whose methods of the same names its methods call, as PySpark's do,
// and it is subclassed and constructed of PySpark's arguments (see `crate::state`).
#[doc = connect_doc!(DataFrameStatFunctions)]
#[pyclass(
    name = "DataFrameStatFunctions",
    module = "pyspark.sql.connect.dataframe",
    subclass,
    weakref,
    dict
)]
pub struct PyStatFunctions {}

impl PyStatFunctions {
    /// The statistics functions of `df`, as `df.stat` makes them: `DataFrameStatFunctions(df)`,
    /// of the DataFrame object, which PySpark's `__init__` keeps as its `df`.
    pub(crate) fn of_frame<'py>(df: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyStatFunctions>> {
        let stat = Bound::new(df.py(), PyStatFunctions {})?;
        Self::__init__(&stat, df)?;
        Ok(stat)
    }

    /// The `df` of the functions, which each of PySpark's methods reads, `self.df`, to call its
    /// method of the same name: of the `AttributeError` of functions whose `__init__` never ran.
    fn frame<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::state::attr(slf.as_any(), pyo3::intern!(slf.py(), "df"))
    }
}

#[pymethods]
impl PyStatFunctions {
    /// Statistics functions of no state, as the `object.__new__` of PySpark's plain class: of
    /// any arguments, which `__init__` takes, so that a subclass of another `__init__` is built as
    /// it is in PySpark. The class shows the signature of `__init__`, as `inspect.signature` of
    /// PySpark's class does.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs), text_signature = "(df)")]
    fn py_new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        PyStatFunctions {}
    }

    /// `DataFrameStatFunctions(df)`, as PySpark Connect's `__init__`, which keeps its argument as
    /// it is, as the `df` of the functions: `df.stat.df` is `df`.
    fn __init__(slf: &Bound<'_, Self>, df: &Bound<'_, PyAny>) -> PyResult<()> {
        crate::state::assign(slf.as_any(), &[(pyo3::intern!(slf.py(), "df"), df)])
    }

    /// The state of the functions, their `__dict__`, as `object.__getstate__` gives that of
    /// PySpark's plain object, which a copy and a pickle keep (see `state::getstate`).
    fn __getstate__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::state::getstate(slf.as_any())
    }

    /// The reduce of the functions, as PySpark's plain object's: `copyreg.__newobj__` of their
    /// class, of their `__dict__`, so a copy is of the same DataFrame, and a deep copy and a pickle
    /// of a copy of it (see `state::reduce_ex`).
    fn __reduce_ex__<'py>(slf: &Bound<'py, Self>, protocol: i32) -> PyResult<Bound<'py, PyAny>> {
        crate::state::reduce_ex(slf.as_any(), protocol)
    }

    // Pearson correlation between two numeric columns, as PySpark's
    // `self.df.corr(col1, col2, method)`, a call of the method of the DataFrame, of a subclass or
    // of any object too, whose `method` is only "pearson".
    #[pyo3(signature = (col1, col2, method=None))]
    #[doc = connect_doc!(DataFrameStatFunctions.corr)]
    fn corr<'py>(
        slf: &Bound<'py, Self>,
        col1: &Bound<'py, PyAny>,
        col2: &Bound<'py, PyAny>,
        method: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::frame(slf)?.call_method1(pyo3::intern!(slf.py(), "corr"), (col1, col2, method))
    }

    // Sample covariance between two numeric columns, as PySpark's `self.df.cov(col1, col2)`.
    #[doc = connect_doc!(DataFrameStatFunctions.cov)]
    fn cov<'py>(
        slf: &Bound<'py, Self>,
        col1: &Bound<'py, PyAny>,
        col2: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::frame(slf)?.call_method1(pyo3::intern!(slf.py(), "cov"), (col1, col2))
    }

    // Contingency table (cross-tabulation) of two columns, as PySpark's
    // `self.df.crosstab(col1, col2)`.
    #[doc = connect_doc!(DataFrameStatFunctions.crosstab)]
    fn crosstab<'py>(
        slf: &Bound<'py, Self>,
        col1: &Bound<'py, PyAny>,
        col2: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::frame(slf)?.call_method1(pyo3::intern!(slf.py(), "crosstab"), (col1, col2))
    }

    // Frequent items for the given columns, as PySpark's `self.df.freqItems(cols, support)`.
    #[pyo3(name = "freqItems", signature = (cols, support=None))]
    #[doc = connect_doc!(DataFrameStatFunctions.freqItems)]
    fn freq_items<'py>(
        slf: &Bound<'py, Self>,
        cols: &Bound<'py, PyAny>,
        support: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::frame(slf)?.call_method1(pyo3::intern!(slf.py(), "freqItems"), (cols, support))
    }

    // Stratified sample without replacement, as PySpark's
    // `self.df.sampleBy(col, fractions, seed)`; `fractions` maps stratum values to sampling
    // fractions.
    #[pyo3(name = "sampleBy", signature = (col, fractions, seed=None))]
    #[doc = connect_doc!(DataFrameStatFunctions.sampleBy)]
    fn sample_by<'py>(
        slf: &Bound<'py, Self>,
        col: &Bound<'py, PyAny>,
        fractions: &Bound<'py, PyAny>,
        seed: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::frame(slf)?.call_method1(pyo3::intern!(slf.py(), "sampleBy"), (col, fractions, seed))
    }

    // Approximate quantiles of a column, or of each of a list or tuple of columns, at the
    // given probabilities, as PySpark's
    // `self.df.approxQuantile(col, probabilities, relativeError)`.
    #[pyo3(name = "approxQuantile")]
    #[doc = connect_doc!(DataFrameStatFunctions.approxQuantile)]
    fn approx_quantile<'py>(
        slf: &Bound<'py, Self>,
        col: &Bound<'py, PyAny>,
        probabilities: &Bound<'py, PyAny>,
        relativeError: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::frame(slf)?.call_method1(
            pyo3::intern!(slf.py(), "approxQuantile"),
            (col, probabilities, relativeError),
        )
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
    let col = if col.is_instance_of::<PyString>() {
        // As PySpark's `F._to_col`, by `col`, of its origin.
        let _origin = crate::origin::with_origin("col")?;
        crate::plan_args::col(col)?
    } else if let Ok(column) = col.cast::<PyColumn>() {
        crate::column::column_of(column)?
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
/// `StatSampleBy` asserts an int, any other type is a bare `AssertionError`. The int is then set
/// in the long field of the proto, which refuses a bool, which is an int, and an int out of the
/// range of a long, as [`crate::stat_arguments::proto_int`] converts it.
pub(crate) fn sample_by_seed(py: Python<'_>, seed: Option<&Bound<'_, PyAny>>) -> PyResult<i64> {
    let Some(seed) = seed else {
        return python_random_seed(py);
    };
    if !seed.is_instance_of::<pyo3::types::PyInt>() {
        return Err(pyo3::exceptions::PyAssertionError::new_err(()));
    }
    crate::stat_arguments::proto_int(seed, "spark.connect.StatSampleBy.seed")
}
