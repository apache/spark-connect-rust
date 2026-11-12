//! PyO3 wrapper for `spark_connect::observation::Observation` (`pyspark.sql.Observation`).

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use spark_connect::dataframe::DataFrame;
use spark_connect::expression::Expression;
use spark_connect::observation::Observation;
use spark_connect::SparkSession;
use spark_connect_core::error::{SparkError, SparkErrorKind};

use crate::column::{raise_pyspark, PyColumn};
use crate::dataframe::PyDataFrame;
use crate::errors::ResultExt;

/// Python wrapper for an Observation of metrics, as PySpark Connect's `Observation`.
#[pyclass(
    name = "Observation",
    module = "pyspark.sql.observation",
    weakref,
    dict
)]
pub struct PyObservation {
    pub(crate) inner: Observation,
}

#[pymethods]
impl PyObservation {
    /// `Observation(name=None)`, as PySpark's: the name is a non-empty string, and an
    /// observation without one is of a random UUID once it is used.
    #[new]
    #[pyo3(signature = (name=None))]
    fn new(py: Python<'_>, name: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let inner = match name {
            None => Observation::unnamed(),
            Some(name) if !name.is_instance_of::<PyString>() => {
                let type_name = name.get_type().name()?.to_string();
                return Err(raise_pyspark(
                    py,
                    "PySparkTypeError",
                    "NOT_EXPECTED_TYPE",
                    &[
                        ("arg_name", "name"),
                        ("expected_type", "str"),
                        ("arg_type", &type_name),
                    ],
                ));
            }
            Some(name) => {
                let name: String = name.extract()?;
                if name.is_empty() {
                    return Err(raise_pyspark(
                        py,
                        "PySparkValueError",
                        "VALUE_NOT_NON_EMPTY_STR",
                        &[("arg_name", "name"), ("arg_value", &name)],
                    ));
                }
                Observation::new(&name)
            }
        };
        Ok(PyObservation { inner })
    }

    /// The name of the observation, as PySpark's `_name`: None for an unnamed observation
    /// before it is used.
    #[getter(_name)]
    fn name(&self) -> Option<String> {
        self.inner.name()
    }

    /// The observed metrics, as PySpark's `Observation.get`: a dict of the metrics that the
    /// actions run so far observed, by their names, and empty before an action runs. The metrics
    /// of an execution that PySpark's client does not convert (see [`check_observed_metrics`])
    /// are never set, so these are those of the executions before it, as PySpark's.
    #[getter]
    fn get<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let metrics = self.inner.get().to_pyerr()?;
        let result = PyDict::new(py);
        for (name, value) in metrics {
            result.set_item(name, crate::dataframe::literal_to_py(py, &value, None)?)?;
        }
        Ok(result)
    }

    /// Observe the metrics of the Columns `exprs` on `df` with this observation, as PySpark's
    /// `Observation._on`, which `DataFrame.observe` of an Observation calls.
    #[pyo3(signature = (df, *exprs))]
    fn _on(
        slf: &Bound<'_, Self>,
        df: PyRef<'_, PyDataFrame>,
        exprs: Vec<PyRef<'_, PyColumn>>,
    ) -> PyResult<PyDataFrame> {
        let exprs: Vec<_> = exprs
            .iter()
            .map(|c| c.column.expression().clone())
            .collect();
        observe(slf.py(), &df.dataframe, slf, exprs)
    }
}

/// `df` observing the metrics `exprs` with `observation`, as PySpark's `Observation._on`, which
/// `DataFrame.observe` calls. The actions of the DataFrames of it set the metrics, as those of
/// an Observation of their plans.
pub(crate) fn observe(
    py: Python<'_>,
    df: &DataFrame,
    observation: &Bound<'_, PyObservation>,
    exprs: Vec<Expression>,
) -> PyResult<PyDataFrame> {
    let inner = observation.borrow().inner.clone();
    let observed = crate::interrupt::interruptible(py, &df.spark_session(), || {
        df.observe_with(&inner, exprs)
    })?;
    Ok(PyDataFrame::new(observed))
}

/// Have the executions of `session` convert the metrics of the Observations of their plans as
/// PySpark's client converts them when it sets them (`LiteralExpression._to_value`): the first
/// metric of a kind that it does not convert (a year-month interval, a nanosecond timestamp),
/// itself or an element of it, raises `UNSUPPORTED_LITERAL` from the action, of the text of its
/// literal. The core then sets neither the metrics of that Observation nor those that come after
/// them in the server's order, which keep the metrics of the executions before, as PySpark's.
/// The metrics of a DataFrame observed by a name are not converted by the action, as PySpark's;
/// `executionInfo` raises for them when they are read.
#[allow(clippy::result_large_err)] // The check of the core returns its Result, of a SparkError.
pub(crate) fn check_observed_metrics(session: &SparkSession) {
    session.set_observation_metrics_check(|metrics| {
        let Some(literal) = metrics
            .values
            .iter()
            .find_map(crate::dataframe::unsupported_literal)
        else {
            return Ok(());
        };
        let text = Python::attach(|py| crate::dataframe::literal_text(py, literal));
        Err(SparkError::classed(
            SparkErrorKind::TypeError,
            "UNSUPPORTED_LITERAL",
            &[("literal", &text)],
        ))
    });
}
