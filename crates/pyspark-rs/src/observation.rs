//! PyO3 wrapper for `spark_connect::observation::Observation` (`pyspark.sql.Observation`).

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use spark_connect::observation::Observation;

use crate::column::{raise_pyspark, PyColumn};
use crate::dataframe::PyDataFrame;
use crate::errors::ResultExt;

/// Python wrapper for an Observation of metrics, as PySpark Connect's `Observation`.
#[pyclass(name = "Observation", module = "pyspark.sql.observation")]
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
    /// actions run so far observed, by their names, and empty before an action runs.
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
        &self,
        py: Python<'_>,
        df: PyRef<'_, PyDataFrame>,
        exprs: Vec<PyRef<'_, PyColumn>>,
    ) -> PyResult<PyDataFrame> {
        let exprs: Vec<_> = exprs
            .iter()
            .map(|c| c.column.expression().clone())
            .collect();
        let (dataframe, observation) = (df.dataframe.clone(), &self.inner);
        py.detach(|| dataframe.observe_with(observation, exprs))
            .to_pyerr()
            .map(PyDataFrame::new)
    }
}
