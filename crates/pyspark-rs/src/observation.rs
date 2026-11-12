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
    module = "pyspark.sql.connect.observation",
    weakref,
    dict
)]
pub struct PyObservation {
    pub(crate) inner: Observation,
    /// The `_result` and `_error` of an Observation unpickled from the state of PySpark's (see
    /// `__setstate__`), which no action sets, and which `get` gives back as PySpark's does.
    restored: Restored,
}

/// The `_result` and `_error` of the state of a pickled Observation, which its Rust state does
/// not hold: none for an Observation that was not unpickled.
#[derive(Default)]
struct Restored {
    /// The metrics of an Observation unpickled from a used one, a dict as PySpark's.
    result: Option<Py<PyAny>>,
    /// The exception that `get` raises.
    error: Option<Py<PyAny>>,
}

impl Restored {
    /// The same objects, as a shallow copy of PySpark's Observation keeps them.
    fn clone_ref(&self, py: Python<'_>) -> Restored {
        Restored {
            result: self.result.as_ref().map(|result| result.clone_ref(py)),
            error: self.error.as_ref().map(|error| error.clone_ref(py)),
        }
    }
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
        Ok(PyObservation {
            inner,
            restored: Restored::default(),
        })
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
    /// An unpickled Observation gives back its `_error` and its `_result`, as PySpark's.
    #[getter]
    fn get<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if let Some(error) = &self.restored.error {
            return Err(PyErr::from_value(error.bind(py).clone()));
        }
        if let Some(result) = &self.restored.result {
            return Ok(result.bind(py).clone());
        }
        let metrics = self.inner.get().to_pyerr()?;
        Ok(metrics_dict(py, metrics)?.into_any())
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

    /// `copy.copy`, as of PySpark's Observation, a plain object: an Observation of the name, the
    /// state and the attributes of this one, which shares the metrics of a used one, as the copy
    /// of PySpark's shares its result dict (see `Observation::shallow_copy`).
    fn __copy__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyObservation>> {
        let py = slf.py();
        let this = slf.borrow();
        let copy = PyObservation {
            inner: this.inner.shallow_copy(),
            restored: this.restored.clone_ref(py),
        };
        let copy = Bound::new(py, copy)?;
        copy_attributes(slf, &copy, None)?;
        Ok(copy)
    }

    /// `copy.deepcopy`, as of PySpark's Observation: an Observation of the name and the state
    /// of this one as they are now, which the actions of this one do not set, with deep copies
    /// of its attributes.
    fn __deepcopy__<'py>(
        slf: &Bound<'py, Self>,
        memo: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyObservation>> {
        let py = slf.py();
        let inner = slf.borrow().inner.detached();
        let copy = Bound::new(
            py,
            PyObservation {
                inner,
                restored: Restored::default(),
            },
        )?;
        // As copy.deepcopy of a plain object, which records the copy in `memo` before it copies
        // the attributes, so that an attribute that refers to this observation refers to the copy.
        memo.set_item(slf.as_ptr() as usize, &copy)?;
        let deepcopy = py.import("copy")?.getattr("deepcopy")?;
        let copied = |object: &Option<Py<PyAny>>| -> PyResult<Option<Py<PyAny>>> {
            object
                .as_ref()
                .map(|object| Ok(deepcopy.call1((object, memo))?.unbind()))
                .transpose()
        };
        let restored = {
            let this = slf.borrow();
            Restored {
                result: copied(&this.restored.result)?,
                error: copied(&this.restored.error)?,
            }
        };
        copy.borrow_mut().restored = restored;
        copy_attributes(slf, &copy, Some(memo))?;
        Ok(copy)
    }

    /// The state that a pickled Observation keeps, that of PySpark's, its `__dict__`: `_name`,
    /// `_result`, a dict of the metrics or None before it is used, `_error`, the exception that
    /// `get` raises or None, and the attributes of this one. So on a Python worker of the server,
    /// whose PySpark has no Rust client, it unpickles as PySpark's Observation, and here as this
    /// one (see `__setstate__`).
    fn __getstate__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDict>> {
        let py = slf.py();
        let this = slf.borrow();
        let (metrics, error) = this.inner.state();
        let result = match &this.restored.result {
            Some(result) => Some(result.bind(py).clone()),
            None => metrics
                .map(|metrics| metrics_dict(py, metrics).map(Bound::into_any))
                .transpose()?,
        };
        let error = match &this.restored.error {
            Some(error) => Some(error.bind(py).clone()),
            None => error.map(|error| {
                crate::errors::spark_error_to_py_exception(error)
                    .value(py)
                    .clone()
                    .into_any()
            }),
        };
        let state = PyDict::new(py);
        state.set_item("_name", this.inner.name())?;
        state.set_item("_result", result)?;
        state.set_item("_error", error)?;
        state.call_method1("update", (slf.getattr(pyo3::intern!(py, "__dict__"))?,))?;
        Ok(state)
    }

    /// The Observation of the state of `__getstate__`, as pickle sets the `__dict__` of
    /// PySpark's: of the name `_name`, whose `get` gives back `_error` and `_result`, which no
    /// action sets, and so used if `_result` is not None, and of the other attributes.
    fn __setstate__<'py>(slf: &Bound<'py, Self>, state: &Bound<'py, PyDict>) -> PyResult<()> {
        let py = slf.py();
        let state = state.copy()?;
        let take = |key: &str| -> PyResult<Option<Bound<'py, PyAny>>> {
            let value = state.get_item(key)?;
            if value.is_some() {
                state.del_item(key)?;
            }
            Ok(value.filter(|value| !value.is_none()))
        };
        let name: Option<String> = take("_name")?.map(|name| name.extract()).transpose()?;
        let result = take("_result")?.map(Bound::unbind);
        let error = take("_error")?.map(Bound::unbind);
        {
            let mut this = slf.borrow_mut();
            this.inner = match name {
                Some(name) => Observation::new(&name),
                None => Observation::unnamed(),
            };
            this.restored = Restored { result, error };
        }
        slf.getattr(pyo3::intern!(py, "__dict__"))?
            .call_method1(pyo3::intern!(py, "update"), (state,))?;
        Ok(())
    }

    /// The `_result` and `_error` of an unpickled Observation, Python objects, which Python's
    /// garbage collector sees, as those of PySpark's: so a reference cycle through them is freed.
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        if let Some(result) = &self.restored.result {
            visit.call(result)?;
        }
        if let Some(error) = &self.restored.error {
            visit.call(error)?;
        }
        Ok(())
    }

    /// Break a reference cycle that the garbage collector found unreachable.
    fn __clear__(&mut self) {
        self.restored = Restored::default();
    }
}

/// A dict of `metrics` by their names, as PySpark's `_result` of an Observation.
fn metrics_dict(
    py: Python<'_>,
    metrics: Vec<(String, spark_connect_proto::expression::Literal)>,
) -> PyResult<Bound<'_, PyDict>> {
    let result = PyDict::new(py);
    for (name, value) in metrics {
        result.set_item(name, crate::dataframe::literal_to_py(py, &value, None)?)?;
    }
    Ok(result)
}

/// Set the attributes of `observation` on `copy`, as `copy` sets the `__dict__` of the copy of a
/// plain object: the same values, or, of a deep copy, their deep copies of `memo`.
fn copy_attributes(
    observation: &Bound<'_, PyObservation>,
    copy: &Bound<'_, PyObservation>,
    memo: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let py = observation.py();
    let mut attributes = observation.getattr(pyo3::intern!(py, "__dict__"))?;
    if let Some(memo) = memo {
        attributes = py
            .import("copy")?
            .call_method1("deepcopy", (attributes, memo))?;
    }
    copy.getattr(pyo3::intern!(py, "__dict__"))?
        .call_method1(pyo3::intern!(py, "update"), (attributes,))?;
    Ok(())
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
    // An Observation unpickled from a used one is used, as PySpark's of a `_result`.
    if observation.borrow().restored.result.is_some() {
        return Err(raise_pyspark(
            py,
            "PySparkAssertionError",
            "REUSE_OBSERVATION",
            &[],
        ));
    }
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
