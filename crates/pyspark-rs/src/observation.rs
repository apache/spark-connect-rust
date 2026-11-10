//! PyO3 wrapper for `spark_connect::observation::Observation` (`pyspark.sql.Observation`).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString, PyWeakrefMethods, PyWeakrefReference};
use spark_connect::dataframe::DataFrame;
use spark_connect::expression::Expression;
use spark_connect::observation::Observation;
use spark_connect::SparkSession;

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
    /// Whether the last execution that observed the metrics failed before PySpark's client would
    /// set them, as it raises at the first metric that it does not convert (`ObservedMetricsCheck`).
    rejected: AtomicBool,
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
            rejected: AtomicBool::new(false),
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
    /// of an execution that failed before PySpark's client would set them (`ObservedMetricsCheck`),
    /// and the metrics of a kind that it does not convert, are not given: PySpark's are those of
    /// the executions before, here none of them.
    #[getter]
    fn get<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        if self.rejected.load(Ordering::SeqCst) {
            return Ok(result);
        }
        let metrics = self.inner.get().to_pyerr()?;
        if metrics
            .iter()
            .any(|(_, value)| crate::dataframe::unsupported_literal(value).is_some())
        {
            return Ok(result);
        }
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

/// The Observations that the DataFrames of each session observe metrics with, by session ID, and
/// their names: held weakly, by their Python objects, so that this keeps none of them alive.
type Observations = HashMap<String, Vec<(String, Py<PyWeakrefReference>)>>;

fn observations() -> &'static Mutex<Observations> {
    static OBSERVATIONS: OnceLock<Mutex<Observations>> = OnceLock::new();
    OBSERVATIONS.get_or_init(Default::default)
}

/// `df` observing the metrics `exprs` with `observation`, as PySpark's `Observation._on`, which
/// `DataFrame.observe` calls. The observation is kept for the actions of the session.
pub(crate) fn observe(
    py: Python<'_>,
    df: &DataFrame,
    observation: &Bound<'_, PyObservation>,
    exprs: Vec<Expression>,
) -> PyResult<PyDataFrame> {
    let inner = observation.borrow().inner.clone();
    let observed = py
        .detach(|| df.observe_with(&inner, exprs).map_err(Box::new))
        .to_pyerr()?;
    if let Some(name) = inner.name() {
        let weak = PyWeakrefReference::new(observation)?.unbind();
        let mut sessions = observations().lock().unwrap_or_else(|e| e.into_inner());
        let kept = sessions
            .entry(df.spark_session().session_id().to_string())
            .or_default();
        kept.push((name, weak));
    }
    Ok(PyDataFrame::new(observed))
}

/// The check of an action, after it ran, that PySpark's client makes when it converts the metrics
/// of the Observations of the plan as the execution sends them: `UNSUPPORTED_LITERAL`, raised from
/// the action, for the first metric of a kind that PySpark's `LiteralExpression._to_value` does
/// not convert (a year-month interval, a nanosecond timestamp). PySpark's client then sets neither
/// those metrics nor the ones that come after them. The metrics are those of the session's last
/// execution, if it is not the one before the action, whose names are those of the session's
/// Observations; the other observed metrics are not converted, as in PySpark.
pub(crate) struct ObservedMetricsCheck {
    before: Option<String>,
}

impl ObservedMetricsCheck {
    /// The check of an action of `session`, which has one only if it has Observations.
    pub(crate) fn before(py: Python<'_>, session: &SparkSession) -> Option<Self> {
        live_observations(py, session).as_ref()?;
        Some(ObservedMetricsCheck {
            before: session
                .last_execution_info()
                .and_then(|info| info.operation_id),
        })
    }

    pub(crate) fn after(self, py: Python<'_>, session: &SparkSession) -> PyResult<()> {
        let Some(info) = session.last_execution_info() else {
            return Ok(());
        };
        if info.operation_id.is_none() || info.operation_id == self.before {
            return Ok(());
        }
        let Some(kept) = live_observations(py, session) else {
            return Ok(());
        };
        let mut error = None;
        for metrics in &info.observed_metrics {
            let observed: Vec<_> = kept
                .iter()
                .filter(|(name, _)| *name == metrics.name)
                .collect();
            if observed.is_empty() {
                continue;
            }
            // The metrics of an error of the server are not converted.
            if error.is_none() && metrics.root_error_idx.is_none() {
                error = metrics
                    .values
                    .iter()
                    .find_map(crate::dataframe::unsupported_literal)
                    .map(|literal| crate::dataframe::literal_text(py, literal));
            }
            for (_, observation) in observed {
                observation
                    .borrow()
                    .rejected
                    .store(error.is_some(), Ordering::SeqCst);
            }
        }
        match error {
            Some(literal) => Err(raise_pyspark(
                py,
                "PySparkTypeError",
                "UNSUPPORTED_LITERAL",
                &[("literal", &literal)],
            )),
            None => Ok(()),
        }
    }
}

/// The Observations of `session` that are alive, by their names, if any; the others are forgotten.
fn live_observations<'py>(
    py: Python<'py>,
    session: &SparkSession,
) -> Option<Vec<(String, Bound<'py, PyObservation>)>> {
    let mut sessions = observations().lock().unwrap_or_else(|e| e.into_inner());
    let kept = sessions.get_mut(session.session_id())?;
    let mut live = Vec::new();
    kept.retain(|(name, weak)| match weak.bind(py).upgrade() {
        Some(observation) => {
            if let Ok(observation) = observation.cast_into() {
                live.push((name.clone(), observation));
            }
            true
        }
        None => false,
    });
    if kept.is_empty() {
        sessions.remove(session.session_id());
    }
    (!live.is_empty()).then_some(live)
}
