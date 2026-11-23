//! PyO3 wrapper for `spark_connect::observation::Observation` (`pyspark.sql.Observation`).

use std::sync::Arc;

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString, PyTuple, PyWeakrefReference};
use spark_connect::expression::Expression;
use spark_connect::observation::{MetricsMirror, Observation};
use spark_connect::SparkSession;
use spark_connect_core::error::{SparkError, SparkErrorKind};

use crate::column::{raise_pyspark, PyColumn};
use crate::dataframe::PyDataFrame;

// Python wrapper for an Observation of metrics, as PySpark Connect's `Observation`, a plain
// class, which is subclassed. Its `_name`, `_result` and `_error` are attributes of its
// `__dict__`, as PySpark's, which a program reads, sets and deletes, and which `get`, `_on` and
// the actions read and set as PySpark's do (see [`PyObservationMirror`]). An Observation of a
// subclass is constructed of the arguments of its `__init__`, and its copies and pickles are of
// its class.
#[doc = connect_doc!(Observation)]
#[pyclass(
    name = "Observation",
    module = "pyspark.sql.connect.observation",
    subclass,
    weakref,
    dict
)]
pub struct PyObservation {
    /// The state of the observation in the core, of its last `_on`, which the plans of the
    /// DataFrames that it observed hold, and whose metrics and errors the actions set in the
    /// attributes of this object.
    pub(crate) inner: Observation,
}

#[pymethods]
impl PyObservation {
    /// An unnamed Observation, as the `object.__new__` of PySpark's plain class: of any
    /// arguments, which `__init__` takes, so that a subclass of another `__init__` is built as it
    /// is in PySpark, and an unpickled one is made without them. The class shows the signature of
    /// `__init__`, as `inspect.signature` of PySpark's class does.
    #[new]
    #[pyo3(
        signature = (*_args, **_kwargs),
        text_signature = "(name=None)"
    )]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        PyObservation {
            inner: Observation::unnamed(),
        }
    }

    /// `Observation(name=None)`, as PySpark's `__init__`: the name is a non-empty string, and an
    /// observation without one is of a random UUID once it is used. As PySpark's, it sets the
    /// attributes `_name`, `_result` and `_error` of the observation, which is unused.
    #[pyo3(signature = (name=None))]
    fn __init__(slf: &Bound<'_, Self>, name: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let py = slf.py();
        if let Some(name) = name {
            if !name.is_instance_of::<PyString>() {
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
            if name.eq("")? {
                return Err(raise_pyspark(
                    py,
                    "PySparkValueError",
                    "VALUE_NOT_NON_EMPTY_STR",
                    &[("arg_name", "name"), ("arg_value", "")],
                ));
            }
        }
        slf.setattr(intern!(py, "_name"), name)?;
        slf.setattr(intern!(py, "_result"), py.None())?;
        slf.setattr(intern!(py, "_error"), py.None())?;
        slf.borrow_mut().inner = Observation::unnamed();
        Ok(())
    }

    // The observed metrics, as PySpark's `Observation.get`: its `_error`, raised if it is not
    // None, else its `_result`, the dict of the metrics that the actions run so far observed, by
    // their names, which the actions update in place, and `NO_OBSERVE_BEFORE_GET` while it is
    // None, before the observation is used.
    #[getter]
    #[doc = connect_doc!(Observation.get)]
    fn get<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let error = slf.getattr(intern!(py, "_error"))?;
        if !error.is_none() {
            return Err(PyErr::from_value(error));
        }
        let result = slf.getattr(intern!(py, "_result"))?;
        if result.is_none() {
            return Err(raise_pyspark(
                py,
                "PySparkAssertionError",
                "NO_OBSERVE_BEFORE_GET",
                &[],
            ));
        }
        Ok(result)
    }

    /// Set the error of collecting the metrics, as PySpark's `_set_error`, which its client calls
    /// when the server reports one: the `_error` that `get` raises.
    fn _set_error(slf: &Bound<'_, Self>, exc: &Bound<'_, PyAny>) -> PyResult<()> {
        slf.setattr(intern!(slf.py(), "_error"), exc)
    }

    // Observe the metrics of the Columns `exprs` on `df` with this observation, as PySpark's
    // `Observation._on`, which `DataFrame.observe` of an Observation calls.
    #[pyo3(signature = (df, *exprs))]
    #[doc = connect_doc!(Observation._on)]
    fn _on(
        slf: &Bound<'_, Self>,
        df: PyRef<'_, PyDataFrame>,
        exprs: Vec<Bound<'_, PyColumn>>,
    ) -> PyResult<PyDataFrame> {
        let exprs = exprs
            .iter()
            .map(|c| Ok(crate::column::column_of(c)?.expression().clone()))
            .collect::<PyResult<Vec<_>>>()?;
        let observed = observe(slf.py(), &df, slf, exprs)?;
        // The cached schema of the DataFrame, which observing does not change.
        Ok(df.of_same_schema(observed))
    }

    /// `copy.copy`, as of PySpark's Observation, a plain object: an Observation of the class and
    /// of the attributes of this one, the same objects, so the copy of a used one shares its
    /// result dict, which the actions of this one update.
    fn __copy__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyObservation>> {
        let copy = crate::reduce::new_of_class(
            &slf.get_type(),
            PyObservation {
                inner: Observation::unnamed(),
            },
        )?;
        copy_attributes(slf, &copy, None)?;
        Ok(copy)
    }

    /// `copy.deepcopy`, as of PySpark's Observation: an Observation of the class and of deep
    /// copies of the attributes of this one, which the actions of this one do not set. The copy
    /// is the Observation of the state that the deep copy of a DataFrame of the call holds, which
    /// the actions of that copy set.
    fn __deepcopy__<'py>(
        slf: &Bound<'py, Self>,
        memo: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyObservation>> {
        let inner = slf.borrow().inner.detached();
        let copy = crate::reduce::new_of_class(&slf.get_type(), PyObservation { inner })?;
        // The copy is the Observation of the state that the deep copy of a DataFrame of the call
        // holds, which a deep copy of that DataFrame copies with the copy.
        keep_handle(&copy)?;
        // As copy.deepcopy of a plain object, which records the copy in `memo` before it copies
        // the attributes, so that an attribute that refers to this observation refers to the copy.
        memo.set_item(slf.as_ptr() as usize, &copy)?;
        copy_attributes(slf, &copy, Some(memo))?;
        mirror_into(&copy)?;
        Ok(copy)
    }

    /// The reduce of pickle, which an Observation has at every protocol: as
    /// `object.__reduce_ex__` gives it at the protocols 2 and up, of its class, which pickle
    /// makes by `copyreg.__newobj__`, and of the state of `__getstate__`. Below protocol 2, the
    /// default reduce of a class with a native base fails, as copyreg's does, so an Observation
    /// would not pickle at all there.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let newobj = py.import("copyreg")?.getattr("__newobj__")?;
        let args = PyTuple::new(py, [slf.get_type()])?;
        let state = Self::__getstate__(slf)?;
        PyTuple::new(py, [newobj, args.into_any(), state.into_any()])
    }

    /// The state that a pickled Observation keeps, that of PySpark's, its `__dict__`: `_name`,
    /// `_result`, `_error` and the other attributes. So on a Python worker of the server, whose
    /// PySpark has no Rust client, it unpickles as PySpark's Observation, and here as this one.
    fn __getstate__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDict>> {
        let attributes = slf.getattr(intern!(slf.py(), "__dict__"))?;
        attributes.cast_into::<PyDict>()?.copy()
    }

    /// The Observation of the state of `__getstate__`, as pickle sets the `__dict__` of
    /// PySpark's: of its attributes, so used if `_result` is not None, which no action sets.
    fn __setstate__<'py>(slf: &Bound<'py, Self>, state: &Bound<'py, PyDict>) -> PyResult<()> {
        let py = slf.py();
        slf.borrow_mut().inner = Observation::unnamed();
        slf.getattr(intern!(py, "__dict__"))?
            .call_method1(intern!(py, "update"), (state,))?;
        Ok(())
    }

    /// The attributes of an Observation are those of its `__dict__`, which Python's garbage
    /// collector sees, as those of PySpark's: so a reference cycle through them, as through its
    /// result dict, is freed. The state of the core holds no Python object of it.
    fn __traverse__(&self, _visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        Ok(())
    }
}

/// The mirror of the state of an Observation of the core of a used Python Observation, which the
/// actions set as PySpark's client sets the attributes of the Observation of their plans: they
/// read its `_result`, which must not be None, update it in place with the metrics, and give the
/// error of collecting them to its `_set_error`. Their errors are those of the actions. The mirror
/// holds the Python Observation weakly, as the plans of its DataFrames hold it.
struct PyObservationMirror {
    observation: Py<PyWeakrefReference>,
}

impl PyObservationMirror {
    /// The Python Observation of this mirror, while it is alive.
    fn observation<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyAny>> {
        self.observation.bind(py).upgrade()
    }
}

impl MetricsMirror for PyObservationMirror {
    fn is_used(&self) -> Result<bool, SparkError> {
        Python::attach(|py| {
            let Some(observation) = self.observation(py) else {
                return Ok(true);
            };
            match observation.getattr(intern!(py, "_result")) {
                Ok(result) => Ok(!result.is_none()),
                Err(error) => Err(crate::errors::callback_error(error)),
            }
        })
    }

    fn update(
        &self,
        keys: &[String],
        values: &[spark_connect_proto::expression::Literal],
    ) -> Result<(), SparkError> {
        Python::attach(|py| {
            let Some(observation) = self.observation(py) else {
                return Ok(());
            };
            let updated = || -> PyResult<()> {
                let result = observation.getattr(intern!(py, "_result"))?;
                // Of metrics that the check of the session found PySpark converts, before they
                // were set (see `check_observed_metrics`).
                let metrics = PyDict::new(py);
                set_metrics(py, &metrics, keys.iter().zip(values))?;
                result.call_method1(intern!(py, "update"), (metrics,))?;
                Ok(())
            };
            updated().map_err(crate::errors::callback_error)
        })
    }

    fn error(&self, error: &SparkError) -> Result<(), SparkError> {
        Python::attach(|py| {
            let Some(observation) = self.observation(py) else {
                return Ok(());
            };
            let exception = crate::errors::spark_error_to_py_exception(error.clone());
            observation
                .call_method1(intern!(py, "_set_error"), (exception.value(py),))
                .map(drop)
                .map_err(crate::errors::callback_error)
        })
    }

    fn name(&self) -> Option<Result<String, SparkError>> {
        Python::attach(|py| {
            let observation = self.observation(py)?;
            // As PySpark's `CollectMetrics`, which sets `str(_name)` in the string field of its
            // name each time it builds the plan: a str that UTF-8 cannot encode fails the request
            // with protobuf's error, and an error of reading the name, or of its `__str__`, as it
            // is.
            let name = || -> PyResult<Result<String, SparkError>> {
                let name = observation.getattr(intern!(py, "_name"))?.str()?;
                Ok(
                    match crate::dataframe::unbuildable_str_error(name.as_any())? {
                        Some(error) => Err(error),
                        None => Ok(name.extract()?),
                    },
                )
            };
            Some(name().unwrap_or_else(|error| Err(crate::errors::callback_error(error))))
        })
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// The state of the core of `observation`, an Observation that `SparkConnectClient` was given for
/// an execution, as PySpark's client sets the Observations of its `observations` by their
/// attributes: that of its last `_on`, else a used one, whose mirror reads its `_result`, so that
/// the execution updates the `_result` of an Observation that no `_on` used, such as one that a
/// pickle restored, and raises PySpark's `AssertionError` of one that is None.
pub(crate) fn state_of_execution(observation: &Bound<'_, PyObservation>) -> PyResult<Observation> {
    let inner = observation.borrow().inner.clone();
    if inner.metrics_mirror().is_some() {
        return Ok(inner);
    }
    let used = Observation::used(&inner.name().unwrap_or_default());
    let weak = PyWeakrefReference::new(observation.as_any())?.unbind();
    let mirror: Arc<dyn MetricsMirror> = Arc::new(PyObservationMirror { observation: weak });
    let _ = used.set_metrics_mirror(mirror);
    Ok(used)
}

/// Make the attributes of `observation` those that the actions set of the state of its core,
/// if that state is used: the metrics that the actions set before the mirror, which a deep copy
/// keeps in its copied `_result`, are not set again.
fn mirror_into(observation: &Bound<'_, PyObservation>) -> PyResult<()> {
    let weak = PyWeakrefReference::new(observation.as_any())?.unbind();
    let mirror: Arc<dyn MetricsMirror> = Arc::new(PyObservationMirror { observation: weak });
    let _ = observation.borrow().inner.set_metrics_mirror(mirror);
    Ok(())
}

/// Set the metrics `metrics` in `dict` by their names, as PySpark's client updates the result of
/// an Observation with them.
fn set_metrics<'a>(
    py: Python<'_>,
    dict: &Bound<'_, PyDict>,
    metrics: impl IntoIterator<Item = (&'a String, &'a spark_connect_proto::expression::Literal)>,
) -> PyResult<()> {
    for (name, value) in metrics {
        dict.set_item(name, crate::dataframe::literal_to_py(py, value, None)?)?;
    }
    Ok(())
}

/// Keep a weak reference to `observation` on its state, by which the deep copy of a DataFrame
/// whose plan holds the state finds it (see [`deep_copy_in`]).
fn keep_handle(observation: &Bound<'_, PyObservation>) -> PyResult<()> {
    let weak = pyo3::types::PyWeakrefReference::new(observation.as_any())?.unbind();
    observation
        .borrow()
        .inner
        .set_handle(std::sync::Arc::new(weak));
    Ok(())
}

/// The Python Observation of `observation`, a state that a plan holds, while it is alive: the
/// one of [`keep_handle`], if it still has this state.
fn python_observation<'py>(
    py: Python<'py>,
    observation: &Observation,
) -> Option<Bound<'py, PyObservation>> {
    let handle = observation.handle()?;
    let weak = handle.downcast_ref::<Py<pyo3::types::PyWeakrefReference>>()?;
    let python = weak.bind(py).upgrade_as::<PyObservation>().ok().flatten()?;
    // `__setstate__` gives an Observation a state of its own.
    let same = python.borrow().inner.same(observation);
    same.then_some(python)
}

/// The Python Observation that `observation`, the clone of a state that a plan holds, keeps
/// alive (see [`held_by_the_plan`]), if it still has this state.
fn owner_observation<'py>(
    py: Python<'py>,
    observation: &Observation,
) -> Option<Bound<'py, PyObservation>> {
    let owner = observation.owner()?.downcast_ref::<Py<PyAny>>()?;
    let python = owner.bind(py).cast::<PyObservation>().ok()?.clone();
    let same = python.borrow().inner.same(observation);
    same.then_some(python)
}

/// The clone of the state of `observation` that a plan holds, which keeps `observation` alive, as
/// PySpark's plan holds the Observation itself: so the plan's Observations are the Python
/// Observations, of their classes and attributes, and a DataFrame reports them to Python's
/// garbage collector (see `crate::gc`).
fn held_by_the_plan(observation: &Bound<'_, PyObservation>) -> Observation {
    let owner: Arc<dyn std::any::Any + Send + Sync> =
        Arc::new(observation.clone().into_any().unbind());
    observation.borrow().inner.with_owner(owner)
}

/// The Python Observation of `observation`, a state that the plan of a DataFrame holds, as the
/// Observation that PySpark's plan holds itself: the one that the plan keeps alive, or that of
/// [`keep_handle`] while it is alive, else a new Observation of the state, of its name, of a
/// result of its metrics once it is used and of its error, which the actions of the plan set from
/// now on, and which later reads give back while it is alive.
pub(crate) fn python_observation_of<'py>(
    py: Python<'py>,
    observation: &Observation,
) -> PyResult<Bound<'py, PyObservation>> {
    if let Some(python) = owner_observation(py, observation) {
        return Ok(python);
    }
    if let Some(python) = python_observation(py, observation) {
        return Ok(python);
    }
    let python = Bound::new(
        py,
        PyObservation {
            inner: observation.clone(),
        },
    )?;
    let (_, error) = observation.state();
    python.setattr(intern!(py, "_name"), observation.name())?;
    python.setattr(intern!(py, "_result"), py.None())?;
    let error = error.map(|error| {
        crate::errors::spark_error_to_py_exception(error)
            .value(py)
            .clone()
    });
    python.setattr(intern!(py, "_error"), error)?;
    // The metrics so far, and the mirror of those that the actions set from now on.
    let weak = PyWeakrefReference::new(python.as_any())?.unbind();
    let mirror: Arc<dyn MetricsMirror> = Arc::new(PyObservationMirror { observation: weak });
    if let Some(metrics) = observation.set_metrics_mirror(mirror) {
        let result = PyDict::new(py);
        set_metrics(
            py,
            &result,
            metrics.iter().map(|(name, value)| (name, value)),
        )?;
        python.setattr(intern!(py, "_result"), result)?;
    }
    keep_handle(&python)?;
    Ok(python)
}

/// The copy of `observation`, an Observation of the plan of a DataFrame that `copy.deepcopy`
/// copies with `memo`, as PySpark's deep copy of a plan copies the Observation that it holds,
/// with the memo of the call: the deep copy of its Python Observation in `memo`, the copy that
/// the call made already, of the Observation itself or for another DataFrame, or that it makes
/// now and gives to the objects that it copies later. So in the copy of a tuple of a DataFrame
/// and its Observation, or of an object that holds both, the actions of the copied DataFrame
/// set the copied Observation. A state that no Python Observation holds any more, which no
/// other object of the call can refer to, has a copy of its own.
pub(crate) fn deep_copy_in(
    py: Python<'_>,
    observation: &Observation,
    memo: &Bound<'_, PyAny>,
) -> PyResult<Observation> {
    let Some(python) = python_observation(py, observation) else {
        return Ok(observation.detached());
    };
    let copy = py
        .import("copy")?
        .call_method1(pyo3::intern!(py, "deepcopy"), (python, memo))?;
    let copy = copy.cast_into::<PyObservation>()?;
    // The copied plan holds the copy, as PySpark's copied plan holds the copied Observation.
    Ok(held_by_the_plan(&copy))
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
/// `DataFrame.observe` calls: `REUSE_OBSERVATION` of an observation whose `_result` is not None,
/// a random UUID as the `_name` of an unnamed one, the check that `df` is not streaming, and then
/// an empty `_result`, of the observation of a new state of the core. The actions of the
/// DataFrames of it set the metrics, as those of an Observation of their plans.
pub(crate) fn observe(
    py: Python<'_>,
    frame: &PyDataFrame,
    observation: &Bound<'_, PyObservation>,
    exprs: Vec<Expression>,
) -> PyResult<PyDataFrame> {
    let df = &frame.dataframe;
    if !observation.getattr(intern!(py, "_result"))?.is_none() {
        return Err(raise_pyspark(
            py,
            "PySparkAssertionError",
            "REUSE_OBSERVATION",
            &[],
        ));
    }
    let mut name = observation.getattr(intern!(py, "_name"))?;
    if name.is_none() {
        name = py.import("uuid")?.call_method0("uuid4")?.str()?.into_any();
        observation.setattr(intern!(py, "_name"), &name)?;
    }
    // The plans of the DataFrame read the `_name` of the observation each time they are built (see
    // `PyObservationMirror::name`), as PySpark's `CollectMetrics` sets `str(_name)` when it builds
    // the plan, after its input, and `_on` does not read it: a name whose `__str__` raises, or a
    // str that UTF-8 cannot encode, fails each request, not `observe`. The state of the core is of
    // the text of a str name, in the place of the name of a plan without the mirror.
    observation.borrow_mut().inner = Observation::new(&crate::plan_args::rendered(&name)?);
    let inner = held_by_the_plan(observation);
    // As PySpark's `Observation._on`, which reads the `isStreaming` of the DataFrame, asked once:
    // of the plan of the DataFrame, which the observed plan is built over and the request of
    // `isStreaming` is built of, and of the expressions of the metrics.
    let parts = df
        .held_parts()
        .into_iter()
        .chain(exprs.iter().map(spark_connect::gc::PartRef::Expression));
    let _in_use = crate::gc::in_use_of_values(py, parts);
    let observed = crate::interrupt::interruptible(py, &df.spark_session(), || {
        df.observe_with_streaming(&inner, exprs, || frame.streaming_once())
    })?;
    observation.setattr(intern!(py, "_result"), PyDict::new(py))?;
    mirror_into(observation)?;
    // The plan holds the Observation itself, as PySpark's does (see `held_by_the_plan`), which a
    // deep copy of the plan copies with the memo of the call. The clones of its state that do not
    // keep it alive find it by its weak reference.
    keep_handle(observation)?;
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
