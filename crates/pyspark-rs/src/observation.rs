//! PyO3 wrapper for `spark_connect::observation::Observation` (`pyspark.sql.Observation`).

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString, PyTuple};
use spark_connect::expression::Expression;
use spark_connect::observation::{MetricsMirror, Observation};
use spark_connect::SparkSession;
use spark_connect_core::error::{SparkError, SparkErrorKind};

use crate::column::{raise_pyspark, PyColumn};
use crate::dataframe::PyDataFrame;
use crate::errors::ResultExt;

/// Python wrapper for an Observation of metrics, as PySpark Connect's `Observation`, a plain
/// class, which is subclassed: an Observation of a subclass is constructed of the arguments of
/// its `__init__`, and its copies and pickles are of its class.
#[pyclass(
    name = "Observation",
    module = "pyspark.sql.connect.observation",
    subclass,
    weakref,
    dict
)]
pub struct PyObservation {
    pub(crate) inner: Observation,
    /// The `_result` and `_error` of an Observation unpickled from the state of PySpark's (see
    /// `__setstate__`), which no action sets, and which `get` gives back as PySpark's does.
    restored: Restored,
    /// A name that UTF-8 cannot encode, as it was given, which `_name` gives back: PySpark keeps
    /// it, and its `CollectMetrics` plan sets it when it builds the plan, which fails the requests
    /// of each DataFrame that it observes (see [`observe`]). `inner` is of its rendering.
    unencodable_name: Option<Py<PyString>>,
}

/// The state of an Observation of the name `name`, None or a str, and the name itself if UTF-8
/// cannot encode it (see `PyObservation::unencodable_name`).
fn named(name: Option<&Bound<'_, PyAny>>) -> PyResult<(Observation, Option<Py<PyString>>)> {
    let Some(name) = name.filter(|name| !name.is_none()) else {
        return Ok((Observation::unnamed(), None));
    };
    let name = name.cast::<PyString>()?;
    Ok(match name.to_cow() {
        Ok(text) => (Observation::new(&text), None),
        Err(_) => (
            Observation::new(&crate::plan_args::rendered(name.as_any())?),
            Some(name.clone().unbind()),
        ),
    })
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
            restored: Restored::default(),
            unencodable_name: None,
        }
    }

    /// `Observation(name=None)`, as PySpark's `__init__`: the name is a non-empty string, and an
    /// observation without one is of a random UUID once it is used. As PySpark's, it makes the
    /// observation anew, unused.
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
        let (inner, unencodable_name) = named(name)?;
        let mut this = slf.borrow_mut();
        this.inner = inner;
        this.restored = Restored::default();
        this.unencodable_name = unencodable_name;
        Ok(())
    }

    /// The name of the observation, as PySpark's `_name`: None for an unnamed observation
    /// before it is used, and a name that UTF-8 cannot encode as it was given.
    #[getter(_name)]
    fn name(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        if let Some(name) = &self.unencodable_name {
            return Some(name.clone_ref(py).into_any());
        }
        self.inner
            .name()
            .map(|name| PyString::new(py, &name).into_any().unbind())
    }

    /// The observed metrics, as PySpark's `Observation.get`: its `_result`, the dict of the metrics
    /// that the actions run so far observed, by their names, and empty before an action runs,
    /// which is one dict, that the actions update in place, as PySpark's client updates it. The
    /// metrics of an execution that PySpark's client does not convert (see
    /// [`check_observed_metrics`]) are never set, so these are those of the executions before it,
    /// as PySpark's. An unpickled Observation gives back its `_error` and its `_result`, as
    /// PySpark's.
    #[getter]
    fn get<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if let Some(error) = &self.restored.error {
            return Err(PyErr::from_value(error.bind(py).clone()));
        }
        if let Some(result) = &self.restored.result {
            return Ok(result.bind(py).clone());
        }
        // The error of collecting the metrics, or NO_OBSERVE_BEFORE_GET of an unused observation.
        self.inner.get().to_pyerr()?;
        match result_dict(py, &self.inner)? {
            Some(result) => Ok(result.into_any()),
            None => Ok(PyDict::new(py).into_any()),
        }
    }

    /// The result of the observation, as PySpark's `_result`: None before it is used, and then
    /// the dict that `get` gives, which the actions that set its metrics update in place.
    #[getter(_result)]
    fn result<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        if let Some(result) = &self.restored.result {
            return Ok(Some(result.bind(py).clone()));
        }
        Ok(result_dict(py, &self.inner)?.map(Bound::into_any))
    }

    /// The error of collecting the metrics, which the server reported, as PySpark's `_error`, or
    /// None: the exception that `get` raises.
    #[getter(_error)]
    fn error<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyAny>> {
        if let Some(error) = &self.restored.error {
            return Some(error.bind(py).clone());
        }
        let (_, error) = self.inner.state();
        error.map(|error| {
            crate::errors::spark_error_to_py_exception(error)
                .value(py)
                .clone()
                .into_any()
        })
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
        observe(slf.py(), &df, slf, exprs)
    }

    /// `copy.copy`, as of PySpark's Observation, a plain object: an Observation of the class, the
    /// name, the state and the attributes of this one, which shares the metrics of a used one, as
    /// the copy of PySpark's shares its result dict (see `Observation::shallow_copy`).
    fn __copy__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyObservation>> {
        let py = slf.py();
        let this = slf.borrow();
        let copy = PyObservation {
            inner: this.inner.shallow_copy(),
            restored: this.restored.clone_ref(py),
            unencodable_name: this
                .unencodable_name
                .as_ref()
                .map(|name| name.clone_ref(py)),
        };
        let copy = crate::reduce::new_of_class(&slf.get_type(), copy)?;
        copy_attributes(slf, &copy, None)?;
        Ok(copy)
    }

    /// `copy.deepcopy`, as of PySpark's Observation: an Observation of the class, the name and
    /// the state of this one as they are now, which the actions of this one do not set, with deep
    /// copies of its attributes.
    fn __deepcopy__<'py>(
        slf: &Bound<'py, Self>,
        memo: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyObservation>> {
        let py = slf.py();
        let inner = slf.borrow().inner.detached();
        let unencodable_name = slf
            .borrow()
            .unencodable_name
            .as_ref()
            .map(|name| name.clone_ref(py));
        let copy = crate::reduce::new_of_class(
            &slf.get_type(),
            PyObservation {
                inner,
                restored: Restored::default(),
                unencodable_name,
            },
        )?;
        // The copy is the Observation of the state that the deep copy of a DataFrame of the call
        // holds, which a deep copy of that DataFrame copies with the copy.
        keep_handle(&copy)?;
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
        // As the deep copy of PySpark's `_result`, of the memo: a copy of the dict that `get` gave,
        // with what was set on it, which the actions of this observation do not update. One that
        // `get` did not give yet is made of the copied metrics when the copy's is read.
        let result = slf.borrow().inner.metrics_mirror();
        if let Some(result) = result.as_deref().and_then(mirrored_dict) {
            let copied = deepcopy.call1((result.bind(py), memo))?;
            if let Ok(copied) = copied.cast_into::<PyDict>() {
                let mirror: Arc<dyn MetricsMirror> = Arc::new(PyMetricsMirror {
                    dict: copied.unbind(),
                });
                let _ = copy.borrow().inner.set_metrics_mirror(mirror);
            }
        }
        copy_attributes(slf, &copy, Some(memo))?;
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
    /// `_result`, a dict of the metrics or None before it is used, `_error`, the exception that
    /// `get` raises or None, and the attributes of this one. So on a Python worker of the server,
    /// whose PySpark has no Rust client, it unpickles as PySpark's Observation, and here as this
    /// one (see `__setstate__`).
    fn __getstate__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDict>> {
        let py = slf.py();
        let this = slf.borrow();
        let (_, error) = this.inner.state();
        let result = match &this.restored.result {
            Some(result) => Some(result.bind(py).clone()),
            None => result_dict(py, &this.inner)?.map(Bound::into_any),
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
        state.set_item("_name", this.name(py))?;
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
        let (inner, unencodable_name) = named(take("_name")?.as_ref())?;
        let result = take("_result")?.map(Bound::unbind);
        let error = take("_error")?.map(Bound::unbind);
        {
            let mut this = slf.borrow_mut();
            this.inner = inner;
            this.unencodable_name = unencodable_name;
            this.restored = Restored { result, error };
        }
        slf.getattr(pyo3::intern!(py, "__dict__"))?
            .call_method1(pyo3::intern!(py, "update"), (state,))?;
        Ok(())
    }

    /// The `_result` and `_error` of an unpickled Observation, Python objects, which Python's
    /// garbage collector sees, as those of PySpark's: so a reference cycle through them is freed.
    /// Only `Option<Py<_>>` fields are read, which are `None` while the object is created, so no
    /// [`crate::gc::GcReady`] is needed.
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

/// The result dict of a used Observation, as PySpark's `_result`: the dict that `get` gives,
/// which the actions that set the metrics of the Observation update in place, as PySpark's client
/// does, the shallow copies of the Observation sharing it.
struct PyMetricsMirror {
    dict: Py<PyDict>,
}

impl MetricsMirror for PyMetricsMirror {
    fn update(&self, keys: &[String], values: &[spark_connect_proto::expression::Literal]) {
        Python::attach(|py| {
            // Of metrics that the check of the session found PySpark converts, before they were
            // set (see `check_observed_metrics`).
            let _ = set_metrics(py, self.dict.bind(py), keys.iter().zip(values));
        });
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// The dict of `mirror`, of an Observation of this binding.
fn mirrored_dict(mirror: &dyn MetricsMirror) -> Option<&Py<PyDict>> {
    mirror
        .as_any()
        .downcast_ref::<PyMetricsMirror>()
        .map(|mirror| &mirror.dict)
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

/// The result dict of `observation`, PySpark's `_result`, or None before it is used: the one that
/// was made of it, else a new one of its metrics so far, which the actions update from now on.
fn result_dict<'py>(
    py: Python<'py>,
    observation: &Observation,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    if let Some(mirror) = observation.metrics_mirror() {
        if let Some(dict) = mirrored_dict(mirror.as_ref()) {
            return Ok(Some(dict.bind(py).clone()));
        }
    }
    let dict = PyDict::new(py);
    let made = Arc::new(PyMetricsMirror {
        dict: dict.clone().unbind(),
    });
    let Some((mirror, metrics)) = observation.set_metrics_mirror(made) else {
        return Ok(None);
    };
    if let Some(metrics) = metrics {
        set_metrics(py, &dict, metrics.iter().map(|(name, value)| (name, value)))?;
    }
    Ok(Some(match mirrored_dict(mirror.as_ref()) {
        Some(kept) => kept.bind(py).clone(),
        None => dict,
    }))
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
/// garbage collector (`DataFrame::exclusive_observation_owners`).
fn held_by_the_plan(observation: &Bound<'_, PyObservation>) -> Observation {
    let owner: Arc<dyn std::any::Any + Send + Sync> =
        Arc::new(observation.clone().into_any().unbind());
    observation.borrow().inner.with_owner(owner)
}

/// The Python Observation of `observation`, a state that the plan of a DataFrame holds, as the
/// Observation that PySpark's plan holds itself: the one that the plan keeps alive, or that of
/// [`keep_handle`] while it is alive, else a new Observation of the state, whose `get` the actions
/// of the plan set as they set the state, and which later reads give back while it is alive.
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
            restored: Restored::default(),
            unencodable_name: None,
        },
    )?;
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
/// `DataFrame.observe` calls. The actions of the DataFrames of it set the metrics, as those of
/// an Observation of their plans.
pub(crate) fn observe(
    py: Python<'_>,
    frame: &PyDataFrame,
    observation: &Bound<'_, PyObservation>,
    mut exprs: Vec<Expression>,
) -> PyResult<PyDataFrame> {
    let df = &frame.dataframe;
    // An Observation unpickled from a used one is used, as PySpark's of a `_result`.
    if observation.borrow().restored.result.is_some() {
        return Err(raise_pyspark(
            py,
            "PySparkAssertionError",
            "REUSE_OBSERVATION",
            &[],
        ));
    }
    // A name that UTF-8 cannot encode fails the requests of the DataFrame, as PySpark's
    // `CollectMetrics` sets it when it builds the plan, before the metrics.
    let mut args = crate::plan_args::PlanArgs::new();
    if let Some(name) = &observation.borrow().unencodable_name {
        args.str(name.bind(py).as_any())?;
    }
    let args = args.before(&mut exprs);
    let inner = held_by_the_plan(observation);
    // As PySpark's `Observation._on`, which reads the `isStreaming` of the DataFrame, asked once.
    let observed = crate::interrupt::interruptible(py, &df.spark_session(), || {
        df.observe_with_streaming(&inner, exprs, || frame.streaming_once())
    })?;
    // The plan holds the Observation itself, as PySpark's does (see `held_by_the_plan`), which a
    // deep copy of the plan copies with the memo of the call. The clones of its state that do not
    // keep it alive find it by its weak reference.
    keep_handle(observation)?;
    Ok(PyDataFrame::new(args.plan(observed)))
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
