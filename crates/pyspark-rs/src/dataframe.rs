//! PyO3 wrapper for spark_connect::dataframe::DataFrame.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt, PyList, PyTuple};
use spark_connect::dataframe::{AsOfJoinOptions, DataFrame, LocalRowIterator, PreparedExecution};
use spark_connect::plan::{AsOfJoinDirection, AsOfJoinType, JoinType};
use spark_connect::udf::{eval_type, CommonInlineUserDefinedFunctionExpression, PythonUDFPayload};

use crate::column::PyColumn;
use crate::errors::{not_expected_type, ResultExt};
use crate::group::PyGroupedData;
use crate::interrupt::{interruptible, interruptible_of};
use crate::options::DefaultedOption;
use crate::row::{FieldConverters, NestedFieldNames};
use crate::stat_arguments::proto_int;
use crate::streaming::{proto_bool, ProtoBool};
use crate::types::PyDataType;

// Python wrapper for a Spark DataFrame.
#[doc = connect_doc!(DataFrame)]
#[pyclass(
    name = "DataFrame",
    module = "pyspark.sql.connect.dataframe",
    subclass,
    weakref,
    dict
)]
pub struct PyDataFrame {
    /// The caches of the core that are of this DataFrame and not of its plan, of `isLocal` and of
    /// the information of its last execution, as PySpark keeps them for each DataFrame, whatever
    /// plan object it holds, and of the schema that the core reads itself (see `DataFrameCaches`):
    /// its requests are of the plan of `_plan` with these. Its `schema` and `columns` are of its
    /// `_cached_schema`, as PySpark's (see `Frame::schema_entry`).
    caches: spark_connect::dataframe::DataFrameCaches,
    /// The plan object of the drop-in that the requests of this DataFrame are made of, when its
    /// `_plan` is not the plan object of a DataFrame of the drop-in of its session, by the
    /// identity of that `_plan`: a plan of the Python client, such as a `SubqueryAlias` of
    /// `pyspark.ml`, of which the plan of the core is built at the first request after it is set,
    /// as PySpark's requests build `self._plan`, or the plan object of a DataFrame of another
    /// session, whose plan has the plan IDs of that DataFrame in the session of this one (see
    /// `Frame::of`). The object and the plan object of it are reported by `__traverse__`, as the
    /// holder of the parts of the plan of the core, as PySpark's DataFrame holds its plan object.
    converted: std::sync::Mutex<Option<(Py<PyAny>, Py<PyNativePlan>)>>,
    /// Set once the value is written. Python's garbage collector may traverse the object while it
    /// is created: the allocation already tracks it, and `object.__new__` allocates the instance
    /// dictionary, which can start a collection, before PyO3 writes the value. Until then the
    /// memory is zeroed, so this flag is false and `__traverse__` reads nothing else (see
    /// [`crate::gc::GcReady`]).
    gc_ready: crate::gc::GcReady,
    /// The answers that PySpark keeps for each DataFrame once it has them, of the class-level
    /// `functools.cache` of `semanticHash`, `isEmpty` and `inputFiles`, and not in its `__dict__`:
    /// later calls do not ask again, also after the session stopped. (`isLocal` is cached in the
    /// core, and `isStreaming` is PySpark's `functools.cached_property`, of the `__dict__`.)
    semantic_hash: std::sync::OnceLock<i32>,
    is_empty: std::sync::OnceLock<bool>,
    input_files: std::sync::OnceLock<Vec<String>>,
    /// The information of the execution of the core that `_execution_info` was last set of, as
    /// PySpark's actions set it (see `Frame::publish_execution`): a later execution of the
    /// DataFrame sets it again, and an assignment of it stays until then.
    published: std::sync::Mutex<Option<spark_connect::session::ExecutionInfo>>,
}

/// The answer kept in `cell`, else that of `ask`, which is kept unless it fails, as
/// `functools.cache` keeps the result of a call.
fn cached<T: Clone>(
    cell: &std::sync::OnceLock<T>,
    ask: impl FnOnce() -> PyResult<T>,
) -> PyResult<T> {
    if let Some(value) = cell.get() {
        return Ok(value.clone());
    }
    let value = ask()?;
    Ok(cell.get_or_init(|| value).clone())
}

/// The number of rows and the truncation of the eager evaluation of `__repr__` and
/// `_repr_html_` of a DataFrame of `session`, its `_session`, as PySpark's `int` of each of the
/// settings that `self._session._client.get_configs(...)` reads, before it reads the plan, if it
/// is enabled.
fn eager_eval(py: Python<'_>, session: &Bound<'_, PyAny>) -> PyResult<Option<(i32, i32)>> {
    let session = request_session(session)?;
    let values = crate::interrupt::interruptible_without_parts(py, &session, || {
        session.conf().get_many(&EAGER_EVAL)
    })?;
    if values[0].as_deref() != Some("true") {
        return Ok(None);
    }
    let int = py.import("builtins")?.getattr("int")?;
    let n: i32 = int.call1((values[1].as_deref(),))?.extract()?;
    let truncate: i32 = int.call1((values[2].as_deref(),))?.extract()?;
    Ok(Some((n, truncate)))
}

/// The settings of the eager evaluation of `__repr__` and `_repr_html_`.
const EAGER_EVAL: [&str; 3] = [
    "spark.sql.repl.eagerEval.enabled",
    "spark.sql.repl.eagerEval.maxNumRows",
    "spark.sql.repl.eagerEval.truncate",
];

/// The relation of a DataFrame as its pickle carries it, the description of the error of a plan
/// that does not build instead, and the holes of the functions of its UDFs (see
/// `PyDataFrame::pickled_relation`).
type PickledRelation<'py> = (
    Bound<'py, pyo3::types::PyBytes>,
    Option<Bound<'py, PyTuple>>,
    Bound<'py, PyList>,
);

/// A DataFrame of the drop-in as its methods read it, as PySpark's methods read `self._plan` and
/// `self._session` at each call, through the attributes, of the instance dict, of a descriptor or
/// of a `__getattr__` of a subclass ([`crate::state::attr`]): the plan object of the drop-in that
/// holds the plan of the DataFrame, of the core, and the parts of it, which is its `_plan`, the
/// plan object of a DataFrame of the drop-in, or else the plan object that the DataFrame made of
/// its `_plan`, a plan of the Python client (see `PyDataFrame::converted`). A method that makes a
/// plan, as PySpark's `DataFrame(plan.Project(self._plan, ...), session=self._session)`, reads
/// this and no more: the session of `_session` is resolved by its requests ([`PlanFrame::request`]),
/// as PySpark's read `self._session.client` then, but for a `_plan` of the Python client, which is
/// built of the client of `_session` when it is read, as the plans of the core are made of the
/// plans of their inputs.
pub(crate) struct PlanFrame<'py> {
    /// The object of the DataFrame.
    slf: Bound<'py, PyDataFrame>,
    /// Its native state.
    this: PyRef<'py, PyDataFrame>,
    /// Its `_plan`.
    plan_value: Bound<'py, PyAny>,
    /// The plan object of the drop-in that holds its plan.
    plan: PyRef<'py, PyNativePlan>,
    /// Its `_session`.
    session: Bound<'py, PyAny>,
    /// The session of `_session`, once it was resolved: of the build of a `_plan` of the Python
    /// client, of which `plan` is made.
    core_session: Option<spark_connect::session::SparkSession>,
    /// The plan of `plan` in the session of the plan of the DataFrame of the method of which this
    /// one is the other input (see [`PlanFrame::same_session`]).
    renumbered: Option<DataFrame>,
}

impl std::ops::Deref for PlanFrame<'_> {
    type Target = PyDataFrame;

    fn deref(&self) -> &PyDataFrame {
        &self.this
    }
}

impl<'py> PlanFrame<'py> {
    /// The DataFrame `slf` as its method reads it, of its `_plan` and then its `_session`, as
    /// PySpark's methods read them, with their errors: the `AttributeError` of one that is not
    /// set, as of a DataFrame whose `__init__` did not run, of the `__getattr__` of the DataFrame,
    /// as PySpark's. Of the plan object of a DataFrame of the drop-in, that object, of any session;
    /// of a plan of the Python client, the plan object of its build of the client of `_session`
    /// ([`request_session`]), which the DataFrame keeps while its `_plan` and that session are the
    /// same, as the core keeps the relation of that build.
    pub(crate) fn of(slf: &Bound<'py, PyDataFrame>) -> PyResult<Self> {
        let py = slf.py();
        let plan_value = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_plan"))?;
        let session = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_session"))?;
        let this = slf.try_borrow()?;
        let (plan, core_session) = match own_plan_object(&plan_value)? {
            Some(own) => (own.try_borrow()?, None),
            None => {
                let core_session = request_session(&session)?;
                let plan = this.conversion_of(&plan_value, &session, &core_session)?;
                (plan, Some(core_session))
            }
        };
        Ok(PlanFrame {
            slf: slf.clone(),
            this,
            plan_value,
            plan,
            session,
            core_session,
            renumbered: None,
        })
    }

    /// The plan of this DataFrame, of the core, of which its methods make theirs: of the session
    /// of its plan object, whose plan IDs it has, which the requests of a DataFrame of another
    /// session give it there (see [`PlanFrame::request`]).
    pub(crate) fn plan(&self) -> &DataFrame {
        self.renumbered.as_ref().unwrap_or(&self.plan.dataframe)
    }

    /// This DataFrame as its request reads it, of the session of the client of its `_session`,
    /// as PySpark's requests read `self._plan.to_proto(self._session.client)`: the DataFrame of
    /// the core of its requests, of its plan, which has its plan IDs in that session (see
    /// `PyDataFrame::conversion_of`), of that session and of the caches of the DataFrame.
    pub(crate) fn request(self) -> PyResult<Frame<'py>> {
        let core_session = match &self.core_session {
            Some(session) => session.clone(),
            None => request_session(&self.session)?,
        };
        let request_plan = match self.plan.dataframe.is_numbered_in(&core_session) {
            true => None,
            false => Some(self.this.conversion_of(
                &self.plan_value,
                &self.session,
                &core_session,
            )?),
        };
        let source = request_plan.as_deref().unwrap_or(&*self.plan);
        let dataframe = source
            .dataframe
            .with_session_and_caches(core_session, &self.this.caches);
        Ok(Frame {
            base: self,
            request_plan,
            dataframe,
        })
    }

    /// The object of the DataFrame.
    pub(crate) fn slf(&self) -> &Bound<'py, PyDataFrame> {
        &self.slf
    }

    /// The `_plan` of the DataFrame, as its method read it.
    pub(crate) fn plan_object(&self) -> &Bound<'py, PyAny> {
        &self.plan_value
    }

    /// The `_session` of the DataFrame, as its method read it.
    pub(crate) fn session_object(&self) -> &Bound<'py, PyAny> {
        &self.session
    }

    /// The new DataFrame of `dataframe`, a DataFrame of the core of a method of this one, as
    /// PySpark's methods make `DataFrame(plan, session=self._session)`: of the `_session` of this
    /// DataFrame.
    pub(crate) fn derived(&self, dataframe: DataFrame) -> NewDataFrame {
        PyDataFrame::new(dataframe).with_session(self.session.clone().unbind())
    }
}

/// A DataFrame of the drop-in as its request reads it (see [`PlanFrame::request`]): the DataFrame
/// of the core of its requests, of its plan, of the session of `_session` and of the caches of the
/// DataFrame. The calls of it without the GIL name the parts of the plan object of that plan
/// ([`Frame::parts`]), as the holder of the plan.
pub(crate) struct Frame<'py> {
    /// The DataFrame as its methods read it.
    base: PlanFrame<'py>,
    /// The plan object that the DataFrame made of its plan in the session of its request, when
    /// that is not the session of its plan object.
    request_plan: Option<PyRef<'py, PyNativePlan>>,
    /// The DataFrame of the core of its requests.
    pub(crate) dataframe: DataFrame,
}

impl<'py> std::ops::Deref for Frame<'py> {
    type Target = PlanFrame<'py>;

    fn deref(&self) -> &PlanFrame<'py> {
        &self.base
    }
}

impl<'py> Frame<'py> {
    /// The DataFrame `slf` as its request reads it, of its `_plan` and then its `_session`, as
    /// PySpark's requests read `self._plan.to_proto(self._session.client)` (see [`PlanFrame::of`]
    /// and [`PlanFrame::request`]).
    pub(crate) fn of(slf: &Bound<'py, PyDataFrame>) -> PyResult<Self> {
        PlanFrame::of(slf)?.request()
    }

    /// The parts of the plan of the requests, which its plan object holds, that the calls without
    /// the GIL of the requests name (see [`crate::gc`]).
    pub(crate) fn parts(&self) -> &crate::gc::Parts {
        &self
            .request_plan
            .as_deref()
            .unwrap_or(&*self.base.plan)
            .parts
    }
}

/// The DataFrame of the core of the plan of `slf`, as PySpark's methods that read `self._plan` and
/// not `self._session` read it, such as its `_col` and the star of `df["*"]`, whose plan ID they
/// take: of the plan object of a DataFrame of the drop-in, its own, of its session; of another
/// `_plan`, the plan of `slf` (see `PlanFrame::of`), which builds it.
fn plan_dataframe(slf: &Bound<'_, PyDataFrame>) -> PyResult<DataFrame> {
    let py = slf.py();
    let plan = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_plan"))?;
    if let Some(own) = own_plan_object(&plan)? {
        return Ok(own.try_borrow()?.dataframe.clone());
    }
    Ok(PlanFrame::of(slf)?.plan().clone())
}

/// The session of the requests of a DataFrame of `session`, its `_session`, of its `client`, as
/// PySpark's requests read `self._session.client` ([`crate::state::session_of_value`]): of a
/// session object of the drop-in, whose `client` is its `_client`, of its `__dict__` at once.
fn request_session(session: &Bound<'_, PyAny>) -> PyResult<spark_connect::session::SparkSession> {
    let py = session.py();
    if let Ok(object) = session.cast_exact::<crate::session::PySparkSession>() {
        let fields = session.getattr(pyo3::intern!(py, "__dict__"))?;
        if let Some(client) = fields
            .cast::<PyDict>()?
            .get_item(pyo3::intern!(py, "_client"))?
        {
            if let Some(own) = object.try_borrow()?.session_of_own_client(&client) {
                return Ok(own);
            }
        }
    }
    crate::state::session_of_value(session, pyo3::intern!(py, "client"))
}

/// A new DataFrame of the drop-in of `plan`, a DataFrame of the core that a method or a
/// constructor made, as PySpark makes `DataFrame(plan, session)`: its native state, of the caches
/// of `plan`, and, once it is a Python object, its `_plan`, a new plan object of `plan` (see
/// [`new_plan_object`]), and its `_session`, the `_session` of the DataFrame of the method, or
/// else the object of the session of `plan`, as PySpark's `__init__` sets them.
pub(crate) struct NewDataFrame {
    state: PyDataFrame,
    plan: DataFrame,
    session: Option<Py<PyAny>>,
    /// The `_cached_schema` that the method sets on the new DataFrame, as PySpark's
    /// `res._cached_schema = ...` of the methods that keep the schema of their input, of `to` and
    /// of the map functions.
    cached_schema: Option<Py<PyAny>>,
}

impl std::ops::Deref for NewDataFrame {
    type Target = PyDataFrame;

    fn deref(&self) -> &PyDataFrame {
        &self.state
    }
}

impl std::ops::DerefMut for NewDataFrame {
    fn deref_mut(&mut self) -> &mut PyDataFrame {
        &mut self.state
    }
}

impl NewDataFrame {
    /// The DataFrame of the core of this new DataFrame.
    pub(crate) fn dataframe(&self) -> &DataFrame {
        &self.plan
    }

    /// This new DataFrame, whose `_session` is `session`, the object that PySpark's
    /// `DataFrame(plan, session)` of its maker is given, which its requests resolve.
    pub(crate) fn with_session(mut self, session: Py<PyAny>) -> Self {
        self.session = Some(session);
        self
    }

    /// This new DataFrame, whose `_cached_schema` is `schema`, as PySpark's method sets it.
    pub(crate) fn with_cached_schema(mut self, schema: Option<Py<PyAny>>) -> Self {
        self.cached_schema = schema;
        self
    }

    /// The Python object of this new DataFrame.
    pub(crate) fn into_py(self, py: Python<'_>) -> PyResult<Py<PyDataFrame>> {
        Ok(self.into_pyobject(py)?.unbind())
    }
}

impl<'py> IntoPyObject<'py> for NewDataFrame {
    type Target = PyDataFrame;
    type Output = Bound<'py, PyDataFrame>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyDataFrame>> {
        let session = match self.session {
            Some(session) => session.into_bound(py),
            None => crate::session::session_object(py, &self.plan.spark_session())?
                .into_bound(py)
                .into_any(),
        };
        // As PySpark's method, which makes the plan object before the DataFrame of it.
        let plan = new_plan_object(py, self.plan)?;
        let df = Bound::new(py, self.state)?;
        // As PySpark's `DataFrame.__init__`, in its order, in the `__dict__` of the new object, of
        // the class of the drop-in, of no descriptor of these.
        let fields = df.as_any().getattr(pyo3::intern!(py, "__dict__"))?;
        let fields = fields.cast::<PyDict>()?;
        init_entries(fields, plan.as_any(), &session)?;
        if let Some(schema) = self.cached_schema {
            // As PySpark's `res._cached_schema = ...` after the DataFrame is made.
            df.as_any()
                .setattr(pyo3::intern!(py, "_cached_schema"), schema)?;
        }
        Ok(df)
    }
}

/// Set the entries of PySpark's `DataFrame.__init__` in `fields`, the `__dict__` of a new
/// DataFrame, in its order: its `_plan` and `_session`, and the state of no request yet.
fn init_entries(
    fields: &Bound<'_, PyDict>,
    plan: &Bound<'_, PyAny>,
    session: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let py = fields.py();
    fields.set_item(pyo3::intern!(py, "_plan"), plan)?;
    fields.set_item(pyo3::intern!(py, "_session"), session)?;
    fields.set_item(pyo3::intern!(py, "_support_repr_html"), false)?;
    fields.set_item(pyo3::intern!(py, "_cached_schema"), py.None())?;
    fields.set_item(pyo3::intern!(py, "_cached_schema_serialized"), py.None())?;
    fields.set_item(pyo3::intern!(py, "_execution_info"), py.None())?;
    Ok(())
}

/// The class of the plan objects of the DataFrames of the drop-in, `_DataFramePlan` of
/// `pyspark.sql.connect.plan`, which subclasses `_NativePlan` and PySpark's `LogicalPlan`.
fn plan_class(py: Python<'_>) -> PyResult<&Bound<'_, pyo3::types::PyType>> {
    static PLAN_OF_DATAFRAME: pyo3::sync::PyOnceLock<Py<pyo3::types::PyType>> =
        pyo3::sync::PyOnceLock::new();
    PLAN_OF_DATAFRAME.import(py, "pyspark.sql.connect.plan", "_DataFramePlan")
}

/// The plan object of `dataframe`, a new DataFrame of the core, which a new DataFrame of the
/// drop-in holds as its `_plan`, as PySpark's DataFrame of a method holds the plan object that the
/// method made: an object of `_DataFramePlan` ([`plan_class`]), whose native base holds the plan
/// of the core and the parts of it, made with the GIL held, with the fields of PySpark's
/// `LogicalPlan` that `_DataFramePlan.__init__` sets, of the plan IDs of the plan, which it takes
/// now, as PySpark's `LogicalPlan.__init__` takes them, in the order in which the plans are made.
pub(crate) fn new_plan_object<'py>(
    py: Python<'py>,
    dataframe: DataFrame,
) -> PyResult<Bound<'py, PyNativePlan>> {
    let (plan_id, root_plan_id) = dataframe.plan_id_and_root();
    let plan = crate::reduce::new_of_class(plan_class(py)?, PyNativePlan::of(py, dataframe))?;
    // Of the `__dict__` of the new object, of a class of the drop-in, of no descriptor of these.
    let fields = plan.as_any().getattr(pyo3::intern!(py, "__dict__"))?;
    let fields = fields.cast::<PyDict>()?;
    fields.set_item(pyo3::intern!(py, "_child"), py.None())?;
    fields.set_item(
        pyo3::intern!(py, "_root_plan_id"),
        root_plan_id.unwrap_or(plan_id),
    )?;
    fields.set_item(pyo3::intern!(py, "_references"), PyList::empty(py))?;
    fields.set_item(
        pyo3::intern!(py, "_plan_id_with_rel"),
        root_plan_id.map(|_| plan_id),
    )?;
    Ok(plan)
}

/// The plan object of `plan`, if it is the plan object of a DataFrame of the drop-in
/// (`_DataFramePlan` itself, [`plan_class`]), whose native base holds the plan of the core: a
/// DataFrame of it is of that plan, as PySpark's `DataFrame(df._plan, session)` is of the plan
/// object of `df`. Of another class, a subclass included, it is a plan of the Python client, whose
/// `plan` builds its relation.
fn own_plan_object<'a, 'py>(
    plan: &'a Bound<'py, PyAny>,
) -> PyResult<Option<&'a Bound<'py, PyNativePlan>>> {
    if !plan.get_type().is(plan_class(plan.py())?) {
        return Ok(None);
    }
    Ok(plan.cast::<PyNativePlan>().ok())
}

impl PyDataFrame {
    /// The native state of a new DataFrame of `caches`, of no other answer yet: its `_plan` and
    /// its `_session` are set once it is a Python object, by `__init__` or by [`NewDataFrame`].
    fn of_caches(caches: spark_connect::dataframe::DataFrameCaches) -> Self {
        PyDataFrame {
            caches,
            converted: std::sync::Mutex::new(None),
            gc_ready: crate::gc::GcReady::new(),
            semantic_hash: std::sync::OnceLock::new(),
            is_empty: std::sync::OnceLock::new(),
            input_files: std::sync::OnceLock::new(),
            published: std::sync::Mutex::new(None),
        }
    }

    /// The new DataFrame of `dataframe`, of the caches of it, whose `_session` is the object of
    /// its session (see [`NewDataFrame`]): of the native state and, once it is a Python object,
    /// of the entries of its `__dict__`, which a value of the class alone does not have.
    #[allow(clippy::new_ret_no_self)]
    pub(crate) fn new(dataframe: DataFrame) -> NewDataFrame {
        NewDataFrame {
            state: PyDataFrame::of_caches(dataframe.caches()),
            plan: dataframe,
            session: None,
            cached_schema: None,
        }
    }

    /// Set `_execution_info`, as PySpark's actions set it, to the `ExecutionInfo` of the last
    /// execution of the core of `slf`, of its metrics and observed metrics, when that is not the
    /// one that it was set of last (see `published`): so an action sets it, and an assignment of
    /// it, as of the callback of a writer, stays until a later execution. Of the caches of the
    /// DataFrame, without a read of its `_plan` or its `_session`.
    pub(crate) fn publish_execution(slf: &Bound<'_, PyDataFrame>) -> PyResult<()> {
        let py = slf.py();
        let this = slf.try_borrow()?;
        // A copy of the observed metrics, of up to `spark_connect_proto::MAX_KEPT_DEPTH` levels.
        let Some(info) = crate::on_grown_stack(|| this.caches.execution_info()) else {
            return Ok(());
        };
        let current = this
            .published
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .is_some_and(|published| same_execution(published, &info));
        if current {
            return Ok(());
        }
        let (plan_metrics, observed) = execution_metrics(py, &info)?;
        let ei = py
            .import(pyo3::intern!(py, "pyspark.sql.metrics"))?
            .getattr(pyo3::intern!(py, "ExecutionInfo"))?
            .call1((plan_metrics, observed, info.operation_id.clone()))?;
        *this
            .published
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(info);
        drop(this);
        crate::state::assign(slf.as_any(), &[(pyo3::intern!(py, "_execution_info"), &ei)])
    }

    /// The plan object that this DataFrame made of `plan`, its `_plan`, for the requests of it in
    /// `session`, the session of `session_object`, its `_session`, by the identity of `plan` (see
    /// `converted`), which it makes at the first read after `_plan` or the session change, as
    /// PySpark's requests build `self._plan` of `self._session.client`: of the plan of the plan
    /// object of a DataFrame of another session, with the plan IDs that it has there
    /// (`DataFrame::of_plan_in`), or of a plan of the Python client, built of the client of
    /// `session_object` ([`dataframe_of_python_plan`]).
    fn conversion_of<'py>(
        &self,
        plan: &Bound<'py, PyAny>,
        session_object: &Bound<'py, PyAny>,
        session: &spark_connect::session::SparkSession,
    ) -> PyResult<PyRef<'py, PyNativePlan>> {
        let py = plan.py();
        let made = self
            .converted
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .filter(|(of, _)| of.bind(py).is(plan))
            .map(|(_, made)| made.bind(py).clone());
        if let Some(made) = made {
            let made = made.try_borrow()?;
            if made.dataframe.is_numbered_in(session) {
                return Ok(made);
            }
        }
        let dataframe = match own_plan_object(plan)? {
            Some(own) => own.try_borrow()?.dataframe.of_plan_in(session.clone()),
            None => dataframe_of_python_plan(plan, session_object, session.clone())?.0,
        };
        let made = Bound::new(py, PyNativePlan::of(py, dataframe))?;
        let previous = self
            .converted
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .replace((plan.clone().unbind(), made.clone().unbind()));
        // Dropped out of the lock, as dropping its references may run Python code, such as a
        // collection, whose `__traverse__` takes the lock.
        drop(previous);
        Ok(made.try_borrow()?)
    }
}

impl<'py> PlanFrame<'py> {
    /// The names of the columns of this DataFrame, as Rust callers need them, of its `columns`, as
    /// PySpark's methods read `self.columns`. Not a method of the Python class, as PySpark's
    /// DataFrame has none of this name: so `df.column_names` is the column of that name, of
    /// `__getattr__`.
    pub(crate) fn column_names(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        column_names_of(&self.slf, py)
    }

    /// Raises as PySpark's `self._df.schema` of the grouped data of this DataFrame, of which they
    /// read the schema, and gives it, as the core takes it.
    pub(crate) fn schema_read(&self, py: Python<'py>) -> PyResult<spark_connect::types::DataType> {
        crate::types::py_to_data_type(&schema_property(&self.slf, py)?)
    }

    /// `other`, a DataFrame of the session of this one, as PySpark's `_check_same_session` takes
    /// the DataFrame of a method: else `SESSION_NOT_SAME`, as for what is not a DataFrame, of
    /// `self._session.session_id != other._session.session_id`, of the `_session` of this one as
    /// its method read it, of the error of that comparison, such as that of the truth of a
    /// Column. Its plan is of the session of the plan of this DataFrame, which gives the plan of a
    /// method of both its plan IDs there.
    pub(crate) fn same_session(&self, other: &Bound<'py, PyAny>) -> PyResult<PlanFrame<'py>> {
        let py = self.slf.py();
        let not_same =
            || crate::column::raise_pyspark(py, "SessionNotSameException", "SESSION_NOT_SAME", &[]);
        let Ok(other) = other.cast::<PyDataFrame>() else {
            return Err(not_same());
        };
        let session_id = pyo3::intern!(py, "session_id");
        let other_session = crate::state::attr(other.as_any(), pyo3::intern!(py, "_session"))?;
        if self
            .session
            .getattr(session_id)?
            .ne(other_session.getattr(session_id)?)?
        {
            return Err(not_same());
        }
        let mut other = PlanFrame::of(other)?;
        let session = self.plan().spark_session();
        if !other.plan().is_numbered_in(&session) {
            other.renumbered = Some(other.plan().of_plan_in(session));
        }
        Ok(other)
    }
}

impl<'py> Frame<'py> {
    /// The schema of a request of the plan of this DataFrame, as PySpark's `_schema` requests it
    /// when its `_cached_schema` is None (see [`schema_entry`]): of no schema that the core cached,
    /// as PySpark requests the schema whenever `_cached_schema` is None, but of one in flight, so
    /// that concurrent first reads ask once.
    fn requested_schema(&self, py: Python<'py>) -> PyResult<spark_connect::types::DataType> {
        self.caches.forget_schema();
        let dataframe = &self.dataframe;
        interruptible_of(py, &dataframe.spark_session(), &[self.parts()], || {
            dataframe.schema()
        })
    }

    /// The pandas DataFrame of the result, as PySpark's `_to_pandas` makes it: the plan is built
    /// first (`to_proto`), so a plan that does not build raises before any request, and the
    /// first plan of the session reads the plan compression settings then; then
    /// `SparkConnectClient.to_pandas` reads the settings of the conversion, in one request,
    /// and sends the plan; the Arrow result with the schema that the server sends is converted
    /// by `pyspark.sql.connect.client.core._to_pandas`. The DataFrame keeps the information of the
    /// execution once the result is converted, as `_to_pandas` sets `_execution_info` after
    /// `SparkConnectClient.to_pandas` returns: a conversion that fails leaves `executionInfo` as
    /// it was.
    pub(crate) fn to_pandas(
        &self,
        py: Python<'_>,
        struct_handling_mode: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let session = self.dataframe.spark_session();
        let dataframe = &self.dataframe;
        let execution = interruptible_of(py, &session, &[self.parts()], || {
            let execution = dataframe.prepare_execution().map_err(Box::new)?;
            // As PySpark's `to_pandas`, which logs the plan before it reads its settings.
            execution.log_plan_of_conversion().map_err(Box::new)?;
            Ok::<_, Box<spark_connect_core::error::SparkError>>(execution)
        })?;
        let settings = crate::interrupt::interruptible_without_parts(py, &session, || {
            session.conf().get_many(&[
                "spark.sql.session.timeZone",
                "spark.sql.execution.pandas.structHandlingMode",
                "spark.sql.execution.arrow.pyspark.selfDestruct.enabled",
            ])
        })?;
        let (batches, schema, info) = interruptible_of(py, &session, &[self.parts()], || {
            execution
                .collect_record_batches_and_info(check_schema_at_receipt)
                .map_err(Box::new)
        })?;
        check_schema_conversion(py, schema.as_ref())?;
        // The conversion of pyarrow and pandas, of calls of Python of many frames, on a stack with
        // room for them, of a thread of a small stack too.
        crate::on_grown_stack(|| {
            self.to_pandas_of(py, batches, schema, info, settings, struct_handling_mode)
        })
    }

    /// The pandas DataFrame of the batches of the result of [`Self::to_pandas`], of its settings.
    fn to_pandas_of(
        &self,
        py: Python<'_>,
        batches: Vec<arrow::record_batch::RecordBatch>,
        schema: Option<spark_connect::types::DataType>,
        info: spark_connect::session::ExecutionInfo,
        settings: Vec<Option<String>>,
        struct_handling_mode: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let mut table = crate::pyarrow_export::record_batches_to_pyarrow(py, batches)?;
        // As SparkConnectClient._execute_and_fetch with self_destruct, which copies each column
        // of each batch into its own allocation before the conversion, with its errors.
        if settings[2].as_deref() == Some("true") {
            table = py
                .import("pyspark.sql.connect.client.core")?
                .call_method1("_self_destruct_table", (table,))?
                .unbind();
        }
        let schema = match &schema {
            Some(schema) => crate::types::schema_to_py(py, schema)?,
            None => py.None(),
        };
        let (metrics, observed) = execution_metrics(py, &info)?;
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("timezone", &settings[0])?;
        match struct_handling_mode {
            Some(mode) => kwargs.set_item("struct_handling_mode", mode.str()?)?,
            None => kwargs.set_item("struct_handling_mode", &settings[1])?,
        }
        kwargs.set_item("self_destruct", &settings[2])?;
        let pdf = py.import("pyspark.sql.connect.client.core")?.call_method(
            "_to_pandas",
            (table, schema, metrics, observed),
            Some(&kwargs),
        )?;
        self.dataframe.keep_execution_info(info);
        self.publish_execution(py)?;
        Ok(pdf.unbind())
    }

    /// The table of `show` and `_show_string`, for the arguments of `function`.
    fn show_table(
        &self,
        py: Python<'_>,
        function: &str,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<String> {
        crate::options::check_keyword_keys(kwargs)?;
        let [n, truncate, vertical] = bind_arguments(
            function,
            args,
            kwargs,
            ["n", "truncate", "vertical"],
            [
                20i32.into_pyobject(py)?.into_any(),
                pyo3::types::PyBool::new(py, true).to_owned().into_any(),
                pyo3::types::PyBool::new(py, false).to_owned().into_any(),
            ],
        )?;
        if n.is_instance_of::<pyo3::types::PyBool>() || !n.is_instance_of::<pyo3::types::PyInt>() {
            return Err(not_expected_type(&n, "n", "int"));
        }
        if !vertical.is_instance_of::<pyo3::types::PyBool>() {
            return Err(not_expected_type(&vertical, "vertical", "bool"));
        }
        let truncate =
            if truncate.is_instance_of::<pyo3::types::PyBool>() && truncate.is_truthy()? {
                20i32.into_pyobject(py)?.into_any()
            } else {
                // As `int(truncate)`, whose ValueError is PySpark's NOT_EXPECTED_TYPE.
                match py.import("builtins")?.getattr("int")?.call1((&truncate,)) {
                    Ok(value) => value,
                    Err(err) if err.is_instance_of::<pyo3::exceptions::PyValueError>(py) => {
                        return Err(not_expected_type(&truncate, "truncate", "bool"));
                    }
                    Err(err) => return Err(err),
                }
            };
        // The proto's int32 fields, as PySpark's `ShowString` sets them: `num_rows`, then
        // `truncate`.
        let n = proto_int::<i32>(&n, "spark.connect.ShowString.num_rows")?;
        let truncate = proto_int::<i32>(&truncate, "spark.connect.ShowString.truncate")?;
        let vertical = vertical.extract::<bool>()?;
        let dataframe = &self.dataframe;
        interruptible_of(py, &dataframe.spark_session(), &[self.parts()], || {
            dataframe
                .show_string(n, truncate, vertical)
                .map_err(Box::new)
        })
    }
}

impl<'py> PlanFrame<'py> {
    /// The relation of this DataFrame as its pickle carries it, the error of a plan that does not
    /// build instead, and the holes of the functions of its UDFs: the relation is built of holes
    /// in place of their commands, as PySpark's pickle carries the functions of its plan objects
    /// ([`crate::udf_command::PickleHoles`]); a plan that does not build, of a value that PySpark
    /// converts when it builds the plan, pickles of its error ([`crate::column::described_error`]),
    /// and a plan of the Python client whose build raised, of that plan, which the unpickle builds
    /// again, as PySpark's does.
    fn pickled_relation(&self, py: Python<'py>) -> PyResult<PickledRelation<'py>> {
        pickled_relation_of(py, self.plan())
    }

    /// `result`, of a plan over this DataFrame of the same schema, of the `_cached_schema` of this
    /// one, the same object, as PySpark's `alias`, `filter`, `limit`, `sort`, the repartitions and
    /// the other methods that keep the schema of their input set
    /// `res._cached_schema = self._cached_schema`, whatever it is.
    pub(crate) fn of_same_schema(&self, result: NewDataFrame) -> PyResult<NewDataFrame> {
        let py = self.slf.py();
        let schema = self
            .slf
            .as_any()
            .getattr(pyo3::intern!(py, "_cached_schema"))?;
        Ok(result.with_cached_schema(Some(schema.unbind())))
    }

    /// The state of the reduce of this DataFrame, as PySpark's `__reduce__` gives it, of its
    /// `_support_repr_html` and its `_cached_schema`, as they are.
    fn reduce_state(&self) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let py = self.slf.py();
        let slf = self.slf.as_any();
        Ok((
            crate::state::attr(slf, pyo3::intern!(py, "_support_repr_html"))?,
            crate::state::attr(slf, pyo3::intern!(py, "_cached_schema"))?,
        ))
    }

    /// A copy of this DataFrame, of `plan` and `session`, as PySpark's copies of a DataFrame are
    /// `DataFrame(plan, session)` of its reduce, of `state`, its `_support_repr_html` and its
    /// `_cached_schema` (see `reduce_state`): as PySpark's reconstruction of the copy looks up its
    /// `__setstate__`, of its `__getattr__`, which reads its columns, of a request of its schema,
    /// as it has none, which sets its `_cached_schema` and `_cached_schema_serialized` (see
    /// `schema_entry`), of the errors of that read, and then sets the state in its `__dict__`.
    /// Its other answers are its own, as PySpark keeps them for each DataFrame.
    fn copy_of(
        &self,
        py: Python<'py>,
        plan: &Bound<'py, PyAny>,
        session: &Bound<'py, PyAny>,
        state: (Bound<'py, PyAny>, Bound<'py, PyAny>),
    ) -> PyResult<Bound<'py, PyDataFrame>> {
        let copy = Bound::new(
            py,
            PyDataFrame::of_caches(spark_connect::dataframe::DataFrameCaches::new()),
        )?;
        let fields = copy.as_any().getattr(pyo3::intern!(py, "__dict__"))?;
        init_entries(fields.cast::<PyDict>()?, plan, session)?;
        property_schema(&copy)?.columns(py)?;
        let (support_repr_html, cached_schema) = state;
        fields.set_item(pyo3::intern!(py, "_support_repr_html"), support_repr_html)?;
        fields.set_item(pyo3::intern!(py, "_cached_schema"), cached_schema)?;
        Ok(copy)
    }

    /// Set `_execution_info` of the execution that a request of this DataFrame made (see
    /// `PyDataFrame::publish_execution`).
    pub(crate) fn publish_execution(&self, py: Python<'py>) -> PyResult<()> {
        let _ = py;
        PyDataFrame::publish_execution(&self.slf)
    }

    /// The column at the 0-based index `index`, as PySpark's `df[index]`: `col` of the name of
    /// the column at that index of `columns`, of Python's list indexing, and of the origin of
    /// `col`.
    fn column_at(&self, py: Python<'_>, index: i64) -> PyResult<spark_connect::column::Column> {
        let columns = self.column_names(py)?;
        let name: String = PyList::new(py, columns)?
            .as_any()
            .get_item(index)?
            .extract()?;
        let _origin = crate::origin::with_origin("col")?;
        Ok(spark_connect::functions::col(&name))
    }

    /// The grouping columns of `groupBy`, `rollup`, `cube` and `groupingSets`, as PySpark's
    /// `_to_cols` of ordinals: Columns, column names and the 1-based ordinals of columns, or
    /// a list or tuple of them.
    fn grouping_columns(
        &self,
        py: Python<'_>,
        cols: Vec<Bound<'_, PyAny>>,
        arg_name: &str,
    ) -> PyResult<Vec<spark_connect::column::Column>> {
        self.grouping_column_list(py, unwrap_sequence(cols)?, arg_name)
    }

    /// The grouping columns of `cols`, each a Column, a column name or the 1-based ordinal of
    /// a column, as PySpark's `_to_cols` of ordinals takes the items of a sequence.
    fn grouping_column_list(
        &self,
        py: Python<'_>,
        cols: Vec<Bound<'_, PyAny>>,
        arg_name: &str,
    ) -> PyResult<Vec<spark_connect::column::Column>> {
        use pyo3::types::{PyBool, PyString};
        cols.iter()
            .map(|c| {
                if let Ok(column) = c.cast::<PyColumn>() {
                    return crate::column::column_of(column);
                }
                if c.is_instance_of::<PyString>() {
                    // As PySpark's `F._to_col`, by `col`, of its origin.
                    let _origin = crate::origin::with_origin("col")?;
                    return crate::plan_args::col(c);
                }
                if c.is_instance_of::<PyInt>() && !c.is_instance_of::<PyBool>() {
                    let ordinal: i64 = c.extract()?;
                    if ordinal < 1 {
                        return Err(crate::column::raise_pyspark(
                            py,
                            "PySparkIndexError",
                            "INDEX_NOT_POSITIVE",
                            &[("index", &ordinal.to_string())],
                        ));
                    }
                    return self.column_at(py, ordinal - 1);
                }
                Err(not_expected_type(c, arg_name, "Column, int or str"))
            })
            .collect()
    }

    /// The sort orders of `sort`, `orderBy` and `sortWithinPartitions`, as PySpark's
    /// `_preapare_cols_for_sort`: Columns, column names and the 1-based ordinals of columns,
    /// descending for a negative one, or a list or tuple of them, each descending for a
    /// false `ascending`, or for a false one of a list of them.
    fn sort_columns(
        &self,
        py: Python<'_>,
        cols: Vec<Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<spark_connect::expression::Expression>> {
        use pyo3::types::{PyBool, PyString};
        crate::options::check_keyword_keys(kwargs)?;
        if cols.is_empty() {
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkValueError",
                "CANNOT_BE_EMPTY",
                &[("item", "cols")],
            ));
        }
        let mut columns = Vec::new();
        for c in unwrap_sequence(cols)? {
            let column = if c.is_instance_of::<PyInt>() && !c.is_instance_of::<PyBool>() {
                let ordinal: i64 = c.extract()?;
                match ordinal {
                    0 => {
                        return Err(crate::column::raise_pyspark(
                            py,
                            "PySparkIndexError",
                            "ZERO_INDEX",
                            &[],
                        ));
                    }
                    ordinal if ordinal > 0 => self.column_at(py, ordinal - 1)?,
                    ordinal => self.column_at(py, -ordinal - 1)?.desc(),
                }
            } else if let Ok(column) = c.cast::<PyColumn>() {
                crate::column::column_of(column)?
            } else if c.is_instance_of::<PyString>() {
                // As PySpark's `F._to_col`, by `col`, of its origin.
                let _origin = crate::origin::with_origin("col")?;
                crate::plan_args::col(&c)?
            } else {
                return Err(not_expected_type(&c, "col", "Column, int or str"));
            };
            columns.push(column);
        }
        let ascending = match kwargs {
            Some(kwargs) => kwargs.get_item("ascending")?,
            None => None,
        };
        if let Some(ascending) = ascending {
            if ascending.is_instance_of::<PyInt>() {
                if !ascending.is_truthy()? {
                    columns = columns.into_iter().map(|c| c.desc()).collect();
                }
            } else if ascending.is_instance_of::<PyList>() {
                // As PySpark's zip of them, which keeps a column for each flag.
                let flags: Vec<bool> = ascending
                    .try_iter()?
                    .map(|flag| flag?.is_truthy())
                    .collect::<PyResult<_>>()?;
                columns = columns
                    .into_iter()
                    .zip(flags)
                    .map(|(c, asc)| if asc { c } else { c.desc() })
                    .collect();
            } else {
                return Err(not_expected_type(
                    &ascending,
                    "ascending",
                    "Column, int or str",
                ));
            }
        }
        Ok(columns.iter().map(|c| c.expression().clone()).collect())
    }
}

/// PySpark's `_schema` of `slf`: its `_cached_schema`, as it is; of none, the schema of a request
/// of the plan, of the server ([`Frame::requested_schema`]), as PySpark's client converts it, of
/// the errors of that conversion (see `check_python_schema`), which it sets as its
/// `_cached_schema`, and of which it sets the pickle, `CPickleSerializer().dumps(self._schema)`,
/// as its `_cached_schema_serialized`, None of a schema that does not pickle, of PySpark's
/// warning. Its `_plan` and its `_session` are read by the request, as PySpark's
/// `self._plan.to_proto(self._session.client)`, after `_cached_schema`.
fn schema_entry<'py>(slf: &Bound<'py, PyDataFrame>) -> PyResult<Bound<'py, PyAny>> {
    let py = slf.py();
    let cached = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_cached_schema"))?;
    if !cached.is_none() {
        return Ok(cached);
    }
    let schema = Frame::of(slf)?.requested_schema(py)?;
    let slf = slf.as_any();
    check_python_schema(py, &schema)?;
    let object = crate::types::schema_to_py(py, &schema)?.into_bound(py);
    crate::state::assign(slf, &[(pyo3::intern!(py, "_cached_schema"), &object)])?;
    let schema = slf.getattr(pyo3::intern!(py, "_schema"))?;
    // Of many levels of a schema of many levels, on a stack with room for them, of a thread of a
    // small stack too.
    let dumped = crate::on_grown_stack(|| {
        py.import(pyo3::intern!(py, "pyspark.serializers"))?
            .getattr(pyo3::intern!(py, "CPickleSerializer"))?
            .call0()?
            .call_method1(pyo3::intern!(py, "dumps"), (schema,))
    });
    let serialized = match dumped {
        Ok(serialized) => serialized,
        Err(err) if err.is_instance_of::<pyo3::exceptions::PyException>(py) => {
            schema_pickle_warning(py, "dumps", &err)?;
            py.None().into_bound(py)
        }
        Err(err) => return Err(err),
    };
    crate::state::assign(
        slf,
        &[(pyo3::intern!(py, "_cached_schema_serialized"), &serialized)],
    )?;
    crate::state::attr(slf, pyo3::intern!(py, "_cached_schema"))
}

/// PySpark's `schema` of `slf`: of `self._schema`, the load of `_cached_schema_serialized` when it
/// is set, which gives new objects of the schema of its pickle, else, or when the load raises, of
/// PySpark's warning, a deep copy of `_schema`.
fn schema_property<'py>(
    slf: &Bound<'py, PyDataFrame>,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let slf = slf.as_any();
    let schema = slf.getattr(pyo3::intern!(py, "_schema"))?;
    let serialized = crate::state::attr(slf, pyo3::intern!(py, "_cached_schema_serialized"))?;
    if !serialized.is_none() {
        let loaded = crate::on_grown_stack(|| {
            py.import(pyo3::intern!(py, "pyspark.serializers"))?
                .getattr(pyo3::intern!(py, "CPickleSerializer"))?
                .call0()?
                .call_method1(pyo3::intern!(py, "loads"), (&serialized,))
        });
        match loaded {
            Ok(loaded) => return Ok(loaded),
            Err(err) if err.is_instance_of::<pyo3::exceptions::PyException>(py) => {
                schema_pickle_warning(py, "loads", &err)?;
            }
            Err(err) => return Err(err),
        }
    }
    crate::on_grown_stack(|| {
        py.import(pyo3::intern!(py, "copy"))?
            .call_method1(pyo3::intern!(py, "deepcopy"), (schema,))
    })
}

/// The schema that the properties `columns` and `dtypes` of `slf` read, as PySpark's read
/// `self._schema`: of the property `_schema`, which, when the plan fails to build with an
/// `AttributeError`, as of a join on a tuple, gives the object that Python then gives of
/// `self._schema`, as PySpark's property raises it: of the `__getattr__` of the DataFrame (or of
/// its Python subclass), the Column of `_schema`.
fn property_schema<'py>(slf: &Bound<'py, PyDataFrame>) -> PyResult<PropertySchema<'py>> {
    let py = slf.py();
    Ok(PropertySchema(
        slf.as_any().getattr(pyo3::intern!(py, "_schema"))?,
    ))
}

/// The names of the columns of `slf`, as Rust callers need them, of its `columns`, as PySpark's
/// methods read `self.columns` ([`property_schema`]).
pub(crate) fn column_names_of(
    slf: &Bound<'_, PyDataFrame>,
    py: Python<'_>,
) -> PyResult<Vec<String>> {
    property_schema(slf)?.columns(py)?.extract()
}

/// The arguments of a method of `*cols`, the items of a single sequence of them, as PySpark's
/// `_to_cols` takes them: a sequence that is not an int, a str or a Column, which is then
/// `tuple(cols[0])`. Any `collections.abc.Sequence` counts, as a range, bytes or a user class.
fn unwrap_sequence(cols: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<Bound<'_, PyAny>>> {
    if let [only] = cols.as_slice() {
        if is_sequence_argument(only)? {
            return only.try_iter()?.collect();
        }
    }
    Ok(cols)
}

/// Whether `only`, the single argument of `*cols`, is a sequence that PySpark unpacks: an
/// `isinstance` of `collections.abc.Sequence`, except for an int, a str and a Column.
fn is_sequence_argument(only: &Bound<'_, PyAny>) -> PyResult<bool> {
    use pyo3::types::{PyInt, PyString, PyTuple};
    if only.is_instance_of::<PyInt>()
        || only.is_instance_of::<PyString>()
        || only.is_instance_of::<PyColumn>()
    {
        return Ok(false);
    }
    if only.is_instance_of::<PyList>() || only.is_instance_of::<PyTuple>() {
        return Ok(true);
    }
    let sequence = only.py().import("collections.abc")?.getattr("Sequence")?;
    only.is_instance(&sequence)
}

/// PySpark's check of the `numPartitions` of `repartition`, `repartitionByRange` and `coalesce`,
/// `not numPartitions > 0`: `VALUE_NOT_POSITIVE`, or the `TypeError` of the comparison, as for
/// None in `coalesce`. The count is converted after, as the proto's int32 ([`proto_int`]).
fn positive_partitions(num_partitions: &Bound<'_, PyAny>) -> PyResult<()> {
    if num_partitions.gt(0)? {
        return Ok(());
    }
    Err(not_positive_partitions(num_partitions)?)
}

/// PySpark's `VALUE_NOT_POSITIVE` error of `numPartitions`, of its `str`.
fn not_positive_partitions(num_partitions: &Bound<'_, PyAny>) -> PyResult<PyErr> {
    Ok(crate::column::raise_pyspark(
        num_partitions.py(),
        "PySparkValueError",
        "VALUE_NOT_POSITIVE",
        &[
            ("arg_name", "numPartitions"),
            ("arg_value", &num_partitions.str()?.to_cow()?),
        ],
    ))
}

/// The `subset` of `dropDuplicates` and `dropDuplicatesWithinWatermark`, as PySpark's: None,
/// or a list or tuple of column names, which its `Deduplicate` plan sets when it builds the plan
/// (`args`).
fn deduplicate_subset(
    args: &mut crate::plan_args::PlanArgs,
    subset: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Vec<String>>> {
    let Some(subset) = subset else {
        return Ok(None);
    };
    if !(subset.is_instance_of::<PyList>() || subset.is_instance_of::<PyTuple>()) {
        return Err(not_expected_type(subset, "subset", "list or tuple"));
    }
    subset
        .try_iter()?
        .map(|column| {
            let column = column?;
            if column.is_instance_of::<pyo3::types::PyString>() {
                args.str(&column)
            } else {
                Err(not_expected_type(&column, "subset", "str"))
            }
        })
        .collect::<PyResult<Vec<String>>>()
        .map(Some)
}

/// Helper to convert arguments to a vector of Columns.
pub(crate) fn to_column_list(
    args: Vec<Bound<'_, PyAny>>,
) -> PyResult<Vec<spark_connect::column::Column>> {
    // pyspark unpacks a single sequence argument: df.select(["a", "b"]) == df.select("a","b").
    // (A single Column/str stays as-is.)
    to_columns_as_given(unwrap_sequence(args)?)
}

/// The columns of `args`, each a Column or a column name, else `NOT_EXPECTED_TYPE`, without
/// unpacking a single list: as PySpark's `_to_cols(cols)` of the tuple of `*cols` itself, which
/// `repartition` and `repartitionByRange` call, so that a list among their columns is an error.
pub(crate) fn to_columns_as_given(
    args: Vec<Bound<'_, PyAny>>,
) -> PyResult<Vec<spark_connect::column::Column>> {
    let mut cols = vec![];
    for arg in args {
        // Try as PyColumn first
        if let Ok(pycol) = arg.extract::<Bound<'_, PyColumn>>() {
            cols.push(crate::column::column_of(&pycol)?);
        } else if arg.is_instance_of::<pyo3::types::PyString>() {
            // A column name, as PySpark's `F._to_col`, by `col`, of its origin.
            let _origin = crate::origin::with_origin("col")?;
            cols.push(crate::plan_args::col(&arg)?);
        } else {
            // As PySpark's `_to_cols`.
            return Err(not_expected_type(&arg, "cols", "Column or str"));
        }
    }
    Ok(cols)
}

/// The join types that `DataFrame.join` accepts, as the message of its `UNSUPPORTED_JOIN_TYPE`.
const JOIN_TYPES_SUPPORTED: &str = "'inner', 'outer', 'full', 'fullouter', 'full_outer', 'leftouter', 'left', 'left_outer', 'rightouter', 'right', 'right_outer', 'leftsemi', 'left_semi', 'semi', 'leftanti', 'left_anti', 'anti', 'cross'";

/// The join types that `DataFrame.lateralJoin` accepts, as the message of its
/// `UNSUPPORTED_JOIN_TYPE`.
const LATERAL_JOIN_TYPES_SUPPORTED: &str = "'inner', 'leftouter', 'left', 'left_outer', 'cross'";

/// Map a pyspark `how` to a JoinType, as the connect client's `plan.Join` (or `plan.LateralJoin`
/// when `lateral`) maps it. The `how` is lowercased with its underscores removed, and no `how`
/// is an inner join. One that is not accepted raises the `AnalysisException` of
/// `UNSUPPORTED_JOIN_TYPE` that PySpark raises locally, naming the normalized `how` (or the
/// `str()` of a `how` that is not a str).
fn parse_join_type(
    py: Python<'_>,
    how: Option<&Bound<'_, PyAny>>,
    lateral: bool,
) -> PyResult<JoinType> {
    let Some(how) = how.filter(|how| !how.is_none()) else {
        return Ok(JoinType::Inner);
    };
    // As PySpark's `how.lower().replace("_", "")` of a str, in Python: of a str that UTF-8
    // cannot encode too, which is no join type, and which the error names as it is.
    let normalized = if how.is_instance_of::<pyo3::types::PyString>() {
        Some(
            how.call_method0("lower")?
                .call_method1("replace", ("_", ""))?,
        )
    } else {
        None
    };
    let text = normalized.as_ref().and_then(|normalized| {
        normalized
            .cast::<pyo3::types::PyString>()
            .ok()?
            .to_cow()
            .ok()
    });
    let join_type = match text.as_deref() {
        Some("inner") => Some(JoinType::Inner),
        Some("cross") => Some(JoinType::Cross),
        Some("left" | "leftouter") => Some(JoinType::LeftOuter),
        Some("right" | "rightouter") if !lateral => Some(JoinType::RightOuter),
        Some("outer" | "full" | "fullouter") if !lateral => Some(JoinType::FullOuter),
        Some("leftsemi" | "semi") if !lateral => Some(JoinType::LeftSemi),
        Some("leftanti" | "anti") if !lateral => Some(JoinType::LeftAnti),
        _ => None,
    };
    if let Some(join_type) = join_type {
        return Ok(join_type);
    }
    let typ = normalized.unwrap_or_else(|| how.clone());
    let supported = if lateral {
        LATERAL_JOIN_TYPES_SUPPORTED
    } else {
        JOIN_TYPES_SUPPORTED
    };
    Err(crate::errors::raise_pyspark_of(
        py,
        "AnalysisException",
        "UNSUPPORTED_JOIN_TYPE",
        &[
            ("typ", &typ),
            (
                "supported",
                pyo3::types::PyString::new(py, supported).as_any(),
            ),
        ],
    ))
}

/// The `AttributeError` that PySpark's join raises for an `on` that is neither a name, a
/// condition nor a list of them: the plan reads `to_plan` of it.
fn no_to_plan(obj: &Bound<'_, PyAny>) -> PyErr {
    let typ = obj
        .get_type()
        .name()
        .map(|name| name.to_string())
        .unwrap_or_default();
    pyo3::exceptions::PyAttributeError::new_err(format!(
        "'{typ}' object has no attribute 'to_plan'"
    ))
}

/// [`no_to_plan`] of `obj` as the error of the core, which the requests of a plan raise, as
/// PySpark's `Join` plan reads `to_plan` of `on` when it builds the plan.
fn no_to_plan_error(obj: &Bound<'_, PyAny>) -> PyResult<spark_connect_core::error::SparkError> {
    let typ = obj.get_type().name()?.to_string();
    Ok(spark_connect_core::error::SparkError::builtin_attribute(
        format!("'{typ}' object has no attribute 'to_plan'"),
    ))
}

/// The join condition of `_joinAsOf` for its `on`, as PySpark's `AsOfJoin` plan sends it: a
/// name or a list of names (`using_columns`), or a Column or a list of Columns, which are
/// joined on their conjunction (`join_expr`). The plan builds them when it builds the plan
/// (`args`), as `DataFrame.join` does its condition: the names in the repeated string field as
/// protobuf converts them, the conjunction of a list as `functools.reduce` of `&`, and the
/// `to_plan` of an `on` of another type, which only a Column has.
fn as_of_join_condition(
    args: &mut crate::plan_args::PlanArgs,
    on: &Bound<'_, PyAny>,
) -> PyResult<(Option<spark_connect::column::Column>, Vec<String>)> {
    if let Ok(column) = on.cast::<PyColumn>() {
        return Ok((Some(crate::column::column_of(column)?), Vec::new()));
    }
    if on.is_instance_of::<pyo3::types::PyString>() {
        return Ok((None, vec![args.str(on)?]));
    }
    // As PySpark's `_collect_references`, which iterates `on` first, and then its `to_plan`.
    if !on.is_instance_of::<PyList>() {
        on.try_iter()?;
        args.record(no_to_plan_error(on)?);
        return Ok((None, Vec::new()));
    }
    let list = on.cast::<PyList>()?;
    // An empty list is no condition at all.
    let Some(first) = list.iter().next() else {
        return Ok((None, Vec::new()));
    };
    if first.is_instance_of::<pyo3::types::PyString>() {
        return Ok((None, args.strs(list.iter())?));
    }
    Ok((list_conjunction(args, list)?, Vec::new()))
}

/// The conjunction of the conditions of `list`, the `on` of a join or an as-of join whose first
/// item is not a name, as PySpark's `Join` and `AsOfJoin` plans make it when they are built,
/// `functools.reduce(lambda c1, c2: c1 & c2, on)`: of Columns, the [`plan_conjunction`] of them,
/// of a Column and an item that the `&` of a Column takes as a literal, the `and` of it, and of
/// an item that is not a Column, its own `&`. The error of an `&`, such as the `AttributeError`
/// of the `_expr` of an object that is neither a Column nor a literal, or of the `to_plan` of a
/// conjunction that is not a Column, fails the requests of the plan (`args`).
fn list_conjunction(
    args: &mut crate::plan_args::PlanArgs,
    list: &Bound<'_, PyList>,
) -> PyResult<Option<spark_connect::column::Column>> {
    let py = list.py();
    let mut items = list.iter();
    let Some(mut reduced) = items.next() else {
        return Ok(None);
    };
    let mut column = match reduced.cast::<PyColumn>() {
        Ok(column) => Some(crate::column::column_of(column)?),
        Err(_) => None,
    };
    for item in items {
        match column.take() {
            Some(left) => {
                let right = match item.cast::<PyColumn>() {
                    Ok(right) => crate::column::column_of(right)?,
                    Err(_) => match crate::column::operand_column(&item) {
                        Ok(right) => right,
                        Err(err) => {
                            args.raised(py, err)?;
                            return Ok(None);
                        }
                    },
                };
                column = Some(plan_conjunction(left, right));
            }
            None => match reduced.bitand(&item) {
                Ok(result) => {
                    column = match result.cast::<PyColumn>() {
                        Ok(column) => Some(crate::column::column_of(column)?),
                        Err(_) => None,
                    };
                    reduced = result;
                }
                Err(err) => {
                    args.raised(py, err)?;
                    return Ok(None);
                }
            },
        }
    }
    if column.is_none() {
        args.record(no_to_plan_error(&reduced)?);
    }
    Ok(column)
}

/// The conjunction `left & right` of the conditions of a join of a list of Columns, as PySpark's
/// `Join` and `AsOfJoin` plans make it when they are built, `functools.reduce(lambda c1, c2:
/// c1 & c2, on)`: so the `and` is of the origin of `__and__` at the call site of each request.
fn plan_conjunction(
    left: spark_connect::column::Column,
    right: spark_connect::column::Column,
) -> spark_connect::column::Column {
    let and = left.and(right).expression().clone();
    spark_connect::column::Column::new(
        spark_connect::expression::Expression::with_plan_build_origin(and, "__and__"),
    )
}

/// The as-of column of `_joinAsOf` of `df`: a Column as it is, or the column of `df` that
/// has this name, as PySpark's `df._col(name)`, whose `ColumnReference` asserts that the name
/// is a str: any other object is a bare `AssertionError`.
fn as_of_column(
    df: &DataFrame,
    column: &Bound<'_, PyAny>,
) -> PyResult<spark_connect::column::Column> {
    if let Ok(column) = column.cast::<PyColumn>() {
        return crate::column::column_of(column);
    }
    crate::plan_args::df_col(df, column, false)
}

/// Resolve a filter/where condition: a Column, or a SQL string parsed via `expr`, as PySpark's
/// `filter`, whose `Filter` plan collects the subqueries of anything else by its `foreach`: an
/// object without one raises the `AttributeError` of it.
fn cond_column(condition: &Bound<'_, PyAny>) -> PyResult<spark_connect::column::Column> {
    if let Ok(c) = condition.cast::<PyColumn>() {
        return crate::column::column_of(c);
    }
    if condition.is_instance_of::<pyo3::types::PyString>() {
        return crate::plan_args::expr(condition);
    }
    condition.getattr("foreach")?;
    Err(no_to_plan(condition))
}

/// An argument of `withWatermark`, as PySpark's: `NOT_EXPECTED_TYPE` for an empty one, or one
/// whose type is not str. Its `WithWatermark` plan sets it when it builds the plan (`args`).
fn watermark_argument(
    args: &mut crate::plan_args::PlanArgs,
    value: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<String> {
    if !value.is_truthy()?
        || !value
            .get_type()
            .is(value.py().get_type::<pyo3::types::PyString>())
    {
        return Err(not_expected_type(value, name, "str"));
    }
    args.str(value)
}

#[pymethods]
impl PyDataFrame {
    /// A DataFrame of no state, as the `object.__new__` of PySpark Connect's `DataFrame.__new__`:
    /// of any arguments, which `__init__` takes, so that a subclass of another `__init__` is built
    /// as it is in PySpark. The class shows the signature of that `__new__`, `(*args, **kwargs)`,
    /// as `inspect.signature` of PySpark's class does.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs), text_signature = "(*args, **kwargs)")]
    fn py_new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        PyDataFrame::of_caches(spark_connect::dataframe::DataFrameCaches::new())
    }

    /// `DataFrame(plan, session)`, as PySpark Connect's `__init__`, which keeps its arguments as
    /// they are, as the `_plan` and the `_session` of the DataFrame, of its checks: no plan
    /// (`None`) is `MISSING_VALID_PLAN`, and no session `NO_ACTIVE_SESSION`. The requests of the
    /// DataFrame are of `_plan` and `_session` as they are then (see `Frame::of`): of the plan of
    /// the plan object of a DataFrame (`DataFrame._plan`), as PySpark's DataFrame of it, also in
    /// another session, where the plan has its plan IDs; of a plan of the Python client
    /// (`pyspark.sql.connect.plan`), such as the plans of `pyspark.ml`, of the relation of the
    /// root of its `to_proto(session.client)`, built at the first request, as PySpark's requests
    /// build it, of the `_plan_id` of the plan. A plan whose build raises makes a DataFrame whose
    /// requests raise the exception of the build, as PySpark's requests build the plan and raise
    /// (see `errors::kept_error`). The other entries are the state of no request yet, as
    /// PySpark's `__init__` sets them: `_support_repr_html`, `_cached_schema`,
    /// `_cached_schema_serialized` and `_execution_info`.
    fn __init__(
        slf: &Bound<'_, Self>,
        plan: &Bound<'_, PyAny>,
        session: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let py = slf.py();
        crate::state::assign(slf.as_any(), &[(pyo3::intern!(py, "_plan"), plan)])?;
        if crate::state::attr(slf.as_any(), pyo3::intern!(py, "_plan"))?.is_none() {
            return Err(crate::errors::raise_pyspark(
                py,
                "PySparkRuntimeError",
                "MISSING_VALID_PLAN",
                &[("operator", "__init__")],
            ));
        }
        crate::state::assign(slf.as_any(), &[(pyo3::intern!(py, "_session"), session)])?;
        if crate::state::attr(slf.as_any(), pyo3::intern!(py, "_session"))?.is_none() {
            return Err(crate::errors::raise_pyspark(
                py,
                "PySparkRuntimeError",
                "NO_ACTIVE_SESSION",
                &[],
            ));
        }
        let none = py.None().into_bound(py);
        let unsupported = pyo3::types::PyBool::new(py, false).to_owned().into_any();
        crate::state::assign(
            slf.as_any(),
            &[
                (pyo3::intern!(py, "_support_repr_html"), &unsupported),
                (pyo3::intern!(py, "_cached_schema"), &none),
                (pyo3::intern!(py, "_cached_schema_serialized"), &none),
                (pyo3::intern!(py, "_execution_info"), &none),
            ],
        )?;
        // The native state of a DataFrame of no request, dropped out of the borrow, as its
        // references may run Python code.
        let previous = std::mem::replace(
            &mut *slf.try_borrow_mut()?,
            PyDataFrame::of_caches(spark_connect::dataframe::DataFrameCaches::new()),
        );
        drop(previous);
        Ok(())
    }

    /// The DataFrame of `plan`, a plan of the Python client that a pickled DataFrame of it carried
    /// in place of the relation that its build did not give, as PySpark's pickle carries its plan
    /// objects: its build again, as PySpark's unpickle builds the plan, of its error at each
    /// request if it raises again (see [`dataframe_of_python_plan`]). Of a DataFrame over the
    /// plan (`derived`), whose relation the pickle could not carry, one that builds now gives a
    /// DataFrame whose requests raise the `RuntimeError` of [`derived_plan_error`], rather than
    /// the DataFrame of the plan alone. Not a stable API.
    #[staticmethod]
    fn _of_python_plan(
        plan: &Bound<'_, PyAny>,
        session: &Bound<'_, crate::session::PySparkSession>,
        derived: bool,
    ) -> PyResult<NewDataFrame> {
        Ok(PyDataFrame::new(
            rebuilt_python_plan(plan, session, derived)?.0,
        ))
    }

    /// As a DataFrame of PySpark Connect, a Python object whose plan Python's garbage collector
    /// sees: its `_plan` and its `_session` are in its `__dict__`, which the collector traverses as
    /// that of a plain object, and the plan object of a DataFrame holds the parts of its plan,
    /// which it reports (see [`PyNativePlan`]), as PySpark's DataFrame holds its plan object. The
    /// plan object that this DataFrame made of its `_plan`, when that is not the plan object of a
    /// DataFrame of its session, and that `_plan`, which it is made of, are reported here (see
    /// `converted`), as the plans of the Python client that the plan holds, such as those of
    /// `pyspark.ml`, the functions of its UDFs and its Observations are reported through it.
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        if !self.gc_ready.get() {
            return Ok(());
        }
        // Its lock is held only while no Python code runs.
        if let Ok(converted) = self.converted.try_lock() {
            if let Some((plan, made)) = converted.as_ref() {
                visit.call(plan)?;
                visit.call(made)?;
            }
        }
        Ok(())
    }

    /// Break a reference cycle that the garbage collector found unreachable: the DataFrame drops
    /// the Python objects that `__traverse__` reported; the collector clears its `__dict__`.
    fn __clear__(&mut self) {
        if self.gc_ready.get() {
            let converted = self
                .converted
                .get_mut()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .take();
            drop(converted);
        }
    }

    /// The bytes of the relation of this DataFrame as it is sent, for the plan of `_plan`, built of
    /// its session (see `errors::of_request_session`).
    fn _relation_bytes<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let this = Frame::of(slf)?;
        let session = this.dataframe.spark_session();
        let relation = crate::errors::of_request_session(&session, || {
            this.dataframe.to_relation().to_pyerr()
        })?;
        Ok(pyo3::types::PyBytes::new(
            py,
            &crate::encode_on_grown_stack(relation),
        ))
    }

    /// A DataFrame pickles, as PySpark Connect's, of its plan, here the bytes of its relation and
    /// the functions of its UDFs, of its session, which gives back the active session of its ID,
    /// and of the `_support_repr_html` and the `_cached_schema` of its state. The functions
    /// unpickle first and then the session, as PySpark's plan and session: so on a Python worker
    /// of the server, which has no active session, a pickled DataFrame raises
    /// `NO_ACTIVE_SESSION`, as PySpark's does, and a plain pickle fails at the first function
    /// that it cannot pickle, a lambda, and else at the session. The unpickled DataFrame has
    /// copies of the Observations of this one (see `DataFrame::from_sent_relation`).
    ///
    /// PySpark's pickle keeps the objects of the plan, which it does not build, and its unpickle
    /// builds the plan, as pickle looks the `__setstate__` of the new DataFrame up through its
    /// `__getattr__`, which reads its columns. So the relation is of holes in place of the
    /// commands of the UDFs, and the pickle carries the functions, the objects of their picklers,
    /// which the requests of the unpickled DataFrame pickle, as PySpark's unpickled plan pickles
    /// them at each request ([`crate::udf_command::PickleHoles`]). A plan that does not build, of
    /// a value that PySpark converts when it builds the plan, pickles of its error instead of its
    /// relation, which the unpickle raises ([`crate::column::described_error`]); of a plan of the
    /// Python client whose build raised, of that plan, which the unpickle builds again, as
    /// PySpark's does, of its own error then.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        if let Some(reduce) = Self::python_plan_reduce(slf)? {
            return Ok(reduce);
        }
        let this = PlanFrame::of(slf)?;
        let session = this.session_object();
        let (relation, build_error, holes) = this.pickled_relation(py)?;
        // The state of PySpark's reduce, its `_support_repr_html` and its `_cached_schema`, as
        // they are.
        let (support_repr_html, cached_schema) = this.reduce_state()?;
        py.import("pyspark.sql.connect.dataframe")?
            .getattr("_reduce_dataframe")?
            .call1((
                session,
                relation,
                this.plan().plan_levels(),
                support_repr_html,
                cached_schema,
                build_error,
                holes,
            ))
    }

    /// The reduce of a pickle, as PySpark Connect's DataFrame pickles its plan before its session,
    /// of the objects of `pyspark.ml` that the plan keeps (see `ml::object::pickle_reach`, of the
    /// plans of its subqueries and of the Python plans of PySpark's own classes of `pyspark.ml`
    /// too): of a plan that keeps a model or a summary (a transform of a model, a DataFrame of a
    /// method of a model or of a summary), the reduce of the objects that PySpark's pickle of the
    /// plan reaches first, the functions of its UDFs before them and then the lock of the model's
    /// `RemoteModelRef`, or the summary dataset of a training summary, whose session a plain
    /// pickle refuses, as PySpark's refuses the session of the summary's `_summary_dataset`, and
    /// then the lock, which cloudpickle, of the function of the session, reaches next (see
    /// `ml::object::reach_reduce`); so a pickle fails at the first object that it cannot pickle,
    /// of the error of its protocol (the TypeError of a lock at 2 and up, its PicklingError below).
    /// Else the reduce of `__reduce__`.
    fn __reduce_ex__<'py>(
        slf: &Bound<'py, Self>,
        protocol: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = protocol;
        let py = slf.py();
        if let Some(reduce) = Self::python_plan_reduce(slf)? {
            return Ok(reduce);
        }
        let reach = crate::ml::object::pickle_reach(py, PlanFrame::of(slf)?.plan())?;
        let reduced = slf.call_method0(pyo3::intern!(py, "__reduce__"))?;
        crate::ml::object::reach_reduce(py, &reach, reduced)
    }

    /// The reduce of a DataFrame whose `_plan` is not the plan object of a DataFrame of the
    /// drop-in, such as a plan of the Python client of `DataFrame(plan, session)`: PySpark's own,
    /// `DataFrame(self._plan, self._session)` of the state of its `_support_repr_html` and its
    /// `_cached_schema`, of the plan as it is, which the pickle carries, and which the unpickle
    /// builds when it reads the columns of the new DataFrame, as PySpark's lookup of its
    /// `__setstate__` does, of the errors of that build; so the objects that the plan holds, such
    /// as those of `pyspark.ml`, pickle as PySpark's, and a pickle builds no plan. None of the plan
    /// object of a DataFrame, whose reduce is of its relation (see `__reduce__`).
    fn python_plan_reduce<'py>(slf: &Bound<'py, Self>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let py = slf.py();
        let plan = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_plan"))?;
        if own_plan_object(&plan)?.is_some() {
            return Ok(None);
        }
        let session = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_session"))?;
        let state = PyDict::new(py);
        state.set_item(
            "_support_repr_html",
            crate::state::attr(slf.as_any(), pyo3::intern!(py, "_support_repr_html"))?,
        )?;
        state.set_item(
            "_cached_schema",
            crate::state::attr(slf.as_any(), pyo3::intern!(py, "_cached_schema"))?,
        )?;
        let args = PyTuple::new(py, [plan, session])?;
        let reduce = (py.get_type::<PyDataFrame>(), args, state).into_pyobject(py)?;
        Ok(Some(reduce.into_any()))
    }

    /// The shallow copy of a DataFrame, as PySpark's `copy.copy` of its `__reduce__`,
    /// `DataFrame(self._plan, self._session)` of its state: the same plan object and session, and
    /// so the same plan and Observations, with the `_support_repr_html` and the `_cached_schema`
    /// of its state, the same objects, the `_cached_schema_serialized` of the schema that its
    /// reconstruction requests (see `copy_of`), and an `executionInfo` of its own, None until it
    /// executes, as that of PySpark's `DataFrame(plan, session)`.
    fn __copy__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDataFrame>> {
        let py = slf.py();
        let this = PlanFrame::of(slf)?;
        let state = this.reduce_state()?;
        this.copy_of(py, this.plan_object(), this.session_object(), state)
    }

    /// The deep copy of a DataFrame, as PySpark's `copy.deepcopy` of its `__reduce__`, which
    /// copies its plan object and then its session with the memo of the call, of which it is
    /// `DataFrame(plan, session)`, and then its state: of the copy of the plan of the plan object
    /// of a DataFrame of the drop-in, the same relation and plan ID, whose Observations are the
    /// copies of those of the plan in `memo` (see `PyNativePlan::__deepcopy__`), with deep copies
    /// of the `_support_repr_html` and the `_cached_schema` of its state, and an `executionInfo`
    /// of its own. So an Observation that the call copies too, in a tuple or an object that holds
    /// both, is copied once, and the actions of the copied DataFrame set the copy; the
    /// Observations of this DataFrame stay as they are. The copy of the session is that of its
    /// `__reduce__`, the active session of its ID, once in a call: so a deep copy raises
    /// `NO_ACTIVE_SESSION` where no session is active, as on another thread, and
    /// `PySparkAssertionError` where another one is, while `copy.copy` keeps the session. A copy in
    /// another session, of a memo that maps the session or of another active session of its ID,
    /// has the plan IDs of this DataFrame there (see `Frame::of`).
    fn __deepcopy__<'py>(
        slf: &Bound<'py, Self>,
        memo: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDataFrame>> {
        let py = slf.py();
        let this = PlanFrame::of(slf)?;
        // As PySpark's: `copy.deepcopy` looks up `__deepcopy__` through `__getattr__`, whose check
        // reads the columns, so the schema is requested unless it is cached, and converted, of the
        // errors of that read (see `schema_entry`), before the plan, the session and the state
        // are copied. On a stopped session, that request raises the SparkConnectException of its
        // closed client, before the session is copied, which raises a PySparkRuntimeError.
        property_schema(this.slf())?.columns(py)?;
        let (support_repr_html, cached_schema) = this.reduce_state()?;
        let deepcopy = py.import("copy")?.getattr(pyo3::intern!(py, "deepcopy"))?;
        let plan = deepcopy.call1((this.plan_object(), memo))?;
        let session = deepcopy.call1((this.session_object(), memo))?;
        let state = (
            deepcopy.call1((support_repr_html, memo))?,
            deepcopy.call1((cached_schema, memo))?,
        );
        this.copy_of(py, &plan, &session, state)
    }

    /// The DataFrame of `relation`, the bytes of a relation of `levels` levels of `session`, as a
    /// pickled DataFrame unpickles, with the `_support_repr_html` and the `_cached_schema` of its
    /// state. As PySpark's unpickle, whose lookup of the `__setstate__` of the new DataFrame reads
    /// its columns, it requests the schema of the relation, whose error it raises, also that of
    /// its conversion (see `check_python_schema`), which sets its `_cached_schema` and
    /// `_cached_schema_serialized`, and then the state sets its `_support_repr_html` and its
    /// `_cached_schema`, the objects that the pickled DataFrame held, or None when it had none,
    /// so that its `schema` requests it again then. `holes` are those of the relation, of the
    /// functions of its UDFs, which each request pickles (see `udf_command::sent_holes`).
    #[staticmethod]
    #[pyo3(signature = (
        session,
        relation,
        levels=1,
        support_repr_html=None,
        cached_schema=None,
        holes=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn _of_relation<'py>(
        py: Python<'py>,
        session: &Bound<'py, crate::session::PySparkSession>,
        relation: &[u8],
        levels: usize,
        support_repr_html: Option<Bound<'py, PyAny>>,
        cached_schema: Option<Bound<'py, PyAny>>,
        holes: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyDataFrame>> {
        let holes = crate::udf_command::sent_holes(holes)?;
        let session_object = session.clone().into_any().unbind();
        let core_session = session.borrow().session.clone();
        use prost::Message;
        // Decoded, and then built or dropped, on a stack that holds its levels.
        // The errors become Python exceptions on the caller's stack, which runs no Python code
        // on the grown one.
        let dataframe = crate::on_grown_stack(|| {
            spark_connect_proto::Relation::decode(relation).map(|relation| {
                DataFrame::from_sent_relation_with_holes(core_session, relation, levels, holes)
            })
        })
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?
        .to_pyerr()?;
        // Of the unpickled session, as PySpark's `DataFrame(plan, session)` of its reduce.
        let mut new = PyDataFrame::new(dataframe);
        new.session = Some(session_object);
        let df = new.into_pyobject(py)?;
        // As PySpark's lookup of the `__setstate__` of the new DataFrame, which reads its columns,
        // of the plan of the unpickled DataFrame, which holds the functions of its holes, and the
        // Observations of a live DataFrame of its plan ID.
        property_schema(&df)?.columns(py)?;
        let support_repr_html = support_repr_html
            .unwrap_or_else(|| pyo3::types::PyBool::new(py, false).to_owned().into_any());
        let cached_schema = cached_schema.unwrap_or_else(|| py.None().into_bound(py));
        let fields = df.as_any().getattr(pyo3::intern!(py, "__dict__"))?;
        fields.set_item(pyo3::intern!(py, "_support_repr_html"), support_repr_html)?;
        fields.set_item(pyo3::intern!(py, "_cached_schema"), cached_schema)?;
        Ok(df)
    }

    // Select columns (accepts Column objects or string names).
    #[pyo3(signature = (*cols))]
    #[doc = connect_doc!(DataFrame.select)]
    fn select(
        slf: &Bound<'_, Self>,
        _py: Python<'_>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        let columns = to_column_list(cols)?;
        Ok(this.derived(this.plan().select(columns)))
    }

    // Select using SQL expressions, as PySpark's `DataFrame.selectExpr`: strings, or lists of
    // them.
    #[pyo3(signature = (*expr))]
    #[doc = connect_doc!(DataFrame.selectExpr)]
    fn selectExpr(slf: &Bound<'_, Self>, expr: Vec<Bound<'_, PyAny>>) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        let expr = match expr.as_slice() {
            [only] if only.is_instance_of::<PyList>() => {
                only.try_iter()?.collect::<PyResult<Vec<_>>>()?
            }
            _ => expr,
        };
        // Each of `F.expr`, which asserts a str.
        let mut columns = Vec::new();
        for element in &expr {
            if element.is_instance_of::<pyo3::types::PyString>() {
                columns.push(crate::plan_args::expr(element)?);
            } else {
                for e in element.try_iter()? {
                    columns.push(crate::plan_args::expr(&e?)?);
                }
            }
        }
        Ok(this.derived(this.plan().select(columns)))
    }

    // Filter rows by a Column condition or a SQL-string condition.
    #[doc = connect_doc!(DataFrame.filter)]
    fn filter(slf: &Bound<'_, Self>, condition: &Bound<'_, PyAny>) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        this.of_same_schema(this.derived(this.plan().filter(cond_column(condition)?)))
    }

    // Alias for filter (reference pyspark name is `where`).
    #[pyo3(name = "where")]
    #[doc = connect_doc!(DataFrame.where)]
    fn where_(slf: &Bound<'_, Self>, condition: &Bound<'_, PyAny>) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        // As PySpark's: a Column or a str, else NOT_EXPECTED_TYPE.
        if !(condition.is_instance_of::<PyColumn>()
            || condition.is_instance_of::<pyo3::types::PyString>())
        {
            return Err(not_expected_type(condition, "condition", "Column or str"));
        }
        this.of_same_schema(this.derived(this.plan().filter(cond_column(condition)?)))
    }

    // Add or replace a column, of a Column, as PySpark's `DataFrame.withColumn`, which checks
    // the Column first. Its `WithColumns` plan asserts a str name, and sets it when it builds
    // the plan, after the Column.
    #[doc = connect_doc!(DataFrame.withColumn)]
    fn withColumn(
        slf: &Bound<'_, Self>,
        colName: &Bound<'_, PyAny>,
        col: &Bound<'_, PyAny>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        let Ok(column) = col.cast::<PyColumn>() else {
            return Err(not_expected_type(col, "col", "Column"));
        };
        if !colName.is_instance_of::<pyo3::types::PyString>() {
            return Err(pyo3::exceptions::PyAssertionError::new_err(()));
        }
        let mut args = crate::plan_args::PlanArgs::new();
        let name = args.str(colName)?;
        Ok(this.derived(
            args.plan(
                this.plan()
                    .with_column(&name, crate::column::column_of(column)?),
            ),
        ))
    }

    // Rename a column, as PySpark's, which renames of the dict `{existing: new}`, so an
    // `existing` that is not hashable raises the `TypeError` of the dict.
    #[doc = connect_doc!(DataFrame.withColumnRenamed)]
    fn withColumnRenamed(
        slf: &Bound<'_, Self>,
        existing: &Bound<'_, PyAny>,
        new: &Bound<'_, PyAny>,
    ) -> PyResult<NewDataFrame> {
        let renames = PyDict::new(existing.py());
        renames.set_item(existing, new)?;
        Self::with_columns_renamed(slf, renames.as_any())
    }

    // Add or replace multiple columns from a {name: Column} mapping, as PySpark's
    // `DataFrame.withColumns`, whose `WithColumns` plan asserts, as the call builds it, that the
    // mapping is not empty, and of str names and Columns.
    #[pyo3(name = "withColumns", signature = (*colsMap))]
    #[doc = connect_doc!(DataFrame.withColumns)]
    fn with_columns(
        slf: &Bound<'_, Self>,
        colsMap: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        let [cols_map] = colsMap.as_slice() else {
            return Err(pyo3::exceptions::PyAssertionError::new_err(()));
        };
        let Ok(cols_map) = cols_map.cast::<PyDict>() else {
            return Err(not_expected_type(cols_map, "colsMap", "dict"));
        };
        let valid = !cols_map.is_empty()
            && cols_map.iter().all(|(name, column)| {
                name.is_instance_of::<pyo3::types::PyString>()
                    && column.is_instance_of::<PyColumn>()
            });
        if !valid {
            return Err(pyo3::exceptions::PyAssertionError::new_err(()));
        }
        // A name that UTF-8 cannot encode fails the plan when it is sent, with the error of its
        // encoding, which the first of them gives.
        let mut args = crate::plan_args::PlanArgs::new();
        let mut pairs = Vec::with_capacity(cols_map.len());
        for (name, column) in cols_map.iter() {
            let column = column.cast::<PyColumn>()?;
            pairs.push((args.str(&name)?, crate::column::column_of(column)?));
        }
        Ok(this.derived(args.plan(this.plan().with_columns(pairs))))
    }

    // Specify a relation hint (e.g. "broadcast"), with optional parameters. Reference
    // hint params may be ints (e.g. `df.hint("rebalance", 3)`), not just strings, so
    // accept any value and stringify it; the core parses a numeric string back into a
    // long literal, matching reference semantics.
    #[pyo3(signature = (name, *parameters))]
    #[doc = connect_doc!(DataFrame.hint)]
    fn hint(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        name: &Bound<'_, PyAny>,
        parameters: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        use pyo3::types::{PyFloat, PyString};
        // As PySpark's: a single list is the parameters, each a str, a float, an int, a
        // Column or a list of str, float or int, sent as `lit` of it.
        let parameters = match parameters.as_slice() {
            [only] if only.is_instance_of::<PyList>() => {
                only.try_iter()?.collect::<PyResult<Vec<_>>>()?
            }
            _ => parameters,
        };
        if !name.is_instance_of::<PyString>() {
            return Err(not_expected_type(name, "name", "str"));
        }
        let primitive = |p: &Bound<'_, PyAny>| {
            p.is_instance_of::<PyString>()
                || p.is_instance_of::<PyFloat>()
                || p.is_instance_of::<PyInt>()
        };
        let invalid = |item_type: &str| {
            crate::column::raise_pyspark(
                py,
                "PySparkTypeError",
                "INVALID_ITEM_FOR_CONTAINER",
                &[
                    ("arg_name", "parameters"),
                    (
                        "allowed_types",
                        "str, float, int, Column, list[str], list[float], list[int]",
                    ),
                    ("item_type", item_type),
                ],
            )
        };
        let type_name =
            |p: &Bound<'_, PyAny>| -> PyResult<String> { Ok(p.get_type().name()?.to_string()) };
        let mut expressions = Vec::with_capacity(parameters.len());
        for p in &parameters {
            if p.is_instance_of::<PyList>() {
                let items: Vec<Bound<'_, PyAny>> = p.try_iter()?.collect::<PyResult<_>>()?;
                if !items.iter().all(primitive) {
                    // As PySpark's, which names the type of the first item.
                    let first = items
                        .first()
                        .map(type_name)
                        .transpose()?
                        .unwrap_or_default();
                    return Err(invalid(&format!("list[{first}]")));
                }
            } else if !(primitive(p) || p.cast::<PyColumn>().is_ok()) {
                return Err(invalid(&type_name(p)?));
            }
            expressions.push(crate::functions::to_column(p)?.expression().clone());
        }
        // Its `Hint` plan sets the name when it builds the plan, before the parameters.
        let mut args = crate::plan_args::PlanArgs::new();
        let name = args.str(name)?;
        let args = args.before(&mut expressions);
        this.of_same_schema(
            this.derived(args.plan(this.plan().hint_expressions(&name, expressions))),
        )
    }

    // Drop columns, of Columns or column names, as PySpark's `DataFrame.drop`.
    #[pyo3(signature = (*cols))]
    #[doc = connect_doc!(DataFrame.drop)]
    fn drop(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        // Its `Drop` plan sets the names when it builds the plan.
        let mut args = crate::plan_args::PlanArgs::new();
        let (mut columns, mut names) = (Vec::new(), Vec::new());
        for column in &cols {
            if let Ok(column) = column.cast::<PyColumn>() {
                columns.push(crate::column::column_of(column)?);
            } else if column.is_instance_of::<pyo3::types::PyString>() {
                names.push(args.str(column)?);
            } else {
                return Err(crate::column::raise_pyspark(
                    py,
                    "PySparkTypeError",
                    "NOT_EXPECTED_TYPE",
                    &[
                        ("expected_type", "Column or str"),
                        ("arg_name", "cols"),
                        ("arg_type", "tuple"),
                    ],
                ));
            }
        }
        Ok(this.derived(args.plan(this.plan().drop_columns(columns, names))))
    }

    // Limit the number of rows, as PySpark's `limit`, which does not check `num`: its plan sets
    // `num` in the proto's int32, whose errors ([`proto_int`]) PySpark raises once it sends the
    // plan, so the DataFrame raises them at each request. The server checks the rest.
    #[doc = connect_doc!(DataFrame.limit)]
    fn limit(slf: &Bound<'_, Self>, num: &Bound<'_, PyAny>) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        let py = num.py();
        let value = proto_int(num, "spark.connect.Limit.limit");
        let df = converted_plan(py, this.plan(), value, |num| this.plan().limit(num))?;
        this.of_same_schema(this.derived(df))
    }

    // Skip the first `num` rows, as PySpark's `offset`, which sends `num` as `limit` does, but
    // does not copy the `_cached_schema` of its input, also when the count fails the plan.
    #[doc = connect_doc!(DataFrame.offset)]
    fn offset(slf: &Bound<'_, Self>, num: &Bound<'_, PyAny>) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        let py = num.py();
        let df = match proto_int(num, "spark.connect.Offset.offset") {
            Ok(num) => this.plan().offset(num),
            Err(err) => this.plan().unbuildable(unbuildable_error(py, err)?),
        };
        Ok(this.derived(df))
    }

    // Remove duplicate rows.
    #[doc = connect_doc!(DataFrame.distinct)]
    fn distinct(slf: &Bound<'_, Self>) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        this.of_same_schema(this.derived(this.plan().distinct()))
    }

    // Remove duplicate rows, optionally on specific columns.
    #[pyo3(signature = (subset=None))]
    #[doc = connect_doc!(DataFrame.dropDuplicates)]
    fn dropDuplicates(
        slf: &Bound<'_, Self>,
        subset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        let mut args = crate::plan_args::PlanArgs::new();
        let subset = deduplicate_subset(&mut args, subset)?;
        let col_refs: Option<Vec<&str>> = subset
            .as_ref()
            .map(|names| names.iter().map(|s| s.as_str()).collect());
        this.of_same_schema(this.derived(args.plan(this.plan().drop_duplicates(col_refs))))
    }

    // Sort rows. `ascending` (a bool or list of bools, via kwargs) mirrors reference
    // pyspark: it wraps each sort column with asc()/desc().
    #[pyo3(signature = (*cols, **kwargs))]
    #[doc = connect_doc!(DataFrame.sort)]
    fn sort(
        slf: &Bound<'_, Self>,
        _py: Python<'_>,
        cols: Vec<Bound<'_, PyAny>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        let exprs = this.sort_columns(_py, cols, kwargs.as_ref())?;
        this.of_same_schema(this.derived(this.plan().sort(exprs)))
    }

    // Alias for sort.
    #[pyo3(signature = (*cols, **kwargs))]
    #[doc = connect_doc!(DataFrame.orderBy)]
    fn orderBy(
        slf: &Bound<'_, Self>,
        _py: Python<'_>,
        cols: Vec<Bound<'_, PyAny>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        let exprs = this.sort_columns(_py, cols, kwargs.as_ref())?;
        this.of_same_schema(this.derived(this.plan().order_by(exprs)))
    }

    // Join with another DataFrame.
    //
    // `on` is a Column condition, a column name, or a list of column names (a "using"
    // join) or of Column conditions (joined on their conjunction), as the connect client's
    // `plan.Join` takes it. Mirrors `pyspark.sql.DataFrame.join(other, on=None, how=None)`
    // (default inner), including its errors.
    #[pyo3(signature = (other, on=None, how=None))]
    #[doc = connect_doc!(DataFrame.join)]
    fn join(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        on: Option<Bound<'_, PyAny>>,
        how: Option<Bound<'_, PyAny>>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        // As PySpark's `_check_same_session`, first.
        let other = this.same_session(other)?;
        // As PySpark's `plan.Join`, `on` is iterated before `how` is checked, so an `on`
        // that is not iterable is the error that is raised first.
        if let Some(on) = &on {
            if !on.is_instance_of::<PyColumn>() && !on.is_instance_of::<pyo3::types::PyString>() {
                on.try_iter()?;
            }
        }
        let join_type = parse_join_type(py, how.as_ref(), false)?;
        let Some(on) = on else {
            return Ok(this.derived(this.plan().join(other.plan(), None, join_type)));
        };
        if let Ok(col) = on.cast::<PyColumn>() {
            return Ok(this.derived(this.plan().join(
                other.plan(),
                Some(crate::column::column_of(col)?),
                join_type,
            )));
        }
        // The names are set in the repeated string field of its `Join` plan as they are when it
        // builds the plan, as protobuf takes them: bytes of UTF-8 too.
        let mut args = crate::plan_args::PlanArgs::new();
        if on.is_instance_of::<pyo3::types::PyString>() {
            let name = args.str(&on)?;
            return Ok(this.derived(args.plan(this.plan().join_using(
                other.plan(),
                vec![name],
                join_type,
            ))));
        }
        if let Ok(list) = on.cast::<PyList>() {
            // An empty list is no condition at all.
            let Some(first) = list.iter().next() else {
                return Ok(this.derived(this.plan().join(other.plan(), None, join_type)));
            };
            if first.is_instance_of::<pyo3::types::PyString>() {
                let names = args.strs(list.iter())?;
                return Ok(this.derived(args.plan(this.plan().join_using(
                    other.plan(),
                    names,
                    join_type,
                ))));
            }
            // A list of conditions is their conjunction, which its `Join` plan computes when it
            // builds the plan.
            let condition = list_conjunction(&mut args, list)?;
            return Ok(this.derived(args.plan(this.plan().join(
                other.plan(),
                condition,
                join_type,
            ))));
        }
        // A tuple, or another iterable that is not a list: its plan reads `to_plan` of it when
        // it builds the plan.
        args.record(no_to_plan_error(&on)?);
        Ok(this.derived(args.plan(this.plan().join(other.plan(), None, join_type))))
    }

    /// `DataFrame._joinAsOf(other, leftAsOfColumn, rightAsOfColumn, on=None, how=None, *,
    /// tolerance=None, allowExactMatches=True, direction="backward")`, the as-of join of
    /// pandas-on-Spark's `merge_asof`, as PySpark's. The as-of columns are Columns, or names of
    /// their DataFrame. `on` is a name, a Column condition, or a list of either. `how` (inner by
    /// default) and `direction` are not checked, as PySpark does not check them: a string that
    /// is none of the join types or directions of the core is sent as it is, and the server
    /// refuses what it does not read.
    #[pyo3(
        name = "_joinAsOf",
        signature = (
            other,
            leftAsOfColumn,
            rightAsOfColumn,
            on=None,
            how=None,
            *,
            tolerance=None,
            allowExactMatches=DefaultedOption::Omitted,
            direction=DefaultedOption::Omitted
        ),
        text_signature = "($self, other, leftAsOfColumn, rightAsOfColumn, on=None, how=None, *, tolerance=None, allowExactMatches=True, direction='backward')"
    )]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn join_as_of(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
        leftAsOfColumn: &Bound<'_, PyAny>,
        rightAsOfColumn: &Bound<'_, PyAny>,
        on: Option<Bound<'_, PyAny>>,
        how: Option<Bound<'_, PyAny>>,
        tolerance: Option<Bound<'_, PyAny>>,
        allowExactMatches: DefaultedOption,
        direction: DefaultedOption,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        // As PySpark's `_check_same_session`, first.
        let other = this.same_session(other)?;
        // As PySpark's `assert isinstance(how, str)`. A str that UTF-8 cannot encode is one, and
        // fails the plan with the error of its encoding, when PySpark sends it.
        let how = how.filter(|how| !how.is_none());
        if let Some(how) = &how {
            if !how.is_instance_of::<pyo3::types::PyString>() {
                return Err(pyo3::exceptions::PyAssertionError::new_err(
                    "how should be a string",
                ));
            }
        }
        let tolerance = match tolerance.filter(|tolerance| !tolerance.is_none()) {
            None => None,
            Some(tolerance) => {
                let column = tolerance.cast::<PyColumn>().map_err(|_| {
                    pyo3::exceptions::PyAssertionError::new_err("tolerance should be Column")
                })?;
                Some(crate::column::column_of(column)?)
            }
        };
        // The as-of columns before `on`, as PySpark converts them before its `AsOfJoin` plan
        // iterates `on`.
        let left_as_of = as_of_column(this.plan(), leftAsOfColumn)?;
        let right_as_of = as_of_column(other.plan(), rightAsOfColumn)?;
        // The plan's values are converted in the order of its protobuf fields, in which its
        // `AsOfJoin` plan sets them when it builds the plan, which gives the first error.
        let mut args = crate::plan_args::PlanArgs::new();
        let (on, using_columns) = match on.filter(|on| !on.is_none()) {
            None => (None, Vec::new()),
            Some(on) => as_of_join_condition(&mut args, &on)?,
        };
        let join_type = match &how {
            None => AsOfJoinType::Inner,
            Some(how) => AsOfJoinType::parse(&args.str(how)?),
        };
        // The flag is a protobuf bool, which PySpark sets as the plan is sent.
        let py = leftAsOfColumn.py();
        let allow_exact_matches = match &allowExactMatches {
            DefaultedOption::Omitted => true,
            DefaultedOption::Given(allow) => args.converted(py, proto_bool(allow.bind(py)))?,
        };
        // PySpark does not check `direction`, so its errors come last, as those of protobuf's
        // string field when the plan is sent.
        let direction = match &direction {
            DefaultedOption::Omitted => AsOfJoinDirection::Backward,
            DefaultedOption::Given(direction) => {
                AsOfJoinDirection::parse(&args.str(direction.bind(py))?)
            }
        };
        let options = AsOfJoinOptions {
            left_as_of,
            right_as_of,
            on,
            using_columns,
            join_type,
            tolerance,
            allow_exact_matches,
            direction,
        };
        let df = this.plan().as_of_join(other.plan(), options).to_pyerr()?;
        Ok(this.derived(args.plan(df)))
    }

    // Cross join.
    #[doc = connect_doc!(DataFrame.crossJoin)]
    fn crossJoin(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        let other = this.same_session(other)?;
        Ok(this.derived(this.plan().cross_join(other.plan())))
    }

    // Lateral (correlated) join. `on` is an optional Column condition; `how` is inner by
    // default and takes the join types of a lateral join. Mirrors `DataFrame.lateralJoin`.
    #[pyo3(signature = (other, on=None, how=None))]
    #[doc = connect_doc!(DataFrame.lateralJoin)]
    fn lateralJoin(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        on: Option<Bound<'_, PyAny>>,
        how: Option<Bound<'_, PyAny>>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        // As PySpark's `_check_same_session`, first. Its `LateralJoin` plan then collects the
        // subqueries of `on` by its `foreach`, which an object that is not a Column lacks, before
        // it checks `how`.
        let other = this.same_session(other)?;
        let on_col = match on {
            None => None,
            Some(on) => match on.cast::<PyColumn>() {
                Ok(column) => Some(crate::column::column_of(column)?),
                Err(_) => {
                    on.getattr("foreach")?;
                    return Err(no_to_plan(&on));
                }
            },
        };
        let join_type = parse_join_type(py, how.as_ref(), true)?;
        Ok(this.derived(this.plan().lateral_join(other.plan(), on_col, join_type)))
    }

    // Transpose (swap rows/columns). `indexColumn` optionally names the header
    // column. Mirrors `DataFrame.transpose(indexColumn=None)`.
    #[pyo3(signature = (indexColumn=None))]
    #[allow(non_snake_case)]
    #[doc = connect_doc!(DataFrame.transpose)]
    fn transpose(
        slf: &Bound<'_, Self>,
        indexColumn: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        // As PySpark's `F._to_col`: a Column or a column name, else NOT_EXPECTED_TYPE of `col`.
        let index = match indexColumn {
            None => None,
            Some(c) => Some(if let Ok(column) = c.cast::<PyColumn>() {
                crate::column::column_of(column)?
            } else if c.is_instance_of::<pyo3::types::PyString>() {
                let _origin = crate::origin::with_origin("col")?;
                crate::plan_args::col(c)?
            } else {
                return Err(not_expected_type(c, "col", "Column or str"));
            }),
        };
        let df = match index {
            Some(column) => this.plan().transpose_with_index(column),
            None => this.plan().transpose(),
        }
        .to_pyerr()?;
        Ok(this.derived(df))
    }

    // Repartition into `numPartitions` by a range partitioning on the given
    // columns. Mirrors `DataFrame.repartitionByRange(numPartitions, *cols)`.
    #[pyo3(signature = (numPartitions, *cols))]
    #[allow(non_snake_case)]
    #[doc = connect_doc!(DataFrame.repartitionByRange)]
    fn repartitionByRange(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        numPartitions: Bound<'_, PyAny>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        // As PySpark's: a positive partition count and columns, or columns alone. The columns
        // are sorted ascending unless they are sort orders, by `repartition_by_range_opt`.
        let (num_partitions, cols) = if numPartitions.is_instance_of::<PyInt>() {
            positive_partitions(&numPartitions)?;
            if cols.is_empty() {
                return Err(crate::column::raise_pyspark(
                    py,
                    "PySparkValueError",
                    "CANNOT_BE_EMPTY",
                    &[("item", "cols")],
                ));
            }
            (Some(numPartitions), cols)
        } else if numPartitions.is_instance_of::<pyo3::types::PyString>()
            || numPartitions.cast::<PyColumn>().is_ok()
        {
            // No partition count is sent, so the server picks it, as PySpark's.
            let mut all = vec![numPartitions];
            all.extend(cols);
            (None, all)
        } else {
            return Err(not_expected_type(
                &numPartitions,
                "numPartitions",
                "Column, int or str",
            ));
        };
        // As PySpark's `_to_cols(cols, sort=True)`, which does not unpack a list among them, and
        // sorts a column that is not a sort order by its `asc`, which is `asc_nulls_first`, of
        // the origin of that.
        let mut exprs = Vec::with_capacity(cols.len());
        for column in to_columns_as_given(cols)? {
            let sorted = match column.expression().node() {
                spark_connect::ExpressionNode::SortOrder(_) => column,
                _ => {
                    let _origin = crate::origin::with_origin("asc_nulls_first")?;
                    column.asc_nulls_first()
                }
            };
            exprs.push(sorted.expression().clone());
        }
        // The count is the proto's int32 of the plan, which PySpark converts once it sends the
        // plan, after the columns: a count that protobuf refuses fails the plan, which is built
        // here with no count, so that the errors of the columns come first.
        let (num_partitions, plan_error) = match num_partitions
            .map(|n| proto_int(&n, "spark.connect.RepartitionByExpression.num_partitions"))
            .transpose()
        {
            Ok(num_partitions) => (num_partitions, None),
            Err(err) => (None, Some(unbuildable_error(py, err)?)),
        };
        let df = this
            .plan()
            .repartition_by_range_opt(num_partitions, exprs)
            .to_pyerr()?;
        // Of the schema that the input cached, as PySpark copies it.
        this.of_same_schema(this.derived(match plan_error {
            Some(error) => df.unbuildable_with_cached_schema(error),
            None => df,
        }))
    }

    // Repartition into `numPartitions` using a column's value directly as the shuffle
    // partition id. Mirrors `DataFrame.repartitionById(numPartitions, partitionIdCol)`.
    #[pyo3(signature = (numPartitions, partitionIdCol))]
    #[allow(non_snake_case)]
    #[doc = connect_doc!(DataFrame.repartitionById)]
    fn repartitionById(
        slf: &Bound<'_, Self>,
        numPartitions: &Bound<'_, PyAny>,
        partitionIdCol: &Bound<'_, PyAny>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        // As PySpark's: an int that is not a bool, and not `numPartitions <= 0`.
        if !numPartitions.is_instance_of::<PyInt>()
            || numPartitions.is_instance_of::<pyo3::types::PyBool>()
        {
            return Err(not_expected_type(numPartitions, "numPartitions", "int"));
        }
        if numPartitions.le(0)? {
            return Err(not_positive_partitions(numPartitions)?);
        }
        // A `ColumnOrName`, as PySpark's `F._to_col`: a string is a column name, by `col`, of its
        // origin, not a literal, and another value is NOT_EXPECTED_TYPE of `col`.
        let col = if let Ok(c) = partitionIdCol.extract::<Bound<'_, PyColumn>>() {
            crate::column::column_of(&c)?
        } else if partitionIdCol.is_instance_of::<pyo3::types::PyString>() {
            let _origin = crate::origin::with_origin("col")?;
            crate::plan_args::col(partitionIdCol)?
        } else {
            return Err(not_expected_type(partitionIdCol, "col", "Column or str"));
        };
        // The count is the proto's int32 of the plan, which PySpark's `RepartitionByExpression`
        // sets after the column and the input, as in `repartition`: a count that protobuf refuses
        // fails the plan, which is built here with no count.
        let df = match proto_int(
            numPartitions,
            "spark.connect.RepartitionByExpression.num_partitions",
        ) {
            Ok(num_partitions) => this.plan().repartition_by_id(num_partitions, col),
            // A count of the plan that is never built: one that the core takes, which refuses a
            // count that is not positive.
            Err(err) => this
                .plan()
                .repartition_by_id(1, col)
                .unbuildable_with_cached_schema(unbuildable_error(numPartitions.py(), err)?),
        };
        this.of_same_schema(this.derived(df))
    }

    // Append a distributed monotonically-increasing index column, as PySpark's
    // `DataFrame.zipWithIndex(indexColName="index")`: `self.select(F.col("*"),
    // F._invoke_function("distributed_sequence_id").alias(indexColName))`, of the origins of
    // `col` and `alias`. The name is that of the alias of a Column ([`PyColumn::alias`]), which
    // the plan sets when it is built: a str or bytes of UTF-8, and another value fails each
    // request of the DataFrame, not the call.
    #[pyo3(
        signature = (indexColName=DefaultedOption::Omitted),
        text_signature = "($self, indexColName='index')"
    )]
    #[allow(non_snake_case)]
    #[doc = connect_doc!(DataFrame.zipWithIndex)]
    fn zipWithIndex(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        indexColName: DefaultedOption,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        let star = {
            let _origin = crate::origin::with_origin("col")?;
            spark_connect::functions::col("*")
        };
        let name = match &indexColName {
            DefaultedOption::Omitted => pyo3::types::PyString::new(py, "index").into_any(),
            DefaultedOption::Given(name) => name.bind(py).clone(),
        };
        let index = crate::column::column_object(
            py,
            spark_connect::column::Column::new(spark_connect::expression::Expression::function(
                "distributed_sequence_id",
                vec![],
            )),
        )?;
        let index =
            PyColumn::alias(&index, py, &PyTuple::new(py, [name])?, None)?.into_pyobject(py)?;
        Ok(this.derived(
            this.plan()
                .select(vec![star, crate::column::column_of(&index)?]),
        ))
    }

    // Zip this DataFrame with another column-wise (side-by-side). Mirrors
    // `DataFrame.zip(other)`.
    #[doc = connect_doc!(DataFrame.zip)]
    fn zip(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        let other = this.same_session(other)?;
        Ok(this.derived(this.plan().zip(other.plan()).to_pyerr()?))
    }

    // Union with another DataFrame.
    #[doc = connect_doc!(DataFrame.union)]
    fn union(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        let other = this.same_session(other)?;
        let dataframe = this.plan().union(other.plan());
        set_operation_result(py, &this, &other, dataframe)
    }

    // Union by name, optionally filling columns missing on one side with null. The flag is
    // converted as PySpark's set operation sets it: None is a `TypeError`.
    #[pyo3(
        signature = (other, allowMissingColumns=ProtoBool::default()),
        text_signature = "($self, other, allowMissingColumns=False)"
    )]
    #[doc = connect_doc!(DataFrame.unionByName)]
    fn unionByName(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        allowMissingColumns: ProtoBool,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        let other = this.same_session(other)?;
        // As PySpark's set operation, which takes the flag when a request sends the plan: a
        // flag that a protobuf bool does not take fails each request, not this call.
        let dataframe = match allowMissingColumns.value(py) {
            Ok(allow_missing_columns) => this
                .plan()
                .union_by_name_opt(other.plan(), allow_missing_columns),
            Err(err) => this
                .plan()
                .union_by_name_unbuildable(other.plan(), unbuildable_error(py, err)?),
        };
        set_operation_result(py, &this, &other, dataframe)
    }

    // Intersect with another DataFrame.
    #[doc = connect_doc!(DataFrame.intersect)]
    fn intersect(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        let other = this.same_session(other)?;
        let dataframe = this.plan().intersect(other.plan());
        set_operation_result(py, &this, &other, dataframe)
    }

    // Intersect keeping duplicates (reference pyspark `intersectAll`).
    #[pyo3(name = "intersectAll")]
    #[doc = connect_doc!(DataFrame.intersectAll)]
    fn intersect_all(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        let other = this.same_session(other)?;
        let dataframe = this.plan().intersect_all(other.plan());
        set_operation_result(py, &this, &other, dataframe)
    }

    // Subtract another DataFrame.
    #[doc = connect_doc!(DataFrame.subtract)]
    fn subtract(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        let other = this.same_session(other)?;
        let dataframe = this.plan().subtract(other.plan());
        set_operation_result(py, &this, &other, dataframe)
    }

    // Repartition. Mirrors reference `repartition(numPartitions, *cols)` where the
    // first argument may instead be a Column/str - `df.repartition("country")` /
    // `df.repartition(col("country"))` repartition by that column at the default
    // partition count (no explicit number).
    #[pyo3(signature = (numPartitions, *cols))]
    #[doc = connect_doc!(DataFrame.repartition)]
    fn repartition(
        slf: &Bound<'_, Self>,
        numPartitions: Bound<'_, PyAny>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        // As PySpark's: a positive partition count, or a partition column (str/Column), at
        // the server's count (passed as 0 = unset). The count is the proto's int32 of the plan,
        // converted after the columns, as in `repartitionByRange`.
        if numPartitions.is_instance_of::<PyInt>() {
            positive_partitions(&numPartitions)?;
            let py = numPartitions.py();
            if cols.is_empty() {
                let n = proto_int(&numPartitions, "spark.connect.Repartition.num_partitions");
                let df = converted_plan(py, this.plan(), n, |n| this.plan().repartition(n))?;
                return this.of_same_schema(this.derived(df));
            }
            // As PySpark's `_to_cols(cols)`, which does not unpack a list among them.
            let exprs: Vec<_> = to_columns_as_given(cols)?
                .iter()
                .map(|c| c.expression().clone())
                .collect();
            // PySpark's `RepartitionByExpression.plan` builds the partition expressions, then the
            // input, and sets the count last: a count that protobuf refuses fails the plan, which
            // is built here with no count, so that the errors of the columns and of the input come
            // first, as in `repartitionByRange`.
            let df = match proto_int(
                &numPartitions,
                "spark.connect.RepartitionByExpression.num_partitions",
            ) {
                Ok(n) => this.plan().repartition_by_expressions(n, exprs),
                // A count of the plan that is never built, which the core takes.
                Err(err) => this
                    .plan()
                    .repartition_by_expressions(1, exprs)
                    .unbuildable_with_cached_schema(unbuildable_error(py, err)?),
            };
            return this.of_same_schema(this.derived(df));
        }
        if !(numPartitions.is_instance_of::<pyo3::types::PyString>()
            || numPartitions.cast::<PyColumn>().is_ok())
        {
            return Err(not_expected_type(
                &numPartitions,
                "numPartitions",
                "Column or str",
            ));
        }
        let mut all = vec![numPartitions];
        all.extend(cols);
        let exprs: Vec<_> = to_columns_as_given(all)?
            .iter()
            .map(|c| c.expression().clone())
            .collect();
        this.of_same_schema(this.derived(this.plan().repartition_by_expressions(0, exprs)))
    }

    // Coalesce into `numPartitions` partitions, as PySpark's `coalesce`: any value that is
    // `> 0`, which is then the proto's int32 of the plan, as in `repartitionByRange`.
    #[doc = connect_doc!(DataFrame.coalesce)]
    fn coalesce(slf: &Bound<'_, Self>, numPartitions: &Bound<'_, PyAny>) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        positive_partitions(numPartitions)?;
        let value = proto_int(numPartitions, "spark.connect.Repartition.num_partitions");
        let df = converted_plan(numPartitions.py(), this.plan(), value, |num_partitions| {
            this.plan().coalesce(num_partitions)
        })?;
        this.of_same_schema(this.derived(df))
    }

    // Convert to DataFrame with new column names.
    #[pyo3(signature = (*cols))]
    #[doc = connect_doc!(DataFrame.toDF)]
    fn toDF(slf: &Bound<'_, Self>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        // As PySpark's, of strings, which its `ToDF` plan sets when it builds the plan.
        let mut args = crate::plan_args::PlanArgs::new();
        let mut names = Vec::with_capacity(cols.len());
        for col in &cols {
            if !col.is_instance_of::<pyo3::types::PyString>() {
                return Err(not_expected_type(col, "cols", "list[str]"));
            }
            names.push(args.str(col)?);
        }
        Ok(this.derived(args.plan(this.plan().to_df(names))))
    }

    // Alias this DataFrame, as PySpark's, whose `SubqueryAlias` plan sets the alias when it
    // builds the plan, as protobuf's string field takes it: bytes of UTF-8 too, and anything
    // else fails the requests of the DataFrame.
    #[doc = connect_doc!(DataFrame.alias)]
    fn alias(slf: &Bound<'_, Self>, alias: &Bound<'_, PyAny>) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        let mut args = crate::plan_args::PlanArgs::new();
        let alias = args.str(alias)?;
        this.of_same_schema(this.derived(args.plan(this.plan().alias(&alias))))
    }

    // Group by columns for aggregation, as PySpark's `GroupedData(df=self, group_type="groupby",
    // grouping_cols=self._to_cols(*cols, allow_ordinal=True))`, of this DataFrame object.
    #[pyo3(signature = (*cols))]
    #[doc = connect_doc!(DataFrame.groupBy)]
    fn groupBy<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
        cols: Vec<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyGroupedData>> {
        let columns = PlanFrame::of(slf)?.grouping_columns(py, cols, "cols")?;
        PyGroupedData::of_grouping(slf.as_any(), "groupby", &columns, None)
    }

    // Rollup multi-dimensional aggregation, as PySpark's `GroupedData(df=self,
    // group_type="rollup", ...)`.
    #[pyo3(signature = (*cols))]
    #[doc = connect_doc!(DataFrame.rollup)]
    fn rollup<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
        cols: Vec<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyGroupedData>> {
        let columns = PlanFrame::of(slf)?.grouping_columns(py, cols, "cols")?;
        PyGroupedData::of_grouping(slf.as_any(), "rollup", &columns, None)
    }

    // Cube multi-dimensional aggregation, as PySpark's `GroupedData(df=self, group_type="cube",
    // ...)`.
    #[pyo3(signature = (*cols))]
    #[doc = connect_doc!(DataFrame.cube)]
    fn cube<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
        cols: Vec<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyGroupedData>> {
        let columns = PlanFrame::of(slf)?.grouping_columns(py, cols, "cols")?;
        PyGroupedData::of_grouping(slf.as_any(), "cube", &columns, None)
    }

    // Grouping sets aggregation: `groupingSets([[a, b], [a], []], "a", "b")`.
    //
    // As PySpark's: `*cols` are the grouping columns, the columns of the result besides the
    // aggregations, of Columns, names and ordinals, as are those of the sets, of
    // `GroupedData(df=self, group_type="grouping_sets", grouping_cols=..., grouping_sets=...)`,
    // whose `_grouping_sets` are lists of the Columns of each set.
    #[pyo3(name = "groupingSets", signature = (groupingSets, *cols))]
    #[doc = connect_doc!(DataFrame.groupingSets)]
    fn grouping_sets<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
        groupingSets: &Bound<'py, PyAny>,
        cols: Vec<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyGroupedData>> {
        let this = PlanFrame::of(slf)?;

        use pyo3::types::{PyInt, PyString};
        // As PySpark's `_to_cols(grouping_set)`: a set is a sequence of columns, or one column.
        let mut sets = Vec::new();
        for set in groupingSets.try_iter()? {
            let set = set?;
            let one = set.is_instance_of::<PyString>()
                || set.is_instance_of::<PyInt>()
                || set.is_instance_of::<PyColumn>();
            let sequence = !one
                && py
                    .import("collections.abc")?
                    .getattr("Sequence")
                    .and_then(|sequence| set.is_instance(&sequence))?;
            let items = if sequence {
                set.try_iter()?.collect::<PyResult<Vec<_>>>()?
            } else {
                vec![set]
            };
            sets.push(this.grouping_column_list(py, items, "groupingSets")?);
        }
        // As PySpark's `_to_cols(cols)`, of the columns themselves, not a list of them.
        let columns = this.grouping_column_list(py, cols, "cols")?;
        drop(this);
        PyGroupedData::of_grouping(slf.as_any(), "grouping_sets", &columns, Some(&sets[..]))
    }

    // Drop duplicates within the event-time watermark.
    #[pyo3(name = "dropDuplicatesWithinWatermark", signature = (subset=None))]
    #[doc = connect_doc!(DataFrame.dropDuplicatesWithinWatermark)]
    fn drop_duplicates_within_watermark(
        slf: &Bound<'_, Self>,
        subset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        let mut args = crate::plan_args::PlanArgs::new();
        let subset = deduplicate_subset(&mut args, subset)?;
        let refs: Option<Vec<&str>> = subset
            .as_ref()
            .map(|names| names.iter().map(|s| s.as_str()).collect());
        Ok(this.derived(args.plan(this.plan().drop_duplicates_within_watermark(refs))))
    }

    // `DataFrame.toJSON`, as PySpark Connect's: a DataFrame of each row as a JSON object
    // string, in a column `value`, of `self.select(F.to_json(F.struct(F.col("*")))
    // .alias("value"))`, so its expressions have the origins of `col` and `alias`.
    #[pyo3(name = "toJSON")]
    #[doc = connect_doc!(DataFrame.toJSON)]
    fn to_json(slf: &Bound<'_, Self>) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        use spark_connect::functions::{col, r#struct, to_json};
        let star = {
            let _origin = crate::origin::with_origin("col")?;
            col("*")
        };
        let json = to_json(r#struct([star]));
        let value = {
            let _origin = crate::origin::with_origin("alias")?;
            json.alias("value")
        };
        Ok(this.derived(this.plan().select([value])))
    }

    // Whether this DataFrame is semantically equal to `other`.
    #[pyo3(name = "sameSemantics")]
    #[doc = connect_doc!(DataFrame.sameSemantics)]
    fn same_semantics(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let this = Frame::of(slf)?;
        // As PySpark's: a DataFrame, else NOT_EXPECTED_TYPE, of the same session.
        if !other.is_instance_of::<PyDataFrame>() {
            return Err(not_expected_type(other, "other", "DataFrame"));
        }
        // Of the request of each, as PySpark's `other._plan.to_proto(other._session.client)`.
        let other_frame = this.same_session(other)?.request()?;
        let other = other_frame.dataframe.clone();
        interruptible_of(
            py,
            &this.dataframe.spark_session(),
            &[this.parts(), other_frame.parts()],
            || this.dataframe.same_semantics(&other),
        )
    }

    // The server-side hash of this DataFrame's logical plan, asked once, as PySpark's.
    #[pyo3(name = "semanticHash")]
    fn semantic_hash(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<i32> {
        let this = Frame::of(slf)?;
        cached(&this.semantic_hash, || {
            interruptible_of(py, &this.dataframe.spark_session(), &[this.parts()], || {
                this.dataframe.semantic_hash().map_err(Box::new)
            })
        })
    }

    // Sort within each partition.
    #[pyo3(name = "sortWithinPartitions", signature = (*cols, **kwargs))]
    #[doc = connect_doc!(DataFrame.sortWithinPartitions)]
    fn sort_within_partitions(
        slf: &Bound<'_, Self>,
        _py: Python<'_>,
        cols: Vec<Bound<'_, PyAny>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        let exprs = this.sort_columns(_py, cols, kwargs.as_ref())?;
        this.of_same_schema(this.derived(this.plan().sort_within_partitions(exprs)))
    }

    // Except all (keeps duplicates).
    #[pyo3(name = "exceptAll")]
    #[doc = connect_doc!(DataFrame.exceptAll)]
    fn except_all(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        let other = this.same_session(other)?;
        let dataframe = this.plan().except_all(other.plan());
        set_operation_result(py, &this, &other, dataframe)
    }

    // Rename multiple columns from an {old: new} mapping.
    #[pyo3(name = "withColumnsRenamed")]
    #[doc = connect_doc!(DataFrame.withColumnsRenamed)]
    fn with_columns_renamed(
        slf: &Bound<'_, Self>,
        colsMap: &Bound<'_, PyAny>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        // As PySpark's, of a dict, whose renames it sends in order.
        let Ok(cols_map) = colsMap.cast::<PyDict>() else {
            return Err(not_expected_type(colsMap, "colsMap", "dict"));
        };
        // Its `WithColumnsRenamed` plan sets the names of each rename, the old one first, in its
        // string fields when it builds the plan, as protobuf takes them: bytes of UTF-8 too.
        let mut args = crate::plan_args::PlanArgs::new();
        let mut renames = Vec::with_capacity(cols_map.len());
        for (existing, new) in cols_map.iter() {
            let existing = args.str(&existing)?;
            renames.push((existing, args.str(&new)?));
        }
        Ok(this.derived(args.plan(this.plan().with_columns_renamed(renames))))
    }

    // The columns of this DataFrame whose names match a regex, as a Column. Mirrors
    // `DataFrame.colRegex`.
    #[pyo3(name = "colRegex")]
    #[doc = connect_doc!(DataFrame.colRegex)]
    fn col_regex(
        slf: &Bound<'_, Self>,
        colName: &Bound<'_, PyAny>,
    ) -> PyResult<crate::column::ColumnOut> {
        let this = PlanFrame::of(slf)?;
        // As PySpark's: a str, else NOT_EXPECTED_TYPE.
        if !colName.is_instance_of::<pyo3::types::PyString>() {
            return Err(not_expected_type(colName, "colName", "str"));
        }
        // Its `UnresolvedRegex` sets the name when it builds the plan.
        let mut args = crate::plan_args::PlanArgs::new();
        let name = args.str(colName)?;
        Ok(crate::column::ColumnOut::new(
            args.column(this.plan().col_regex(&name)),
        ))
    }

    // Basic statistics for the given columns (count/mean/stddev/min/max), of column names or
    // a list of them, as PySpark's `DataFrame.describe`.
    #[pyo3(signature = (*cols))]
    #[doc = connect_doc!(DataFrame.describe)]
    fn describe(slf: &Bound<'_, Self>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        let mut args = crate::plan_args::PlanArgs::new();
        let cols = crate::stat_arguments::describe_columns(&mut args, &cols)?;
        Ok(this.derived(args.plan(this.plan().describe(cols))))
    }

    // Summary statistics (percentiles etc.), of strings, as PySpark's `DataFrame.summary`.
    #[pyo3(signature = (*statistics))]
    #[doc = connect_doc!(DataFrame.summary)]
    fn summary(slf: &Bound<'_, Self>, statistics: Vec<Bound<'_, PyAny>>) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        let mut args = crate::plan_args::PlanArgs::new();
        let statistics = crate::stat_arguments::summary_statistics(&mut args, &statistics)?;
        Ok(this.derived(args.plan(this.plan().summary(statistics))))
    }

    // Persist with the default MEMORY_AND_DISK storage level.
    #[doc = connect_doc!(DataFrame.cache)]
    fn cache<'py>(slf: &Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        // As PySpark's, which returns the DataFrame itself.
        let this = Frame::of(slf)?;
        let df = &this.dataframe;
        interruptible_of(py, &df.spark_session(), &[this.parts()], || {
            df.cache().map_err(Box::new)
        })?;
        Ok(slf.clone())
    }

    // Mark the DataFrame as non-persistent. As PySpark's `unpersist(blocking=False)`, which sets
    // `blocking` in the bool field of its request unless it is None, as
    // [`crate::streaming::proto_bool`] converts it: None sends no `blocking`.
    #[pyo3(
        signature = (blocking=ProtoBool::default()),
        text_signature = "($self, blocking=False)"
    )]
    #[doc = connect_doc!(DataFrame.unpersist)]
    fn unpersist<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
        blocking: ProtoBool,
    ) -> PyResult<Bound<'py, Self>> {
        let blocking = blocking.value_unless_none(py)?;
        // As PySpark's, which returns the DataFrame itself.
        let this = Frame::of(slf)?;
        let df = &this.dataframe;
        interruptible_of(py, &df.spark_session(), &[this.parts()], || {
            df.unpersist_opt(blocking).map_err(Box::new)
        })?;
        Ok(slf.clone())
    }

    // Print the plan with Python's `print`, as PySpark's `DataFrame.explain`, which prints
    // `_explain_string(extended, mode)`.
    #[pyo3(signature = (extended=None, mode=None))]
    #[doc = connect_doc!(DataFrame.explain)]
    fn explain(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        extended: Option<&Bound<'_, PyAny>>,
        mode: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let plan = Self::explain_string(slf, py, extended, mode)?;
        py.import("builtins")?.call_method1("print", (plan,))?;
        Ok(())
    }

    /// The plan that `explain` prints, as PySpark's `DataFrame._explain_string`: `extended` is
    /// a bool, for the "extended" or the "simple" mode, or a mode, and `mode` a mode, but not
    /// both; the mode is "simple" without either.
    #[pyo3(name = "_explain_string", signature = (extended=None, mode=None))]
    fn explain_string(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        extended: Option<&Bound<'_, PyAny>>,
        mode: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        let this = Frame::of(slf)?;
        use pyo3::types::{PyBool, PyString};
        let explain_mode: String = match (extended, mode) {
            (Some(_), Some(_)) => {
                return Err(crate::column::raise_pyspark(
                    py,
                    "PySparkValueError",
                    "CANNOT_SET_TOGETHER",
                    &[("arg_list", "extended and mode")],
                ));
            }
            (None, None) => "simple".to_string(),
            (Some(extended), None) if extended.is_instance_of::<PyBool>() => {
                if extended.extract::<bool>()? {
                    "extended"
                } else {
                    "simple"
                }
                .to_string()
            }
            // A mode that UTF-8 cannot encode is none of the modes, which the core refuses.
            (Some(arg), None) | (None, Some(arg)) if arg.is_instance_of::<PyString>() => arg
                .cast::<PyString>()?
                .to_cow()
                .map(|mode| mode.into_owned())
                .unwrap_or_default(),
            (Some(arg), None) | (None, Some(arg)) => {
                return Err(crate::column::raise_pyspark(
                    py,
                    "PySparkTypeError",
                    "NOT_EXPECTED_TYPE",
                    &[
                        ("expected_type", "bool or str"),
                        ("arg_name", "extended (optional) and mode (optional)"),
                        ("arg_type", &arg.get_type().str()?.to_string()),
                    ],
                ));
            }
        };
        interruptible_of(py, &this.dataframe.spark_session(), &[this.parts()], || {
            this.dataframe
                .explain_string(&explain_mode)
                .map_err(Box::new)
        })
    }

    // Register this DataFrame as a temporary view, of a name that PySpark's `CreateView` sets in
    // its string field as it is, as [`crate::str_argument`] converts it, as the other views do.
    #[pyo3(name = "createOrReplaceTempView")]
    #[doc = connect_doc!(DataFrame.createOrReplaceTempView)]
    fn create_or_replace_temp_view(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        name: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let this = Frame::of(slf)?;
        let name = crate::str_argument(name)?;
        interruptible_of(py, &this.dataframe.spark_session(), &[this.parts()], || {
            this.dataframe
                .create_or_replace_temp_view(&name)
                .map_err(Box::new)
        })?;
        // As PySpark's, which sets `_execution_info` of the command.
        this.publish_execution(py)
    }

    // Drop rows containing nulls, as PySpark's `DataFrame.dropna`.
    #[pyo3(
        signature = (how=None, thresh=None, subset=None),
        text_signature = "($self, how='any', thresh=None, subset=None)"
    )]
    #[doc = connect_doc!(DataFrame.dropna)]
    fn dropna(
        slf: &Bound<'_, Self>,
        how: Option<&Bound<'_, PyAny>>,
        thresh: Option<&Bound<'_, PyAny>>,
        subset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        crate::stat_arguments::dropna(this.plan(), how, thresh, subset)
    }

    // Fill null values, as PySpark's `DataFrame.fillna`: `value` is a scalar, over
    // `subset`, or a {column: value} mapping.
    #[pyo3(signature = (value, subset=None))]
    #[doc = connect_doc!(DataFrame.fillna)]
    fn fillna(
        slf: &Bound<'_, Self>,
        value: &Bound<'_, PyAny>,
        subset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        crate::stat_arguments::fillna(this.plan(), value, subset)
    }

    // Statistical functions (`df.stat`): a new object at each access, as PySpark's
    // `DataFrameStatFunctions(self)`, of this DataFrame object, its `df`.
    #[getter]
    #[pyo3(name = "stat")]
    #[doc = connect_doc!(DataFrame.stat)]
    fn stat<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, crate::stat::PyStatFunctions>> {
        crate::stat::PyStatFunctions::of_frame(slf.as_any())
    }

    // `df.crosstab(col1, col2)` - pyspark exposes it on DataFrame as well as
    // `df.stat.crosstab`.
    #[doc = connect_doc!(DataFrame.crosstab)]
    fn crosstab(
        slf: &Bound<'_, Self>,
        col1: &Bound<'_, PyAny>,
        col2: &Bound<'_, PyAny>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        crate::stat_arguments::crosstab(&this.plan().stat(), col1, col2)
    }

    // `df.freqItems(cols, support=None)` (also on df.stat).
    #[pyo3(name = "freqItems", signature = (cols, support=None))]
    #[doc = connect_doc!(DataFrame.freqItems)]
    fn freq_items(
        slf: &Bound<'_, Self>,
        cols: &Bound<'_, PyAny>,
        support: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        crate::stat_arguments::freq_items(&this.plan().stat(), cols, support)
    }

    // `df.approxQuantile(col, probabilities, relativeError)` (also on df.stat): the
    // quantiles of a column, or a list of them of each of a list or tuple of columns.
    #[pyo3(name = "approxQuantile")]
    #[doc = connect_doc!(DataFrame.approxQuantile)]
    fn approx_quantile(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        col: &Bound<'_, PyAny>,
        probabilities: &Bound<'_, PyAny>,
        relativeError: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let this = Frame::of(slf)?;
        crate::stat_arguments::approx_quantile(
            py,
            &this.dataframe.stat(),
            col,
            probabilities,
            relativeError,
        )
    }

    // Observe metrics, as PySpark's `DataFrame.observe(observation, *exprs)`: the Columns
    // `exprs`, which an Observation receives by its `_on`, of a subclass too, or which are
    // observed by a name.
    #[pyo3(signature = (observation, *exprs))]
    #[doc = connect_doc!(DataFrame.observe)]
    fn observe<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
        observation: &Bound<'py, PyAny>,
        exprs: Vec<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if exprs.is_empty() {
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkValueError",
                "CANNOT_BE_EMPTY",
                &[("item", "exprs")],
            ));
        }
        let mut columns = Vec::with_capacity(exprs.len());
        for expr in &exprs {
            match expr.cast::<PyColumn>() {
                Ok(column) => columns.push(crate::column::column_of(column)?.expression().clone()),
                Err(_) => {
                    return Err(crate::column::raise_pyspark(
                        py,
                        "PySparkTypeError",
                        "NOT_EXPECTED_TYPE",
                        &[
                            ("arg_name", "exprs"),
                            ("expected_type", "list[Column]"),
                            ("arg_type", "tuple"),
                        ],
                    ));
                }
            }
        }
        if observation.is_instance_of::<crate::observation::PyObservation>() {
            // As PySpark's, which gives the DataFrame of the `_on` of the Observation, of a
            // subclass too, and then sets `res._cached_schema = self._cached_schema`.
            let mut arguments = vec![slf.clone().into_any()];
            arguments.extend(exprs);
            let observed = observation.call_method1(
                pyo3::intern!(py, "_on"),
                pyo3::types::PyTuple::new(py, arguments)?,
            )?;
            let attribute = pyo3::intern!(py, "_cached_schema");
            observed.setattr(attribute, slf.as_any().getattr(attribute)?)?;
            return Ok(observed);
        }
        let this = PlanFrame::of(slf)?;
        if observation.is_instance_of::<pyo3::types::PyString>() {
            // Its `CollectMetrics` plan sets the name when it builds the plan, before the metrics.
            let mut args = crate::plan_args::PlanArgs::new();
            let name = args.str(observation)?;
            let args = args.before(&mut columns);
            let observed =
                this.of_same_schema(this.derived(args.plan(this.plan().observe(&name, columns))))?;
            return Ok(observed.into_pyobject(py)?.into_any());
        }
        let type_name = observation.get_type().name()?.to_string();
        Err(crate::column::raise_pyspark(
            py,
            "PySparkTypeError",
            "NOT_EXPECTED_TYPE",
            &[
                ("arg_name", "observation"),
                ("expected_type", "Observation or str"),
                ("arg_type", &type_name),
            ],
        ))
    }

    // Register as a temporary view (session-scoped).
    #[pyo3(name = "createTempView")]
    #[doc = connect_doc!(DataFrame.createTempView)]
    fn create_temp_view(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        name: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let this = Frame::of(slf)?;
        let name = crate::str_argument(name)?;
        interruptible_of(py, &this.dataframe.spark_session(), &[this.parts()], || {
            this.dataframe.create_temp_view(&name).map_err(Box::new)
        })?;
        // As PySpark's, which sets `_execution_info` of the command.
        this.publish_execution(py)
    }

    // Register as a global temporary view.
    #[pyo3(name = "createGlobalTempView")]
    #[doc = connect_doc!(DataFrame.createGlobalTempView)]
    fn create_global_temp_view(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        name: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let this = Frame::of(slf)?;
        let name = crate::str_argument(name)?;
        interruptible_of(py, &this.dataframe.spark_session(), &[this.parts()], || {
            this.dataframe
                .create_global_temp_view(&name)
                .map_err(Box::new)
        })?;
        // As PySpark's, which sets `_execution_info` of the command.
        this.publish_execution(py)
    }

    // Register as (or replace) a global temporary view.
    #[pyo3(name = "createOrReplaceGlobalTempView")]
    #[doc = connect_doc!(DataFrame.createOrReplaceGlobalTempView)]
    fn create_or_replace_global_temp_view(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        name: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let this = Frame::of(slf)?;
        let name = crate::str_argument(name)?;
        interruptible_of(py, &this.dataframe.spark_session(), &[this.parts()], || {
            this.dataframe
                .create_or_replace_global_temp_view(&name)
                .map_err(Box::new)
        })?;
        // As PySpark's, which sets `_execution_info` of the command.
        this.publish_execution(py)
    }

    // Deprecated alias for createOrReplaceTempView, as PySpark's, which warns that it is
    // deprecated, of its line of `pyspark.sql.connect.dataframe` (see `warn_as_pyspark`), and
    // then replaces the view as `createOrReplaceTempView` does.
    #[pyo3(name = "registerTempTable")]
    #[doc = connect_doc!(DataFrame.registerTempTable)]
    fn register_temp_table(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        name: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        crate::errors::warn_as_pyspark(
            py,
            "pyspark.sql.connect.dataframe",
            2259,
            "Deprecated in 2.0, use createOrReplaceTempView instead.",
            &py.get_type::<pyo3::exceptions::PyFutureWarning>(),
        )?;
        Self::create_or_replace_temp_view(slf, py, name)
    }

    // Reliable checkpoint (materialize + truncate lineage), at once if `eager`, as PySpark's.
    #[pyo3(signature = (eager=DefaultedOption::Omitted), text_signature = "($self, eager=True)")]
    #[doc = connect_doc!(DataFrame.checkpoint)]
    fn checkpoint(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        eager: DefaultedOption,
    ) -> PyResult<NewDataFrame> {
        let this = Frame::of(slf)?;
        let eager = checkpoint_eager(py, &eager)?;
        let checkpointed =
            interruptible_of(py, &this.dataframe.spark_session(), &[this.parts()], || {
                this.dataframe.checkpoint_with(eager).map_err(Box::new)
            })?;
        // As PySpark's, which sets `_execution_info` of the command.
        this.publish_execution(py)?;
        Ok(this.derived(checkpointed))
    }

    // Local checkpoint, at once if `eager`, kept at `storageLevel`, else at the server's
    // default, as PySpark's.
    #[pyo3(
        name = "localCheckpoint",
        signature = (eager=DefaultedOption::Omitted, storageLevel=None),
        text_signature = "($self, eager=True, storageLevel=None)"
    )]
    #[doc = connect_doc!(DataFrame.localCheckpoint)]
    fn local_checkpoint(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        eager: DefaultedOption,
        storageLevel: Option<Bound<'_, PyAny>>,
    ) -> PyResult<NewDataFrame> {
        let this = Frame::of(slf)?;
        let eager = checkpoint_eager(py, &eager)?;
        let level = storageLevel
            .filter(|level| !level.is_none())
            .map(|level| crate::storagelevel::storage_level_proto(&level))
            .transpose()?;
        let checkpointed =
            interruptible_of(py, &this.dataframe.spark_session(), &[this.parts()], || {
                this.dataframe
                    .local_checkpoint_with(eager, level)
                    .map_err(Box::new)
            })?;
        // As PySpark's, which sets `_execution_info` of the command.
        this.publish_execution(py)?;
        Ok(this.derived(checkpointed))
    }

    // Pearson correlation between two columns; PySpark's `method` is only "pearson".
    #[pyo3(signature = (col1, col2, method=None))]
    #[doc = connect_doc!(DataFrame.corr)]
    fn corr(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        col1: &Bound<'_, PyAny>,
        col2: &Bound<'_, PyAny>,
        method: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<f64> {
        let this = Frame::of(slf)?;
        crate::stat_arguments::corr(py, &this.dataframe.stat(), col1, col2, method)
    }

    // Sample covariance between two columns.
    #[doc = connect_doc!(DataFrame.cov)]
    fn cov(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        col1: &Bound<'_, PyAny>,
        col2: &Bound<'_, PyAny>,
    ) -> PyResult<f64> {
        let this = Frame::of(slf)?;
        crate::stat_arguments::cov(py, &this.dataframe.stat(), col1, col2)
    }

    // Stratified sample without replacement per key. Without a seed, the client draws one
    // from Python's `random`, as PySpark Connect's `sampleBy` does.
    #[pyo3(name = "sampleBy", signature = (col, fractions, seed=None))]
    #[doc = connect_doc!(DataFrame.sampleBy)]
    fn sample_by(
        slf: &Bound<'_, Self>,
        col: &Bound<'_, PyAny>,
        fractions: &Bound<'_, PyAny>,
        seed: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        let py = col.py();
        let (col, fractions) = crate::stat::sample_by_arguments(col, fractions)?;
        // The seed is converted as the proto's long as PySpark sends the plan, after its checks.
        let (seed, plan_error) = match crate::stat::sample_by_seed(py, seed) {
            Ok(seed) => (seed, None),
            Err(err) => (0, Some(unbuildable_error(py, err)?)),
        };
        let df = this
            .plan()
            .stat()
            .sample_by(col, fractions, Some(seed))
            .to_pyerr()?;
        Ok(this.derived(match plan_error {
            Some(error) => df.unbuildable(error),
            None => df,
        }))
    }

    // Replace values, as PySpark's `DataFrame.replace`: `to_replace` is a {old: new} dict, or
    // a value or a list or tuple of them with `value`, a value or a list or tuple of them.
    #[pyo3(signature = (to_replace, value=ReplaceValue::Omitted, subset=None))]
    #[doc = connect_doc!(DataFrame.replace)]
    fn replace(
        slf: &Bound<'_, Self>,
        to_replace: &Bound<'_, PyAny>,
        value: ReplaceValue,
        subset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        replace_values(this.plan(), to_replace, value, subset)
    }

    // Reconcile to a target schema (StructType or DDL string). As PySpark's, a schema that is
    // not a DataType is kept as it is, and the DataFrame fails at each action, which PySpark's
    // client raises when it builds the plan; a DDL string is not parsed. As PySpark's, the
    // schema is the cached schema of the DataFrame whatever it is, as the object that it is, so a
    // DataType that is not a struct, which the core does not cache, is kept as it is too, while
    // its plan is sent; and as PySpark's `ToSchema` converts the type object when it builds the
    // plan, a `StructType` changed in place after the call is the schema of the DataFrame, and
    // the schema that its requests send, as it is then.
    #[doc = connect_doc!(DataFrame.to)]
    fn to(slf: &Bound<'_, Self>, schema: &Bound<'_, PyAny>) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        if schema.is_none() {
            // As PySpark's `assert schema is not None`.
            return Err(pyo3::exceptions::PyAssertionError::new_err(()));
        }
        // As PySpark's `ToSchema`, which converts the schema object when it builds the plan: each
        // request reads the object as it is then (see `crate::udf_command::data_type_pickler`),
        // of the errors of PySpark's `pyspark_types_to_proto_types`, its `UNSUPPORTED_OPERATION`
        // and those of the reads of the elements of a struct whose fields hold one that is not a
        // `StructField`. The DataFrame caches the type that stands for the object at the call
        // (see `crate::types::lenient_type`), and its `_cached_schema` is the object, as PySpark's
        // `res._cached_schema = schema`.
        if schema.is_instance_of::<crate::types::PyDataType>() {
            let stand_in = crate::types::lenient_type(schema)?;
            let reader = crate::udf_command::data_type_pickler(schema.clone().unbind());
            let result = this.derived(this.plan().to_read_at_build(stand_in, reader));
            return Ok(result.with_cached_schema(Some(schema.clone().unbind())));
        }
        // Of a value that is not a data type, such as a str, PySpark's conversion of the plan
        // raises its `UNSUPPORTED_OPERATION` at each request.
        let operation = crate::plan_args::data_type_operation(schema)?;
        let error = spark_connect_core::error::SparkError::classed(
            spark_connect_core::error::SparkErrorKind::ValueError,
            "UNSUPPORTED_OPERATION",
            &[("operation", &operation)],
        );
        let result = this.derived(this.plan().unbuildable(error));
        Ok(result.with_cached_schema(Some(schema.clone().unbind())))
    }

    // Set the metadata of a column, as PySpark's: a dict of any JSON values, which it sends as
    // `json.dumps(metadata)`, and `NOT_EXPECTED_TYPE` for one that is not a dict.
    #[pyo3(name = "withMetadata")]
    #[doc = connect_doc!(DataFrame.withMetadata)]
    fn with_metadata(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        columnName: &Bound<'_, PyAny>,
        metadata: &Bound<'_, PyAny>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        if !metadata.is_instance_of::<PyDict>() {
            return Err(not_expected_type(metadata, "metadata", "dict"));
        }
        // As PySpark's, which takes `self[columnName]` and `json.dumps(metadata)`, in this order,
        // for its `WithColumns` plan, which asserts a str name, and sets it when it builds the
        // plan. So an item that `__getitem__` does not take fails first, and an int reads the
        // columns before the assertion fails.
        Self::__getitem__(slf, py, columnName)?;
        let json: String = py
            .import("json")?
            .call_method1("dumps", (metadata,))?
            .extract()?;
        if !columnName.is_instance_of::<pyo3::types::PyString>() {
            return Err(pyo3::exceptions::PyAssertionError::new_err(()));
        }
        let mut args = crate::plan_args::PlanArgs::new();
        let name = args.str(columnName)?;
        Ok(this.derived(args.plan(this.plan().with_metadata_json(&name, &json))))
    }

    // Define an event-time watermark, as PySpark's: `NOT_EXPECTED_TYPE` for an empty event time
    // or delay, or one that is not a str.
    #[pyo3(name = "withWatermark")]
    #[doc = connect_doc!(DataFrame.withWatermark)]
    fn with_watermark(
        slf: &Bound<'_, Self>,
        eventTime: &Bound<'_, PyAny>,
        delayThreshold: &Bound<'_, PyAny>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        let mut args = crate::plan_args::PlanArgs::new();
        let event_time = watermark_argument(&mut args, eventTime, "eventTime")?;
        let delay_threshold = watermark_argument(&mut args, delayThreshold, "delayThreshold")?;
        Ok(this.derived(args.plan(this.plan().with_watermark(&event_time, &delay_threshold))))
    }

    // Randomly split into multiple DataFrames by the given weights.
    // As PySpark's: `VALUE_NOT_POSITIVE` for a negative weight and for weights of a sum that is
    // not positive, of their `str`; without a seed, the client chooses one of every split; a
    // seed goes through `int()`, so a float or a str is accepted.
    #[pyo3(name = "randomSplit", signature = (weights, seed=None))]
    #[doc = connect_doc!(DataFrame.randomSplit)]
    fn random_split(
        slf: &Bound<'_, Self>,
        weights: &Bound<'_, PyAny>,
        seed: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<NewDataFrame>> {
        let this = PlanFrame::of(slf)?;
        let py = weights.py();
        let not_positive = |arg_name: &str, value: &Bound<'_, PyAny>| -> PyResult<PyErr> {
            Ok(crate::column::raise_pyspark(
                py,
                "PySparkValueError",
                "VALUE_NOT_POSITIVE",
                &[
                    ("arg_name", arg_name),
                    ("arg_value", &value.str()?.to_cow()?),
                ],
            ))
        };
        // Keep Python's checks so arg_value uses str() of the original objects, as in
        // PySpark. The core also validates the extracted weights for native Rust callers.
        for weight in weights.try_iter()? {
            let weight = weight?;
            if weight.lt(0.0)? {
                return Err(not_positive("weights", &weight)?);
            }
        }
        // Drawn before the sum is checked, as PySpark does, so that a failing sum uses one too.
        let seed = match seed {
            Some(seed) => seed.clone(),
            None => python_random_seed(py)?.into_pyobject(py)?.into_any(),
        };
        let total = py.import("builtins")?.call_method1("sum", (weights,))?;
        if total.le(0)? {
            return Err(not_positive("sum(weights)", &total)?);
        }
        // As PySpark's, which iterates the weights once more for their proportions, and adds
        // them up from 0.0 for the bounds of the splits, in Python: so any iterable is taken, as
        // a set or the keys of a dict, and a weight that does not add to a float, such as a
        // Decimal, raises the TypeError of the addition. The core computes the same bounds of
        // the weights as floats.
        let mut given = Vec::new();
        let mut proportions = Vec::new();
        for weight in weights.try_iter()? {
            let weight = weight?;
            proportions.push(weight.div(&total)?);
            given.push(weight);
        }
        let mut bound = pyo3::types::PyFloat::new(py, 0.0).into_any();
        for proportion in &proportions {
            bound = bound.add(proportion)?;
        }
        let float_weights = given
            .iter()
            .map(|weight| weight.extract::<f64>())
            .collect::<PyResult<Vec<_>>>()?;
        // Converted once the sum is checked, as PySpark's plans do with int(seed). The long of the
        // proto is converted as the plan is sent: a value out of its range fails each split.
        let seed = py.import("builtins")?.getattr("int")?.call1((&seed,))?;
        let (seed, plan_error) = match seed.extract::<i64>() {
            Ok(seed) => (seed, None),
            Err(_) => (
                0,
                Some(unbuildable_error(
                    py,
                    crate::session::value_out_of_range(&seed),
                )?),
            ),
        };
        this.plan()
            .random_split(float_weights, Some(seed))
            .to_pyerr()?
            .into_iter()
            .map(|df| {
                // Of the schema that the input cached, as PySpark copies it to each split.
                this.of_same_schema(this.derived(match &plan_error {
                    Some(error) => df.unbuildable_with_cached_schema(error.clone()),
                    None => df,
                }))
            })
            .collect::<PyResult<Vec<_>>>()
    }

    // The list of files that make up this DataFrame, asked once, as PySpark's.
    #[pyo3(name = "inputFiles")]
    fn input_files(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<Vec<String>> {
        let this = Frame::of(slf)?;
        cached(&this.input_files, || {
            interruptible_of(py, &this.dataframe.spark_session(), &[this.parts()], || {
                this.dataframe.input_files().map_err(Box::new)
            })
        })
    }

    // Whether the server answers that this DataFrame is local (a local relation).
    #[pyo3(name = "isLocal")]
    fn is_local(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<bool> {
        let this = Frame::of(slf)?;
        interruptible_of(py, &this.dataframe.spark_session(), &[this.parts()], || {
            this.dataframe.is_local_on_server().map_err(Box::new)
        })
    }

    // Whether this (cached) DataFrame is cached.
    #[pyo3(name = "is_cached")]
    #[getter]
    fn is_cached(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<bool> {
        let this = Frame::of(slf)?;
        interruptible_of(py, &this.dataframe.spark_session(), &[this.parts()], || {
            this.dataframe.is_cached().map_err(Box::new)
        })
    }

    // Whether this DataFrame contains one or more sources that continuously return data, asked
    // once, as PySpark's.
    // Its request, of `isStreaming`, PySpark's `functools.cached_property`, which the module
    // `pyspark.sql.connect.dataframe` installs on the class, of its function, and which keeps the
    // answer in the `__dict__` of the DataFrame. Not a stable API.
    fn _is_streaming(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<bool> {
        let this = Frame::of(slf)?;
        crate::client_log::of_cached_property("isStreaming", || {
            interruptible_of(py, &this.dataframe.spark_session(), &[this.parts()], || {
                this.dataframe.is_streaming().map_err(Box::new)
            })
        })
    }

    // The storage level currently in use for this DataFrame.
    #[pyo3(name = "storageLevel")]
    #[getter]
    #[doc = connect_doc!(DataFrame.storageLevel)]
    fn storage_level(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let this = Frame::of(slf)?;
        let sl = interruptible_of(py, &this.dataframe.spark_session(), &[this.parts()], || {
            this.dataframe.storage_level().map_err(Box::new)
        })?;
        let cls = py.import("pyspark.storagelevel")?.getattr("StorageLevel")?;
        Ok(cls
            .call1((
                sl.use_disk,
                sl.use_memory,
                sl.use_off_heap,
                sl.deserialized,
                sl.replication,
            ))?
            .unbind())
    }

    // The SparkSession of this DataFrame, as PySpark Connect's `sparkSession`, which is
    // `self._session`, of the `__dict__` of the DataFrame.
    #[pyo3(name = "sparkSession")]
    #[getter]
    #[doc = connect_doc!(DataFrame.sparkSession)]
    fn spark_session<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::state::attr(slf.as_any(), pyo3::intern!(slf.py(), "_session"))
    }

    // The EXISTS subquery of this DataFrame, as a Column. Mirrors `DataFrame.exists()`.
    #[doc = connect_doc!(DataFrame.exists)]
    fn exists(slf: &Bound<'_, Self>) -> PyResult<crate::column::ColumnOut> {
        let this = PlanFrame::of(slf)?;
        Ok(crate::column::ColumnOut::new(this.plan().exists()))
    }

    // The scalar subquery of this DataFrame, as a Column. Mirrors `DataFrame.scalar()`.
    #[doc = connect_doc!(DataFrame.scalar)]
    fn scalar(slf: &Bound<'_, Self>) -> PyResult<crate::column::ColumnOut> {
        let this = PlanFrame::of(slf)?;
        Ok(crate::column::ColumnOut::new(this.plan().scalar()))
    }

    // Return a `TableArg` wrapping this DataFrame, for use as a table-valued function
    // argument. Mirrors `DataFrame.asTable()`.
    #[pyo3(name = "asTable")]
    #[doc = connect_doc!(DataFrame.asTable)]
    fn as_table<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, crate::tablearg::PyTableArg>> {
        let this = PlanFrame::of(slf)?;
        crate::tablearg::table_arg_object(
            py,
            spark_connect::table_arg::TableArg::new(this.plan().clone()),
            Some(this.plan().spark_session().session_id().to_string()),
        )
    }

    // Deprecated alias for dropDuplicates.
    #[pyo3(name = "drop_duplicates", signature = (subset=None))]
    #[doc = connect_doc!(DataFrame.drop_duplicates)]
    fn drop_duplicates(
        slf: &Bound<'_, Self>,
        subset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<NewDataFrame> {
        Self::dropDuplicates(slf, subset)
    }

    // Alias for groupBy.
    #[pyo3(signature = (*cols))]
    #[doc = connect_doc!(DataFrame.groupby)]
    fn groupby<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
        cols: Vec<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyGroupedData>> {
        Self::groupBy(slf, py, cols)
    }

    // Apply a function to this DataFrame and return its result:
    // `df.transform(func, *args, **kwargs)` == `func(df, *args, **kwargs)`.
    #[pyo3(signature = (func, *args, **kwargs))]
    #[doc = connect_doc!(DataFrame.transform)]
    fn transform(
        slf: &Bound<'_, Self>,
        func: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        // As PySpark's, which passes `self`: a subclass of DataFrame keeps its methods and type.
        let py = slf.py();
        let mut call_args: Vec<Py<PyAny>> = vec![slf.clone().into_any().unbind()];
        for a in args.iter() {
            call_args.push(a.unbind());
        }
        let tuple = PyTuple::new(py, call_args)?;
        let result = func.call(tuple, kwargs)?;
        // As PySpark's assert that the function returned a DataFrame.
        if !result.is_instance_of::<PyDataFrame>() {
            return Err(pyo3::exceptions::PyAssertionError::new_err(format!(
                "Func returned an instance of type [{}], should have been DataFrame.",
                result.get_type().repr()?
            )));
        }
        Ok(result.unbind())
    }

    // Unpivot (wide-to-long), as PySpark's `DataFrame.unpivot`: `ids` and `values` are each a
    // column or a column name, or a list or tuple of them, and `values=None` unpivots all
    // non-id columns.
    #[pyo3(signature = (ids, values, variableColumnName, valueColumnName))]
    #[doc = connect_doc!(DataFrame.unpivot)]
    fn unpivot(
        slf: &Bound<'_, Self>,
        ids: &Bound<'_, PyAny>,
        values: Option<&Bound<'_, PyAny>>,
        variableColumnName: &Bound<'_, PyAny>,
        valueColumnName: &Bound<'_, PyAny>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        if ids.is_none() {
            return Err(pyo3::exceptions::PyAssertionError::new_err(
                "ids must not be None",
            ));
        }
        let ids = unpivot_columns(ids)?;
        let values = values.map(unpivot_columns).transpose()?;
        // The names are of the string fields of its `Unpivot` plan, which sets them as they
        // are when it builds the plan, after the columns.
        let mut args = crate::plan_args::PlanArgs::new();
        let variable = args.str(variableColumnName)?;
        let value = args.str(valueColumnName)?;
        Ok(this.derived(args.plan(this.plan().unpivot(ids, values, &variable, &value))))
    }

    // The same function as `unpivot`, as PySpark's `DataFrame.melt`.
    #[pyo3(signature = (ids, values, variableColumnName, valueColumnName))]
    #[doc = connect_doc!(DataFrame.melt)]
    fn melt(
        slf: &Bound<'_, Self>,
        ids: &Bound<'_, PyAny>,
        values: Option<&Bound<'_, PyAny>>,
        variableColumnName: &Bound<'_, PyAny>,
        valueColumnName: &Bound<'_, PyAny>,
    ) -> PyResult<NewDataFrame> {
        Self::unpivot(slf, ids, values, variableColumnName, valueColumnName)
    }

    // Persist the DataFrame at `storageLevel`, as PySpark's `DataFrame.persist()`:
    // MEMORY_AND_DISK_DESER when it is omitted, and no storage level for None, so that the
    // server caches at its `spark.sql.defaultCacheStorageLevel`.
    #[pyo3(signature = (storageLevel=crate::storagelevel::PersistLevel::Omitted))]
    #[doc = connect_doc!(DataFrame.persist)]
    fn persist<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
        storageLevel: crate::storagelevel::PersistLevel,
    ) -> PyResult<Bound<'py, Self>> {
        let level = storageLevel.proto(py)?;
        // As PySpark's, which returns the DataFrame itself.
        let this = Frame::of(slf)?;
        let df = &this.dataframe;
        interruptible_of(py, &df.spark_session(), &[this.parts()], || {
            df.persist(level).map_err(Box::new)
        })?;
        Ok(slf.clone())
    }

    /// Column access by attribute (`df.colname` -> Column), by this DataFrame's plan ID.
    /// JVM attribute names raise JVM_ATTRIBUTE_NOT_SUPPORTED. Dunder names that are not
    /// columns raise ATTRIBUTE_NOT_SUPPORTED; with PYSPARK_VALIDATE_COLUMN_NAME_LEGACY=1,
    /// all non-column names do. Single-underscore names otherwise resolve to columns.
    fn __getattr__(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        name: &Bound<'_, pyo3::types::PyString>,
    ) -> PyResult<crate::column::ColumnOut> {
        // A name that UTF-8 cannot encode is none of those below, nor a column of the server,
        // which the validation reads: it is the column of the name, which fails its requests, as
        // PySpark's `_col`, unless it is validated. The DataFrame is read where PySpark's reads
        // `self.columns` and `self._plan`, of the errors of their reads.
        let legacy = std::env::var("PYSPARK_VALIDATE_COLUMN_NAME_LEGACY").as_deref() == Ok("1");
        let Ok(text) = name.to_cow() else {
            if legacy || name.call_method1("startswith", ("__",))?.is_truthy()? {
                property_schema(slf)?.columns(py)?;
                return Err(crate::errors::raise_pyspark_of(
                    py,
                    "PySparkAttributeError",
                    "ATTRIBUTE_NOT_SUPPORTED",
                    &[("attr_name", name.as_any())],
                ));
            }
            return Ok(crate::column::ColumnOut::new(crate::plan_args::df_col(
                &plan_dataframe(slf)?,
                name.as_any(),
                false,
            )?));
        };
        let text = text.as_ref();
        // As PySpark Connect's: the attributes of the JVM of a classic DataFrame are not
        // supported, and a name is validated as a column only for a dunder name, which Python
        // may look up for an attribute, or with PYSPARK_VALIDATE_COLUMN_NAME_LEGACY.
        if ["_jseq", "_jdf", "_jmap", "_jcols", "rdd"].contains(&text) {
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkAttributeError",
                "JVM_ATTRIBUTE_NOT_SUPPORTED",
                &[("attr_name", text)],
            ));
        }
        // As PySpark's `name not in self.columns`, of the property, whose read of a plan that
        // fails to build with an AttributeError is of the Column of `_schema`, which raises
        // `NOT_ITERABLE`.
        if (legacy || text.starts_with("__"))
            && !property_schema(slf)?.columns(py)?.contains(text)?
        {
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkAttributeError",
                "ATTRIBUTE_NOT_SUPPORTED",
                &[("attr_name", text)],
            ));
        }
        // As PySpark's `self._col(name)`, of the name `*` too, of `self._plan`.
        Ok(crate::column::ColumnOut::new(
            plan_dataframe(slf)?.column_reference(text),
        ))
    }

    /// Private bridge used by pyspark.sql.dataframe.DataFrame.__getattr__ for
    /// explicit column lookup. The __getattr__ special method itself is exposed
    /// only as a slot; this ordinary method remains callable by the public base.
    fn _get_column_by_attribute(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        name: &Bound<'_, pyo3::types::PyString>,
    ) -> PyResult<crate::column::ColumnOut> {
        Self::__getattr__(slf, py, name)
    }

    /// Class attributes and all column names, matching Python Connect, of its `self.columns`,
    /// which an AttributeError of the property makes a Column, which raises `NOT_ITERABLE`.
    fn __dir__(slf: &Bound<'_, Self>) -> PyResult<Vec<String>> {
        let py = slf.py();
        // Inspect the class rather than the instance to avoid calling __dir__ again.
        let mut attrs = py.get_type::<Self>().dir()?.extract::<Vec<String>>()?;
        for name in slf.getattr(pyo3::intern!(py, "columns"))?.try_iter()? {
            attrs.push(name?.extract()?);
        }
        attrs.sort();
        attrs.dedup();
        Ok(attrs)
    }

    /// Bracket completion keeps column order, duplicates and non-identifiers, as PySpark's
    /// `self.columns`.
    fn _ipython_key_completions_<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        slf.getattr(pyo3::intern!(slf.py(), "columns"))
    }

    // Aggregate over the whole DataFrame (shorthand for `groupBy().agg(...)`).
    //
    // Mirrors `pyspark.sql.DataFrame.agg(*exprs)`.
    #[pyo3(signature = (*exprs))]
    #[doc = connect_doc!(DataFrame.agg)]
    fn agg(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        exprs: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        if exprs.is_empty() {
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkValueError",
                "CANNOT_BE_EMPTY",
                &[("item", "exprs")],
            ));
        }
        // As PySpark's: a dict of column names and functions, the aggregation of each, as of
        // `groupBy().agg`, else Columns, which it asserts.
        if let [only] = exprs.as_slice() {
            if let Ok(dict) = only.cast::<PyDict>() {
                // As PySpark's, whose `groupBy().agg()` of none of the entries asserts that
                // it has expressions.
                if dict.is_empty() {
                    return Err(pyo3::exceptions::PyAssertionError::new_err(
                        "exprs should not be empty",
                    ));
                }
                let expressions = crate::group::dict_aggregates(dict)?;
                return Ok(this.derived(this.plan().agg(expressions)));
            }
        }
        let mut expressions = Vec::with_capacity(exprs.len());
        for expr in &exprs {
            let Ok(column) = expr.cast::<PyColumn>() else {
                return Err(pyo3::exceptions::PyAssertionError::new_err(
                    "all exprs should be Expression",
                ));
            };
            expressions.push(crate::column::column_of(column)?.expression().clone());
        }
        Ok(this.derived(this.plan().agg(expressions)))
    }

    // Collect all rows into memory.
    #[doc = connect_doc!(DataFrame.collect)]
    fn collect(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let this = Frame::of(slf)?;
        // Release the GIL across the blocking RPC so other Python threads run.
        let (rows, schema) =
            interruptible_of(py, &this.dataframe.spark_session(), &[this.parts()], || {
                this.dataframe
                    .collect_with_schema_checked(check_schema_at_receipt)
                    .map_err(Box::new)
            })?;
        // As PySpark's `_to_table`, which sets `_execution_info` before the rows are converted.
        this.publish_execution(py)?;
        rows_to_py(py, &this.dataframe, rows, schema.as_ref())
    }

    // Return an iterator over rows without materializing the entire result in memory.
    //
    // Mirrors `pyspark.sql.DataFrame.toLocalIterator(prefetchPartitions=False)`, a generator:
    // the execution starts with the first row, and its errors are raised by the iteration.
    // Yields Row objects lazily as batches are fetched from the server. As PySpark Connect's,
    // which ignores `prefetchPartitions`, a batch is fetched when the iteration reaches it, so
    // the progress handlers of the execution are called by `next()`, on the iterating thread.
    #[pyo3(name = "toLocalIterator")]
    #[pyo3(
        signature = (prefetchPartitions=None),
        text_signature = "($self, prefetchPartitions=False)"
    )]
    #[doc = connect_doc!(DataFrame.toLocalIterator)]
    fn to_local_iterator<'py>(
        slf: &Bound<'py, Self>,
        prefetchPartitions: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Any value, as PySpark Connect's, which ignores it.
        let _ = prefetchPartitions;
        let py = slf.py();
        // The rows hold this DataFrame, as PySpark's generator holds `self` until it ends.
        let rows = PyLocalRowIterator::new(slf)?;
        py.import("pyspark.sql.connect.dataframe")?
            .getattr("_local_rows")?
            .call1((rows,))
    }

    /// The `pyarrow.Table` of the result as the server sends it, uncast, and the schema that the
    /// server sends with it, a `StructType`, or None: PySpark Connect's private `_to_table`, of
    /// which its `toArrow` is made. A result without batches raises the `AssertionError` of its
    /// `assert table is not None`; a Spark server sends at least one batch. The metadata of the
    /// schema and of the fields of the table are those that PySpark reads of the server's stream
    /// (`pyspark.sql.connect.client.core._table_as_received`).
    fn _to_table(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let this = Frame::of(slf)?;
        let (batches, schema) =
            interruptible_of(py, &this.dataframe.spark_session(), &[this.parts()], || {
                this.dataframe
                    .collect_record_batches_with_schema_checked(check_schema_at_receipt)
                    .map_err(Box::new)
            })?;
        if batches.is_empty() {
            return Err(pyo3::exceptions::PyAssertionError::new_err(()));
        }
        let table = crate::pyarrow_export::record_batches_to_pyarrow(py, batches)?;
        let table = py
            .import("pyspark.sql.connect.client.core")?
            .call_method1("_table_as_received", (table,))?
            .unbind();
        let schema = match &schema {
            Some(schema) => crate::types::schema_to_py(py, schema)?,
            None => py.None(),
        };
        Ok((table, schema))
    }

    // Collect the DataFrame into a `pyarrow.Table`.
    //
    // Mirrors PySpark Connect's `toArrow()`: the Arrow schema of the `schema` of the DataFrame
    // first, `to_arrow_schema(self.schema, error_on_duplicated_field_names_in_struct=True,
    // timezone="UTC")`, then the table of the result, cast to it. The schema is that of the
    // analysis of the plan, which the DataFrame keeps, so `columns` asks nothing afterwards; a
    // type that PySpark does not convert to Arrow, and duplicate names of the fields of a
    // nested struct, raise before the query runs, so before any error of its execution.
    //
    // The result is decoded from the server's Arrow IPC stream and handed to `pyarrow` through
    // arrow-rs's reader conversion, sharing compatible decoded Arrow value buffers. Bitmap
    // alignment and schemas requiring the IPC compatibility fallback can still copy.
    #[pyo3(name = "toArrow")]
    #[doc = connect_doc!(DataFrame.toArrow)]
    fn to_arrow(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let kwargs = PyDict::new(py);
        kwargs.set_item("error_on_duplicated_field_names_in_struct", true)?;
        kwargs.set_item("timezone", "UTC")?;
        // Of `self.schema`, as PySpark's (see `property_schema`).
        let arrow_schema = py
            .import("pyspark.sql.pandas.types")?
            .getattr("to_arrow_schema")?
            .call((slf.getattr(pyo3::intern!(py, "schema"))?,), Some(&kwargs))?;
        let this = Frame::of(slf)?;
        let dataframe = &this.dataframe;
        let (batches, _) =
            interruptible_of(py, &dataframe.spark_session(), &[this.parts()], || {
                dataframe
                    .collect_record_batches_with_schema_checked(check_schema_at_receipt)
                    .map_err(Box::new)
            })?;
        this.publish_execution(py)?;
        let table = crate::pyarrow_export::record_batches_to_pyarrow(py, batches)?;
        let table = table.bind(py);
        // A result without batches has no columns: the empty table of the schema.
        if table.getattr("num_columns")?.extract::<usize>()? == 0 && arrow_schema.len()? > 0 {
            return Ok(arrow_schema.call_method0("empty_table")?.unbind());
        }
        Ok(table.call_method1("cast", (arrow_schema,))?.unbind())
    }

    /// The Arrow C stream of the DataFrame, of the Arrow PyCapsule interface, as PySpark's
    /// `__arrow_c_stream__`: `pyspark.sql.interchange.SparkArrowCStreamer`, which reads the rows of
    /// one partition at a time, of the Arrow schema of the schema, and `requested_schema`, a
    /// PyCapsule of a schema, if any, which the stream is cast to as far as Arrow can.
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__(
        slf: &Bound<'_, Self>,
        requested_schema: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        Ok(py
            .import("pyspark.sql.interchange")?
            .getattr("SparkArrowCStreamer")?
            .call1((slf,))?
            .call_method1("__arrow_c_stream__", (requested_schema,))?
            .unbind())
    }

    // Collect the DataFrame into a pandas DataFrame.
    //
    // Mirrors `pyspark.sql.DataFrame.toPandas()`, as `SparkConnectClient.to_pandas` converts
    // the Arrow result: each column by its Spark type, in the session time zone.
    #[doc = connect_doc!(DataFrame.toPandas)]
    fn toPandas(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let this = Frame::of(slf)?;
        this.to_pandas(py, None)
    }

    /// Internal Arrow-based pandas conversion used by pandas-on-Spark
    /// (`pyspark.sql.connect.dataframe.DataFrame._to_pandas`): `toPandas`, with
    /// `pandasStructHandlingMode` overriding `spark.sql.execution.pandas.structHandlingMode`,
    /// as in `SparkConnectClient.to_pandas`.
    #[pyo3(signature = (**kwargs))]
    fn _to_pandas(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let this = Frame::of(slf)?;
        crate::options::check_keyword_keys(kwargs)?;
        let mode = match kwargs {
            Some(kwargs) => kwargs.get_item("pandasStructHandlingMode")?,
            None => None,
        };
        this.to_pandas(py, mode)
    }

    // Get the count of rows.
    #[doc = connect_doc!(DataFrame.count)]
    fn count(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<i64> {
        let this = Frame::of(slf)?;
        interruptible_of(py, &this.dataframe.spark_session(), &[this.parts()], || {
            this.dataframe.count().map_err(Box::new)
        })
    }

    // Print the first `n` rows as the server formats them. Mirrors `DataFrame.show`, which
    // prints the table of `_show_string` with Python's `print`.
    #[pyo3(signature = (*args, **kwargs), text_signature = "($self, n=20, truncate=True, vertical=False)")]
    #[doc = connect_doc!(DataFrame.show)]
    fn show(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let this = Frame::of(slf)?;
        let table = this.show_table(py, "DataFrame.show", args, kwargs)?;
        py.import("builtins")?.call_method1("print", (table,))?;
        Ok(())
    }

    /// The table that `show` prints: the first `n` rows (20 by default), values truncated
    /// to 20 characters for `truncate=True`, to `int(truncate)` otherwise (0 for none), and
    /// vertical with a line per value. Mirrors `DataFrame._show_string`, with its errors,
    /// for any value given, None included.
    #[pyo3(name = "_show_string", signature = (*args, **kwargs), text_signature = "($self, n=20, truncate=True, vertical=False)")]
    fn show_string(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<String> {
        let this = Frame::of(slf)?;
        this.show_table(py, "DataFrame._show_string", args, kwargs)
    }

    // Execution info from the most recent action, as a `pyspark.sql.metrics.ExecutionInfo`
    // (plan metrics, observed-metric flows, and the ExecutePlan `operation_id`). Mirrors
    // `DataFrame.executionInfo`, which returns `self._execution_info`: None before any action ran,
    // the one object that the last execution of the DataFrame set (see
    // `Frame::publish_execution`), or that the callback of a writer of it set.
    #[getter]
    #[pyo3(name = "executionInfo")]
    #[doc = connect_doc!(DataFrame.executionInfo)]
    fn execution_info<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        PyDataFrame::publish_execution(slf)?;
        crate::state::attr(slf.as_any(), pyo3::intern!(py, "_execution_info"))
    }

    // Convert to a pandas-on-Spark DataFrame, as PySpark's `DataFrame.pandas_api`: of the
    // `InternalFrame` of this DataFrame, of the index of the columns of `index_col`
    // (`pyspark.pandas.namespace._get_index_map`), of the vendored `pyspark.pandas`.
    #[pyo3(name = "pandas_api")]
    #[pyo3(signature = (index_col=None))]
    #[doc = connect_doc!(DataFrame.pandas_api)]
    fn pandas_api(
        slf: Bound<'_, Self>,
        py: Python<'_>,
        index_col: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let (index_spark_columns, index_names) = py
            .import("pyspark.pandas.namespace")?
            .getattr("_get_index_map")?
            .call1((&slf, index_col))?
            .extract::<(Bound<'_, PyAny>, Bound<'_, PyAny>)>()?;
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("spark_frame", &slf)?;
        kwargs.set_item("index_spark_columns", index_spark_columns)?;
        kwargs.set_item("index_names", index_names)?;
        let internal = py
            .import("pyspark.pandas.internal")?
            .getattr("InternalFrame")?
            .call((), Some(&kwargs))?;
        Ok(py
            .import("pyspark.pandas.frame")?
            .getattr("DataFrame")?
            .call1((internal,))?
            .unbind())
    }

    // The plot accessor of the DataFrame, as PySpark's `DataFrame.plot`: the
    // `PySparkPlotAccessor` of `pyspark.sql.plot`, of this DataFrame.
    #[getter]
    #[doc = connect_doc!(DataFrame.plot)]
    fn plot(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(py
            .import("pyspark.sql.plot")?
            .getattr("PySparkPlotAccessor")?
            .call1((slf,))?
            .unbind())
    }

    // The schema of this DataFrame as a `StructType` (a property, mirroring
    // `DataFrame.schema`), as PySpark's: of `self._schema`, the load of the pickle of
    // `_cached_schema_serialized`, of new objects, or else a deep copy of `_schema` (see
    // `Frame::schema_property`).
    #[getter]
    #[doc = connect_doc!(DataFrame.schema)]
    fn schema<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        schema_property(slf, slf.py())
    }

    /// PySpark's `_schema`: the `_cached_schema` of this DataFrame, else the schema of a request,
    /// which it keeps as its `_cached_schema`, with its pickle as its
    /// `_cached_schema_serialized` (see `Frame::schema_entry`). An `AttributeError` of it, as of a
    /// plan that fails to build with one, falls back to the `__getattr__` of the DataFrame, the
    /// Column of `_schema`, as PySpark's property does.
    #[getter]
    fn _schema<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        schema_entry(slf)
    }

    // Get the first row.
    #[doc = connect_doc!(DataFrame.first)]
    fn first(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let this = Frame::of(slf)?;
        let (row, schema) =
            interruptible_of(py, &this.dataframe.spark_session(), &[this.parts()], || {
                this.dataframe
                    .first_with_schema_checked(check_schema_at_receipt)
                    .map_err(Box::new)
            })?;
        Ok(rows_to_py(
            py,
            &this.dataframe,
            row.into_iter().collect(),
            schema.as_ref(),
        )?
        .pop())
    }

    // The first row, or None, or with `n`, a list of the first `n` rows, as PySpark's `head`,
    // whose `head(n)` is `take(n)`.
    #[pyo3(signature = (n=None))]
    #[doc = connect_doc!(DataFrame.head)]
    fn head(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        n: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        use pyo3::IntoPyObjectExt;
        match n {
            None => Self::first(slf, py)?.into_py_any(py),
            Some(n) => Self::take(slf, py, n)?.into_py_any(py),
        }
    }

    // Get the first n rows, as PySpark's `DataFrame.take`, which is `limit(num).collect()`: `num`
    // is converted as `limit` converts it, and the server checks the rest.
    #[doc = connect_doc!(DataFrame.take)]
    fn take(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        num: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let this = Frame::of(slf)?;
        let num = proto_int(num, "spark.connect.Limit.limit")?;
        let (rows, schema) =
            interruptible_of(py, &this.dataframe.spark_session(), &[this.parts()], || {
                this.dataframe
                    .limit(num)
                    .collect_with_schema_checked(check_schema_at_receipt)
                    .map_err(Box::new)
            })?;
        rows_to_py(py, &this.dataframe, rows, schema.as_ref())
    }

    // Column names (a property, mirroring `DataFrame.columns`). A schema that is an object, of
    // `to` that is not a DataType or of a plan that fails to build (see `property_schema`), gives
    // the names of its fields as it is, as PySpark's `[field.name for field in
    // self._schema.fields]`: an AttributeError of it falls back to a Column, as in PySpark, and
    // a Column raises `NOT_ITERABLE`.
    #[getter]
    #[doc = connect_doc!(DataFrame.columns)]
    fn columns(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        Ok(property_schema(slf)?.columns(py)?.into_any().unbind())
    }

    // Get the last `num` rows as a list of Rows (eager, like pyspark `tail`), of `num` as the
    // proto's int32 of the plan, converted as `limit` converts it.
    #[doc = connect_doc!(DataFrame.tail)]
    fn tail(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        num: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let this = Frame::of(slf)?;
        let num = proto_int(num, "spark.connect.Tail.limit")?;
        let (rows, schema) =
            interruptible_of(py, &this.dataframe.spark_session(), &[this.parts()], || {
                this.dataframe
                    .tail(num)
                    .collect_with_schema_checked(check_schema_at_receipt)
                    .map_err(Box::new)
            })?;
        rows_to_py(py, &this.dataframe, rows, schema.as_ref())
    }

    // Sample a fraction of rows, as PySpark's `DataFrame.sample(withReplacement=None,
    // fraction=None, seed=None)`, which takes `sample(fraction[, seed])` positionally as
    // well: a bool and a float, a float, or a float and the seed.
    #[pyo3(signature = (withReplacement=None, fraction=None, seed=None))]
    #[doc = connect_doc!(DataFrame.sample)]
    fn sample(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        withReplacement: Option<&Bound<'_, PyAny>>,
        fraction: Option<&Bound<'_, PyAny>>,
        seed: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        use pyo3::types::{PyBool, PyFloat};
        // The seed is `int(seed)`, as PySpark takes it, which fails at the call; the long of the
        // proto is converted as the plan is sent, so that its error is raised then.
        let int = |value: &Bound<'_, PyAny>| -> PyResult<Result<i64, PyErr>> {
            let int = py.import("builtins")?.getattr("int")?.call1((value,))?;
            Ok(int
                .extract::<i64>()
                .map_err(|_| crate::session::value_out_of_range(&int)))
        };
        let (replace, frac, seed) = match (withReplacement, fraction) {
            (Some(w), Some(f)) if w.is_instance_of::<PyBool>() && f.is_instance_of::<PyFloat>() => {
                (w.extract()?, f.extract()?, seed.map(int).transpose()?)
            }
            (None, Some(f)) if f.is_instance_of::<PyFloat>() => {
                (false, f.extract()?, seed.map(int).transpose()?)
            }
            // `sample(fraction[, seed])`: the fraction is the first argument, and the seed the
            // second.
            (Some(w), f) if w.is_instance_of::<PyFloat>() => {
                (false, w.extract()?, f.map(int).transpose()?)
            }
            _ => {
                let type_name = |value: Option<&Bound<'_, PyAny>>| -> PyResult<String> {
                    Ok(match value {
                        Some(value) => value.get_type().name()?.to_string(),
                        None => "NoneType".to_string(),
                    })
                };
                let arg_types = [withReplacement, fraction, seed]
                    .into_iter()
                    .map(type_name)
                    .collect::<PyResult<Vec<_>>>()?
                    .join(", ");
                return Err(crate::column::raise_pyspark(
                    py,
                    "PySparkTypeError",
                    "NOT_EXPECTED_TYPE",
                    &[
                        ("expected_type", "bool, float or int"),
                        (
                            "arg_name",
                            "withReplacement (optional), fraction (required) and seed (optional)",
                        ),
                        ("arg_type", &arg_types),
                    ],
                ));
            }
        };
        let (seed, plan_error) = match seed {
            Some(Ok(seed)) => (seed, None),
            Some(Err(err)) => (0, Some(unbuildable_error(py, err)?)),
            None => (python_random_seed(py)?, None),
        };
        let df = this.plan().sample_opt(frac, replace, Some(seed));
        // Of the schema that the input cached, as PySpark copies it.
        this.of_same_schema(this.derived(match plan_error {
            Some(error) => df.unbuildable_with_cached_schema(error),
            None => df,
        }))
    }

    // Check if the DataFrame is empty, asked once, as PySpark's.
    fn isEmpty(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<bool> {
        let this = Frame::of(slf)?;
        cached(&this.is_empty, || {
            let empty =
                interruptible_of(py, &this.dataframe.spark_session(), &[this.parts()], || {
                    this.dataframe.is_empty().map_err(Box::new)
                })?;
            // As PySpark's `select().take(1)`, whose `collect` reads the binary setting once the
            // rows are fetched, though they have no columns.
            read_binary_as_bytes(py, &this.dataframe)?;
            Ok(empty)
        })
    }

    // Print the schema through Python stdout, honoring the maximum nesting depth.
    #[pyo3(signature = (level=None))]
    #[doc = connect_doc!(DataFrame.printSchema)]
    fn printSchema(slf: &Bound<'_, Self>, level: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let py = slf.py();
        // Reference's `if level:` selects the default for false values before
        // resolving the schema; positive fractional depths behave as ceil(level).
        let level = match level {
            Some(level) if level.is_truthy()? => Some(level),
            _ => None,
        };
        // As PySpark's `print(self.schema.treeString(level))`, of the property `schema`, whose
        // `treeString` raises its own errors, and that of a Column is a Column, whose call raises.
        let schema = slf.as_any().getattr(pyo3::intern!(py, "schema"))?;
        let tree = match level {
            Some(level) => schema.call_method1("treeString", (level,))?,
            None => schema.call_method0("treeString")?,
        };
        py.import("builtins")?.getattr("print")?.call1((tree,))?;
        Ok(())
    }

    // Get the schema as a list of (name, type) tuples. A schema that is an object (see
    // `property_schema`) gives them as it is, as PySpark's `dtypes` reads its fields.
    #[getter]
    #[doc = connect_doc!(DataFrame.dtypes)]
    fn dtypes(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        Ok(property_schema(slf)?.dtypes(py)?.into_any().unbind())
    }

    // Additional set operation: unionAll (alias for union).
    #[doc = connect_doc!(DataFrame.unionAll)]
    fn unionAll(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        let other = this.same_session(other)?;
        let dataframe = this.plan().union(other.plan());
        set_operation_result(py, &this, &other, dataframe)
    }

    // The V1 writer interface (`df.write`): PySpark Connect's `DataFrameWriter(self._plan,
    // self._session, cb)`, whose callback sets the execution information of this DataFrame.
    #[getter]
    #[doc = connect_doc!(DataFrame.write)]
    fn write<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let module = py.import("pyspark.sql.connect.readwriter")?;
        let callback = module.call_method1("_execution_info_callback", (slf,))?;
        let plan = slf.getattr(pyo3::intern!(py, "_plan"))?;
        let session = slf.getattr(pyo3::intern!(py, "_session"))?;
        module
            .getattr(pyo3::intern!(py, "DataFrameWriter"))?
            .call1((plan, session, callback))
    }

    // The V2 writer interface (`df.writeTo("table")`): PySpark Connect's
    // `DataFrameWriterV2(self._plan, self._session, table, cb)`.
    #[pyo3(name = "writeTo")]
    #[doc = connect_doc!(DataFrame.writeTo)]
    fn write_to<'py>(
        slf: &Bound<'py, Self>,
        table: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let module = py.import("pyspark.sql.connect.readwriter")?;
        let callback = module.call_method1("_execution_info_callback", (slf,))?;
        let plan = slf.getattr(pyo3::intern!(py, "_plan"))?;
        let session = slf.getattr(pyo3::intern!(py, "_session"))?;
        module
            .getattr(pyo3::intern!(py, "DataFrameWriterV2"))?
            .call1((plan, session, table, callback))
    }

    // The streaming writer interface (`df.writeStream`): PySpark Connect's
    // `DataStreamWriter(self._plan, self._session)`.
    #[getter]
    #[pyo3(name = "writeStream")]
    #[doc = connect_doc!(DataFrame.writeStream)]
    fn write_stream<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        static WRITER: pyo3::sync::PyOnceLock<Py<pyo3::types::PyType>> =
            pyo3::sync::PyOnceLock::new();
        let plan = slf.getattr(pyo3::intern!(py, "_plan"))?;
        let session = slf.getattr(pyo3::intern!(py, "_session"))?;
        WRITER
            .import(
                py,
                "pyspark.sql.connect.streaming.readwriter",
                "DataStreamWriter",
            )?
            .call1((plan, session))
    }

    // Null-handling functions (`df.na.drop()/fill()/replace()`): a new object at each access,
    // as PySpark's `DataFrameNaFunctions(self)`, of this DataFrame object, its `df`.
    #[getter]
    #[doc = connect_doc!(DataFrame.na)]
    fn na<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDataFrameNaFunctions>> {
        PyDataFrameNaFunctions::of_frame(slf.as_any())
    }

    // Merge-into (`df.mergeInto("table", cond)`): PySpark Connect's
    // `MergeIntoWriter(plan=self._plan, session=self._session, table=table, condition=condition,
    // callback=cb)`, of the callback of the DataFrame, as `write`'s.
    #[pyo3(name = "mergeInto")]
    #[doc = connect_doc!(DataFrame.mergeInto)]
    fn merge_into<'py>(
        slf: &Bound<'py, Self>,
        table: &Bound<'py, PyAny>,
        condition: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        static CALLBACK_OF: pyo3::sync::PyOnceLock<Py<PyAny>> = pyo3::sync::PyOnceLock::new();
        static WRITER: pyo3::sync::PyOnceLock<Py<pyo3::types::PyType>> =
            pyo3::sync::PyOnceLock::new();
        let callback = CALLBACK_OF
            .import(
                py,
                "pyspark.sql.connect.readwriter",
                "_execution_info_callback",
            )?
            .call1((slf,))?;
        let kwargs = PyDict::new(py);
        kwargs.set_item(
            pyo3::intern!(py, "plan"),
            slf.getattr(pyo3::intern!(py, "_plan"))?,
        )?;
        kwargs.set_item(
            pyo3::intern!(py, "session"),
            slf.getattr(pyo3::intern!(py, "_session"))?,
        )?;
        kwargs.set_item(pyo3::intern!(py, "table"), table)?;
        kwargs.set_item(pyo3::intern!(py, "condition"), condition)?;
        kwargs.set_item(pyo3::intern!(py, "callback"), callback)?;
        WRITER
            .import(py, "pyspark.sql.connect.merge", "MergeIntoWriter")?
            .call((), Some(&kwargs))
    }

    /// As PySpark's `DataFrame.__getitem__`: a column by name (`df["col"]`), of this DataFrame
    /// by its plan ID, as its star of `df["*"]`, or by position (`df[0]`), the rows that match
    /// a Column (`df[df.a > 1]`, as `filter`), or the projection of a list or tuple of columns
    /// (`df[["a", "b"]]`, as `select`). With `PYSPARK_VALIDATE_COLUMN_NAME_LEGACY=1`, a name is
    /// validated first, as PySpark's: by `verify_col_name` of the cached schema, which is
    /// requested unless it is cached, and else by the server, of `select(item).isLocal()`, whose
    /// request raises its error, such as `UNRESOLVED_COLUMN.WITH_SUGGESTION`.
    fn __getitem__(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        item: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        use pyo3::IntoPyObjectExt;

        let name = if item.is_instance_of::<pyo3::types::PyString>() {
            // The star of this DataFrame, or its column of the name, `_col(item)`, which sets the
            // name when it builds the plan, of `self._plan` (see `plan_dataframe`).
            let column = if item.eq("*")? {
                plan_dataframe(slf)?.star()
            } else {
                let legacy =
                    std::env::var("PYSPARK_VALIDATE_COLUMN_NAME_LEGACY").as_deref() == Ok("1");
                let session = || crate::state::attr(slf.as_any(), pyo3::intern!(py, "_session"));
                if legacy && !session()?.hasattr("is_mock_session")? {
                    // `self._schema`, its `_cached_schema` or the schema of a request (see
                    // `schema_entry`).
                    let schema = property_schema(slf)?.0;
                    let verified = py
                        .import("pyspark.sql.connect.types")?
                        .getattr("verify_col_name")?
                        .call1((item, schema))?
                        .is_truthy()?;
                    if !verified {
                        let selected = Self::select(slf, py, vec![item.clone()])?;
                        Self::is_local(&selected.into_pyobject(py)?, py)?;
                    }
                }
                crate::plan_args::df_col(&plan_dataframe(slf)?, item, false)?
            };
            return crate::column::ColumnOut::new(column).into_py_any(py);
        } else if item.is_instance_of::<PyColumn>() {
            return Self::filter(slf, item)?.into_py_any(py);
        } else if item.is_instance_of::<PyList>() || item.is_instance_of::<PyTuple>() {
            let cols = item.try_iter()?.collect::<PyResult<Vec<_>>>()?;
            return Self::select(slf, py, cols)?.into_py_any(py);
        } else if item.is_instance_of::<PyInt>() {
            let columns = column_names_of(slf, py)?;
            // Use Python list indexing for negative indices, bools (an int
            // subclass), and IndexError even for integers larger than isize.
            PyList::new(py, columns)?
                .as_any()
                .get_item(item)?
                .extract::<String>()?
        } else {
            let arg_type = item.get_type().name()?.to_string();
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkTypeError",
                "NOT_EXPECTED_TYPE",
                &[
                    ("expected_type", "Column, int, list, str or tuple"),
                    ("arg_name", "item"),
                    ("arg_type", &arg_type),
                ],
            ));
        };
        // As PySpark's `F.col(self.columns[item])`, of the origin of `col`.
        let _origin = crate::origin::with_origin("col")?;
        crate::column::ColumnOut::new(spark_connect::functions::col(&name)).into_py_any(py)
    }

    /// Private column accessor used by pandas-on-Spark (`sdf._col(name)`); mirrors the
    /// connect DataFrame's `_col`, the column of this DataFrame by its plan ID. Backtick-quoted
    /// names arrive from `scol_for`.
    fn _col(slf: &Bound<'_, Self>, item: &Bound<'_, PyAny>) -> PyResult<crate::column::ColumnOut> {
        Ok(crate::column::ColumnOut::new(crate::plan_args::df_col(
            &plan_dataframe(slf)?,
            item,
            false,
        )?))
    }

    /// As PySpark's `DataFrame.__repr__`: the table of `_show_string` when the session's
    /// `spark.sql.repl.eagerEval.enabled` is `true`, with its `maxNumRows` rows and its
    /// `truncate`, and `DataFrame[name: type, ...]` otherwise, of its `dtypes`. Like PySpark, it
    /// reads the three settings, in one request, on every call.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        {
            // As PySpark's: once `_repr_html_` was called, the string of the schema.
            let supported =
                crate::state::attr(slf.as_any(), pyo3::intern!(py, "_support_repr_html"))?;
            if !supported.is_truthy()? {
                let session = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_session"))?;
                if let Some((n, truncate)) = eager_eval(py, &session)? {
                    let this = Frame::of(slf)?;
                    let dataframe = &this.dataframe;
                    return interruptible_of(
                        py,
                        &dataframe.spark_session(),
                        &[this.parts()],
                        || dataframe.show_string(n, truncate, false).map_err(Box::new),
                    );
                }
            }
        }
        // As PySpark's `", ".join("%s: %s" % c for c in self.dtypes)`: an AttributeError of the
        // property, of a schema that is an object (see `property_schema`), makes it a Column, which
        // the join cannot iterate, as it cannot iterate the Column of `_schema`.
        let mut fields = Vec::new();
        for pair in slf.getattr(pyo3::intern!(py, "dtypes"))?.try_iter()? {
            let pair = pair?;
            let name = pair.get_item(0)?.str()?.to_string();
            let dtype = pair.get_item(1)?.str()?.to_string();
            fields.push(format!("{name}: {dtype}"));
        }
        Ok(format!("DataFrame[{}]", fields.join(", ")))
    }

    /// The HTML table of the DataFrame, which notebooks show, as PySpark Connect's
    /// `_repr_html_`: with `spark.sql.repl.eagerEval.enabled`, the table of the first
    /// `maxNumRows` rows, values truncated to `truncate` characters, which the server formats
    /// (`HtmlString`); else None. Once called, `__repr__` is the string of the schema.
    fn _repr_html_(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<Option<String>> {
        // As PySpark's `if not self._support_repr_html: self._support_repr_html = True`, before it
        // reads the settings of the session and then the plan.
        let attribute = pyo3::intern!(py, "_support_repr_html");
        if !crate::state::attr(slf.as_any(), attribute)?.is_truthy()? {
            let supported = pyo3::types::PyBool::new(py, true).to_owned().into_any();
            crate::state::assign(slf.as_any(), &[(attribute, &supported)])?;
        }
        let session = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_session"))?;
        match eager_eval(py, &session)? {
            Some((n, truncate)) => {
                let this = Frame::of(slf)?;
                interruptible_of(py, &this.dataframe.spark_session(), &[this.parts()], || {
                    this.dataframe.html_string(n, truncate).map_err(Box::new)
                })
                .map(Some)
            }
            None => Ok(None),
        }
    }

    // `DataFrame.mapInPandas` — map an iterator of pandas DataFrames, in a stage of the
    // resource profile `profile`, if any, as PySpark's.
    #[pyo3(
        name = "mapInPandas",
        signature = (func, schema, barrier = ProtoBool::default(), profile = None),
        text_signature = "($self, func, schema, barrier=False, profile=None)"
    )]
    #[doc = connect_doc!(DataFrame.mapInPandas)]
    fn map_in_pandas(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        func: Bound<'_, PyAny>,
        schema: Bound<'_, PyAny>,
        barrier: ProtoBool,
        profile: Option<Bound<'_, PyAny>>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        validate_map_function(py, &func, eval_type::SQL_MAP_PANDAS_ITER_UDF)?;
        // As PySpark's `_map_partitions`: a DDL string is parsed, of `self._session`, before the
        // columns of the input are read.
        let return_type = return_type_of(py, this.session_object(), &schema)?;
        let cols = this.column_names(py)?;
        let udf = map_udf_of(
            py,
            &func,
            &schema,
            return_type,
            eval_type::SQL_MAP_PANDAS_ITER_UDF,
            &cols,
        )?;
        map_result_of(
            &schema,
            map_partitions_of(py, this.plan(), udf, &barrier, profile)?,
        )
    }

    // `DataFrame.mapInArrow` — map an iterator of Arrow batches, in a stage of the
    // resource profile `profile`, if any, as PySpark's.
    #[pyo3(
        name = "mapInArrow",
        signature = (func, schema, barrier = ProtoBool::default(), profile = None),
        text_signature = "($self, func, schema, barrier=False, profile=None)"
    )]
    #[doc = connect_doc!(DataFrame.mapInArrow)]
    fn map_in_arrow(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        func: Bound<'_, PyAny>,
        schema: Bound<'_, PyAny>,
        barrier: ProtoBool,
        profile: Option<Bound<'_, PyAny>>,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        validate_map_function(py, &func, eval_type::SQL_MAP_ARROW_ITER_UDF)?;
        // As PySpark's `_map_partitions`: a DDL string is parsed, of `self._session`, before the
        // columns of the input are read.
        let return_type = return_type_of(py, this.session_object(), &schema)?;
        let cols = this.column_names(py)?;
        let udf = map_udf_of(
            py,
            &func,
            &schema,
            return_type,
            eval_type::SQL_MAP_ARROW_ITER_UDF,
            &cols,
        )?;
        map_result_of(
            &schema,
            map_partitions_of(py, this.plan(), udf, &barrier, profile)?,
        )
    }

    // `DataFrame.foreach`, as PySpark Connect's: a UDF over a struct of the columns of each
    // row calls `f` with the row. The UDF is Python, as it runs on the server
    // (`pyspark.sql._foreach`).
    #[doc = connect_doc!(DataFrame.foreach)]
    fn foreach(slf: &Bound<'_, Self>, f: &Bound<'_, PyAny>) -> PyResult<()> {
        slf.py()
            .import("pyspark.sql._foreach")?
            .call_method1("foreach", (slf, f))?;
        Ok(())
    }

    // `DataFrame.foreachPartition`, as PySpark Connect's: a `mapInArrow` function calls `f`
    // with an iterator of the rows of each partition.
    #[pyo3(name = "foreachPartition")]
    #[doc = connect_doc!(DataFrame.foreachPartition)]
    fn foreach_partition(slf: &Bound<'_, Self>, f: &Bound<'_, PyAny>) -> PyResult<()> {
        slf.py()
            .import("pyspark.sql._foreach")?
            .call_method1("foreachPartition", (slf, f))?;
        Ok(())
    }

    // `DataFrame.nearestByJoin(other, rankingExpression, numResults, mode, direction, *,
    // joinType="inner")`, as PySpark's, which checks its arguments as the Scala
    // `Dataset.validateNearestByJoinArgs` does: an `AnalysisException` of
    // `NEAREST_BY_JOIN.NUM_RESULTS_OUT_OF_RANGE`, `UNSUPPORTED_JOIN_TYPE`, `UNSUPPORTED_MODE` or
    // `UNSUPPORTED_DIRECTION`. As PySpark, it compares `joinType.lower().replace("_", "")`,
    // `mode.lower()` and `direction.lower()` to the values that it takes, so a value that has no
    // `lower` raises its `AttributeError`, bytes as the join type the `TypeError` of `replace`,
    // and another value, such as bytes as the mode or a str of a lone surrogate, is named as it
    // is in the error.
    #[pyo3(
        name = "nearestByJoin",
        signature = (other, rankingExpression, numResults, mode, direction, *, joinType = DefaultedOption::Omitted),
        text_signature = "($self, other, rankingExpression, numResults, mode, direction, *, joinType='inner')"
    )]
    #[allow(clippy::too_many_arguments)]
    #[doc = connect_doc!(DataFrame.nearestByJoin)]
    fn nearest_by_join(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        rankingExpression: &Bound<'_, PyColumn>,
        numResults: &Bound<'_, PyAny>,
        mode: &Bound<'_, PyAny>,
        direction: &Bound<'_, PyAny>,
        joinType: DefaultedOption,
    ) -> PyResult<NewDataFrame> {
        let this = PlanFrame::of(slf)?;
        const MAX_NUM_RESULTS: i64 = 100_000;
        const MODES: [&str; 2] = ["approx", "exact"];
        const DIRECTIONS: [&str; 2] = ["distance", "similarity"];
        let other = this.same_session(other)?;
        let join_type = match &joinType {
            DefaultedOption::Omitted => pyo3::types::PyString::new(py, "inner").into_any(),
            DefaultedOption::Given(join_type) => join_type.bind(py).clone(),
        };
        // As PySpark's `int(numResults)`.
        let num_results = py
            .import("builtins")?
            .getattr("int")?
            .call1((numResults,))?;
        let analysis = |class: &str, params: &[(&str, &str)]| {
            crate::column::raise_pyspark(py, "AnalysisException", class, params)
        };
        let in_range = num_results
            .extract::<i64>()
            .is_ok_and(|n| (1..=MAX_NUM_RESULTS).contains(&n));
        if !in_range {
            return Err(analysis(
                "NEAREST_BY_JOIN.NUM_RESULTS_OUT_OF_RANGE",
                &[
                    ("numResults", &num_results.str()?.to_cow()?),
                    ("min", "1"),
                    ("max", &MAX_NUM_RESULTS.to_string()),
                ],
            ));
        }
        // Whether `value` is one of `values`, by Python's `in` of the tuple of them.
        let one_of = |value: &Bound<'_, PyAny>, values: &[&str]| -> PyResult<bool> {
            PyTuple::new(py, values)?.contains(value)
        };
        let unsupported = |class: &str, params: &[(&str, &Bound<'_, PyAny>)]| {
            crate::errors::raise_pyspark_of(py, "AnalysisException", class, params)
        };
        let normalized = join_type
            .call_method0("lower")?
            .call_method1("replace", ("_", ""))?;
        if !one_of(&normalized, &["inner", "leftouter", "left"])? {
            let supported = pyo3::types::PyString::new(py, "'INNER', 'LEFT OUTER'");
            return Err(unsupported(
                "NEAREST_BY_JOIN.UNSUPPORTED_JOIN_TYPE",
                &[("joinType", &join_type), ("supported", supported.as_any())],
            ));
        }
        let supported = |values: &[&str]| {
            pyo3::types::PyString::new(py, &format!("'{}'", values.join("', '"))).into_any()
        };
        if !one_of(&mode.call_method0("lower")?, &MODES)? {
            return Err(unsupported(
                "NEAREST_BY_JOIN.UNSUPPORTED_MODE",
                &[("mode", mode), ("supported", &supported(&MODES))],
            ));
        }
        if !one_of(&direction.call_method0("lower")?, &DIRECTIONS)? {
            return Err(unsupported(
                "NEAREST_BY_JOIN.UNSUPPORTED_DIRECTION",
                &[
                    ("direction", direction),
                    ("supported", &supported(&DIRECTIONS)),
                ],
            ));
        }
        Ok(this.derived(this.plan().nearest_by_join(
            other.plan(),
            crate::column::column_of(rankingExpression)?,
            num_results.extract::<i32>()?,
            &mode.extract::<String>()?,
            &direction.extract::<String>()?,
            &join_type.extract::<String>()?,
        )))
    }

    // `DataFrame.metadataColumn`.
    #[pyo3(name = "metadataColumn")]
    #[doc = connect_doc!(DataFrame.metadataColumn)]
    fn metadata_column(
        slf: &Bound<'_, Self>,
        colName: &Bound<'_, PyAny>,
    ) -> PyResult<crate::column::ColumnOut> {
        let this = PlanFrame::of(slf)?;
        // As PySpark's: a str, else NOT_EXPECTED_TYPE.
        if !colName.is_instance_of::<pyo3::types::PyString>() {
            return Err(not_expected_type(colName, "colName", "str"));
        }
        Ok(crate::column::ColumnOut::new(crate::plan_args::df_col(
            this.plan(),
            colName,
            true,
        )?))
    }
}

// Python wrapper for `DataFrameNaFunctions` (`df.na`). As PySpark's plain class, it keeps its
// state in its `__dict__`, its `df`, whose methods its methods call, as PySpark's do, and it is
// subclassed and constructed of PySpark's arguments (see `crate::state`).
#[doc = connect_doc!(DataFrameNaFunctions)]
#[pyclass(
    name = "DataFrameNaFunctions",
    module = "pyspark.sql.connect.dataframe",
    subclass,
    weakref,
    dict
)]
pub struct PyDataFrameNaFunctions {}

impl PyDataFrameNaFunctions {
    /// The null functions of `df`, as `df.na` makes them: `DataFrameNaFunctions(df)`, of the
    /// DataFrame object, which PySpark's `__init__` keeps as its `df`.
    fn of_frame<'py>(df: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDataFrameNaFunctions>> {
        let na = Bound::new(df.py(), PyDataFrameNaFunctions {})?;
        Self::__init__(&na, df)?;
        Ok(na)
    }

    /// The `df` of the functions, which each of PySpark's methods reads, `self.df`, to call its
    /// method: of the `AttributeError` of functions whose `__init__` never ran.
    fn frame<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::state::attr(slf.as_any(), pyo3::intern!(slf.py(), "df"))
    }
}

#[pymethods]
impl PyDataFrameNaFunctions {
    /// Null functions of no state, as the `object.__new__` of PySpark's plain class: of any
    /// arguments, which `__init__` takes, so that a subclass of another `__init__` is built as it
    /// is in PySpark. The class shows the signature of `__init__`, as `inspect.signature` of
    /// PySpark's class does.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs), text_signature = "(df)")]
    fn py_new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        PyDataFrameNaFunctions {}
    }

    /// `DataFrameNaFunctions(df)`, as PySpark Connect's `__init__`, which keeps its argument as
    /// it is, as the `df` of the functions: `df.na.df` is `df`.
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

    // Drop rows containing nulls, as PySpark's `DataFrameNaFunctions.drop`, which is
    // `self.df.dropna(how=how, thresh=thresh, subset=subset)`, a call of the method of the
    // DataFrame, of a subclass or of any object too, of its default `how` of "any".
    #[pyo3(
        signature = (how=crate::options::DefaultedOption::Omitted, thresh=None, subset=None),
        text_signature = "($self, how='any', thresh=None, subset=None)"
    )]
    #[doc = connect_doc!(DataFrameNaFunctions.drop)]
    fn drop<'py>(
        slf: &Bound<'py, Self>,
        how: crate::options::DefaultedOption,
        thresh: Option<&Bound<'py, PyAny>>,
        subset: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let how = match how {
            crate::options::DefaultedOption::Omitted => pyo3::intern!(py, "any").clone().into_any(),
            crate::options::DefaultedOption::Given(how) => how.into_bound(py),
        };
        let kwargs = PyDict::new(py);
        kwargs.set_item(pyo3::intern!(py, "how"), how)?;
        kwargs.set_item(pyo3::intern!(py, "thresh"), thresh)?;
        kwargs.set_item(pyo3::intern!(py, "subset"), subset)?;
        Self::frame(slf)?.call_method(pyo3::intern!(py, "dropna"), (), Some(&kwargs))
    }

    // Fill nulls, as PySpark's `DataFrameNaFunctions.fill`, which is
    // `self.df.fillna(value=value, subset=subset)`.
    #[pyo3(signature = (value, subset=None))]
    #[doc = connect_doc!(DataFrameNaFunctions.fill)]
    fn fill<'py>(
        slf: &Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
        subset: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let kwargs = PyDict::new(py);
        kwargs.set_item(pyo3::intern!(py, "value"), value)?;
        kwargs.set_item(pyo3::intern!(py, "subset"), subset)?;
        Self::frame(slf)?.call_method(pyo3::intern!(py, "fillna"), (), Some(&kwargs))
    }

    // Replace values, as PySpark's `DataFrameNaFunctions.replace`, which is
    // `self.df.replace(to_replace, value, subset)`, of PySpark's `_NoValue` of an omitted value.
    #[pyo3(signature = (to_replace, value=ReplaceValue::Omitted, subset=None))]
    #[doc = connect_doc!(DataFrameNaFunctions.replace)]
    fn replace<'py>(
        slf: &Bound<'py, Self>,
        to_replace: &Bound<'py, PyAny>,
        value: ReplaceValue,
        subset: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let value = match value {
            ReplaceValue::Omitted => no_value(py)?.clone(),
            ReplaceValue::Given(value) => value.into_bound(py),
        };
        Self::frame(slf)?.call_method1(pyo3::intern!(py, "replace"), (to_replace, value, subset))
    }
}

/// PySpark's `pyspark._globals._NoValue`, the default `value` of `replace`, which tells an
/// omitted value apart from None.
fn no_value(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    static NO_VALUE: pyo3::sync::PyOnceLock<Py<PyAny>> = pyo3::sync::PyOnceLock::new();
    NO_VALUE.import(py, "pyspark._globals", "_NoValue")
}

/// The `value` argument of `DataFrame.replace`, where PySpark tells an omitted value apart
/// from None: `replace(to_replace)` requires a dict `to_replace`, while
/// `replace(to_replace, None)` replaces with nulls. PySpark's `_NoValue`, its default, is an
/// omitted value too, as `DataFrameNaFunctions.replace` gives it.
pub enum ReplaceValue {
    Omitted,
    Given(Py<PyAny>),
}

impl<'a, 'py> FromPyObject<'a, 'py> for ReplaceValue {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if obj.is(no_value(obj.py())?) {
            return Ok(ReplaceValue::Omitted);
        }
        Ok(ReplaceValue::Given(obj.to_owned().unbind()))
    }
}

/// `df.replace(to_replace, value, subset)`, checked and built as PySpark's
/// `DataFrame.replace`: each value of `to_replace` (a {old: new} dict, or a value or a list or
/// tuple of them) is replaced by `value` (a value or a list or tuple of them, or None), all
/// bools, numbers or strings, sent as literals with an int as a float, in the columns of
/// `subset` (a column name, or a list or tuple of them).
fn replace_values(
    dataframe: &DataFrame,
    to_replace: &Bound<'_, PyAny>,
    value: ReplaceValue,
    subset: Option<&Bound<'_, PyAny>>,
) -> PyResult<NewDataFrame> {
    use pyo3::types::{PyBool, PyFloat, PyString};
    use spark_connect::expression::LiteralExpression;

    let py = to_replace.py();
    let raise = |class_name: &str, error_class: &str, params: &[(&str, &str)]| {
        crate::column::raise_pyspark(py, class_name, error_class, params)
    };
    let type_name = |v: &Bound<'_, PyAny>| -> String {
        v.get_type()
            .name()
            .map(|name| name.to_string())
            .unwrap_or_default()
    };
    let is_sequence =
        |v: &Bound<'_, PyAny>| v.is_instance_of::<PyList>() || v.is_instance_of::<PyTuple>();
    // A bool, a float, an int or a str: a bool is an int.
    let is_scalar = |v: &Bound<'_, PyAny>| {
        v.is_instance_of::<PyFloat>()
            || v.is_instance_of::<PyInt>()
            || v.is_instance_of::<PyString>()
    };
    let is_dict = to_replace.is_instance_of::<PyDict>();
    let value = match value {
        ReplaceValue::Given(value) => value.into_bound(py),
        ReplaceValue::Omitted if is_dict => py.None().into_bound(py),
        ReplaceValue::Omitted => {
            return Err(raise(
                "PySparkTypeError",
                "ARGUMENT_REQUIRED",
                &[("arg_name", "value"), ("condition", "`to_replace` is dict")],
            ));
        }
    };

    if !(is_scalar(to_replace) || is_sequence(to_replace) || is_dict) {
        return Err(raise(
            "PySparkTypeError",
            "NOT_EXPECTED_TYPE",
            &[
                ("expected_type", "bool, dict, float, int, str or tuple"),
                ("arg_name", "to_replace"),
                ("arg_type", &type_name(to_replace)),
            ],
        ));
    }
    if !(is_scalar(&value) || is_sequence(&value) || value.is_none() || is_dict) {
        return Err(raise(
            "PySparkTypeError",
            "NOT_EXPECTED_TYPE",
            &[
                (
                    "expected_type",
                    "bool, float, int, list, None, str or tuple",
                ),
                ("arg_name", "value"),
                ("arg_type", &type_name(&value)),
            ],
        ));
    }
    if is_sequence(to_replace) && is_sequence(&value) && to_replace.len()? != value.len()? {
        return Err(raise(
            "PySparkValueError",
            "LENGTH_SHOULD_BE_THE_SAME",
            &[
                ("arg1", "to_replace"),
                ("arg2", "value"),
                ("arg1_length", &to_replace.len()?.to_string()),
                ("arg2_length", &value.len()?.to_string()),
            ],
        ));
    }
    // The names of a list or a tuple are set in the repeated string field of the plan as they
    // are, as protobuf converts them, when its `NAReplace` plan builds the plan, before the
    // replacements (`args`).
    let mut args = crate::plan_args::PlanArgs::new();
    let subset: Option<Bound<'_, PyAny>> = match subset {
        None => None,
        Some(subset) if subset.is_instance_of::<PyString>() || is_sequence(subset) => {
            Some(subset.clone())
        }
        Some(subset) => {
            return Err(raise(
                "PySparkTypeError",
                "NOT_EXPECTED_TYPE",
                &[
                    ("arg_name", "subset"),
                    ("expected_type", "list, str or tuple"),
                    ("arg_type", &type_name(subset)),
                ],
            ));
        }
    };

    // The replacements as a dict, as PySpark builds it, so that equal values are one key. Of a
    // dict and a value, PySpark's `DataFrame.replace`, which `na.replace` calls, warns of the line
    // of `pyspark.sql.connect.dataframe` of its `warnings.warn`.
    let replacements = if is_dict {
        if !value.is_none() {
            crate::errors::warn_as_pyspark(
                py,
                "pyspark.sql.connect.dataframe",
                1541,
                "to_replace is a dict and value is not None. value will be ignored.",
                &py.get_type::<pyo3::exceptions::PyUserWarning>(),
            )?;
        }
        to_replace.cast::<PyDict>()?.clone()
    } else {
        let olds: Vec<Bound<'_, PyAny>> = if is_scalar(to_replace) {
            vec![to_replace.clone()]
        } else {
            to_replace.try_iter()?.collect::<PyResult<_>>()?
        };
        let news: Vec<Bound<'_, PyAny>> = if is_scalar(&value) || value.is_none() {
            vec![value.clone(); olds.len()]
        } else {
            value.try_iter()?.collect::<PyResult<_>>()?
        };
        let replacements = PyDict::new(py);
        for (old, new) in olds.iter().zip(news.iter()) {
            replacements.set_item(old, new)?;
        }
        replacements
    };

    // All of the values to replace, and all of their replacements but None, are bools,
    // numbers or strings.
    let olds: Vec<Bound<'_, PyAny>> = replacements.keys().iter().collect();
    let news: Vec<Bound<'_, PyAny>> = replacements
        .values()
        .iter()
        .filter(|new| !new.is_none())
        .collect();
    let all_of = |is_type: &dyn Fn(&Bound<'_, PyAny>) -> bool| {
        olds.iter().all(is_type) && news.iter().all(is_type)
    };
    if !(all_of(&|v| v.is_instance_of::<PyBool>())
        || all_of(&|v| v.is_instance_of::<PyString>())
        || all_of(&|v| v.is_instance_of::<PyFloat>() || v.is_instance_of::<PyInt>()))
    {
        return Err(raise("PySparkValueError", "MIXED_TYPE_REPLACEMENT", &[]));
    }

    let subset = match subset {
        None => None,
        Some(subset) if subset.is_instance_of::<PyString>() => Some(vec![args.str(&subset)?]),
        Some(subset) => Some(args.strs(subset.try_iter()?.collect::<PyResult<Vec<_>>>()?)?),
    };
    // The literals of `F.lit`, whose strings the plan sets when it builds the plan.
    let mut literal = |v: &Bound<'_, PyAny>| -> PyResult<LiteralExpression> {
        Ok(if v.is_none() {
            LiteralExpression::null(spark_connect::types::DataType::Null)
        } else if v.is_instance_of::<PyBool>() {
            LiteralExpression::boolean(v.extract()?)
        } else if v.is_instance_of::<PyString>() {
            LiteralExpression::string(args.str(v)?)
        } else {
            // As `float(v)`, for an int as well.
            LiteralExpression::double(v.extract()?)
        })
    };
    let replacements = replacements
        .iter()
        .map(|(old, new)| Ok((literal(&old)?, literal(&new)?)))
        .collect::<PyResult<Vec<_>>>()?;
    let subset: Option<Vec<&str>> = subset
        .as_ref()
        .map(|subset| subset.iter().map(String::as_str).collect());
    Ok(PyDataFrame::new(
        args.plan(dataframe.replace_literals(replacements, subset)),
    ))
}

/// The columns of the `ids` or `values` of `df.unpivot`, as PySpark's: a list or tuple of
/// columns or column names, or one of them.
fn unpivot_columns(cols: &Bound<'_, PyAny>) -> PyResult<Vec<spark_connect::column::Column>> {
    if cols.is_instance_of::<PyList>() || cols.is_instance_of::<PyTuple>() {
        cols.try_iter()?.map(|col| column_or_name(&col?)).collect()
    } else {
        Ok(vec![column_or_name(cols)?])
    }
}

/// A column, or a column by its name, as PySpark's `_to_col`.
pub(crate) fn column_or_name(col: &Bound<'_, PyAny>) -> PyResult<spark_connect::column::Column> {
    if let Ok(column) = col.cast::<PyColumn>() {
        return crate::column::column_of(column);
    }
    if col.is_instance_of::<pyo3::types::PyString>() {
        // As PySpark's `F._to_col`, by `col`, of its origin.
        let _origin = crate::origin::with_origin("col")?;
        return crate::plan_args::col(col);
    }
    Err(not_expected_type(col, "col", "Column or str"))
}

// ---- Shared helpers for closure-based (UDF) DataFrame/GroupedData methods ----

/// cloudpickle an arbitrary Python object to bytes (mirrors the scalar UDF path,
/// where the client serializes the callable for server-side execution).
///
/// Uses `CloudPickleSerializer().dumps` as PySpark does, which raises a `pickle.PicklingError`
/// ("Could not serialize object: ...") for an object that does not pickle, rather than the
/// bare pickling error of `cloudpickle`.
pub(crate) fn py_cloudpickle(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    let serializer = py
        .import("pyspark.serializers")?
        .getattr("CloudPickleSerializer")?
        .call0()?;
    serializer.call_method1("dumps", (obj,))?.extract()
}

/// The running Python's "major.minor" (recorded in the UDF payload).
pub(crate) fn py_version(py: Python<'_>) -> String {
    (|| -> Option<String> {
        let vi = py.import("sys").ok()?.getattr("version_info").ok()?;
        let major: i64 = vi.getattr("major").ok()?.extract().ok()?;
        let minor: i64 = vi.getattr("minor").ok()?.extract().ok()?;
        Some(format!("{major}.{minor}"))
    })()
    .unwrap_or_else(|| "3.11".to_string())
}

/// The arguments of a call, bound to `names` positionally and by keyword as Python binds a
/// signature of those names with `defaults`, raising Python's `TypeError`s of `function`
/// for too many, repeated or unexpected arguments. For a method that tells a value of None
/// from an argument not given.
fn bind_arguments<'py, const N: usize>(
    function: &str,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
    names: [&str; N],
    defaults: [Bound<'py, PyAny>; N],
) -> PyResult<[Bound<'py, PyAny>; N]> {
    if args.len() > N {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{function}() takes from 1 to {} positional arguments but {} were given",
            N + 1,
            args.len() + 1
        )));
    }
    let mut values: [Option<Bound<'py, PyAny>>; N] = std::array::from_fn(|_| None);
    for (i, value) in args.iter().enumerate() {
        values[i] = Some(value);
    }
    if let Some(kwargs) = kwargs {
        for (key, value) in kwargs.iter() {
            let key: String = key.extract()?;
            let Some(i) = names.iter().position(|name| *name == key) else {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "{function}() got an unexpected keyword argument '{key}'"
                )));
            };
            if values[i].is_some() {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "{function}() got multiple values for argument '{key}'"
                )));
            }
            values[i] = Some(value);
        }
    }
    let mut defaults = defaults.into_iter();
    Ok(values.map(|value| {
        let default = defaults.next().expect("a default per name");
        value.unwrap_or(default)
    }))
}

/// Resolve the return type of a map or group function (a `DataType` wrapper or a DDL string): a
/// DDL string into the type that the server of `session` parses it to, as PySpark's `_parse_ddl`,
/// and a type object into None, which [`map_udf_of`] lowers when it builds the function, as the
/// `returnType` of PySpark's `UserDefinedFunction` reads it then, after the columns of the input.
/// Another object raises as [`check_return_type_argument`].
pub(crate) fn resolve_datatype(
    py: Python<'_>,
    session: &spark_connect::session::SparkSession,
    schema: &Bound<'_, PyAny>,
) -> PyResult<Option<spark_connect::types::DataType>> {
    if schema.extract::<PyRef<PyDataType>>().is_ok() {
        return Ok(None);
    }
    if schema.is_instance_of::<pyo3::types::PyString>() {
        return crate::types::schema_data_type(py, session, schema).map(Some);
    }
    Err(not_expected_type(schema, "returnType", "DataType or str"))
}

/// The return type of a map of a function of the return type `schema`, as [`resolve_datatype`]
/// gives it: of a DDL string, of the session of the client of `session`, the `_session` of the
/// DataFrame of the map, which parses it, as PySpark's `self._session._parse_ddl(schema)`, which
/// is read only then.
pub(crate) fn return_type_of(
    py: Python<'_>,
    session: &Bound<'_, PyAny>,
    schema: &Bound<'_, PyAny>,
) -> PyResult<Option<spark_connect::types::DataType>> {
    if schema.is_instance_of::<pyo3::types::PyString>() {
        return resolve_datatype(py, &request_session(session)?, schema);
    }
    if schema.extract::<PyRef<PyDataType>>().is_ok() {
        return Ok(None);
    }
    Err(not_expected_type(schema, "returnType", "DataType or str"))
}

/// Raises as the constructor of PySpark's `UserDefinedFunction` does for a return type that is
/// neither a `DataType` nor a str, `NOT_EXPECTED_TYPE` of `returnType`, before any request.
pub(crate) fn check_return_type_argument(schema: &Bound<'_, PyAny>) -> PyResult<()> {
    if schema.extract::<PyRef<PyDataType>>().is_ok()
        || schema.is_instance_of::<pyo3::types::PyString>()
    {
        return Ok(());
    }
    Err(not_expected_type(schema, "returnType", "DataType or str"))
}

/// The DataFrame of `mapInPandas` or `mapInArrow` of `df`, of the function `udf`, as PySpark's
/// `MapPartitions` plan sets its values when it is built: the function, then the flag `barrier`,
/// as protobuf converts it, and then the id of the resource profile `profile`, if any, which it
/// reads of `profile.id` then, so the profile is created on the server by the first request that
/// builds the plan, as `collect` (`schema` is of the DataFrame's cached schema), not by this call.
/// A flag that protobuf does not take fails each request, after the function is pickled, and the
/// id is not read, as PySpark's build stops at the flag; the DataFrame keeps its cached schema,
/// as PySpark's `_map_partitions` sets it of the schema.
fn map_partitions_of(
    py: Python<'_>,
    df: &DataFrame,
    udf: CommonInlineUserDefinedFunctionExpression,
    barrier: &ProtoBool,
    profile: Option<Bound<'_, PyAny>>,
) -> PyResult<DataFrame> {
    let mut args = crate::plan_args::PlanArgs::new();
    let is_barrier = args.converted(py, barrier.value(py))?;
    let reader = match profile.filter(|profile| !profile.is_none()) {
        Some(profile) if args.error().is_none() => Some(profile_id_reader(profile.unbind())),
        _ => None,
    };
    Ok(args.plan(df.map_in_python_with_profile_reader(udf, is_barrier, reader)))
}

/// The reader of `profile.id`, which a request calls when it builds the plan of `mapInPandas` or
/// `mapInArrow`, as PySpark's `MapPartitions` reads it then, of the int32 field of the proto: the
/// `AttributeError` of an object that has no `id`, the error of the command that creates the
/// profile, or that of a value that the field does not take, is the request's, as it is.
fn profile_id_reader(profile: Py<PyAny>) -> spark_connect::udf::CommandPickler {
    spark_connect::udf::CommandPickler::new(profile, |profile: &Py<PyAny>| {
        Python::attach(|py| {
            let id = profile.bind(py).getattr("id")?;
            let id: i32 = proto_int(&id, "spark.connect.MapPartitions.profile_id")?;
            Ok(id.to_le_bytes().to_vec())
        })
        .map_err(crate::errors::callback_error)
    })
}

/// The DataFrame `dataframe` of a map or group-map function of `schema`, the schema that the call
/// was given, as PySpark's `_map_partitions`, `applyInPandas`, `applyInArrow` and those of a
/// cogroup cache it, `res._cached_schema = schema` of a `StructType`: the object that it is, so
/// `schema` gives a copy of it as it is then and `columns` its names then, also once it was
/// changed in place, as the plan sends it as it is then (`udf_command::udf_pickler`); of a DDL
/// string, which PySpark parses into an object of its own, the struct of its type as the core
/// parsed it, of new objects.
pub(crate) fn map_result_of(
    schema: &Bound<'_, PyAny>,
    dataframe: DataFrame,
) -> PyResult<NewDataFrame> {
    let py = schema.py();
    let cached = if schema.is_instance_of::<crate::types::PyStructType>() {
        Some(schema.clone().unbind())
    } else if schema.is_instance_of::<pyo3::types::PyString>() {
        known_schema_object(py, &dataframe)?
    } else {
        None
    };
    Ok(PyDataFrame::new(dataframe).with_cached_schema(cached))
}

/// Validate the function of a map UDF for its eval type with
/// `pyspark.sql.pandas.functions._validate_vectorized_udf`, as the reference client does
/// before it builds the UDF.
pub(crate) fn validate_map_function(
    py: Python<'_>,
    func: &Bound<'_, PyAny>,
    eval_type_id: i32,
) -> PyResult<()> {
    py.import("pyspark.sql.pandas.functions")?
        .call_method1("_validate_vectorized_udf", (func, eval_type_id))?;
    Ok(())
}

/// The eval type of a grouped map function from its type hints, by `infer` of
/// `pyspark.sql.pandas.typehints` (the iterator forms, for example), as the reference
/// client infers it. `default` is used when the hints give none, and when they cannot be
/// read, after the reference client's `warning`, of the line `line` of its
/// `pyspark.sql.connect.group` (see `warn_as_pyspark`).
pub(crate) fn infer_group_map_eval_type(
    py: Python<'_>,
    func: &Bound<'_, PyAny>,
    infer: &str,
    warning: &str,
    line: u32,
    default: i32,
) -> PyResult<i32> {
    match py
        .import("pyspark.sql.pandas.typehints")?
        .call_method1(infer, (func,))
    {
        Ok(inferred) if inferred.is_none() => Ok(default),
        Ok(inferred) => inferred.extract(),
        Err(err) if err.is_instance_of::<pyo3::exceptions::PyException>(py) => {
            crate::errors::warn_as_pyspark(
                py,
                "pyspark.sql.connect.group",
                line,
                warning,
                &py.get_type::<pyo3::exceptions::PyUserWarning>(),
            )?;
            Ok(default)
        }
        Err(err) => Err(err),
    }
}

/// The name of the UDF of `func`, as PySpark's `UserDefinedFunction` names it: the
/// function's `__name__`, or the name of its class for a callable object without one.
fn udf_name(func: &Bound<'_, PyAny>) -> PyResult<String> {
    if func.hasattr("__name__")? {
        func.getattr("__name__")?.extract()
    } else {
        Ok(func.get_type().name()?.to_string())
    }
}

/// Build a map/group UDF expression: its command is `(func, schema)`, cloudpickled by each
/// request that builds the plan, and it carries the resolved schema as the output type (mirrors
/// the scalar UDF command layout).
/// The return type is resolved here, after the caller read the columns of the input, as the
/// `returnType` of PySpark's `UserDefinedFunction` parses a DDL string when the map node builds
/// the function, which `applyInPandasWithState` and `transformWithState` do.
pub(crate) fn build_map_udf(
    py: Python<'_>,
    session: &spark_connect::session::SparkSession,
    func: &Bound<'_, PyAny>,
    schema: &Bound<'_, PyAny>,
    eval_type_id: i32,
    input_cols: &[String],
) -> PyResult<CommonInlineUserDefinedFunctionExpression> {
    let return_type = resolve_datatype(py, session, schema)?;
    map_udf_of(py, func, schema, return_type, eval_type_id, input_cols)
}

/// The map/group UDF expression of [`build_map_udf`], of `return_type`, the type of `schema`
/// that [`resolve_datatype`] gives: for a function whose DDL string PySpark parses before it
/// reads the columns of the input, as `mapInPandas`, `applyInPandas` and their Arrow forms do,
/// and cogroups, which bind no columns.
pub(crate) fn map_udf_of(
    py: Python<'_>,
    func: &Bound<'_, PyAny>,
    schema: &Bound<'_, PyAny>,
    return_type: Option<spark_connect::types::DataType>,
    eval_type_id: i32,
    input_cols: &[String],
) -> PyResult<CommonInlineUserDefinedFunctionExpression> {
    let name = udf_name(func)?;
    // As PySpark's `MapPartitions`, `GroupMap` and the other map nodes do when they build the
    // UDF, the return type is checked for the eval type: a map or group function returns a
    // struct, so another type raises `INVALID_RETURN_TYPE_FOR_PANDAS_UDF` here, not at the action.
    // A type object is checked as it is, whose Arrow type of metadata that `json.dumps` refuses is
    // its `NOT_IMPLEMENTED`, and then lowered, as the live type of the object: an edit in place
    // of a StructType's fields or metadata after it was built is kept, as PySpark does.
    let checked = match &return_type {
        Some(return_type) => crate::types::data_type_to_py(py, return_type)?.into_bound(py),
        None => schema.clone(),
    };
    py.import("pyspark.sql.udf")?
        .getattr("UserDefinedFunction")?
        .call_method1("_check_return_type", (checked, eval_type_id))?;
    let return_type = match return_type {
        Some(return_type) => return_type,
        // A struct whose fields hold an element that is not a `StructField`, which the build
        // reads as PySpark's does (see `crate::types::py_to_proto_data_type`), stands as the types
        // that stand for its elements until then.
        None => {
            let (lowered, maybe_foreign) =
                crate::types::lower_noting_subclasses(schema, crate::types::MetadataOf::Strict);
            match crate::types::foreign_of_lowering(schema, maybe_foreign)? {
                true => crate::types::lenient_type(schema)?,
                false => lowered?,
            }
        }
    };
    // The worker reads the pickled type as a `DataType`, so a DDL string is pickled as the
    // type it parses to, as the reference client pickles the type of `_parse_ddl`.
    let pickled_type = if schema.is_instance_of::<pyo3::types::PyString>() {
        crate::types::data_type_to_py(py, &return_type)?.into_bound(py)
    } else {
        schema.clone()
    };
    // As PySpark's `PythonUDF.to_plan`, each request that builds the plan converts the output
    // type of the schema and pickles it with the function, not this call: so a `StructType`
    // changed in place after the call is the type that the server gets, and the command's.
    let tup = PyTuple::new(py, [func.clone(), pickled_type])?;
    let pickler = crate::udf_command::udf_pickler(tup.into_any().unbind());
    let payload =
        PythonUDFPayload::pickled_at_build(return_type, eval_type_id, pickler, py_version(py));
    // The worker reconstructs the input pandas/Arrow DataFrame of a map or group function from
    // the UDF's argument columns; without them the input columns are unnamed (KeyError in the
    // user fn). Mirror the reference Connect client, which passes the child's columns as the
    // function arguments, and none for a cogroup function, which takes the whole cogroups: each
    // the `ColumnReference` of the name, of `to_expr` of a UDF call, so a column named `*` or
    // `s.*` is that column, not a star.
    let arguments = input_cols
        .iter()
        .map(spark_connect::expression::Expression::column)
        .collect();
    Ok(CommonInlineUserDefinedFunctionExpression::new(
        name, true, arguments, payload,
    ))
}

/// The converters of the fields of a result of `schema` that deserialize its values of
/// user-defined types, as PySpark does when it collects them
/// (`pyspark.sql.types._udt_field_converters`); None without a user-defined type.
fn udt_field_converters(
    py: Python<'_>,
    schema: Option<&spark_connect::types::DataType>,
) -> PyResult<Option<FieldConverters>> {
    let Some(schema) = schema.filter(|schema| crate::types::has_converted_values(schema)) else {
        return Ok(None);
    };
    // As PySpark's, which converts the rows of the schema of the server, of a new object of each
    // user-defined type (see `schema_to_py`).
    let schema = crate::types::schema_to_py(py, schema)?;
    let converters = py
        .import("pyspark.sql.types")?
        .call_method1("_udt_field_converters", (schema,))?;
    if converters.is_none() {
        return Ok(None);
    }
    Ok(Some(std::sync::Arc::new(converters.extract()?)))
}

/// Whether `data_type` holds a year-month interval, also nested in an array, a map, a struct or
/// a user-defined type (`pyspark.sql.types._has_type`).
fn has_year_month_interval(data_type: &spark_connect::types::DataType) -> bool {
    has_type(data_type, &|t| {
        matches!(t, spark_connect::types::DataType::YearMonthInterval { .. })
    })
}

/// Whether `data_type` is of `is`, also nested in an array, a map, a struct or a user-defined
/// type.
fn has_type(
    data_type: &spark_connect::types::DataType,
    is: &dyn Fn(&spark_connect::types::DataType) -> bool,
) -> bool {
    use spark_connect::types::DataType;
    if is(data_type) {
        return true;
    }
    match data_type {
        DataType::Struct { fields } => fields.iter().any(|field| has_type(&field.data_type, is)),
        DataType::Array { element_type, .. } => has_type(element_type, is),
        DataType::Map {
            key_type,
            value_type,
            ..
        } => has_type(key_type, is) || has_type(value_type, is),
        DataType::Udt {
            sql_type: Some(sql_type),
            ..
        } => has_type(sql_type, is),
        _ => false,
    }
}

/// Raises as PySpark's `collect` does for a result of `schema` with a calendar interval: it
/// checks the Arrow schema of the result with `from_arrow_schema`, which has no type of the
/// Arrow `month_day_nano_interval` of a calendar interval. Its `toLocalIterator` does not.
fn check_arrow_schema(
    py: Python<'_>,
    schema: Option<&spark_connect::types::DataType>,
) -> PyResult<()> {
    let calendar = |t: &spark_connect::types::DataType| {
        matches!(t, spark_connect::types::DataType::CalendarInterval)
    };
    if schema.is_some_and(|schema| has_type(schema, &calendar)) {
        return Err(crate::column::raise_pyspark(
            py,
            "PySparkTypeError",
            "UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION",
            &[("data_type", "month_day_nano_interval")],
        ));
    }
    Ok(())
}

/// The proto text of the first nanosecond-precision timestamp of `data_type`, `TIMESTAMP_NTZ(p)`
/// or `TIMESTAMP_LTZ(p)` with p > 6, in the order of PySpark's conversion of a schema from its
/// proto: the message that PySpark's `proto_schema_to_pyspark_data_type` names when it fails.
pub(crate) fn nanosecond_timestamp_text(
    data_type: &spark_connect::types::DataType,
) -> Option<String> {
    use spark_connect::types::DataType;
    match data_type {
        DataType::TimestampNtzNanos { precision } => Some(format!(
            "timestamp_ntz_nanos {{\n  precision: {precision}\n}}\n"
        )),
        DataType::TimestampLtzNanos { precision } => Some(format!(
            "timestamp_ltz_nanos {{\n  precision: {precision}\n}}\n"
        )),
        DataType::Struct { fields } => fields
            .iter()
            .find_map(|field| nanosecond_timestamp_text(&field.data_type)),
        DataType::Array { element_type, .. } => nanosecond_timestamp_text(element_type),
        DataType::Map {
            key_type,
            value_type,
            ..
        } => nanosecond_timestamp_text(key_type).or_else(|| nanosecond_timestamp_text(value_type)),
        DataType::Udt {
            sql_type: Some(sql_type),
            ..
        } => nanosecond_timestamp_text(sql_type),
        _ => None,
    }
}

/// A type that PySpark's conversion of a schema from its proto, `proto_schema_to_pyspark_data_type`,
/// fails to build.
enum UnconvertibleType {
    /// A nanosecond-precision timestamp, which PySpark has no type of: the proto text of it.
    NanosecondTimestamp(String),
    /// A spatial type of an SRID that PySpark's `GeometryType` or `GeographyType` does not take.
    SpatialSrid(i32),
}

/// The first type of `data_type` that PySpark's conversion of a schema from its proto fails to
/// build, in the order in which it converts them (the fields of a struct, the element of an
/// array, the key and then the value of a map): a nanosecond-precision timestamp (see
/// [`nanosecond_timestamp_text`]), or a spatial type of an SRID that PySpark does not take (see
/// [`crate::types::pyspark_converts_srid`]). PySpark builds a user-defined type of its Python
/// class, not of its SQL type, of which only a nanosecond timestamp counts.
fn unconvertible_type(data_type: &spark_connect::types::DataType) -> Option<UnconvertibleType> {
    use spark_connect::types::DataType;
    let spatial = |srid: i32, geographic: bool| {
        (!crate::types::pyspark_converts_srid(srid, geographic))
            .then_some(UnconvertibleType::SpatialSrid(srid))
    };
    match data_type {
        DataType::Geometry { srid } => spatial(*srid, false),
        DataType::Geography { srid } => spatial(*srid, true),
        DataType::Struct { fields } => fields
            .iter()
            .find_map(|field| unconvertible_type(&field.data_type)),
        DataType::Array { element_type, .. } => unconvertible_type(element_type),
        DataType::Map {
            key_type,
            value_type,
            ..
        } => unconvertible_type(key_type).or_else(|| unconvertible_type(value_type)),
        _ => nanosecond_timestamp_text(data_type).map(UnconvertibleType::NanosecondTimestamp),
    }
}

/// Raises as PySpark does for a schema that it cannot convert from the proto of the schema (see
/// [`unconvertible_type`]): `UNSUPPORTED_OPERATION` for a nanosecond-precision timestamp, and
/// `ST_INVALID_SRID_VALUE` for a spatial type of an SRID that PySpark does not take. The
/// collecting APIs and every schema surface raise (the schema, columns, dtypes, printSchema,
/// repr, a DDL string of a schema), though the Rust core keeps these values, and count, show and
/// describe work.
pub(crate) fn check_schema_conversion(
    py: Python<'_>,
    schema: Option<&spark_connect::types::DataType>,
) -> PyResult<()> {
    match schema.and_then(unconvertible_type) {
        Some(UnconvertibleType::NanosecondTimestamp(text)) => Err(crate::column::raise_pyspark(
            py,
            "PySparkValueError",
            "UNSUPPORTED_OPERATION",
            &[("operation", &format!("data type {text}"))],
        )),
        Some(UnconvertibleType::SpatialSrid(srid)) => {
            Err(crate::types::invalid_schema_srid(py, srid))
        }
        None => Ok(()),
    }
}

/// Raises as PySpark's conversion of `schema`, a schema of the server, raises where its
/// `_schema` makes the type objects of it: the errors of [`check_schema_conversion`], and those of
/// its user-defined types, of which `UserDefinedType.fromJson` makes `UDT()` of the class that
/// the `pyClass` names (see `types::schema_to_py`), so that a class whose `__init__` takes
/// arguments raises its `TypeError`. The properties that read the schema, the column lookups,
/// the copies and the unpickles raise it so.
pub(crate) fn check_python_schema(
    py: Python<'_>,
    schema: &spark_connect::types::DataType,
) -> PyResult<()> {
    check_schema_conversion(py, Some(schema))?;
    if crate::types::has_udt(schema) {
        crate::types::schema_to_py(py, schema)?;
    }
    Ok(())
}

/// The `_cached_schema` of PySpark's DataFrame of `dataframe`, of the schema that the core caches,
/// which is of the server: its type objects, as PySpark converts the schema of the server, of a
/// new object of each user-defined type (see `types::schema_to_py`), and not of the objects that
/// the client was given. None when it has none, or when PySpark would not have one, as its
/// conversion of the schema raised (see [`check_python_schema`]).
fn known_schema_object(py: Python<'_>, dataframe: &DataFrame) -> PyResult<Option<Py<PyAny>>> {
    schema_object_of(py, dataframe.known_schema())
}

/// The Python object of `schema`, a schema that a DataFrame cached, of the types of PySpark, when
/// they hold it (see `known_schema_object`).
fn schema_object_of(
    py: Python<'_>,
    schema: Option<spark_connect::types::DataType>,
) -> PyResult<Option<Py<PyAny>>> {
    let Some(schema) = schema else {
        return Ok(None);
    };
    if check_schema_conversion(py, Some(&schema)).is_err() {
        return Ok(None);
    }
    match crate::types::schema_to_py(py, &schema) {
        Ok(schema) => Ok(Some(schema)),
        Err(err) if err.is_instance_of::<pyo3::exceptions::PyException>(py) => Ok(None),
        Err(err) => Err(err),
    }
}

/// The result of a set operation of `this` and `other`, `dataframe`, of the `_cached_schema` of
/// PySpark's `_merge_cached_schema`: `this.schema` when `this` has a `_cached_schema` and it is
/// equal to that of `other`, as PySpark compares them with `==`, else None.
fn set_operation_result(
    py: Python<'_>,
    this: &PlanFrame<'_>,
    other: &PlanFrame<'_>,
    dataframe: DataFrame,
) -> PyResult<NewDataFrame> {
    let attribute = pyo3::intern!(py, "_cached_schema");
    let mine = this.slf.as_any().getattr(attribute)?;
    let merged = !mine.is_none() && mine.eq(other.slf.as_any().getattr(attribute)?)?;
    let schema = if merged {
        Some(schema_property(&this.slf, py)?.unbind())
    } else {
        None
    };
    Ok(this.derived(dataframe).with_cached_schema(schema))
}

/// The schema that the properties of a DataFrame read, as PySpark's read its `_schema` (see
/// [`property_schema`]): its `_cached_schema` or the schema of a request, or the Column of
/// `_schema` of the `__getattr__` of the DataFrame, which PySpark's properties read as they read
/// any `_schema`.
struct PropertySchema<'py>(Bound<'py, PyAny>);

impl<'py> PropertySchema<'py> {
    /// As PySpark's `columns`, `[field.name for field in self._schema.fields]`.
    fn columns(self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let names = PyList::empty(py);
        for field in self.0.getattr(pyo3::intern!(py, "fields"))?.try_iter()? {
            names.append(field?.getattr(pyo3::intern!(py, "name"))?)?;
        }
        Ok(names)
    }

    /// As PySpark's `dtypes`, of the `simpleString()` of the type of each field.
    fn dtypes(self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        object_dtypes(py, &self.0)
    }
}

/// The warning of PySpark's logger of a pickle of the schema of a DataFrame that failed, of
/// `operation`, `dumps` or `loads`, and of `err`, as PySpark's `_schema` and `schema` log it.
fn schema_pickle_warning(py: Python<'_>, operation: &str, err: &PyErr) -> PyResult<()> {
    let logger = py
        .import(pyo3::intern!(py, "pyspark.sql.connect.logging"))?
        .getattr(pyo3::intern!(py, "logger"))?;
    let error = err.value(py).str()?;
    logger.call_method1(
        pyo3::intern!(py, "warning"),
        (format!(
            "DataFrame schema pickle {operation} failed with exception: {error}."
        ),),
    )?;
    Ok(())
}

/// The `(str(field.name), field.dataType.simpleString())` of each field of `schema`, a schema of
/// `to` that is not a DataType, as PySpark's `dtypes` reads them: an object without `fields`, or
/// whose fields lack these attributes, raises the error of Python for it.
fn object_dtypes<'py>(py: Python<'py>, schema: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let pairs = PyList::empty(py);
    for field in schema.getattr("fields")?.try_iter()? {
        let field = field?;
        let name = field.getattr("name")?.str()?;
        let simple = field.getattr("dataType")?.call_method0("simpleString")?;
        pairs.append((name, simple))?;
    }
    Ok(pairs)
}

/// The DataFrame of `plan`, a plan of the Python client of `session`, an object of the session
/// `core_session`, as the requests of `DataFrame(plan, session)` make it, and the error that it
/// keeps when its build failed: of the relation of `plan.to_proto(session.client)`, of
/// the plan ID of the plan. A plan whose build raises an `Exception` gives a DataFrame that keeps
/// it, of which each request raises a new exception ([`crate::errors::kept_error`]), as each of
/// PySpark's requests builds the plan and raises; another exception of the build, such as the
/// `KeyboardInterrupt` of a Ctrl-C while it runs or a `SystemExit`, is raised now, as it would be
/// where PySpark's program is then, which goes on after it.
pub(crate) fn dataframe_of_python_plan(
    plan: &Bound<'_, PyAny>,
    session: &Bound<'_, PyAny>,
    core_session: spark_connect::session::SparkSession,
) -> PyResult<(DataFrame, Option<spark_connect_core::error::SparkError>)> {
    use prost::Message;
    let py = plan.py();
    let client = session.getattr("client")?;
    // The plan, as PySpark Connect's DataFrame keeps it, keeps the models whose cached objects on
    // the server the relation refers to (`__source_instance__`).
    let owner: std::sync::Arc<dyn std::any::Any + Send + Sync> =
        std::sync::Arc::new(plan.clone().unbind());
    // Of the session of the build, the plans that the Columns of the subqueries of the plan build
    // again (`errors::of_request_session`), as PySpark's plan builds them of its client.
    let built = crate::errors::of_request_session(&core_session, || {
        plan.call_method1("to_proto", (client,))
            .and_then(|built| built.getattr("root"))
            .and_then(|relation| relation.call_method0("SerializeToString"))
            .and_then(|relation| {
                // With the type URLs of placeholders of its encoding, of which the core finds
                // those of its relations as it walks them (see `plan::BuiltOutside`).
                spark_connect::plan::BuiltOutside::decoded(
                    relation.cast::<pyo3::types::PyBytes>()?.as_bytes(),
                    spark_connect_proto::Relation::decode,
                )
                .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
            })
            .and_then(|relation| {
                DataFrame::from_relation(
                    core_session.clone(),
                    relation,
                    Some(std::sync::Arc::clone(&owner)),
                )
                .to_pyerr()
            })
    });
    let (dataframe, kept) = match built {
        Ok(dataframe) => (dataframe, None),
        Err(exception) if exception.is_instance_of::<pyo3::exceptions::PyException>(py) => {
            let (error, owner) = crate::errors::kept_error(py, plan, exception)?;
            (
                DataFrame::from_unbuilt_relation(core_session, Some(owner), error.clone()),
                Some(error),
            )
        }
        Err(exception) => return Err(exception),
    };
    // The plan ID of the plan, of the counter of the core (`LogicalPlan._fresh_plan_id`), of no
    // other node; one that is not an int, of an object that is not such a plan, leaves the plan
    // ID of the core.
    if let Some(plan_id) = plan
        .getattr(pyo3::intern!(py, "_plan_id"))
        .ok()
        .filter(|id| id.is_exact_instance_of::<pyo3::types::PyInt>())
        .and_then(|id| id.extract::<i64>().ok())
    {
        dataframe.assign_plan_id(plan_id);
    }
    Ok((dataframe, kept))
}

/// The DataFrame of `plan`, a plan of the Python client whose build failed when an object over it
/// was pickled, which the pickle carried, as PySpark's pickle carries its plan objects: built
/// again, as PySpark builds the plan when the unpickled object is used, of its error if it raises
/// again ([`dataframe_of_python_plan`]). Of an object `derived` from the DataFrame of the plan,
/// such as a DataFrame over it or a Column of its subquery, whose own plan the pickle could not
/// carry, a plan that builds now gives a DataFrame whose requests raise the error of
/// [`derived_plan_error`]. With the error of each request of the DataFrame, if it has one.
pub(crate) fn rebuilt_python_plan(
    plan: &Bound<'_, PyAny>,
    session: &Bound<'_, crate::session::PySparkSession>,
    derived: bool,
) -> PyResult<(DataFrame, Option<spark_connect_core::error::SparkError>)> {
    let (dataframe, kept) =
        dataframe_of_python_plan(plan, session.as_any(), session.borrow().session.clone())?;
    if kept.is_some() || !derived {
        return Ok((dataframe, kept));
    }
    let error = derived_plan_error();
    Ok((
        DataFrame::from_unbuilt_relation(session.borrow().session.clone(), None, error.clone()),
        Some(error),
    ))
}

/// The error of the requests of an unpickled object over a plan of the Python client that failed
/// to build when the object was pickled and builds when it was unpickled: the pickle carries the
/// plan, and not the plan of the object over it, which it could not build.
pub(crate) fn derived_plan_error() -> spark_connect_core::error::SparkError {
    spark_connect_core::error::SparkError::builtin_runtime_msg(
        "the plan of the Python client that the pickled object was made over failed to build \
         when it was pickled and builds when it is unpickled, so the pickle has no plan of the \
         object itself",
    )
}

pub(crate) fn pickled_relation_of<'py>(
    py: Python<'py>,
    dataframe: &DataFrame,
) -> PyResult<PickledRelation<'py>> {
    use pyo3::types::PyBytes;
    let (holes, copies) = crate::udf_command::PickleHoles::of(py, dataframe.picklers());
    let built = if copies.is_empty() {
        crate::origin::pickling(|| dataframe.to_relation())
    } else {
        let marked = dataframe.with_pickler_copies(&copies);
        crate::origin::pickling(|| marked.to_relation())
    };
    Ok(match built {
        Ok(relation) => {
            let relation = crate::encode_on_grown_stack(relation);
            let holes = holes.holes_of(py, &relation)?;
            (PyBytes::new(py, &relation), None, holes)
        }
        // Of the DataFrame of the plan of a kept error itself, or of one over it.
        Err(error) => {
            match crate::column::described_error_of(py, &error, !dataframe.is_unbuilt_with(&error))?
            {
                Some(description) => (PyBytes::new(py, b""), Some(description), PyList::empty(py)),
                None => return Err(crate::errors::spark_error_to_py_exception(error)),
            }
        }
    })
}

/// The native base of the plan object of a DataFrame of the core, `DataFrame._plan`, of the class
/// `_DataFramePlan` of `pyspark.sql.connect.plan`, which subclasses it and PySpark's `LogicalPlan`:
/// the plan of the core that it stands for, and its reference to the parts of that plan, which
/// `__traverse__` reports (see [`crate::gc`]), as PySpark's plan object holds the plan objects of
/// its inputs and the functions of its UDFs. It holds no DataFrame, as PySpark's plan object holds
/// none: so a DataFrame that keeps an object over its plan, such as its stream writer, is freed at
/// its last reference, as PySpark's is. Its `__dict__` holds the fields of PySpark's
/// `LogicalPlan`, which `_DataFramePlan.__init__` sets.
#[pyclass(
    name = "_NativePlan",
    module = "pyspark.sql.connect.plan",
    subclass,
    weakref,
    dict
)]
pub struct PyNativePlan {
    /// The plan of the core, a DataFrame of its session.
    pub(crate) dataframe: DataFrame,
    /// Set once the value is written (see [`crate::gc::GcReady`]).
    gc_ready: crate::gc::GcReady,
    /// The reference of this object to the parts of its plan, which `__traverse__` reports, as
    /// PySpark's plan object holds its plan objects (see [`crate::gc::Parts`]).
    parts: crate::gc::Parts,
}

impl PyNativePlan {
    /// The native part of the plan object of `dataframe`, of its reference to the parts of the
    /// plan, made with the GIL held.
    pub(crate) fn of(py: Python<'_>, dataframe: DataFrame) -> Self {
        PyNativePlan {
            parts: crate::gc::Parts::of(py, dataframe.held_parts()),
            dataframe,
            gc_ready: crate::gc::GcReady::new(),
        }
    }
}

#[pymethods]
impl PyNativePlan {
    /// `_NativePlan(df)`, of the plan of `df`, a DataFrame of the core: the `__init__` of the class
    /// that subclasses it sets the fields of PySpark's plan object.
    #[new]
    #[pyo3(signature = (df, *_args, **_kwargs))]
    fn py_new(
        df: &Bound<'_, PyAny>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let df = df.cast::<PyDataFrame>()?;
        let dataframe = PlanFrame::of(df)?.plan().clone();
        Ok(PyNativePlan::of(df.py(), dataframe))
    }

    /// The parts of the plan, which this object holds, as PySpark's plan object holds the plan
    /// objects of its inputs and the functions of its UDFs (see [`crate::gc`]); its `__dict__`,
    /// PySpark's fields, is traversed as that of a plain object.
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        if !self.gc_ready.get() {
            return Ok(());
        }
        self.parts.visit(&visit)
    }

    /// Break a reference cycle that the garbage collector found unreachable: the plan object drops
    /// its plan and its reference to the parts of it.
    fn __clear__(&mut self) {
        if self.gc_ready.get() {
            self.parts.release();
            self.dataframe.clear_plan();
        }
    }

    /// The plan ID of this plan, by which the columns of its DataFrame refer to it, and that of its
    /// relation within the `WithRelations` that sends it with the plans of its subqueries, if it
    /// has subqueries, for its `_root_plan_id` and `_plan_id_with_rel`, as those of PySpark's plan of a DataFrame.
    fn _plan_ids(&self) -> (i64, Option<i64>) {
        self.dataframe.plan_id_and_root()
    }

    /// The bytes of the placeholder of the relation of this plan, which `plan(session)` gives in
    /// its place: one extension that holds the relation encoded, which the core builds
    /// now, as PySpark's `plan(session)` builds the relation, with its errors, and replaces the
    /// placeholder with when the plan reaches it, also after the DataFrame and this object are gone. It is built
    /// of the session of `client`, the client that the plan is built of, as PySpark's plan builds
    /// the plan objects of the subqueries of its Columns of it, or else of the session of the
    /// plan (see `errors::of_request_session`).
    #[pyo3(signature = (client=None))]
    fn _relation_placeholder<'py>(
        &self,
        py: Python<'py>,
        client: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        use prost::Message;
        let session = client
            .and_then(crate::session::session_of_client)
            .unwrap_or_else(|| self.dataframe.spark_session());
        let placeholder = crate::errors::of_request_session(&session, || {
            self.dataframe.standalone_placeholder_relation().to_pyerr()
        })?;
        Ok(pyo3::types::PyBytes::new(py, &placeholder.encode_to_vec()))
    }

    /// The reduce of this plan object, as PySpark's plan object of a DataFrame pickles of its fields
    /// and of no session: the plan of `_column_reduce.sent_plan`
    /// of the relation of the plan as it was sent, of its holes and of the error of a plan
    /// that does not build, as the pickle of a DataFrame carries them (see `PyDataFrame::__reduce__`), a `LogicalPlan` on the
    /// client and on a Python worker of the server, whose PySpark has no class of the drop-in,
    /// which `DataFrame(plan, session)` and PySpark's DataFrame of a plan over it build alike, of
    /// `observations`, the Observations of the plan by their names. So a plan of the Python client
    /// over this plan, such as a `SubqueryAlias`, pickles plainly too, and
    /// unpickles on the worker, as PySpark's plan does; one of a model fails as the DataFrame's
    /// pickle does (see `PyDataFrame::__reduce_ex__`). Not a stable API.
    #[pyo3(signature = (observations=None))]
    fn _plan_reduce<'py>(
        slf: &Bound<'py, Self>,
        observations: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let (relation, build_error, holes) = pickled_relation_of(py, &slf.borrow().dataframe)?;
        let reduce_module = py.import("pyspark.sql.connect._column_reduce")?;
        let fill = if holes.is_empty() {
            py.None().into_bound(py)
        } else {
            reduce_module.getattr("fill_holes")?
        };
        let observations = observations.unwrap_or_else(|| py.None().into_bound(py));
        let args = (relation, build_error, holes, fill, observations).into_pyobject(py)?;
        let reduce = PyTuple::new(py, [reduce_module.getattr("sent_plan")?, args.into_any()])?;
        // As PySpark's pickle of the plan of a model, which fails at the objects of `pyspark.ml`
        // that it keeps, as the pickle of the DataFrame does (see `PyDataFrame::__reduce_ex__`).
        let dataframe = slf.borrow().dataframe.clone();
        let reach = crate::ml::object::pickle_reach(py, &dataframe)?;
        crate::ml::object::reach_reduce(py, &reach, reduce.into_any())
    }

    /// The Observations of this plan by their names, as PySpark Connect's `LogicalPlan.observations`
    /// of a `DataFrame._plan`, which its DataFrame gives the
    /// executions of `SparkSession.client` (`client.to_table(plan, df._plan.observations)`): those
    /// of `observe` on the DataFrame of this plan and on the inputs that PySpark's plans hold as their child,
    /// both sides of a join and of a set operation, and those of the DataFrames under a plan of
    /// Python (`DataFrame(plan, session)`), such as a `SubqueryAlias`, in the order of PySpark's
    /// dict. A DataFrame observed by a name has none, as in PySpark.
    fn _observations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let observations = PyDict::new(py);
        // The core gives them in the reverse of the order of PySpark's dict, which merges the
        // Observations of a plan after those of its inputs (see `DataFrame::observations`).
        for (_, observation) in self.dataframe.observations().into_iter().rev() {
            let python = crate::observation::python_observation_of(py, &observation)?;
            // As PySpark's key, `str(observation._name)`, of a name that UTF-8 cannot encode too.
            let name = python.getattr(pyo3::intern!(py, "_name"))?.str()?;
            observations.set_item(name, python)?;
        }
        Ok(observations)
    }

    /// The shallow copy of the plan object, as PySpark's `copy.copy` of a plan object, of its
    /// class and the same fields: of the same plan.
    fn __copy__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let native = PyNativePlan::of(py, slf.borrow().dataframe.clone());
        let copy = crate::reduce::new_of_class(&slf.get_type(), native)?;
        let fields = slf.getattr(pyo3::intern!(py, "__dict__"))?;
        copy.getattr(pyo3::intern!(py, "__dict__"))?
            .call_method1(pyo3::intern!(py, "update"), (fields,))?;
        Ok(copy)
    }

    /// The deep copy of the plan object, as PySpark's `copy.deepcopy` of a plan object, of its
    /// class and of deep copies of its fields, with `memo`: of the plan, of deep copies of the
    /// functions of its UDFs and of its Observations, as PySpark's copies the plan objects that
    /// hold them, of the same session, as PySpark's plan holds none. The objects of `pyspark.ml`
    /// that the plan keeps fail it as PySpark's (see `PyDataFrame::__deepcopy__`).
    fn __deepcopy__<'py>(
        slf: &Bound<'py, Self>,
        memo: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let dataframe = slf.borrow().dataframe.clone();
        let reach = crate::ml::object::pickle_reach(py, &dataframe)?;
        if let Some(error) = crate::ml::object::reach_deepcopy_error(py, &reach, memo)? {
            return Err(error);
        }
        let observations = dataframe
            .observations()
            .into_iter()
            .map(|(name, observation)| {
                Ok((
                    name,
                    crate::observation::deep_copy_in(py, &observation, memo)?,
                ))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let copies = crate::udf_command::deep_copied_picklers(py, dataframe.picklers(), memo)?;
        let session = dataframe.spark_session();
        let copied = crate::on_grown_stack(|| {
            dataframe.deep_copy_with_pickler_copies(session, observations, &copies)
        })
        .to_pyerr()?;
        let copy = crate::reduce::new_of_class(&slf.get_type(), PyNativePlan::of(py, copied))?;
        memo.set_item(slf.as_ptr() as usize, &copy)?;
        let fields = py.import("copy")?.call_method1(
            pyo3::intern!(py, "deepcopy"),
            (slf.getattr(pyo3::intern!(py, "__dict__"))?, memo),
        )?;
        copy.getattr(pyo3::intern!(py, "__dict__"))?
            .call_method1(pyo3::intern!(py, "update"), (fields,))?;
        Ok(copy)
    }
}

/// The error of a value that protobuf refuses in a field of a plan, which PySpark's plan raises as
/// a request sends it: a builtin `TypeError`, `OverflowError` or `ValueError`, as [`proto_int`]
/// and [`proto_bool`] raise them. Another error is raised at the call.
pub(crate) fn unbuildable_error(
    py: Python<'_>,
    err: PyErr,
) -> PyResult<spark_connect_core::error::SparkError> {
    use spark_connect_core::error::{SparkError, SparkErrorKind};
    let kind = if err.is_instance_of::<pyo3::exceptions::PyTypeError>(py) {
        SparkErrorKind::BuiltinTypeError
    } else if err.is_instance_of::<pyo3::exceptions::PyOverflowError>(py) {
        SparkErrorKind::BuiltinOverflowError
    } else if err.is_instance_of::<pyo3::exceptions::PyValueError>(py)
        && !err.is_instance_of::<pyo3::exceptions::PyUnicodeError>(py)
    {
        SparkErrorKind::BuiltinValueError
    } else {
        return Err(err);
    };
    let message = err.value(py).str()?.to_string();
    Ok(SparkError {
        kind,
        ..SparkError::value_msg(message)
    })
}

/// The plan of `df` with `plan` of the value `value`, which PySpark converts as it sends the plan:
/// a value that protobuf refuses makes `df` fail each request with its error, as
/// [`unbuildable_error`] gives it, with the schema that `df` cached, as PySpark's `limit`,
/// `coalesce`, `repartition` and `repartitionById` copy the `_cached_schema` of their input.
/// Another error is raised at the call.
fn converted_plan<T>(
    py: Python<'_>,
    df: &DataFrame,
    value: PyResult<T>,
    plan: impl FnOnce(T) -> DataFrame,
) -> PyResult<DataFrame> {
    match value {
        Ok(value) => Ok(plan(value)),
        Err(err) => Ok(df.unbuildable_with_cached_schema(unbuildable_error(py, err)?)),
    }
}

/// The error that PySpark's plan raises when it sends the str `text` of a protobuf string field,
/// which fails to convert as [`crate::options::str_argument`] gives it: a str that UTF-8 cannot
/// encode, bytes that are not UTF-8, or an object that is neither. None for a value that converts.
pub(crate) fn unbuildable_str_error(
    text: &Bound<'_, PyAny>,
) -> PyResult<Option<spark_connect_core::error::SparkError>> {
    use pyo3::types::PyBytes;
    use spark_connect_core::error::SparkError;
    let err = match crate::options::str_argument(text) {
        Ok(_) => return Ok(None),
        Err(err) => err,
    };
    let py = text.py();
    if err.is_instance_of::<pyo3::exceptions::PyUnicodeEncodeError>(py) {
        // The UTF-32 bytes of the str, which keep the surrogates, as `unencodable_str` takes them.
        let utf32le = text.call_method1("encode", ("utf-32-le", "surrogatepass"))?;
        let utf32le = utf32le.cast::<PyBytes>()?;
        return Ok(Some(SparkError::unencodable_str(utf32le.as_bytes())));
    }
    if err.is_instance_of::<pyo3::exceptions::PyUnicodeDecodeError>(py) {
        let bytes = text.cast::<PyBytes>()?;
        return Ok(Some(SparkError::invalid_utf8(0, bytes.as_bytes())));
    }
    Ok(Some(unbuildable_error(py, err)?))
}

/// The `eager` flag of a checkpoint, which PySpark passes to the constructor of its command: true
/// when it is omitted, and false for None, which the constructor leaves unset. Another value is
/// converted as [`proto_bool`] converts it.
fn checkpoint_eager(py: Python<'_>, eager: &DefaultedOption) -> PyResult<bool> {
    match eager {
        DefaultedOption::Omitted => Ok(true),
        DefaultedOption::Given(value) => {
            let value = value.bind(py);
            if value.is_none() {
                Ok(false)
            } else {
                proto_bool(value)
            }
        }
    }
}

/// The error that [`check_schema_conversion`] raises for a schema that it cannot convert, as the
/// core error of the same PySpark error class and parameters.
fn unconvertible_schema_error(
    schema: &spark_connect::types::DataType,
) -> Option<spark_connect_core::error::SparkError> {
    use spark_connect_core::error::{SparkError, SparkErrorKind};
    unconvertible_type(schema).map(|unconvertible| match unconvertible {
        UnconvertibleType::NanosecondTimestamp(text) => SparkError::classed(
            SparkErrorKind::ValueError,
            "UNSUPPORTED_OPERATION",
            &[("operation", &format!("data type {text}"))],
        ),
        UnconvertibleType::SpatialSrid(srid) => SparkError::classed(
            SparkErrorKind::IllegalArgument,
            "ST_INVALID_SRID_VALUE",
            &[("srid", &srid.to_string())],
        ),
    })
}

/// The check of the schema that the collecting APIs run as the server sends it, as
/// [`unconvertible_schema_error`]: PySpark raises the error of a schema that it cannot convert
/// when it receives the schema, before the rows and any later failure of the execution.
fn check_schema_at_receipt(
    schema: &spark_connect::types::DataType,
) -> spark_connect_core::error::Result<()> {
    match unconvertible_schema_error(schema) {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// Raises as PySpark's Arrow conversion of the rows does for a result of `schema` with a
/// year-month interval, which Spark Connect cannot collect: the error is raised even when the
/// result has no rows.
fn check_collectable(
    py: Python<'_>,
    schema: Option<&spark_connect::types::DataType>,
) -> PyResult<()> {
    if schema.is_some_and(has_year_month_interval) {
        return Err(crate::column::raise_pyspark(
            py,
            "PySparkNotImplementedError",
            "NOT_IMPLEMENTED",
            &[(
                "feature",
                "Collecting a year-month interval value in Spark Connect",
            )],
        ));
    }
    Ok(())
}

/// The binary result setting, as PySpark's `DataFrame._get_binary_as_bytes` reads it,
/// including when the server sends only an Arrow schema or the result has no binary fields.
/// Collect reads it after execution; a local iterator reads it once its plan is built, before
/// the execution starts.
fn read_binary_as_bytes(py: Python<'_>, dataframe: &DataFrame) -> PyResult<bool> {
    let session = dataframe.spark_session();
    let value = crate::interrupt::interruptible_without_parts(py, &session, || {
        session
            .conf()
            .get_with_default("spark.sql.execution.pyspark.binaryAsBytes", Some("true"))
    })?;
    Ok(value.is_some_and(|value| value.eq_ignore_ascii_case("true")))
}

/// The seed of a sample without one, drawn from Python's `random` as PySpark's
/// `random.randint(0, sys.maxsize)`, so that `random.seed` makes the rows repeat.
pub(crate) fn python_random_seed(py: Python<'_>) -> PyResult<i64> {
    let max_size = py.import("sys")?.getattr("maxsize")?;
    py.import("random")?
        .call_method1("randint", (0, max_size))?
        .extract()
}

/// The collected `rows` of a result of `schema`, with configured binary values and
/// user-defined values materialized before constructing the tuples returned to Python.
/// Reads the binary setting after execution, including with only an Arrow schema.
fn rows_to_py(
    py: Python<'_>,
    dataframe: &DataFrame,
    rows: Vec<spark_connect::row::Row>,
    schema: Option<&spark_connect::types::DataType>,
) -> PyResult<Vec<Py<PyAny>>> {
    check_schema_conversion(py, schema)?;
    check_arrow_schema(py, schema)?;
    let binary_as_bytes = read_binary_as_bytes(py, dataframe)?;
    check_collectable(py, schema)?;
    let converters = udt_field_converters(py, schema)?;
    let Some(first) = rows.first() else {
        return Ok(Vec::new());
    };
    let fields = PyList::new(py, first.fields())?;
    let mut nested_fields = NestedFieldNames::default();
    if let Some(schema) = schema {
        nested_fields.note_schema(schema);
    }
    // Drop each native payload after conversion instead of retaining two full results.
    rows.into_iter()
        .map(|row| {
            crate::row::row_values_to_py(
                py,
                &fields,
                row.values(),
                converters.as_ref(),
                binary_as_bytes,
                &mut nested_fields,
            )
        })
        .collect()
}

/// Python wrapper for LocalRowIterator yielding Row objects lazily: the rows that the generator
/// of `toLocalIterator` yields from. Implements `__iter__`, `__next__` and `close`.
#[pyclass(name = "LocalRowIterator", module = "pyspark.sql.dataframe")]
pub struct PyLocalRowIterator {
    /// The reference of the rows to the parts of the plan of `frame`, of which `dataframe` is a
    /// clone, which `__traverse__` reports (see [`crate::gc::Parts`]). The clone that the
    /// execution holds while it is prepared is not covered by it, which keeps the parts alive
    /// meanwhile.
    parts: crate::gc::Parts,
    dataframe: DataFrame,
    /// The DataFrame of `toLocalIterator`, as the generator of PySpark's holds `self` until it
    /// ends: the generator of the drop-in holds this iterator until then.
    frame: Py<PyDataFrame>,
    /// The execution of the rows once its plan is built, which the first row sends.
    prepared: Option<PreparedExecution>,
    /// The rows of the execution, which the first row starts, as PySpark's generator does.
    iterator: Option<LocalRowIterator>,
    /// Whether the iteration ended, with its last row or an error, as a generator ends.
    finished: bool,
    /// The converters of the fields of user-defined types of the rows, once the schema of the
    /// result is known: with the first row.
    converters: Option<Option<FieldConverters>>,
    /// Whether the conversion of the first record batch of the result was checked, once it
    /// arrived.
    batch_checked: bool,
    /// Read before the execution starts, after the plan is built, as the body of PySpark's
    /// generator reads it.
    binary_as_bytes: Option<bool>,
    /// Top-level names shared only by rows of the current Arrow record batch.
    fields: Option<(std::sync::Arc<[String]>, Py<PyList>)>,
    /// Nested names belong to schema paths and stay shared across iterator batches.
    nested_fields: NestedFieldNames,
}

impl PyLocalRowIterator {
    /// The rows of the DataFrame `frame`, of a clone of its plan.
    pub fn new(frame: &Bound<'_, PyDataFrame>) -> PyResult<Self> {
        let dataframe = Frame::of(frame)?.dataframe.clone();
        let parts = crate::gc::Parts::of(frame.py(), dataframe.held_parts());
        let frame = frame.clone().unbind();
        Ok(PyLocalRowIterator {
            parts,
            dataframe,
            frame,
            prepared: None,
            iterator: None,
            finished: false,
            converters: None,
            batch_checked: false,
            binary_as_bytes: None,
            fields: None,
            nested_fields: NestedFieldNames::default(),
        })
    }

    /// The next row, sending the prepared execution for the first; it fetches the next batch of
    /// the result when the batch that the iterator holds is used up.
    fn next_row(&mut self) -> Option<spark_connect_core::error::Result<spark_connect::row::Row>> {
        if let Some(prepared) = self.prepared.take() {
            match prepared.into_local_iterator(false) {
                Ok(iterator) => self.iterator = Some(iterator),
                Err(error) => return Some(Err(error)),
            }
        }
        self.iterator.as_mut()?.next()
    }

    /// Prepare and fetch one Python row; the public wrapper closes on exhaustion or error.
    fn next_python_row(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let session = self.dataframe.spark_session();
        let binary_as_bytes = match self.binary_as_bytes {
            Some(value) => value,
            None => {
                // As the body of PySpark's generator: the plan is built first (`to_proto`), so a
                // plan that does not build raises before any request, then the binary setting is
                // read, and then the execution starts.
                let prepared = interruptible_of(py, &session, &[&self.parts], || {
                    self.dataframe.prepare_execution().map_err(Box::new)
                })?;
                let value = read_binary_as_bytes(py, &self.dataframe)?;
                self.prepared = Some(prepared);
                self.binary_as_bytes = Some(value);
                value
            }
        };
        // A row of the batch that the iterator holds is taken with the GIL held: only the fetch
        // of a batch waits for the server, which releases it.
        let next = match self
            .iterator
            .as_mut()
            .and_then(LocalRowIterator::next_decoded)
        {
            Some(row) => Some(Ok(row)),
            // The first fetch sends the execution, of the request context of the client; the
            // others continue it, as PySpark's generator reads nothing of the client again.
            None if self.prepared.is_some() => {
                let _in_use = crate::gc::in_use(py, &[&self.parts]);
                interruptible(py, &session, || out_of_the_wait(self.next_row()))?
            }
            None => {
                let _in_use = crate::gc::in_use(py, &[&self.parts]);
                crate::interrupt::interruptible_continuation(py, &session, || {
                    out_of_the_wait(self.next_row())
                })?
            }
        };
        if let Some(iterator) = self.iterator.as_ref() {
            if self.converters.is_none() {
                if let Some(schema) = iterator.schema() {
                    // As PySpark's conversion of the schema of the response that carries it,
                    // before the rows and a later failure of the execution.
                    check_schema_conversion(py, Some(&schema))?;
                    self.converters = Some(udt_field_converters(py, Some(&schema))?);
                    self.nested_fields.note_schema(&schema);
                }
            }
            if !self.batch_checked && iterator.has_received_batch() {
                // As PySpark's `ArrowTableToRowsConversion.convert` of each record batch as it
                // arrives, also of one without rows, as the server sends for an empty result: its
                // year-month error is raised with the first record batch, before the error of a
                // later response, and not for an execution that fails before any.
                self.batch_checked = true;
                check_collectable(py, iterator.schema().as_ref())?;
            }
        }
        match next {
            Some(Ok(row)) => {
                let converters = self.converters.clone().flatten();
                let fields = match row.shared_fields() {
                    Some(native_fields) => match self.fields.as_ref() {
                        Some((cached, fields)) if std::sync::Arc::ptr_eq(cached, native_fields) => {
                            fields.bind(py).clone()
                        }
                        _ => {
                            let fields = PyList::new(py, row.fields())?;
                            self.fields = Some((
                                std::sync::Arc::clone(native_fields),
                                fields.clone().unbind(),
                            ));
                            fields
                        }
                    },
                    None => {
                        self.fields = None;
                        PyList::new(py, row.fields())?
                    }
                };
                crate::row::row_values_to_py(
                    py,
                    &fields,
                    row.values(),
                    converters.as_ref(),
                    binary_as_bytes,
                    &mut self.nested_fields,
                )
                .map(Some)
            }
            Some(Err(error)) => Err(crate::errors::spark_error_to_py_exception(error)),
            None => Ok(None),
        }
    }
}

/// `next`, the next row of an iterator of rows, of its error as the error of the wait of the call
/// when it is the exception of Python code of the client that is a `BaseException` only, as the
/// `KeyboardInterrupt` of a handler of the logs at a response, which goes out of PySpark's
/// generator before the row and the checks of its schema, and which the wait converts, as the
/// `except KeyboardInterrupt` of PySpark's execution, into the interrupt of the execution (see
/// `crate::interrupt::interruptible`). The other errors of a row are raised once the schema of the
/// result that came before them is checked.
#[allow(clippy::type_complexity)]
fn out_of_the_wait(
    next: Option<spark_connect_core::error::Result<spark_connect::row::Row>>,
) -> Result<
    Option<spark_connect_core::error::Result<spark_connect::row::Row>>,
    Box<spark_connect_core::error::SparkError>,
> {
    match next {
        Some(Err(error)) if crate::errors::is_base_exception_only(&error) => Err(Box::new(error)),
        next => Ok(next),
    }
}

impl PyLocalRowIterator {
    /// The iterator, for a call that runs or closes it. The fetch of a batch releases the GIL
    /// with the iterator borrowed, so that a second thread can call it meanwhile: its state is
    /// never shared, as the borrow is exclusive, and the second call raises the `ValueError` of
    /// a generator that is already executing, as PySpark's generator does, rather than the
    /// `RuntimeError` of the borrow.
    fn not_executing<'py>(slf: &Bound<'py, Self>) -> PyResult<PyRefMut<'py, Self>> {
        slf.try_borrow_mut()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("generator already executing"))
    }

    /// End the iteration, releasing its execution without the GIL.
    fn finish(&mut self, py: Python<'_>) {
        let iterator = self.iterator.take();
        self.prepared = None;
        self.finished = true;
        self.fields = None;
        self.nested_fields.clear();
        crate::gc::detach_without_parts(py, move || drop(iterator));
    }
}

#[pymethods]
impl PyLocalRowIterator {
    /// Make this object an iterator by returning self.
    fn __iter__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }

    /// Close the iterator, as a generator does: the rows that are left are not fetched, so the
    /// execution of the query is released, and the iteration ends. Closing it again does nothing.
    fn close(slf: &Bound<'_, Self>) -> PyResult<()> {
        Self::not_executing(slf)?.finish(slf.py());
        Ok(())
    }

    /// Fetch the next row from the iterator.
    ///
    /// The first call builds the plan, then reads the binary result setting, then starts the
    /// execution, as PySpark's generator does. The Config RPC and the fetch of each batch release
    /// the GIL; the rows of a fetched batch are converted with it held.
    fn __next__(slf: &Bound<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        let py = slf.py();
        let mut this = Self::not_executing(slf)?;
        if this.finished {
            return Ok(None);
        }
        match this.next_python_row(py) {
            Ok(Some(row)) => Ok(Some(row)),
            result => {
                this.finish(py);
                result
            }
        }
    }

    /// As the generator of PySpark's `toLocalIterator`, which holds its DataFrame, a Python object
    /// that Python's garbage collector sees: the DataFrame is reported, and the parts of the plan
    /// of which the rows hold a clone, such as the functions of its UDFs (see [`crate::gc`]). So a
    /// reference cycle through the rows and a function of the plan, such as a bound method of an
    /// object that keeps the rows, is freed. The iterator has no `__dict__`: it is tracked when it
    /// is allocated, and its value is written before any allocation that could start a
    /// collection, so no [`crate::gc::GcReady`] is needed.
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        visit.call(&self.frame)?;
        self.parts.visit(&visit)
    }

    /// Break a reference cycle that the garbage collector found unreachable: the iterator drops
    /// its clone of the plan, and its reference to its parts.
    fn __clear__(&mut self) {
        self.parts.release();
        self.prepared = None;
        self.dataframe.clear_plan();
    }
}

/// The metrics and observed metrics of an execution, as the lists of PySpark's client: of the
/// `PlanMetrics` that `SparkConnectClient._build_metrics` makes, and of the `PlanObservedMetrics`
/// that `_execute_and_fetch_as_iterator` makes of the `Expression.Literal` messages that the
/// server sent, which their `pairs` converts when it is read.
pub(crate) fn execution_metrics<'py>(
    py: Python<'py>,
    info: &spark_connect::session::ExecutionInfo,
) -> PyResult<(
    Bound<'py, pyo3::types::PyList>,
    Bound<'py, pyo3::types::PyList>,
)> {
    let metrics_mod = py.import("pyspark.sql.metrics")?;
    let plan_metrics_cls = metrics_mod.getattr("PlanMetrics")?;
    let metric_value_cls = metrics_mod.getattr("MetricValue")?;
    let plan_metrics = pyo3::types::PyList::empty(py);
    if let Some(m) = &info.metrics {
        for x in &m.metrics {
            let values = pyo3::types::PyList::empty(py);
            for (k, v) in &x.execution_metrics {
                values.append(metric_value_cls.call1((k, v.value, &v.metric_type))?)?;
            }
            plan_metrics.append(plan_metrics_cls.call1((&x.name, x.plan_id, x.parent, values))?)?;
        }
    }
    let observed_cls = py
        .import("pyspark.sql.connect.client.core")?
        .getattr("PlanObservedMetrics")?;
    // The metrics as PySpark's client keeps them, the `Expression.Literal` messages that the
    // server sent, of `pyspark.sql.connect.proto`, which `PlanObservedMetrics.pairs` converts
    // when it is read, and so raises the `UNSUPPORTED_LITERAL` of a literal that PySpark does not
    // convert there. They are parsed of any depth, as the transport has read their response
    // within the limit of upb's parser (see `spark_connect_proto::ClientCodec`).
    let literal_cls = py
        .import("pyspark.sql.connect.proto")?
        .getattr("Expression")?
        .getattr("Literal")?;
    let observed = pyo3::types::PyList::empty(py);
    for o in &info.observed_metrics {
        let values = o
            .values
            .iter()
            .map(|literal| {
                use prost::Message;
                crate::pbmsg::parse_deep(&literal_cls, &literal.encode_to_vec())
            })
            .collect::<PyResult<Vec<_>>>()?;
        observed.append(observed_cls.call1((&o.name, values, o.keys.clone()))?)?;
    }
    Ok((plan_metrics, observed))
}

/// Whether `a` and `b` are the information of one execution, of which `executionInfo` gives one
/// object: of the same operation, which the core makes for each execution, with the same metrics
/// and observed metrics. An observed metric of a NaN, which is not equal to itself, compares by
/// its encoding.
fn same_execution(
    a: &spark_connect::session::ExecutionInfo,
    b: &spark_connect::session::ExecutionInfo,
) -> bool {
    use prost::Message;
    a.operation_id == b.operation_id
        && a.metrics == b.metrics
        && a.observed_metrics.len() == b.observed_metrics.len()
        // Of literals of up to `spark_connect_proto::MAX_KEPT_DEPTH` levels.
        && (a.observed_metrics.is_empty()
            || crate::on_grown_stack(|| {
                a.observed_metrics
                    .iter()
                    .zip(&b.observed_metrics)
                    .all(|(x, y)| x == y || x.encode_to_vec() == y.encode_to_vec())
            }))
}

/// Convert an observed-metric literal to a Python value, as the metrics of an Observation.
/// Mirrors `LiteralExpression._to_value`: the kinds it decodes map to the same Python values,
/// and for any other kind (which the reference rejects with `UNSUPPORTED_LITERAL`, and which
/// `check_observed_metrics` refuses before) this returns an `_UnsupportedLiteral` marker.
/// `PlanObservedMetrics` keeps the literals themselves (see [`execution_metrics`]).
///
/// `data_type` is the literal's type when the enclosing literal carries it; a struct's
/// field names come from it, else from the literal's own `data_type`, else from the
/// deprecated `struct_type`.
pub(crate) fn literal_to_py(
    py: Python<'_>,
    literal: &spark_connect_proto::expression::Literal,
    data_type: Option<&spark_connect_proto::DataType>,
) -> PyResult<Py<PyAny>> {
    // A level of a literal of many levels, of the observed metrics of a response, on a stack with
    // room.
    crate::pbmsg::grow(|| literal_to_py_level(py, literal, data_type))
}

/// [`literal_to_py`] of one level of a literal.
fn literal_to_py_level(
    py: Python<'_>,
    literal: &spark_connect_proto::expression::Literal,
    data_type: Option<&spark_connect_proto::DataType>,
) -> PyResult<Py<PyAny>> {
    use pyo3::IntoPyObjectExt;
    use spark_connect_proto::data_type::Kind;
    use spark_connect_proto::expression::literal::LiteralType;
    let data_type = data_type.or(literal.data_type.as_ref());
    let datetime = || py.import("datetime");
    Ok(match &literal.literal_type {
        None | Some(LiteralType::Null(_)) => py.None(),
        Some(LiteralType::Boolean(b)) => b.into_py_any(py)?,
        Some(LiteralType::Byte(v))
        | Some(LiteralType::Short(v))
        | Some(LiteralType::Integer(v)) => v.into_py_any(py)?,
        Some(LiteralType::Long(v)) => v.into_py_any(py)?,
        Some(LiteralType::Float(v)) => (*v as f64).into_py_any(py)?,
        Some(LiteralType::Double(v)) => v.into_py_any(py)?,
        Some(LiteralType::String(s)) => s.into_py_any(py)?,
        Some(LiteralType::Binary(b)) => pyo3::types::PyBytes::new(py, b).into_any().unbind(),
        Some(LiteralType::Decimal(d)) => py
            .import("decimal")?
            .getattr("Decimal")?
            .call1((&d.value,))?
            .unbind(),
        // DateType.fromInternal: days since the epoch (ordinal 719163).
        Some(LiteralType::Date(days)) => datetime()?
            .getattr("date")?
            .call_method1("fromordinal", (*days as i64 + 719_163,))?
            .unbind(),
        // TimeType.fromInternal: nanoseconds of the day, truncated to microseconds.
        Some(LiteralType::Time(t)) => {
            let nano = t.nano;
            datetime()?
                .getattr("time")?
                .call1((
                    nano.div_euclid(3_600_000_000_000),
                    nano.rem_euclid(3_600_000_000_000) / 60_000_000_000,
                    nano.rem_euclid(60_000_000_000) / 1_000_000_000,
                    nano.rem_euclid(1_000_000_000) / 1_000,
                ))?
                .unbind()
        }
        // TimestampType.fromInternal: a naive local datetime, built from whole seconds so
        // the microseconds are exact.
        Some(LiteralType::Timestamp(ts)) => {
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("microsecond", ts.rem_euclid(1_000_000))?;
            datetime()?
                .getattr("datetime")?
                .call_method1("fromtimestamp", (ts.div_euclid(1_000_000),))?
                .call_method("replace", (), Some(&kwargs))?
                .unbind()
        }
        // TimestampNTZType.fromInternal: the UTC wall clock, without a zone.
        Some(LiteralType::TimestampNtz(ts)) => {
            let dt = datetime()?;
            let utc = dt.getattr("timezone")?.getattr("utc")?;
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("microsecond", ts.rem_euclid(1_000_000))?;
            kwargs.set_item("tzinfo", py.None())?;
            dt.getattr("datetime")?
                .call_method1("fromtimestamp", (ts.div_euclid(1_000_000), utc))?
                .call_method("replace", (), Some(&kwargs))?
                .unbind()
        }
        // DayTimeIntervalType.fromInternal: timedelta(microseconds=v).
        Some(LiteralType::DayTimeInterval(micros)) => {
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("microseconds", micros)?;
            datetime()?
                .getattr("timedelta")?
                .call((), Some(&kwargs))?
                .unbind()
        }
        Some(LiteralType::Array(a)) => {
            let element_type = match data_type.and_then(|t| t.kind.as_ref()) {
                Some(Kind::Array(t)) => t.element_type.as_deref(),
                _ => None,
            };
            let items = a
                .elements
                .iter()
                .map(|e| literal_to_py(py, e, element_type))
                .collect::<PyResult<Vec<_>>>()?;
            items.into_py_any(py)?
        }
        Some(LiteralType::Map(m)) => {
            let (key_type, value_type) = match data_type.and_then(|t| t.kind.as_ref()) {
                Some(Kind::Map(t)) => (t.key_type.as_deref(), t.value_type.as_deref()),
                _ => (None, None),
            };
            let dict = pyo3::types::PyDict::new(py);
            for (k, v) in m.keys.iter().zip(&m.values) {
                dict.set_item(
                    literal_to_py(py, k, key_type)?,
                    literal_to_py(py, v, value_type)?,
                )?;
            }
            dict.into_any().unbind()
        }
        Some(LiteralType::Struct(s)) => {
            #[allow(deprecated)]
            let struct_type = match data_type.and_then(|t| t.kind.as_ref()) {
                Some(Kind::Struct(t)) => Some(t),
                _ => match s.struct_type.as_ref().and_then(|t| t.kind.as_ref()) {
                    Some(Kind::Struct(t)) => Some(t),
                    _ => None,
                },
            };
            let fields = struct_type.map(|t| t.fields.as_slice()).unwrap_or_default();
            let names = (0..s.elements.len())
                .map(|i| match fields.get(i) {
                    Some(f) => f.name.clone(),
                    None => format!("_{}", i + 1),
                })
                .collect::<Vec<_>>();
            let values = s
                .elements
                .iter()
                .enumerate()
                .map(|(i, e)| {
                    literal_to_py(py, e, fields.get(i).and_then(|f| f.data_type.as_ref()))
                })
                .collect::<PyResult<Vec<_>>>()?;
            py.import("pyspark.sql.types")?
                .getattr("_create_row")?
                .call1((names, values))?
                .unbind()
        }
        Some(_) => unsupported_literal_marker(py, literal)?,
    })
}

/// The `_UnsupportedLiteral` marker of `literal`, a literal of a kind that `literal_to_py` does
/// not convert, of its text: the value of such a metric of an Observation, whose metrics
/// `check_observed_metrics` refuses before they are converted, raising PySpark's
/// `UNSUPPORTED_LITERAL`. `PlanObservedMetrics` keeps the literals themselves, which its `pairs`
/// converts with PySpark's `LiteralExpression._to_value` (see [`execution_metrics`]).
fn unsupported_literal_marker(
    py: Python<'_>,
    literal: &spark_connect_proto::expression::Literal,
) -> PyResult<Py<PyAny>> {
    Ok(py
        .import("pyspark.sql.connect.client.core")?
        .getattr("_UnsupportedLiteral")?
        .call1((literal_text(py, literal),))?
        .unbind())
}

/// The first literal of `literal`, itself or an element, whose kind `literal_to_py` does not
/// convert, in the order in which PySpark's `LiteralExpression._to_value` converts them: the
/// literal that PySpark's `UNSUPPORTED_LITERAL` names.
pub(crate) fn unsupported_literal(
    literal: &spark_connect_proto::expression::Literal,
) -> Option<&spark_connect_proto::expression::Literal> {
    use spark_connect_proto::expression::literal::LiteralType;
    match &literal.literal_type {
        None
        | Some(LiteralType::Null(_))
        | Some(LiteralType::Boolean(_))
        | Some(LiteralType::Byte(_))
        | Some(LiteralType::Short(_))
        | Some(LiteralType::Integer(_))
        | Some(LiteralType::Long(_))
        | Some(LiteralType::Float(_))
        | Some(LiteralType::Double(_))
        | Some(LiteralType::String(_))
        | Some(LiteralType::Binary(_))
        | Some(LiteralType::Decimal(_))
        | Some(LiteralType::Date(_))
        | Some(LiteralType::Time(_))
        | Some(LiteralType::Timestamp(_))
        | Some(LiteralType::TimestampNtz(_))
        | Some(LiteralType::DayTimeInterval(_)) => None,
        Some(LiteralType::Array(array)) => array.elements.iter().find_map(unsupported_literal),
        // A key before its value, as a dict comprehension evaluates them.
        Some(LiteralType::Map(map)) => map.keys.iter().zip(&map.values).find_map(|(key, value)| {
            unsupported_literal(key).or_else(|| unsupported_literal(value))
        }),
        Some(LiteralType::Struct(fields)) => fields.elements.iter().find_map(unsupported_literal),
        Some(_) => Some(literal),
    }
}

/// The text of `literal` as PySpark's `str` of its protobuf message gives it, which its
/// `UNSUPPORTED_LITERAL` shows; the Rust debug text of the message without `protobuf`.
pub(crate) fn literal_text(
    py: Python<'_>,
    literal: &spark_connect_proto::expression::Literal,
) -> String {
    use prost::Message;
    let text = || -> PyResult<String> {
        let literal_cls = py
            .import("pyspark.sql.connect.proto")?
            .getattr("Expression")?
            .getattr("Literal")?;
        crate::pbmsg::parse_deep(&literal_cls, &literal.encode_to_vec())?
            .str()?
            .extract()
    };
    text().unwrap_or_else(|_| format!("{literal:?}"))
}

#[cfg(test)]
mod execution_info_tests {
    use super::same_execution;
    use spark_connect::session::ExecutionInfo;
    use spark_connect_proto::execute_plan_response::{Metrics, ObservedMetrics};
    use spark_connect_proto::expression::literal::LiteralType;
    use spark_connect_proto::expression::Literal;

    fn info(operation_id: &str, value: f64) -> ExecutionInfo {
        ExecutionInfo {
            metrics: Some(Metrics::default()),
            observed_metrics: vec![ObservedMetrics {
                name: "metric".to_string(),
                values: vec![Literal {
                    literal_type: Some(LiteralType::Double(value)),
                    ..Default::default()
                }],
                keys: vec!["value".to_string()],
                ..Default::default()
            }],
            operation_id: Some(operation_id.to_string()),
        }
    }

    #[test]
    fn the_information_of_one_execution_is_the_same_with_a_nan_observed_too() {
        assert!(same_execution(&info("a", 1.0), &info("a", 1.0)));
        assert!(same_execution(&info("a", f64::NAN), &info("a", f64::NAN)));
    }

    #[test]
    fn another_operation_or_other_metrics_are_of_another_execution() {
        assert!(!same_execution(&info("a", 1.0), &info("b", 1.0)));
        assert!(!same_execution(&info("a", 1.0), &info("a", 2.0)));
        assert!(!same_execution(&info("a", 1.0), &ExecutionInfo::default()));
        let mut without_metrics = info("a", 1.0);
        without_metrics.metrics = None;
        assert!(!same_execution(&info("a", 1.0), &without_metrics));
    }
}
