// SPDX-License-Identifier: Apache-2.0
//! The server-side objects of Spark Connect ML in Python, as official pyspark-client 4.3.0 has
//! them: the values of a fetch, as `deserialize_param` returns them; the arguments of a fetch, as
//! `serialize` sends them; and the writers and readers of the native classes (`MLWriter` and
//! `MLReader`, as `RemoteMLWriter` and `RemoteMLReader`).

use super::*;
use pyo3::types::PyString;
use spark_connect::ml::{
    FetchArg, FetchOutput, MlOperator, MlRef, MlTarget, MlValue, OperatorType,
};
use spark_connect::SparkSession;
use std::collections::HashMap;

/// The Python value of `value`, as `deserialize_param` of `pyspark.ml.connect.serialize` returns
/// it: an ML vector or matrix of `pyspark.ml.linalg`, and another value as
/// `LiteralExpression._to_value` returns it (a struct as a `Row`).
pub(super) fn ml_value_to_py(py: Python<'_>, value: &MlValue) -> PyResult<Py<PyAny>> {
    use pyo3::IntoPyObjectExt;
    use spark_connect::ml::{MlMatrix, MlVector};
    let linalg = || py.import("pyspark.ml.linalg");
    let types = || py.import("pyspark.sql.types");
    Ok(match value {
        MlValue::Null => py.None(),
        MlValue::Bool(v) => v.into_py_any(py)?,
        MlValue::Byte(v) | MlValue::Short(v) | MlValue::Int(v) => v.into_py_any(py)?,
        MlValue::Long(v) => v.into_py_any(py)?,
        MlValue::Float(v) => f64::from(*v).into_py_any(py)?,
        MlValue::Double(v) => v.into_py_any(py)?,
        MlValue::Decimal(v) => py
            .import("decimal")?
            .getattr("Decimal")?
            .call1((v,))?
            .unbind(),
        MlValue::String(v) => v.into_py_any(py)?,
        MlValue::Binary(v) => pyo3::types::PyBytes::new(py, v).into_any().unbind(),
        MlValue::Date(v) => types()?
            .getattr("DateType")?
            .call0()?
            .call_method1("fromInternal", (*v,))?
            .unbind(),
        MlValue::Timestamp(v) => types()?
            .getattr("TimestampType")?
            .call0()?
            .call_method1("fromInternal", (*v,))?
            .unbind(),
        MlValue::TimestampNtz(v) => types()?
            .getattr("TimestampNTZType")?
            .call0()?
            .call_method1("fromInternal", (*v,))?
            .unbind(),
        MlValue::Vector(MlVector::Dense(values)) => linalg()?
            .getattr("DenseVector")?
            .call1((values.clone(),))?
            .unbind(),
        MlValue::Vector(MlVector::Sparse {
            size,
            indices,
            values,
        }) => linalg()?
            .getattr("SparseVector")?
            .call1((*size, indices.clone(), values.clone()))?
            .unbind(),
        MlValue::Matrix(MlMatrix::Dense {
            num_rows,
            num_cols,
            values,
            is_transposed,
        }) => linalg()?
            .getattr("DenseMatrix")?
            .call1((*num_rows, *num_cols, values.clone(), *is_transposed))?
            .unbind(),
        MlValue::Matrix(MlMatrix::Sparse {
            num_rows,
            num_cols,
            col_ptrs,
            row_indices,
            values,
            is_transposed,
        }) => linalg()?
            .getattr("SparseMatrix")?
            .call1((
                *num_rows,
                *num_cols,
                col_ptrs.clone(),
                row_indices.clone(),
                values.clone(),
                *is_transposed,
            ))?
            .unbind(),
        MlValue::Array(items) => PyList::new(
            py,
            items
                .iter()
                .map(|v| ml_value_to_py(py, v))
                .collect::<PyResult<Vec<_>>>()?,
        )?
        .into_any()
        .unbind(),
        MlValue::Map(entries) => {
            let dict = PyDict::new(py);
            for (k, v) in entries {
                dict.set_item(ml_value_to_py(py, k)?, ml_value_to_py(py, v)?)?;
            }
            dict.into_any().unbind()
        }
        MlValue::Struct(fields) => {
            let names: Vec<&str> = fields.iter().map(|(n, _)| n.as_str()).collect();
            let values = fields
                .iter()
                .map(|(_, v)| ml_value_to_py(py, v))
                .collect::<PyResult<Vec<_>>>()?;
            types()?
                .getattr("_create_row")?
                .call1((names, values))?
                .unbind()
        }
    })
}

/// The argument `value` of a fetch, as `serialize` of `pyspark.ml.connect.serialize` sends it: a
/// DataFrame as its relation, an ML vector or matrix as itself, and another value as the literal
/// of `LiteralExpression._from_value`, which raises PySpark's error of a value that cannot be one.
pub(super) fn fetch_arg_of(value: &Bound<'_, PyAny>) -> PyResult<FetchArg> {
    if let Ok(df) = value.extract::<PyRef<'_, PyDataFrame>>() {
        return Ok(FetchArg::Input(df.dataframe.clone()));
    }
    let py = value.py();
    let linalg = py.import("pyspark.ml.linalg")?;
    for class in ["SparseVector", "DenseVector"] {
        if value.is_instance(&linalg.getattr(class)?)? {
            if let Converted::Vector(vector) = convert("value", Kind::Vector, value)? {
                return Ok(FetchArg::Value(MlValue::Vector(vector)));
            }
        }
    }
    for class in ["SparseMatrix", "DenseMatrix"] {
        if value.is_instance(&linalg.getattr(class)?)? {
            if let Converted::Matrix(matrix) = convert("value", Kind::Matrix, value)? {
                return Ok(FetchArg::Value(MlValue::Matrix(matrix)));
            }
        }
    }
    let column = crate::functions::to_value_column(value)?;
    match column.expression().node() {
        spark_connect::expression::ExpressionNode::Literal(literal) => {
            Ok(FetchArg::Literal(literal.to_literal_proto()))
        }
        _ => Err(crate::column::raise_pyspark(
            py,
            "PySparkTypeError",
            "UNSUPPORTED_DATA_TYPE",
            &[("data_type", &value.get_type().name()?.to_string())],
        )),
    }
}

/// The Python value of the result of a fetch: a value, or the string of a `summary` result.
pub(super) fn fetch_output_to_py(py: Python<'_>, output: FetchOutput) -> PyResult<Py<PyAny>> {
    use pyo3::IntoPyObjectExt;
    match output {
        FetchOutput::Value(value) => ml_value_to_py(py, &value),
        FetchOutput::Summary(text) => text.into_py_any(py),
        FetchOutput::Objects(objects) => Err(PyValueError::new_err(format!(
            "an ML method returned {} objects where a value was expected",
            objects.len()
        ))),
    }
}

/// Fetch the method `method` of `target`, of the Python arguments `args`, interruptibly.
pub(super) fn fetch_py(
    py: Python<'_>,
    target: &MlTarget,
    method: &str,
    args: &[Bound<'_, PyAny>],
) -> PyResult<FetchOutput> {
    let args = args
        .iter()
        .map(fetch_arg_of)
        .collect::<PyResult<Vec<_>>>()?;
    interruptible(py, target.session(), || {
        spark_connect::ml::fetch(target, method, args).map_err(Box::new)
    })
}

/// The DataFrame of the method `method` of `target`, of the Python arguments `args`.
pub(super) fn attribute_relation_py(
    target: &MlTarget,
    method: &str,
    args: &[Bound<'_, PyAny>],
) -> PyResult<PyDataFrame> {
    let args = args
        .iter()
        .map(fetch_arg_of)
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyDataFrame::new(
        spark_connect::ml::attribute_relation(target, method, args).to_pyerr()?,
    ))
}

/// The session of the active Spark session of Python, which PySpark's `RemoteMLWriter.save` and
/// `RemoteMLReader.load` use (`SparkSession.getActiveSession()`), or of `session` if given.
fn session_of(py: Python<'_>, session: Option<&Py<PyAny>>) -> PyResult<SparkSession> {
    if let Some(session) = session {
        return Ok(session
            .bind(py)
            .extract::<PyRef<'_, crate::session::PySparkSession>>()?
            .session
            .clone());
    }
    // As PySpark's `assert session is not None` of no active session.
    let active = crate::session::active_session(py)
        .ok_or_else(|| pyo3::exceptions::PyAssertionError::new_err(()))?;
    let session = active.borrow(py).session.clone();
    Ok(session)
}

/// What a writer saves: an operator that is not fitted (an estimator, a transformer or an
/// evaluator), of its params, or a fitted model's object, of its params.
#[derive(Clone)]
pub(super) enum WriteTarget {
    Operator(MlOperator, Params),
    Object(MlRef, Params),
}

/// PySpark's `MLWriter` of a native class (`RemoteMLWriter` over Spark Connect): `overwrite()`,
/// `option(key, value)` (of its lowercase key and string value), `session(spark)` and
/// `save(path)`, which sends the `Write`.
#[pyclass(name = "MLWriter", module = "pyspark.ml.util")]
pub struct PyMLWriter {
    target: WriteTarget,
    #[pyo3(get)]
    #[allow(non_snake_case)]
    shouldOverwrite: bool,
    options: HashMap<String, String>,
    session: Option<Py<PyAny>>,
    /// The error of PySpark's serialization of the params, which `save` raises.
    error: Option<PyErr>,
}

impl PyMLWriter {
    pub(super) fn new(target: WriteTarget) -> Self {
        Self {
            target,
            shouldOverwrite: false,
            options: HashMap::new(),
            session: None,
            error: None,
        }
    }

    pub(super) fn with_error(mut self, error: Option<PyErr>) -> Self {
        self.error = error;
        self
    }
}

#[pymethods]
impl PyMLWriter {
    /// Save to `path`, as `RemoteMLWriter.save`: of the active session of an operator, and of the
    /// session that holds a model's object.
    fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        if let Some(error) = &self.error {
            return Err(error.clone_ref(py));
        }
        match &self.target {
            WriteTarget::Object(object, params) => {
                let session = object.session().clone();
                interruptible(py, &session, || {
                    spark_connect::ml::write_object(
                        object,
                        params,
                        path,
                        self.shouldOverwrite,
                        &self.options,
                    )
                    .map_err(Box::new)
                })
            }
            WriteTarget::Operator(operator, params) => {
                let session = session_of(py, self.session.as_ref())?;
                interruptible(py, &session, || {
                    spark_connect::ml::write_operator(
                        &session,
                        operator,
                        params,
                        path,
                        self.shouldOverwrite,
                        &self.options,
                    )
                    .map_err(Box::new)
                })
            }
        }
    }

    /// Overwrite a saved one of the path, as `MLWriter.overwrite`.
    fn overwrite(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.shouldOverwrite = true;
        slf
    }

    /// An option of the writer, as `MLWriter.option`, of its lowercase key and string value.
    fn option<'py>(
        mut slf: PyRefMut<'py, Self>,
        key: &str,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let value = value.str()?.to_string();
        slf.options.insert(key.to_lowercase(), value);
        Ok(slf)
    }

    /// The options, as `MLWriter.optionMap`.
    #[getter]
    #[allow(non_snake_case)]
    fn optionMap<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (k, v) in &self.options {
            dict.set_item(k, v)?;
        }
        Ok(dict)
    }

    /// The session to save with, as `BaseReadWrite.session`.
    fn session<'py>(mut slf: PyRefMut<'py, Self>, spark: Bound<'py, PyAny>) -> PyRefMut<'py, Self> {
        slf.session = Some(spark.unbind());
        slf
    }

    /// The session to save with, as `BaseReadWrite.sparkSession`: the one that `session` set, else
    /// the active one.
    #[getter]
    #[allow(non_snake_case)]
    fn sparkSession(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.session {
            Some(session) => Ok(session.clone_ref(py)),
            None => Ok(py
                .import("pyspark.sql")?
                .getattr("SparkSession")?
                .call_method0("getActiveSession")?
                .unbind()),
        }
    }

    /// As `RemoteMLWriter.sc`.
    #[getter]
    fn sc(&self) -> PyResult<()> {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Accessing SparkContext is not supported on Connect",
        ))
    }
}

/// The constructor of a native class of what a `Read` returns: its object (of a model), uid and
/// params.
pub(super) type FromRead =
    for<'py> fn(Python<'py>, spark_connect::ml::ReadResult) -> PyResult<Py<PyAny>>;

/// PySpark's `MLReader` of a native class (`RemoteMLReader` over Spark Connect): `load(path)`
/// sends the `Read` of the JVM class, and `session(spark)` sets the session to load with.
#[pyclass(name = "MLReader", module = "pyspark.ml.util")]
pub struct PyMLReader {
    jvm_class: &'static str,
    op_type: OperatorType,
    from_read: FromRead,
    session: Option<Py<PyAny>>,
}

impl PyMLReader {
    pub(super) fn new(jvm_class: &'static str, op_type: OperatorType, from_read: FromRead) -> Self {
        Self {
            jvm_class,
            op_type,
            from_read,
            session: None,
        }
    }
}

#[pymethods]
impl PyMLReader {
    /// Load from `path`, as `RemoteMLReader.load`, of the active session.
    fn load(&self, py: Python<'_>, path: &str) -> PyResult<Py<PyAny>> {
        let session = session_of(py, self.session.as_ref())?;
        let read = interruptible(py, &session, || {
            spark_connect::ml::read(&session, self.jvm_class, self.op_type, path).map_err(Box::new)
        })?;
        (self.from_read)(py, read)
    }

    /// The session to load with, as `BaseReadWrite.session`.
    fn session<'py>(mut slf: PyRefMut<'py, Self>, spark: Bound<'py, PyAny>) -> PyRefMut<'py, Self> {
        slf.session = Some(spark.unbind());
        slf
    }

    /// As `BaseReadWrite.sparkSession`.
    #[getter]
    #[allow(non_snake_case)]
    fn sparkSession(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.session {
            Some(session) => Ok(session.clone_ref(py)),
            None => Ok(py
                .import("pyspark.sql")?
                .getattr("SparkSession")?
                .call_method0("getActiveSession")?
                .unbind()),
        }
    }

    /// As `RemoteMLReader.sc` over Spark Connect.
    #[getter]
    fn sc(&self) -> PyResult<()> {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Accessing SparkContext is not supported on Connect",
        ))
    }
}

/// The Python string of a param name, for the errors.
#[allow(dead_code)]
fn py_str<'py>(py: Python<'py>, text: &str) -> Bound<'py, PyString> {
    PyString::new(py, text)
}

/// The summary dataset of the fit of a model, as its PySpark class's `_summary_dataset` gives it.
#[derive(Clone, Copy)]
#[allow(dead_code)] // The areas' models use each.
pub(crate) enum SummaryDataset {
    /// The model has no training summary.
    None,
    /// The training dataset itself (the classification models, LinearRegression and
    /// GeneralizedLinearRegression).
    Train,
    /// The transform of the training dataset by the model, of no params (the clustering models).
    Transformed,
}

/// The state of a native model: its core model, the session of its fit, the training summary of
/// its fit (PySpark's `_summary`, which a remote fit sets and a load does not), and the defaults
/// that it holds itself (see [`params_api::OwnDefaults`]: of its estimator, or of the uid that a
/// loaded model was made of).
pub(super) struct ModelCore {
    pub(super) model: Box<dyn Model>,
    pub(super) session_id: String,
    pub(super) summary: Option<MlTarget>,
    /// The object of the training summary, PySpark's `_summary`, which the fit makes (see
    /// [`ModelCore::summary_object`]).
    summary_object: std::sync::OnceLock<Py<PyAny>>,
    pub(super) own_defaults: params_api::OwnDefaults,
    /// The values of the `functools.cached_property` members, by name (see [`ModelCore::cached`]).
    cache: std::sync::Mutex<Vec<(&'static str, Py<PyAny>)>>,
}

/// The maker of the training summary object of a model, of the model and of the target of its
/// summary (`model._summaryCls(...)` of `try_remote_fit`, which may fetch, as of `numClasses`).
pub(super) type SummaryMaker = fn(Python<'_>, &ModelCore, MlTarget) -> PyResult<Py<PyAny>>;

/// The maker of the value of a `functools.cached_property` member of a model (`trees`).
#[allow(dead_code)] // The tree ensembles' `trees` use it.
pub(super) type CachedMaker = fn(Python<'_>, &ModelCore) -> PyResult<Py<PyAny>>;

/// The Python value of the Python literal `text`, the default of an argument of a method of a
/// model or a summary (see the macros).
pub(crate) fn py_default(text: &str) -> Py<PyAny> {
    Python::attach(|py| {
        let code = std::ffi::CString::new(text).expect("a Python literal has no NUL");
        py.eval(&code, None, None)
            .expect("the default of an argument is a Python literal")
            .unbind()
    })
}

impl ModelCore {
    /// The model of a fit of `df`, whose training summary, if the class has one, is of the summary
    /// dataset of `summary`, as PySpark's `try_remote_fit` sets `_summary` (see
    /// `spark_connect::ml::training_summary`).
    pub(super) fn fitted(
        py: Python<'_>,
        model: Box<dyn Model>,
        df: &DataFrame,
        summary: SummaryDataset,
        make_summary: Option<SummaryMaker>,
    ) -> PyResult<Self> {
        let session_id = df.spark_session().session_id().to_string();
        let summary = match (summary, model.ml_ref()) {
            (SummaryDataset::None, _) | (_, None) => None,
            (SummaryDataset::Train, Some(object)) => {
                Some(spark_connect::ml::training_summary(object, df))
            }
            (SummaryDataset::Transformed, Some(object)) => {
                let dataset = py
                    .detach(|| spark_connect::ml::summary_transform(object, df).map_err(Box::new))
                    .to_pyerr()?;
                Some(spark_connect::ml::training_summary(object, &dataset))
            }
        };
        let core = Self {
            model,
            session_id,
            summary,
            summary_object: std::sync::OnceLock::new(),
            own_defaults: Vec::new(),
            cache: Default::default(),
        };
        // The summary object, of the fit, as `try_remote_fit` sets `_summary` right after it.
        if let (Some(make), Some(target)) = (make_summary, core.summary.clone()) {
            let object = make(py, &core, target)?;
            let _ = core.summary_object.set(object);
        }
        Ok(core)
    }

    /// The model of the server-side object `object` that a fetch returns, of the model's JVM class
    /// `jvm_class`: of a new uid of the class `class` (`Identifiable._randomUID`), without set
    /// params, a training summary, own defaults or cached members, as PySpark makes a model of a
    /// bare object (the trees of an ensemble, the models of a OneVsRestModel).
    pub(super) fn of_object(
        py: Python<'_>,
        jvm_class: &str,
        class: &str,
        object: spark_connect::ml::MlRef,
    ) -> PyResult<Self> {
        let session_id = object.session().session_id().to_string();
        let uid = params_api::random_uid(py, class)?;
        let operator = MlOperator::with_uid(jvm_class, &uid, OperatorType::Model);
        Ok(Self {
            model: Box::new(spark_connect::ml::ObjectModel::new(
                object,
                operator,
                Params::new(),
            )),
            session_id,
            summary: None,
            summary_object: std::sync::OnceLock::new(),
            own_defaults: Vec::new(),
            cache: Default::default(),
        })
    }

    /// The model that a `Read` loads: of its object, the model's JVM class `jvm_class`, its uid and
    /// params, without a training summary, as PySpark's loaded models have none, and of the
    /// defaults that a model of the class `class` derives from the uid that it is made of (see
    /// [`params_api::uid_defaults`]).
    pub(super) fn loaded(
        py: Python<'_>,
        jvm_class: &str,
        class: &str,
        read: spark_connect::ml::ReadResult,
    ) -> PyResult<Self> {
        let object = read.object.ok_or_else(|| {
            PyValueError::new_err(format!("the server loaded no object of {jvm_class}"))
        })?;
        let session_id = object.session().session_id().to_string();
        let operator = MlOperator::with_uid(jvm_class, &read.uid, OperatorType::Model);
        Ok(Self {
            model: Box::new(spark_connect::ml::ObjectModel::new(
                object,
                operator,
                read.params,
            )),
            session_id,
            summary: None,
            summary_object: std::sync::OnceLock::new(),
            own_defaults: params_api::uid_defaults(py, class)?,
            cache: Default::default(),
        })
    }

    pub(super) fn duplicate(&self, py: Python<'_>) -> Self {
        Self {
            model: self.model.clone_box(),
            session_id: self.session_id.clone(),
            summary: self.summary.clone(),
            // A copy shares the summary object, as `copy.copy` of PySpark's `Params.copy` shares
            // its `_summary`.
            summary_object: match self.summary_object.get() {
                Some(object) => std::sync::OnceLock::from(object.clone_ref(py)),
                None => std::sync::OnceLock::new(),
            },
            own_defaults: params_api::clone_own_defaults(py, &self.own_defaults),
            // A copy shares the values cached before it, as of the `__dict__` that `copy.copy`
            // copies.
            cache: std::sync::Mutex::new(params_api::clone_own_defaults(
                py,
                &self.cache.lock().expect("the cache is not poisoned"),
            )),
        }
    }

    /// The value of the cached member `key`, as a `functools.cached_property`: made by `make`
    /// once, and the same object after.
    #[allow(dead_code)] // The tree ensembles' `trees` use it.
    pub(super) fn cached(
        &self,
        py: Python<'_>,
        key: &'static str,
        make: impl FnOnce() -> PyResult<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let lock = || self.cache.lock().expect("the cache is not poisoned");
        if let Some(value) = params_api::own_default_of(py, &lock(), key) {
            return Ok(value);
        }
        // Made without the lock, as `make` runs Python.
        let value = make()?;
        let mut cache = lock();
        if let Some(existing) = params_api::own_default_of(py, &cache, key) {
            return Ok(existing);
        }
        cache.push((key, value.clone_ref(py)));
        Ok(value)
    }

    /// The target of the training summary of the model `slf`, or the RuntimeError of
    /// `HasTrainingSummary.summary` over Spark Connect of a model without one (a loaded one), of
    /// the name of the class of `slf`.
    pub(super) fn training_summary_target(&self, slf: &Bound<'_, PyAny>) -> PyResult<MlTarget> {
        match &self.summary {
            Some(target) => Ok(target.clone()),
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "No training summary available for this {}",
                slf.get_type().name()?
            ))),
        }
    }

    /// The training summary of the model `slf`, as `HasTrainingSummary.summary` over Spark Connect
    /// returns `_summary`: the object that the fit made, the same on every call (of `make`, if the
    /// class makes none at the fit), or PySpark's RuntimeError of a model without one (see
    /// [`ModelCore::training_summary_target`]).
    pub(super) fn summary_object(
        &self,
        slf: &Bound<'_, PyAny>,
        make: SummaryMaker,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if let Some(object) = self.summary_object.get() {
            return Ok(object.clone_ref(py));
        }
        let object = make(py, self, self.training_summary_target(slf)?)?;
        Ok(self.summary_object.get_or_init(|| object).clone_ref(py))
    }

    /// The target of the methods of the model's object.
    pub(super) fn target(&self) -> PyResult<MlTarget> {
        self.model.model_target().to_pyerr()
    }

    pub(super) fn uid(&self) -> String {
        self.model.operator().uid.clone()
    }

    /// The error of PySpark's serialization of the model's params (an int that it cannot send).
    pub(super) fn serialization_error(&self, py: Python<'_>) -> Option<PyErr> {
        let unsendable = self
            .model
            .params()
            .to_proto()
            .params
            .keys()
            .any(|name| name.starts_with("\u{0}unsendable:"));
        unsendable.then(|| crate::functions::value_not_between(py))
    }

    /// The transform of `df` by the model, of `df` as it is, as `try_remote_transform_relation` of
    /// `pyspark.ml` sends it (see `PyMLModel::transform_dataframe`).
    pub(super) fn transform(&self, py: Python<'_>, df: &DataFrame) -> PyResult<DataFrame> {
        if let Some(error) = self.serialization_error(py) {
            return Err(error);
        }
        let mut model = self.model.clone_box();
        let session = df.spark_session();
        if session.session_id() != self.session_id {
            let placeholder =
                DataFrame::from_sent_relation(session.clone(), df.placeholder_relation());
            let relation = spark_connect_proto::Relation {
                rel_type: Some(spark_connect_proto::relation::RelType::MlRelation(
                    Box::new(model.build_ml_relation(&placeholder).to_pyerr()?),
                )),
                ..Default::default()
            };
            DataFrame::from_relation(session, relation, None).to_pyerr()
        } else {
            py.detach(|| model.transform(df).map_err(Box::new))
                .to_pyerr()
        }
    }

    /// The writer of the model's object, of its params, as `RemoteMLWriter` writes a model.
    pub(super) fn writer(&self, py: Python<'_>) -> PyResult<PyMLWriter> {
        let object = self
            .model
            .ml_ref()
            .ok_or_else(|| PyValueError::new_err("the model has no server-side object to save"))?;
        Ok(PyMLWriter::new(WriteTarget::Object(
            object.clone(),
            self.model.params().clone(),
        ))
        .with_error(self.serialization_error(py)))
    }
}

/// Set the param `jvm` of the model params `params` to `value`, as the setters of a model set it:
/// a value of the converter `identity` as the literal of `LiteralExpression._from_value`, and an
/// int that PySpark cannot send under the reserved key that keeps it (see
/// [`params_api::unsendable_key`]).
pub(super) fn put_model_param(
    py: Python<'_>,
    params: Params,
    jvm: &str,
    value: Converted,
) -> PyResult<Params> {
    let params = params.clear_param(&params_api::unsendable_key(jvm));
    Ok(match value {
        Converted::Null => params.set_param_null(jvm),
        Converted::Unsendable(text) => params
            .clear_param(jvm)
            .set_param_string(&params_api::unsendable_key(jvm), &text),
        Converted::Int(value) => params.set_param_int_narrow(jvm, value),
        Converted::Float(value) => params.set_param_double(jvm, value),
        Converted::Str(value) => params.set_param_string(jvm, &value),
        Converted::Bool(value) => params.set_param_bool(jvm, value),
        Converted::StrList(values) => params.set_param_string_array(jvm, values),
        Converted::FloatList(values) => params.set_param_double_array(jvm, values),
        Converted::IntList(values) => params.set_param_int_array(jvm, values),
        Converted::Vector(value) => params.set_param_vector(jvm, value),
        Converted::FloatListList(value) => params.set_param_double_arrays(jvm, value),
        Converted::Matrix(value) => params.set_param_matrix(jvm, value),
        Converted::Raw(value) => {
            let column = crate::functions::to_value_column(value.bind(py))?;
            match column.expression().node() {
                spark_connect::expression::ExpressionNode::Literal(literal) => {
                    params.set_param_literal(jvm, literal.to_literal_proto())
                }
                _ => {
                    return Err(crate::column::raise_pyspark(
                        py,
                        "PySparkTypeError",
                        "UNSUPPORTED_DATA_TYPE",
                        &[("data_type", &value.bind(py).get_type().name()?.to_string())],
                    ))
                }
            }
        }
    })
}
