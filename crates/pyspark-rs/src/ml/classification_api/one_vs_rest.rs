// SPDX-License-Identifier: Apache-2.0
//! The typed model of OneVsRest: OneVsRestModel, a container of the models of the classes of a fit,
//! with its classifier and its other params as Python values (as PySpark's `PipelineModel`).
use super::*;
use crate::errors::ResultExt;
use crate::ml::pipeline_api::{load_metadata_dict, load_params_instance, metadata_str, writable};
use spark_connect::ml::{
    join_path, save_metadata, Metadata, OneVsRestModel as CoreOneVsRestModel, SubModel,
};
use spark_connect::SparkSession;
use std::sync::{Mutex, PoisonError};

/// The fit of a OneVsRest, of what its model's submodels are made of: the dataset of the fit's
/// columns, and the binary training set of each class (of its binary label, PySpark's
/// `mc2b$<class>`), which the classifier of the class was fitted on, of which the class's model's
/// training summary is.
pub(crate) struct OneVsRestFit {
    pub(crate) train: DataFrame,
    pub(crate) datasets: Vec<DataFrame>,
}

/// OneVsRestModel: the models of the classes of a fit, with the classifier and the other params
/// that PySpark's OneVsRestModel holds. Its `transform` is PySpark's, which composes the transforms
/// of its `models` (`OneVsRestModel._transform`), and its `models` are the typed submodels of the
/// fit, as PySpark's `OneVsRest._fit` makes them, or the list that a constructor or an assignment
/// gives.
#[pyclass(
    subclass,
    name = "OneVsRestModel",
    module = "pyspark.ml.connect.classification"
)]
pub struct PyOneVsRestModel {
    core: object::ModelCore,
    /// The Python values of the identity params: the classifier.
    raw: Vec<(&'static str, Py<PyAny>)>,
    /// The `models` attribute, which PySpark keeps as a plain attribute: the object that the fit,
    /// the load, the copy, the constructor or an assignment gives.
    models: Mutex<Option<Py<PyAny>>>,
    /// The order of the defaults of PySpark's `_defaultParamMap` of the model: of its
    /// constructor (`MODEL_DEFAULTS`) of a fit and of `OneVsRestModel(models)`, and of the names of
    /// the params (none) of a load, as its `_resetUid` makes them.
    default_order: &'static [&'static str],
    /// The order of the set params of a load (the classifier, then the params of the metadata in
    /// its order); none of a fit, whose `_copyValues` sets them in the order of their names.
    set_order: params_api::SetOrder,
}

impl PyOneVsRestModel {
    /// The model of a fit of `model`, the core's composite of the classes, with the classifier
    /// `classifier`. As PySpark's model, it has a uid of its own.
    pub(crate) fn fitted(
        py: Python<'_>,
        model: Box<dyn Model>,
        fit: OneVsRestFit,
        classifier: Py<PyAny>,
    ) -> PyResult<Self> {
        let mut core =
            object::ModelCore::fitted(py, model, &fit.train, object::SummaryDataset::None, None)?;
        core.model.operator_mut().uid = params_api::random_uid(py, "OneVsRestModel")?;
        let this = Self {
            core,
            raw: vec![("classifier", classifier)],
            models: Mutex::new(None),
            default_order: MODEL_DEFAULTS,
            set_order: params_api::SetOrder::default(),
        };
        // As PySpark's classifier fits make their models, with their summaries, during the fit.
        // The model keeps no dataset of the fit but those of its core's models of the classes
        // (see `ModelCore`), which its traverse reports, as PySpark's model holds its datasets
        // through the summaries of its `models` alone.
        let models = build_submodels(py, &this, &fit)?;
        this.hold_models(models.into_any());
        Ok(this)
    }

    /// The classifier, if the param is set.
    fn classifier(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.raw
            .iter()
            .find(|(name, _)| *name == "classifier")
            .map(|(_, value)| value.clone_ref(py))
    }

    /// The `models` attribute, if one is held.
    fn models_value(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.models
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
            .map(|models| models.clone_ref(py))
    }

    /// Hold `models` as the `models` attribute.
    fn hold_models(&self, models: Py<PyAny>) {
        *self.models.lock().unwrap_or_else(PoisonError::into_inner) = Some(models);
    }
}

/// The defaults of the classifier `classifier`, which PySpark's submodels have as the classifier's
/// copy has them (see `params_api::fit_defaults`).
fn classifier_defaults(classifier: &Bound<'_, PyAny>) -> PyResult<params_api::OwnDefaults> {
    if let Ok(value) = classifier.cast::<crate::ml::PyLinearSVC>() {
        return params_api::fit_defaults(value);
    }
    if let Ok(value) = classifier.cast::<crate::ml::PyNaiveBayes>() {
        return params_api::fit_defaults(value);
    }
    if let Ok(value) = classifier.cast::<crate::ml::PyFMClassifier>() {
        return params_api::fit_defaults(value);
    }
    if let Ok(value) = classifier.cast::<crate::ml::PyMultilayerPerceptronClassifier>() {
        return params_api::fit_defaults(value);
    }
    if let Ok(value) = classifier.cast::<crate::ml::PyLogisticRegression>() {
        return params_api::fit_defaults(value);
    }
    Ok(Vec::new())
}

/// A generic model of a submodel of a classifier without a typed class.
fn generic_submodel(
    py: Python<'_>,
    model: Box<dyn Model>,
    binary: &DataFrame,
) -> PyResult<Py<PyAny>> {
    let session_id = binary.spark_session().session_id().to_string();
    let model = crate::ml::PyMLModel::new(py, model, session_id);
    Ok(crate::ml::object::new_py(py, model)?.into_any())
}

/// The typed submodel of the class `index`, as PySpark's `OneVsRest._fit` has it: the model of the
/// classifier's copy with the binary label of the class, of the classifier's uid, params and
/// defaults, and with the binary training set as its summary dataset.
fn submodel(
    py: Python<'_>,
    index: usize,
    sub: SubModel,
    fit: &OneVsRestFit,
    defaults: &params_api::OwnDefaults,
) -> PyResult<Py<PyAny>> {
    let Some(binary) = fit.datasets.get(index) else {
        return Err(PyValueError::new_err(format!(
            "the fit of the OneVsRest has no binary training set of the class {index}"
        )));
    };
    let label = format!("mc2b${index}");
    // PySpark's submodel has the binary label of the class as its labelCol param.
    let with_label = |mut model: Box<dyn Model>| -> Box<dyn Model> {
        let params = std::mem::take(model.params_mut()).set_param_string("labelCol", &label);
        *model.params_mut() = params;
        model
    };
    match sub {
        SubModel::LinearSVC(model) => {
            let mut typed = PyLinearSVCModel::fitted(
                py,
                with_label(Box::new(model)),
                binary,
                object::SummaryDataset::Train,
            )?;
            params_api::copy_defaults(
                py,
                params_api::clone_own_defaults(py, defaults),
                &mut typed,
            )?;
            Ok(crate::ml::object::new_py(py, typed)?.into_any())
        }
        SubModel::NaiveBayes(model) => {
            let mut typed = PyNaiveBayesModel::fitted(
                py,
                with_label(Box::new(model)),
                binary,
                object::SummaryDataset::None,
            )?;
            params_api::copy_defaults(
                py,
                params_api::clone_own_defaults(py, defaults),
                &mut typed,
            )?;
            Ok(crate::ml::object::new_py(py, typed)?.into_any())
        }
        SubModel::FMClassification(model) => {
            let mut typed = PyFMClassificationModel::fitted(
                py,
                with_label(Box::new(model)),
                binary,
                object::SummaryDataset::Train,
            )?;
            params_api::copy_defaults(
                py,
                params_api::clone_own_defaults(py, defaults),
                &mut typed,
            )?;
            Ok(crate::ml::object::new_py(py, typed)?.into_any())
        }
        SubModel::MultilayerPerceptronClassification(model) => {
            let mut typed = PyMultilayerPerceptronClassificationModel::fitted(
                py,
                with_label(Box::new(model)),
                binary,
                object::SummaryDataset::Train,
            )?;
            params_api::copy_defaults(
                py,
                params_api::clone_own_defaults(py, defaults),
                &mut typed,
            )?;
            Ok(crate::ml::object::new_py(py, typed)?.into_any())
        }
        SubModel::LogisticRegression(model) => {
            let mut typed = PyLogisticRegressionModel::fitted(
                py,
                with_label(Box::new(model)),
                binary,
                object::SummaryDataset::Train,
            )?;
            params_api::copy_defaults(
                py,
                params_api::clone_own_defaults(py, defaults),
                &mut typed,
            )?;
            Ok(crate::ml::object::new_py(py, typed)?.into_any())
        }
        SubModel::DecisionTreeClassification(model) => {
            let mut typed = PyDecisionTreeClassificationModel::fitted(
                py,
                with_label(Box::new(model)),
                binary,
                object::SummaryDataset::None,
            )?;
            params_api::copy_defaults(
                py,
                params_api::clone_own_defaults(py, defaults),
                &mut typed,
            )?;
            Ok(crate::ml::object::new_py(py, typed)?.into_any())
        }
        SubModel::RandomForestClassification(model) => {
            let mut typed = PyRandomForestClassificationModel::fitted(
                py,
                with_label(Box::new(model)),
                binary,
                object::SummaryDataset::Train,
            )?;
            params_api::copy_defaults(
                py,
                params_api::clone_own_defaults(py, defaults),
                &mut typed,
            )?;
            Ok(crate::ml::object::new_py(py, typed)?.into_any())
        }
        SubModel::GBTClassification(model) => {
            let mut typed = PyGBTClassificationModel::fitted(
                py,
                with_label(Box::new(model)),
                binary,
                object::SummaryDataset::None,
            )?;
            params_api::copy_defaults(
                py,
                params_api::clone_own_defaults(py, defaults),
                &mut typed,
            )?;
            Ok(crate::ml::object::new_py(py, typed)?.into_any())
        }
        SubModel::Other(model) => generic_submodel(py, with_label(model), binary),
    }
}

/// The submodels of the model `this` of the fit `fit`, typed as PySpark's classifier fits make
/// them, of the composite's submodels, in the order of the classes. They are made by the fit, with
/// the summaries that the classes' fits make.
fn build_submodels(
    py: Python<'_>,
    this: &PyOneVsRestModel,
    fit: &OneVsRestFit,
) -> PyResult<Py<PyList>> {
    let Some(composite) = this.core.model.downcast_ref::<CoreOneVsRestModel>() else {
        return Err(PyValueError::new_err(
            "the model of a OneVsRestModel is not a composite of its classes",
        ));
    };
    let defaults = match this.classifier(py) {
        Some(classifier) => classifier_defaults(classifier.bind(py))?,
        None => Vec::new(),
    };
    let list = PyList::empty(py);
    for (index, sub) in composite.models().into_iter().enumerate() {
        list.append(submodel(py, index, sub, fit, &defaults)?)?;
    }
    Ok(list.unbind())
}

/// The `models` of `slf`, the object that the fit, the load, the copy, the constructor or an
/// assignment gave it.
fn submodels(slf: &Bound<'_, PyOneVsRestModel>) -> PyResult<Py<PyAny>> {
    let this = slf.try_borrow()?;
    match this.models_value(slf.py()) {
        Some(models) => Ok(models),
        None => Err(PyValueError::new_err(
            "the models of a OneVsRestModel are not held",
        )),
    }
}

impl params_api::ParamsHolder for PyOneVsRestModel {
    fn set_value(&self, py: Python<'_>, name: &str) -> PyResult<Option<Py<PyAny>>> {
        if let Some((_, value)) = self.raw.iter().find(|(n, _)| *n == name) {
            return Ok(Some(value.clone_ref(py)));
        }
        params_api::literal_value(py, self.core.model.params(), name)
    }

    fn put(&mut self, py: Python<'_>, name: &'static str, value: Converted) -> PyResult<()> {
        self.raw.retain(|(n, _)| *n != name);
        let params = self.core.model.params().clone();
        *self.core.model.params_mut() = match value {
            Converted::Raw(value) => {
                self.raw.push((name, value));
                params.clear_param(name)
            }
            value => object::put_model_param(py, params, name, value)?,
        };
        self.set_order.put(name);
        Ok(())
    }

    fn own_default(&self, py: Python<'_>, name: &str) -> Option<Py<PyAny>> {
        params_api::own_default_of(py, &self.core.own_defaults, name)
    }

    fn put_own_default(&mut self, name: &'static str, value: Py<PyAny>) -> bool {
        params_api::put_own_default_in(&mut self.core.own_defaults, name, value);
        true
    }

    fn unset(&mut self, name: &'static str) -> PyResult<()> {
        self.raw.retain(|(n, _)| *n != name);
        let params = self.core.model.params().clone().clear_param(name);
        *self.core.model.params_mut() = params;
        self.set_order.unset(name);
        Ok(())
    }

    /// The defaults in the order of the constructor (of a fit and of `OneVsRestModel(models)`) or
    /// of the names (of a load), and the set params in the order of their setting.
    fn map_order(&self) -> Option<(Vec<&'static str>, Vec<&'static str>)> {
        Some((self.default_order.to_vec(), self.set_order.names()))
    }
}

impl params_api::NativeInstance for PyOneVsRestModel {
    fn state_attributes(&self) -> &'static [&'static str] {
        &["models"]
    }

    fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        let raw = self
            .raw
            .iter()
            .map(|(name, value)| (*name, value.clone_ref(py)))
            .collect();
        crate::ml::object::new_py(
            py,
            Self {
                core: self.core.duplicate(py),
                raw,
                models: Mutex::new(self.models_value(py)),
                default_order: self.default_order,
                set_order: self.set_order.clone(),
            },
        )
    }

    fn uid(&self) -> String {
        self.core.uid()
    }

    fn set_uid(&mut self, uid: &str) {
        self.core.model.operator_mut().uid = uid.to_string();
    }
}

native_pymethods! {
    PyOneVsRestModel;
    persistence: python;
    params: [
        "classifier" => ("getClassifier", get_classifier, param_classifier),
        "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
        "labelCol" => ("getLabelCol", get_label_col, param_label_col),
        "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
        "rawPredictionCol" => (
            "getRawPredictionCol",
            get_raw_prediction_col,
            param_raw_prediction_col
        ),
        "weightCol" => ("getWeightCol", get_weight_col, param_weight_col),
    ];

    /// The `models` attribute, as PySpark's: the typed models of the classifier, each of the uid of
    /// the classifier, or the object that was assigned to it.
    #[getter]
    fn models(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        submodels(slf)
    }

    #[setter]
    fn set_models(slf: &Bound<'_, Self>, value: Py<PyAny>) -> PyResult<()> {
        slf.try_borrow()?.hold_models(value);
        Ok(())
    }

    /// Visit the Python objects that the model holds, of Python's garbage collector, as PySpark's
    /// model holds its attributes and its param maps: those of its core (see
    /// `ModelCore::traverse`, of the binary training sets of the core's models of the classes), its
    /// classifier and its `models`.
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        if !self.core.gc_ready.get() {
            return Ok(());
        }
        self.core.traverse(&visit)?;
        params_api::visit_values(&self.raw, &visit)?;
        if let Ok(models) = self.models.try_lock() {
            if let Some(models) = models.as_ref() {
                visit.call(models)?;
            }
        }
        Ok(())
    }

    /// Break a reference cycle that the garbage collector found unreachable: the model drops the
    /// Python objects that `__traverse__` reported.
    fn __clear__(&mut self) {
        if !self.core.gc_ready.get() {
            return;
        }
        self.core.clear();
        self.raw.clear();
        if let Ok(models) = self.models.get_mut() {
            *models = None;
        }
    }

    /// A model of the given `models`, as PySpark's `OneVsRestModel(models)`: of a new uid and no
    /// set params, with the defaults of the class, and of no classifier. The uid is of the class
    /// of the object, as `Identifiable._randomUID` names it (`MyOneVsRestModel_...` of a Python
    /// subclass).
    #[new]
    #[classmethod]
    #[pyo3(signature = (models=None))]
    fn new(cls: &Bound<'_, pyo3::types::PyType>, models: Option<Py<PyAny>>) -> PyResult<Self> {
        let models = models.unwrap_or_else(|| cls.py().None());
        let core = object::ModelCore::unbound_named(
            cls.py(),
            "org.apache.spark.ml.classification.OneVsRestModel",
            "OneVsRestModel",
            &cls.name()?.to_string(),
        )?;
        Ok(Self {
            core,
            raw: Vec::new(),
            models: Mutex::new(Some(models)),
            default_order: MODEL_DEFAULTS,
            set_order: params_api::SetOrder::default(),
        })
    }

    /// A copy of the same uid and params, with `extra`, as `OneVsRestModel.copy`: `Params.copy`,
    /// whose `models` are copies of the models, each with `extra`
    /// (`[model.copy(extra) for model in self.models]`).
    #[pyo3(name = "copy", signature = (extra=None))]
    fn __params_copy<'py>(
        slf: &Bound<'py, Self>,
        extra: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let extra = crate::ml::pipeline_api::extra_or_empty(py, extra);
        let that = params_api::copy(slf, Some(&extra))?;
        let models = PyList::empty(py);
        for model in submodels(slf)?.bind(py).try_iter()? {
            models.append(model?.call_method1("copy", (&extra,))?)?;
        }
        that.try_borrow()?.hold_models(models.into_any().unbind());
        Ok(that)
    }

    /// The writer of the model, as `OneVsRestModel.write` over Spark Connect (`RemoteMLWriter`,
    /// which saves it as `OneVsRestModelWriter` does).
    fn write(slf: &Bound<'_, Self>) -> PyResult<Py<object::PyMLWriter>> {
        object::PyMLWriter::new(object::WriteTarget::Python(save_one_vs_rest_model))
            .object_of(slf.as_any())
    }

    /// Save to `path`, as `MLWritable.save`: `self.write().save(path)`.
    fn save(slf: &Bound<'_, Self>, path: &Bound<'_, PyAny>) -> PyResult<()> {
        slf.call_method0("write")?.call_method1("save", (path,))?;
        Ok(())
    }

    /// The reader of the class, as `OneVsRestModel.read` over Spark Connect (`RemoteMLReader`,
    /// which loads it as `OneVsRestModelReader` does).
    #[classmethod]
    fn read(cls: &Bound<'_, pyo3::types::PyType>) -> PyResult<Py<object::PyMLReader>> {
        let reader = object::PyMLReader::python(load_one_vs_rest_model);
        reader.object_of(cls.as_any())
    }

    /// Load from `path`, as `MLReadable.load`: `cls.read().load(path)`.
    #[classmethod]
    fn load(
        cls: &Bound<'_, pyo3::types::PyType>,
        path: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        Ok(cls.call_method0("read")?.call_method1("load", (path,))?.unbind())
    }

    #[pyo3(name = "setFeaturesCol")]
    fn set_features_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        params_api::set_by_name(&slf, "featuresCol", value)?;
        Ok(slf)
    }

    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        params_api::set_by_name(&slf, "predictionCol", value)?;
        Ok(slf)
    }

    #[pyo3(name = "setRawPredictionCol")]
    fn set_raw_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        params_api::set_by_name(&slf, "rawPredictionCol", value)?;
        Ok(slf)
    }
}

/// Register the typed classes of the family.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::ml::object::register_class::<PyOneVsRestModel>(m)?;
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Persistence, as `OneVsRestWriter`, `OneVsRestModelWriter` and their readers of PySpark (the
// Python format of `_OneVsRestSharedReadWrite`, which the core writes and reads too).
// ---------------------------------------------------------------------------------------------

/// The classes of the metadata, PySpark's, of which PySpark loads what this saves.
const ONE_VS_REST_CLASS: &str = "pyspark.ml.classification.OneVsRest";
const ONE_VS_REST_MODEL_CLASS: &str = "pyspark.ml.classification.OneVsRestModel";

/// The params of a OneVsRestModel that have a default, in the order in which PySpark's constructor
/// of the model sets them in its `_defaultParamMap` (see the core's `MODEL_DEFAULTS`).
const MODEL_DEFAULTS: &[&str] = &[
    "predictionCol",
    "featuresCol",
    "labelCol",
    "rawPredictionCol",
];

/// The `paramMap` and the `defaultParamMap` of the metadata of a OneVsRest or its model
/// `instance`, as `DefaultParamsWriter` makes them of PySpark's maps of it, in their order: of
/// `extractParamMap()` (the defaults, of their set values, then the other set params) but the
/// classifier, which its own writer saves (`extractJsonParams`), and of `_defaultParamMap`. So a
/// OneVsRest of its constructor writes its params in the order of the constructor's defaults, and
/// one of a load or a copy in the order of their names.
fn param_maps<'py>(
    instance: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    let py = instance.py();
    let param_map = PyDict::new(py);
    for (param, value) in instance
        .call_method0("extractParamMap")?
        .cast_into::<PyDict>()?
        .iter()
    {
        let name = param.getattr("name")?;
        if name.extract::<String>()? != "classifier" {
            param_map.set_item(name, value)?;
        }
    }
    let default_map = PyDict::new(py);
    for (param, value) in instance
        .getattr("_defaultParamMap")?
        .cast_into::<PyDict>()?
        .iter()
    {
        default_map.set_item(param.getattr("name")?, value)?;
    }
    Ok((param_map, default_map))
}

/// `json.dumps(value, separators=(",", ":"))`, as `DefaultParamsWriter` writes the metadata.
fn dumps_compact(value: &Bound<'_, PyAny>) -> PyResult<String> {
    let py = value.py();
    let kwargs = PyDict::new(py);
    kwargs.set_item("separators", (",", ":"))?;
    py.import("json")?
        .getattr("dumps")?
        .call((value,), Some(&kwargs))?
        .extract()
}

/// The check of the elements of a OneVsRest that its write saves, as `validateParams`: its
/// classifier, and of a model its models, must write.
fn validate_one_vs_rest(instance: &Bound<'_, PyAny>, models: bool) -> PyResult<()> {
    let mut elems = vec![instance.call_method0("getClassifier")?];
    if models {
        for model in instance.getattr("models")?.try_iter()? {
            elems.push(model?);
        }
    }
    for elem in elems {
        if !writable(&elem)? {
            return Err(PyValueError::new_err(format!(
                "OneVsRest write will fail because it contains {} which is not writable.",
                elem.getattr("uid")?.str()?
            )));
        }
    }
    Ok(())
}

/// `obj.write().session(spark).save(path)`, as `MLWritable.save` of the session `spark`.
fn save_at(obj: &Bound<'_, PyAny>, spark: &Bound<'_, PyAny>, path: String) -> PyResult<()> {
    obj.call_method0("write")?
        .call_method1("session", (spark,))?
        .call_method1("save", (path,))?;
    Ok(())
}

/// Save the metadata `metadata` to `path`, as `saveMetadata` does.
fn save_metadata_at(
    py: Python<'_>,
    session: &SparkSession,
    path: &str,
    class: &str,
    uid: &str,
    param_map: String,
    default_map: String,
    num_classes: Option<usize>,
) -> PyResult<()> {
    // Of strings.
    crate::interrupt::interruptible_without_parts(py, session, || {
        let version = session.version()?;
        let mut metadata = Metadata::new(class, uid, &version, param_map, default_map);
        if let Some(num_classes) = num_classes {
            metadata
                .extra
                .push(("numClasses".to_string(), num_classes.to_string()));
        }
        save_metadata(session, path, &metadata.to_json()).map_err(Box::new)
    })
}

/// Save the OneVsRest `instance` to `path`, as `OneVsRestWriter.saveImpl`: its metadata, and its
/// classifier at `classifier`.
pub(crate) fn save_one_vs_rest(
    instance: &Bound<'_, PyAny>,
    spark: &Bound<'_, PyAny>,
    session: &SparkSession,
    path: &str,
    _options: &Bound<'_, PyAny>,
) -> PyResult<()> {
    validate_one_vs_rest(instance, false)?;
    let (param_map, default_map) = param_maps(instance)?;
    let uid: String = instance.getattr("uid")?.extract()?;
    let class = crate::ml::pipeline_api::metadata_class::<crate::ml::PyOneVsRest>(
        instance,
        ONE_VS_REST_CLASS,
    )?;
    save_metadata_at(
        instance.py(),
        session,
        path,
        &class,
        &uid,
        dumps_compact(param_map.as_any())?,
        dumps_compact(default_map.as_any())?,
        None,
    )?;
    save_at(
        &instance.call_method0("getClassifier")?,
        spark,
        join_path(path, "classifier"),
    )
}

/// Save the OneVsRestModel `instance` to `path`, as `OneVsRestModelWriter.saveImpl`: its metadata
/// with `numClasses`, its classifier at `classifier` and each model at `model_<index>`.
fn save_one_vs_rest_model(
    instance: &Bound<'_, PyAny>,
    spark: &Bound<'_, PyAny>,
    session: &SparkSession,
    path: &str,
    _options: &Bound<'_, PyAny>,
) -> PyResult<()> {
    validate_one_vs_rest(instance, true)?;
    let (param_map, default_map) = param_maps(instance)?;
    let uid: String = instance.getattr("uid")?.extract()?;
    let models = instance.getattr("models")?;
    let class = crate::ml::pipeline_api::metadata_class::<PyOneVsRestModel>(
        instance,
        ONE_VS_REST_MODEL_CLASS,
    )?;
    save_metadata_at(
        instance.py(),
        session,
        path,
        &class,
        &uid,
        dumps_compact(param_map.as_any())?,
        dumps_compact(default_map.as_any())?,
        Some(models.len()?),
    )?;
    save_at(
        &instance.call_method0("getClassifier")?,
        spark,
        join_path(path, "classifier"),
    )?;
    for (index, model) in models.try_iter()?.enumerate() {
        save_at(&model?, spark, join_path(path, &format!("model_{index}")))?;
    }
    Ok(())
}

/// The static name of the param `name` of a OneVsRest, as the table of its params has it.
fn static_param_name(name: &str) -> PyResult<&'static str> {
    ONE_VS_REST_PARAMS
        .iter()
        .find(|(param, _)| *param == name)
        .map(|(param, _)| *param)
        .ok_or_else(|| PyValueError::new_err(format!("Cannot find param with name {name}.")))
}

/// Set the params of the `paramMap` of the metadata `metadata` in `target`, as
/// `DefaultParamsReader.getAndSetParams` does, without the classifier. A null of a param is set as
/// it is.
fn set_metadata_params<T: params_api::ParamsHolder>(
    target: &Bound<'_, T>,
    metadata: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let py = target.py();
    let param_map = metadata.get_item("paramMap")?;
    for item in param_map.call_method0("items")?.try_iter()? {
        let item = item?;
        let name = item.get_item(0)?;
        let name_text: String = name.extract()?;
        if name_text == "classifier" {
            continue;
        }
        let value = item.get_item(1)?;
        if value.is_none() {
            let name = static_param_name(&name_text)?;
            params_api::ParamsHolder::put(
                &mut *target.try_borrow_mut()?,
                py,
                name,
                Converted::Null,
            )?;
        } else {
            params_api::set_by_name(target, &name_text, &value)?;
        }
    }
    Ok(())
}

/// Load the OneVsRest at `path`, as `OneVsRestReader.load`: its classifier, its uid and params.
pub(crate) fn load_one_vs_rest(
    py: Python<'_>,
    spark: &Bound<'_, PyAny>,
    session: &SparkSession,
    path: &str,
) -> PyResult<Py<PyAny>> {
    let metadata = load_metadata_dict(py, session, path)?;
    let classifier = load_params_instance(py, spark, session, &join_path(path, "classifier"))?;
    let estimator = crate::ml::object::new_object(
        py,
        PyOneVsRest {
            inner: OneVsRest::new(),
            raw: Vec::new(),
            set_order: params_api::SetOrder::default(),
            gc_ready: crate::gc::GcReady::new(),
        },
    )?;
    params_api::ParamsHolder::put(
        &mut *estimator.try_borrow_mut()?,
        py,
        "classifier",
        Converted::Raw(classifier.unbind()),
    )?;
    set_metadata_params(&estimator, &metadata)?;
    let uid = metadata_str(&metadata.get_item("uid")?)?;
    params_api::NativeInstance::set_uid(&mut *estimator.try_borrow_mut()?, &uid);
    Ok(estimator.into_any().unbind())
}

/// The core model of the loaded model `model`, of one of the native classification models that the
/// classifier of a OneVsRest makes; None of a model of another class.
fn core_submodel(model: &Bound<'_, PyAny>) -> PyResult<Option<Box<dyn Model>>> {
    macro_rules! core_of {
        ($($class:ty),* $(,)?) => {$(
            if let Ok(model) = model.cast::<$class>() {
                return Ok(Some(model.try_borrow()?.core_model()));
            }
        )*};
    }
    core_of!(
        PyLinearSVCModel,
        PyLogisticRegressionModel,
        PyDecisionTreeClassificationModel,
        PyRandomForestClassificationModel,
        PyGBTClassificationModel,
        PyNaiveBayesModel,
        PyFMClassificationModel,
        PyMultilayerPerceptronClassificationModel,
    );
    Ok(None)
}

/// Load the OneVsRestModel at `path`, as `OneVsRestModelReader.load`: its metadata, of any class
/// (PySpark's of the Python format, or the JVM's, as a classic `JavaMLWriter` writes it), its
/// classifier, its models (`numClasses` of them, each loaded of its own format), and then
/// `OneVsRestModel(subModels)._resetUid(metadata["uid"])` of the classifier and the params of the
/// metadata. Each is loaded once, as PySpark loads it: the core's composite of the classes (of
/// which a fit makes the typed models) is of the core models of the models just loaded, when they
/// are of the native classes; a model of models of another class has none, as one that
/// `OneVsRestModel(models)` constructs.
fn load_one_vs_rest_model(
    py: Python<'_>,
    spark: &Bound<'_, PyAny>,
    session: &SparkSession,
    path: &str,
) -> PyResult<Py<PyAny>> {
    let metadata = load_metadata_dict(py, session, path)?;
    let classifier = load_params_instance(py, spark, session, &join_path(path, "classifier"))?;
    let num_classes: usize = metadata.get_item("numClasses")?.extract()?;
    let models = PyList::empty(py);
    for index in 0..num_classes {
        let model_path = join_path(path, &format!("model_{index}"));
        models.append(load_params_instance(py, spark, session, &model_path)?)?;
    }
    let core_models = models
        .iter()
        .map(|model| core_submodel(&model))
        .collect::<PyResult<Option<Vec<_>>>>()?;
    let core = match core_models {
        Some(core_models) => {
            let mut composite = CoreOneVsRestModel::new(session, core_models);
            if let Ok(native) = native_classifier(&classifier, false) {
                composite = composite.set_classifier(native);
            }
            let df = session.range(1).to_pyerr()?;
            object::ModelCore::fitted(
                py,
                Box::new(composite),
                &df,
                object::SummaryDataset::None,
                None,
            )?
        }
        None => object::ModelCore::unbound(
            py,
            "org.apache.spark.ml.classification.OneVsRestModel",
            "OneVsRestModel",
        )?,
    };
    // As `ovaModel.set(ovaModel.classifier, classifier)` after the `_resetUid` of the load, whose
    // defaults are of the names of the params, and before the params of the metadata.
    let model = PyOneVsRestModel {
        core,
        raw: vec![("classifier", classifier.unbind())],
        models: Mutex::new(Some(models.into_any().unbind())),
        default_order: &[],
        set_order: params_api::SetOrder::of(&["classifier"]),
    };
    let model = crate::ml::object::new_object(py, model)?;
    let uid = metadata_str(&metadata.get_item("uid")?)?;
    params_api::NativeInstance::set_uid(&mut *model.try_borrow_mut()?, &uid);
    set_metadata_params(&model, &metadata)?;
    Ok(model.into_any().unbind())
}
