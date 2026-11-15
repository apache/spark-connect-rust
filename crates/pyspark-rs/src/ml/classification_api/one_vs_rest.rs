// SPDX-License-Identifier: Apache-2.0
//! The typed model of OneVsRest: OneVsRestModel, a container of the models of the classes of a fit,
//! with its classifier and its other params as Python values (as PySpark's `PipelineModel`).
use super::*;
use crate::errors::ResultExt;
use crate::interrupt::interruptible;
use crate::ml::pipeline_api::{load_metadata_dict, load_params_instance, metadata_str, writable};
use spark_connect::column::{lit_double, when};
use spark_connect::ml::{
    join_path, save_metadata, Metadata, OneVsRestModel as CoreOneVsRestModel, SubModel,
};
use spark_connect::SparkSession;
use std::sync::{Mutex, PoisonError};

/// The fit of a OneVsRest, of what its model's submodels are made of: the training DataFrame, the
/// prefix of the names of the binary labels that the core fits them with, and the columns of
/// the fit.
#[derive(Clone)]
pub(crate) struct OneVsRestFit {
    pub(crate) train: DataFrame,
    pub(crate) label_prefix: String,
    pub(crate) label: String,
    pub(crate) features: String,
    pub(crate) weight: Option<String>,
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
    /// The fit that the submodels are made of; None of a loaded or constructed model.
    fit: Option<OneVsRestFit>,
    /// The `models` attribute, which PySpark keeps as a plain attribute: the object that the fit,
    /// the load, the copy, the constructor or an assignment gives.
    models: Mutex<Option<Py<PyAny>>>,
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
            fit: Some(fit),
            models: Mutex::new(None),
        };
        // As PySpark's classifier fits make their models, with their summaries, during the fit.
        let models = build_submodels(py, &this)?;
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

/// The column of the name `name` exactly, a name with a dot too.
fn ml_column(name: &str) -> spark_connect::Column {
    spark_connect::column::col(&format!("`{}`", name.replace('`', "``")))
}

/// The binary training set of the class `index`, which PySpark's `OneVsRest._fit` fits the class
/// with: the label, the features and the weight, and the binary label of the class, as the core
/// names it on the server (the summary dataset of the class's model).
fn binary_frame(fit: &OneVsRestFit, index: usize) -> DataFrame {
    let mut columns = vec![ml_column(&fit.label), ml_column(&fit.features)];
    if let Some(weight) = &fit.weight {
        columns.push(ml_column(weight));
    }
    fit.train.select(columns).with_column(
        &format!("{}_label_{}", fit.label_prefix, index),
        when(
            ml_column(&fit.label).eq(lit_double(index as f64)),
            lit_double(1.0),
        )
        .otherwise(lit_double(0.0)),
    )
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
    let model = crate::ml::PyMLModel {
        model,
        session_id: binary.spark_session().session_id().to_string(),
    };
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
    let binary = binary_frame(fit, index);
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
                &binary,
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
                &binary,
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
                &binary,
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
                &binary,
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
                &binary,
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
                &binary,
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
                &binary,
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
                &binary,
                object::SummaryDataset::None,
            )?;
            params_api::copy_defaults(
                py,
                params_api::clone_own_defaults(py, defaults),
                &mut typed,
            )?;
            Ok(crate::ml::object::new_py(py, typed)?.into_any())
        }
        SubModel::Other(model) => generic_submodel(py, with_label(model), &binary),
    }
}

/// The submodels of the model `this`, typed as PySpark's classifier fits make them, of the
/// composite's submodels, in the order of the classes. They are made by the fit, with the
/// summaries that the classes' fits make.
fn build_submodels(py: Python<'_>, this: &PyOneVsRestModel) -> PyResult<Py<PyList>> {
    let Some(fit) = this.fit.as_ref() else {
        return Err(PyValueError::new_err(
            "the models of a loaded OneVsRestModel are not held",
        ));
    };
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
        Ok(())
    }
}

impl params_api::NativeInstance for PyOneVsRestModel {
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
                fit: self.fit.clone(),
                models: Mutex::new(self.models_value(py)),
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

    /// A model of the given `models`, as PySpark's `OneVsRestModel(models)`: of a new uid and no
    /// set params, with the defaults of the class, and of no classifier.
    #[new]
    fn new(py: Python<'_>, models: Py<PyAny>) -> PyResult<Self> {
        let core = object::ModelCore::unbound(
            py,
            "org.apache.spark.ml.classification.OneVsRestModel",
            "OneVsRestModel",
        )?;
        Ok(Self {
            core,
            raw: Vec::new(),
            fit: None,
            models: Mutex::new(Some(models)),
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
        let writer = object::PyMLWriter::new(object::WriteTarget::Python(
            slf.clone().into_any().unbind(),
            save_one_vs_rest_model,
        ));
        object::new_py(slf.py(), writer)
    }

    /// Save to `path`, as `MLWritable.save`: `self.write().save(path)`.
    fn save(slf: &Bound<'_, Self>, path: &str) -> PyResult<()> {
        slf.call_method0("write")?.call_method1("save", (path,))?;
        Ok(())
    }

    /// The reader of the class, as `OneVsRestModel.read` over Spark Connect (`RemoteMLReader`,
    /// which loads it as `OneVsRestModelReader` does).
    #[classmethod]
    fn read(cls: &Bound<'_, pyo3::types::PyType>) -> PyResult<Py<object::PyMLReader>> {
        let reader = object::PyMLReader::python(load_one_vs_rest_model);
        object::new_py(cls.py(), reader)
    }

    /// Load from `path`, as `MLReadable.load`: `cls.read().load(path)`.
    #[classmethod]
    fn load(cls: &Bound<'_, pyo3::types::PyType>, path: &str) -> PyResult<Py<PyAny>> {
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
    m.add_class::<PyOneVsRestModel>()?;
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Persistence, as `OneVsRestWriter`, `OneVsRestModelWriter` and their readers of PySpark (the
// Python format of `_OneVsRestSharedReadWrite`, which the core writes and reads too).
// ---------------------------------------------------------------------------------------------

/// The classes of the metadata, PySpark's, of which PySpark loads what this saves.
const ONE_VS_REST_CLASS: &str = "pyspark.ml.classification.OneVsRest";
const ONE_VS_REST_MODEL_CLASS: &str = "pyspark.ml.classification.OneVsRestModel";

/// The params of a OneVsRest that its metadata has of their defaults, in the order of PySpark's
/// `_defaultParamMap` of the estimator (see the core's `ESTIMATOR_DEFAULTS`).
const ESTIMATOR_DEFAULTS: &[&str] = &[
    "parallelism",
    "predictionCol",
    "featuresCol",
    "labelCol",
    "rawPredictionCol",
];

/// The params of a OneVsRestModel that its metadata has of their defaults, in the order of
/// PySpark's `_defaultParamMap` of the model (see the core's `MODEL_DEFAULTS`).
const MODEL_DEFAULTS: &[&str] = &[
    "predictionCol",
    "featuresCol",
    "labelCol",
    "rawPredictionCol",
];

/// The `paramMap` and the `defaultParamMap` of a OneVsRest or its model `slf`, as
/// `DefaultParamsWriter` makes them: of each of `defaults` in its order, the set value or the
/// default (`extractParamMap`), and then the set `weightCol`, which has no default. The classifier
/// is no param of these, which its own writer saves.
fn param_maps<'py, T: params_api::ParamsHolder>(
    slf: &Bound<'py, T>,
    defaults: &[&str],
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    let py = slf.py();
    let param_map = PyDict::new(py);
    let default_map = PyDict::new(py);
    for &name in defaults {
        let Some(default) = params_api::default_of(slf, name)? else {
            continue;
        };
        let set = params_api::ParamsHolder::set_value(&*slf.try_borrow()?, py, name)?;
        default_map.set_item(name, &default)?;
        param_map.set_item(name, set.unwrap_or(default))?;
    }
    let weight = params_api::ParamsHolder::set_value(&*slf.try_borrow()?, py, "weightCol")?;
    if let Some(weight) = weight {
        param_map.set_item("weightCol", weight)?;
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
    interruptible(py, session, || {
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
    _options: &std::collections::HashMap<String, String>,
) -> PyResult<()> {
    validate_one_vs_rest(instance, false)?;
    let (param_map, default_map) = param_maps(instance.cast::<PyOneVsRest>()?, ESTIMATOR_DEFAULTS)?;
    let uid: String = instance.getattr("uid")?.extract()?;
    save_metadata_at(
        instance.py(),
        session,
        path,
        ONE_VS_REST_CLASS,
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
    _options: &std::collections::HashMap<String, String>,
) -> PyResult<()> {
    validate_one_vs_rest(instance, true)?;
    let (param_map, default_map) =
        param_maps(instance.cast::<PyOneVsRestModel>()?, MODEL_DEFAULTS)?;
    let uid: String = instance.getattr("uid")?.extract()?;
    let models = instance.getattr("models")?;
    save_metadata_at(
        instance.py(),
        session,
        path,
        ONE_VS_REST_MODEL_CLASS,
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

/// Load the OneVsRestModel at `path`, as `OneVsRestModelReader.load`: its classifier, its models
/// (`numClasses` of them, each loaded of its own format), and its uid and params.
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
    // The composite of the classes, which the core transforms with, of the same save.
    let composite = CoreOneVsRestModel::load(session, path).to_pyerr()?;
    let df = session.range(1).to_pyerr()?;
    let core = object::ModelCore::fitted(
        py,
        Box::new(composite),
        &df,
        object::SummaryDataset::None,
        None,
    )?;
    let model = PyOneVsRestModel {
        core,
        raw: vec![("classifier", classifier.unbind())],
        fit: None,
        models: Mutex::new(Some(models.into_any().unbind())),
    };
    let model = crate::ml::object::new_object(py, model)?;
    set_metadata_params(&model, &metadata)?;
    Ok(model.into_any().unbind())
}
