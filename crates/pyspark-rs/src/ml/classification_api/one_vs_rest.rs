// SPDX-License-Identifier: Apache-2.0
//! The typed model of OneVsRest: OneVsRestModel, a container of the models of the classes of a fit,
//! with its classifier and its other params as Python values (as PySpark's `PipelineModel`).
use super::*;
use spark_connect::column::{lit_double, when};
use spark_connect::ml::{OneVsRestModel as CoreOneVsRestModel, OperatorType, ReadResult, SubModel};
use std::sync::OnceLock;

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
/// that PySpark's OneVsRestModel holds. Its transform is the composite model's of the core, and its
/// `models` are the typed submodels of the composite, as PySpark's `OneVsRest._fit` makes them.
#[pyclass(name = "OneVsRestModel", module = "pyspark.ml.connect.classification")]
pub struct PyOneVsRestModel {
    core: object::ModelCore,
    /// The Python values of the identity params: the classifier.
    raw: Vec<(&'static str, Py<PyAny>)>,
    /// The fit that the submodels are made of; None of a loaded model.
    fit: Option<OneVsRestFit>,
    /// The `models`, made at the first access, then the same list.
    models: OnceLock<Py<PyList>>,
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
            models: OnceLock::new(),
        };
        // As PySpark's classifier fits make their models, with their summaries, during the fit.
        let models = build_submodels(py, &this)?;
        let _ = this.models.set(models);
        Ok(this)
    }

    fn __from_read<'py>(py: Python<'py>, read: ReadResult) -> PyResult<Py<PyAny>> {
        let core = object::ModelCore::loaded(
            py,
            "org.apache.spark.ml.classification.OneVsRestModel",
            "OneVsRestModel",
            read,
        )?;
        Ok(Py::new(
            py,
            Self {
                core,
                raw: Vec::new(),
                fit: None,
                models: OnceLock::new(),
            },
        )?
        .into_any())
    }

    /// The classifier, if the param is set.
    fn classifier(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.raw
            .iter()
            .find(|(name, _)| *name == "classifier")
            .map(|(_, value)| value.clone_ref(py))
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
    Ok(Py::new(py, model)?.into_any())
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
            Ok(Py::new(py, typed)?.into_any())
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
            Ok(Py::new(py, typed)?.into_any())
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
            Ok(Py::new(py, typed)?.into_any())
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
            Ok(Py::new(py, typed)?.into_any())
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
            Ok(Py::new(py, typed)?.into_any())
        }
        SubModel::DecisionTreeClassification(model) => {
            generic_submodel(py, with_label(Box::new(model)), &binary)
        }
        SubModel::RandomForestClassification(model) => {
            generic_submodel(py, with_label(Box::new(model)), &binary)
        }
        SubModel::GBTClassification(model) => {
            generic_submodel(py, with_label(Box::new(model)), &binary)
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

/// The `models` of `slf`: the typed submodels that the fit made.
fn submodels(slf: &Bound<'_, PyOneVsRestModel>) -> PyResult<Py<PyList>> {
    let this = slf.try_borrow()?;
    match this.models.get() {
        Some(list) => Ok(list.clone_ref(slf.py())),
        None => Err(PyValueError::new_err(
            "the models of a loaded OneVsRestModel are not held",
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
        Py::new(
            py,
            Self {
                core: self.core.duplicate(py),
                raw,
                fit: self.fit.clone(),
                models: OnceLock::new(),
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

    /// The models of the classes, as PySpark's `models` attribute: the typed models of the
    /// classifier, each of the uid of the classifier.
    #[getter]
    fn models(slf: &Bound<'_, Self>) -> PyResult<Py<PyList>> {
        submodels(slf)
    }

    /// The transform of `dataset`, as `Transformer.transform`: of a copy of the model of a
    /// non-empty param map `params`.
    #[pyo3(signature = (dataset, params=None))]
    fn transform<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        params: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let map = params.filter(|p| p.cast::<PyDict>().is_ok_and(|d| !d.is_empty()));
        if let Some(map) = map {
            let copy = params_api::copy(slf, Some(map))?;
            return Ok(copy.call_method1("transform", (dataset,))?.unbind());
        }
        crate::ml::check_params(params)?;
        let input = crate::ml::dataset_of(dataset, "transform")?;
        let df = slf.try_borrow()?.core.transform(slf.py(), &input)?;
        Ok(Py::new(slf.py(), PyDataFrame::new(df))?.into_any())
    }

    /// A copy of the same uid and params, with `extra`, as `Params.copy`.
    #[pyo3(name = "copy", signature = (extra=None))]
    fn __params_copy<'py>(
        slf: &Bound<'py, Self>,
        extra: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        params_api::copy(slf, extra)
    }

    /// The writer of the model, as `JavaMLWritable.write` (`RemoteMLWriter`).
    fn write(slf: &Bound<'_, Self>) -> PyResult<object::PyMLWriter> {
        slf.try_borrow()?.core.writer(slf.py())
    }

    /// Save to `path`, as `MLWritable.save`: `self.write().save(path)`.
    fn save(slf: &Bound<'_, Self>, path: &str) -> PyResult<()> {
        slf.call_method0("write")?.call_method1("save", (path,))?;
        Ok(())
    }

    /// The reader of the class, as `JavaMLReadable.read` (`RemoteMLReader`).
    #[classmethod]
    fn read(cls: &Bound<'_, pyo3::types::PyType>) -> object::PyMLReader {
        let _ = cls;
        object::PyMLReader::new(
            "org.apache.spark.ml.classification.OneVsRestModel",
            OperatorType::Model,
            PyOneVsRestModel::__from_read,
        )
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
