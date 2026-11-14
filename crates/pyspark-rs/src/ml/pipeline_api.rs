// SPDX-License-Identifier: Apache-2.0
//! PySpark's `Pipeline` and `PipelineModel` of `pyspark.ml.pipeline`, natively, as PySpark 4.3.0
//! has them over Spark Connect: of stages of any estimator or transformer (a native class,
//! PySpark's own or the user's), the fit of `Pipeline._fit`, the transform of
//! `PipelineModel._transform`, PySpark's `copy`, and the persistence of `PipelineWriter` and
//! `PipelineReader` in PySpark's Python format, which each of PySpark and this client loads of the
//! other.
use super::*;
use spark_connect::ml::{
    join_path, json_object, load_metadata_text, save_metadata, stage_path, Metadata,
};
use spark_connect::SparkSession;

/// The class of the metadata of a Pipeline, PySpark's, of which PySpark loads what this saves.
const PIPELINE_CLASS: &str = "pyspark.ml.pipeline.Pipeline";
/// The class of the metadata of a PipelineModel (see [`PIPELINE_CLASS`]).
const PIPELINE_MODEL_CLASS: &str = "pyspark.ml.pipeline.PipelineModel";

/// PySpark's `Pipeline`: its `stages` (a param of the converter `identity`, the object as it was
/// set), and its uid.
#[pyclass(name = "Pipeline", module = "pyspark.ml.connect.pipeline")]
pub struct PyPipeline {
    uid: String,
    stages: Option<Py<PyAny>>,
}

impl params_api::ParamsHolder for PyPipeline {
    fn set_value(&self, py: Python<'_>, name: &str) -> PyResult<Option<Py<PyAny>>> {
        Ok(match name {
            "stages" => self.stages.as_ref().map(|stages| stages.clone_ref(py)),
            _ => None,
        })
    }

    fn put(&mut self, py: Python<'_>, name: &'static str, value: Converted) -> PyResult<()> {
        match (name, value) {
            ("stages", Converted::Raw(value)) => {
                self.stages = Some(value);
                Ok(())
            }
            // The converter `identity` keeps a None as it is.
            ("stages", Converted::Null) => {
                self.stages = Some(py.None());
                Ok(())
            }
            (name, _) => Err(PyValueError::new_err(format!(
                "Cannot find param with name {name}."
            ))),
        }
    }

    fn unset(&mut self, name: &'static str) -> PyResult<()> {
        if name == "stages" {
            self.stages = None;
        }
        Ok(())
    }
}

impl params_api::NativeInstance for PyPipeline {
    fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        Py::new(
            py,
            Self {
                uid: self.uid.clone(),
                stages: self.stages.as_ref().map(|stages| stages.clone_ref(py)),
            },
        )
    }

    fn uid(&self) -> String {
        self.uid.clone()
    }

    fn set_uid(&mut self, uid: &str) {
        self.uid = uid.to_string();
    }
}

const PIPELINE_PARAMS: &[(&str, Kind)] = &[("stages", Kind::Identity)];

native_pymethods! {
    PyPipeline;
    persistence: python;
    params: [
        "stages" => ("getStages", get_stages, param_stages),
    ];

    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, stages=None)")]
    fn new(
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let mut this = Self {
            uid: params_api::random_uid(py, "Pipeline")?,
            stages: None,
        };
        for (_, _, value) in bind_params("__init__", "Pipeline", &[], PIPELINE_PARAMS, args, kwargs)? {
            this.stages = Some(value.unbind());
        }
        Ok(this)
    }

    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, stages=None)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        for (name, _, value) in
            bind_params("setParams", "Pipeline", &[], PIPELINE_PARAMS, args, kwargs)?
        {
            params_api::set_by_name(&slf, name, &value)?;
        }
        Ok(slf)
    }

    #[pyo3(name = "setStages")]
    fn set_stages<'py>(slf: Bound<'py, Self>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        params_api::set_by_name(&slf, "stages", value)?;
        Ok(slf)
    }

    /// The fit of the stages, as `Estimator.fit` (a param map fits a copy, a list or tuple of
    /// them fits once each) and `Pipeline._fit`.
    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<Py<PyAny>>> {
        fit_with(slf, dataset, params, || pipeline_fit(slf, dataset))
    }

    /// The fits of a copy of the pipeline, one per param map, as `Estimator.fitMultiple`.
    #[pyo3(name = "fitMultiple")]
    #[allow(non_snake_case)]
    fn fit_multiple<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        paramMaps: &Bound<'py, PyAny>,
    ) -> PyResult<params_api::PyFitMultipleIterator> {
        params_api::fit_multiple(slf.as_any(), dataset, paramMaps)
    }

    /// A copy of the pipeline, as `Pipeline.copy`: of the same uid and params, with `extra`, of
    /// the copies of its stages, each with `extra`.
    #[pyo3(name = "copy", signature = (extra=None))]
    fn copy<'py>(slf: &Bound<'py, Self>, extra: Option<&Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let extra = extra_or_empty(py, extra);
        let that = params_api::copy(slf, Some(&extra))?;
        let stages = copied_stages(&that.call_method0("getStages")?, &extra)?;
        that.call_method1("setStages", (stages,))
    }

    /// The writer of the pipeline, as `Pipeline.write` over Spark Connect (`RemoteMLWriter`,
    /// which saves it as `PipelineWriter` does).
    fn write(slf: &Bound<'_, Self>) -> object::PyMLWriter {
        object::PyMLWriter::new(object::WriteTarget::Python(
            slf.clone().into_any().unbind(),
            save_pipeline,
        ))
    }

    /// Save to `path`, as `MLWritable.save`: `self.write().save(path)`.
    fn save(slf: &Bound<'_, Self>, path: &str) -> PyResult<()> {
        slf.call_method0("write")?.call_method1("save", (path,))?;
        Ok(())
    }

    /// The reader of the class, as `Pipeline.read` over Spark Connect (`PipelineReader`).
    #[classmethod]
    fn read(cls: &Bound<'_, pyo3::types::PyType>) -> object::PyMLReader {
        let _ = cls;
        object::PyMLReader::python(load_pipeline)
    }

    /// Load from `path`, as `MLReadable.load`: `cls.read().load(path)`.
    #[classmethod]
    fn load(cls: &Bound<'_, pyo3::types::PyType>, path: &str) -> PyResult<Py<PyAny>> {
        Ok(cls.call_method0("read")?.call_method1("load", (path,))?.unbind())
    }
}

/// PySpark's `PipelineModel`: its `stages` (an attribute of the instance, as PySpark's
/// constructor sets it), of the transformers and the models of a fit, and its uid. It has no
/// params.
#[pyclass(name = "PipelineModel", module = "pyspark.ml.connect.pipeline")]
pub struct PyPipelineModel {
    uid: String,
    #[pyo3(get, set)]
    stages: Py<PyAny>,
}

impl PyPipelineModel {
    /// The model of the stages `stages`, of a new uid, as `PipelineModel(stages)`.
    fn of_stages(py: Python<'_>, stages: Py<PyAny>) -> PyResult<Self> {
        Ok(Self {
            uid: params_api::random_uid(py, "PipelineModel")?,
            stages,
        })
    }
}

impl params_api::ParamsHolder for PyPipelineModel {
    fn set_value(&self, _py: Python<'_>, _name: &str) -> PyResult<Option<Py<PyAny>>> {
        Ok(None)
    }

    fn put(&mut self, _py: Python<'_>, name: &'static str, _value: Converted) -> PyResult<()> {
        Err(PyValueError::new_err(format!(
            "Cannot find param with name {name}."
        )))
    }
}

impl params_api::NativeInstance for PyPipelineModel {
    fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        Py::new(
            py,
            Self {
                uid: self.uid.clone(),
                stages: self.stages.clone_ref(py),
            },
        )
    }

    fn uid(&self) -> String {
        self.uid.clone()
    }

    fn set_uid(&mut self, uid: &str) {
        self.uid = uid.to_string();
    }
}

native_pymethods! {
    PyPipelineModel;
    persistence: python;
    params: [];

    #[new]
    fn new(py: Python<'_>, stages: Py<PyAny>) -> PyResult<Self> {
        Self::of_stages(py, stages)
    }

    /// The transform of `dataset` by each stage in order, as `Transformer.transform` (a param map
    /// transforms of a copy) and `PipelineModel._transform`.
    #[pyo3(signature = (dataset, params=None))]
    fn transform<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        params: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if let Some(map) = params.filter(|p| p.cast::<PyDict>().is_ok_and(|d| !d.is_empty())) {
            let copy = slf.call_method1("copy", (map,))?;
            return copy.call_method1("transform", (dataset,));
        }
        check_params(params)?;
        let stages = slf.try_borrow()?.stages.clone_ref(slf.py()).into_bound(slf.py());
        let mut dataset = dataset.clone();
        for stage in stages.try_iter()? {
            dataset = stage?.call_method1("transform", (&dataset,))?;
        }
        Ok(dataset)
    }

    /// A copy of the model, as `PipelineModel.copy`: of a new uid, of the copies of its stages,
    /// each with `extra`.
    #[pyo3(name = "copy", signature = (extra=None))]
    fn copy(slf: &Bound<'_, Self>, extra: Option<&Bound<'_, PyAny>>) -> PyResult<Py<Self>> {
        let py = slf.py();
        let extra = extra_or_empty(py, extra);
        let stages = slf.try_borrow()?.stages.clone_ref(py).into_bound(py);
        let stages = copied_stages(&stages, &extra)?;
        Py::new(py, Self::of_stages(py, stages.into_any().unbind())?)
    }

    /// The writer of the model, as `PipelineModel.write` over Spark Connect (`RemoteMLWriter`,
    /// which saves it as `PipelineModelWriter` does).
    fn write(slf: &Bound<'_, Self>) -> object::PyMLWriter {
        object::PyMLWriter::new(object::WriteTarget::Python(
            slf.clone().into_any().unbind(),
            save_pipeline_model,
        ))
    }

    fn save(slf: &Bound<'_, Self>, path: &str) -> PyResult<()> {
        slf.call_method0("write")?.call_method1("save", (path,))?;
        Ok(())
    }

    #[classmethod]
    fn read(cls: &Bound<'_, pyo3::types::PyType>) -> object::PyMLReader {
        let _ = cls;
        object::PyMLReader::python(load_pipeline_model)
    }

    #[classmethod]
    fn load(cls: &Bound<'_, pyo3::types::PyType>, path: &str) -> PyResult<Py<PyAny>> {
        Ok(cls.call_method0("read")?.call_method1("load", (path,))?.unbind())
    }
}

/// `extra`, or `dict()` of None, as `copy` takes it.
fn extra_or_empty<'py>(py: Python<'py>, extra: Option<&Bound<'py, PyAny>>) -> Bound<'py, PyAny> {
    match extra.filter(|extra| !extra.is_none()) {
        Some(extra) => extra.clone(),
        None => PyDict::new(py).into_any(),
    }
}

/// `[stage.copy(extra) for stage in stages]`.
fn copied_stages<'py>(
    stages: &Bound<'py, PyAny>,
    extra: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let copies = PyList::empty(stages.py());
    for stage in stages.try_iter()? {
        copies.append(stage?.call_method1("copy", (extra,))?)?;
    }
    Ok(copies)
}

/// The fit of the pipeline `slf` of `dataset`, as `Pipeline._fit`: up to its last estimator, a
/// transformer transforms and an estimator fits (and its model transforms, but of the last), and
/// the stages after it are kept as they are, in a new PipelineModel.
fn pipeline_fit(slf: &Bound<'_, PyPipeline>, dataset: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let py = slf.py();
    // The native classes are PySpark's estimators and transformers of the import of their package.
    py.import("pyspark.ml.connect")?;
    let base = py.import("pyspark.ml.base")?;
    let estimator = base.getattr("Estimator")?;
    let transformer = base.getattr("Transformer")?;
    let stages = slf.call_method0("getStages")?;
    let stages = stages.try_iter()?.collect::<PyResult<Vec<_>>>()?;
    for stage in &stages {
        if !(stage.is_instance(&estimator)? || stage.is_instance(&transformer)?) {
            return Err(PyTypeError::new_err(format!(
                "Cannot recognize a pipeline stage of type {}.",
                stage.get_type().str()?
            )));
        }
    }
    let mut last_estimator = None;
    for (i, stage) in stages.iter().enumerate() {
        if stage.is_instance(&estimator)? {
            last_estimator = Some(i);
        }
    }
    let mut dataset = dataset.clone();
    let transformers = PyList::empty(py);
    for (i, stage) in stages.iter().enumerate() {
        match last_estimator {
            Some(last) if i <= last => {
                if stage.is_instance(&transformer)? {
                    transformers.append(stage)?;
                    dataset = stage.call_method1("transform", (&dataset,))?;
                } else {
                    let model = stage.call_method1("fit", (&dataset,))?;
                    transformers.append(&model)?;
                    if i < last {
                        dataset = model.call_method1("transform", (&dataset,))?;
                    }
                }
            }
            _ => transformers.append(stage)?,
        }
    }
    let model = PyPipelineModel::of_stages(py, transformers.into_any().unbind())?;
    Ok(Py::new(py, model)?.into_any())
}

/// Save the pipeline `instance`, as `PipelineWriter.saveImpl`: of its stages.
fn save_pipeline(
    instance: &Bound<'_, PyAny>,
    spark: &Bound<'_, PyAny>,
    session: &SparkSession,
    path: &str,
    _options: &std::collections::HashMap<String, String>,
) -> PyResult<()> {
    let stages = instance.call_method0("getStages")?;
    save_with_stages(instance, PIPELINE_CLASS, &stages, spark, session, path)
}

/// Save the model `instance`, as `PipelineModelWriter.saveImpl`: of its stages.
fn save_pipeline_model(
    instance: &Bound<'_, PyAny>,
    spark: &Bound<'_, PyAny>,
    session: &SparkSession,
    path: &str,
    _options: &std::collections::HashMap<String, String>,
) -> PyResult<()> {
    let stages = instance.getattr("stages")?;
    save_with_stages(
        instance,
        PIPELINE_MODEL_CLASS,
        &stages,
        spark,
        session,
        path,
    )
}

/// Save `instance` of the class `class` and its stages `stages` to `path`, as
/// `PipelineSharedReadWrite.validateStages` and `saveImpl`: the metadata of the uids of the stages
/// (`stageUids`, of the language "Python"), and each stage, of its own writer and the session
/// `spark`, at `stages/<index>_<uid>`.
pub(super) fn save_with_stages(
    instance: &Bound<'_, PyAny>,
    class: &str,
    stages: &Bound<'_, PyAny>,
    spark: &Bound<'_, PyAny>,
    session: &SparkSession,
    path: &str,
) -> PyResult<()> {
    let py = instance.py();
    let stages = stages.try_iter()?.collect::<PyResult<Vec<_>>>()?;
    for stage in &stages {
        // As `validateStages`, whose `MLWritable` is an object that writes.
        if !stage.hasattr("write")? {
            return Err(PyValueError::new_err((
                "Pipeline write will fail on this pipeline because stage %s of type %s is not \
                 MLWritable",
                stage.getattr("uid")?.unbind(),
                stage.get_type().unbind(),
            )));
        }
    }
    let uids = stages
        .iter()
        .map(|stage| stage.getattr("uid")?.extract::<String>())
        .collect::<PyResult<Vec<_>>>()?;
    let uid: String = instance.getattr("uid")?.extract()?;
    // As PySpark's `{"stageUids": stageUids, "language": "Python"}`, of its order.
    let param_map = json_object(&[
        ("stageUids", serde_json::json!(uids).to_string()),
        ("language", serde_json::json!("Python").to_string()),
    ]);
    interruptible(py, session, || {
        let version = session.version()?;
        let metadata = Metadata::new(class, &uid, &version, param_map, "{}".to_string());
        save_metadata(session, path, &metadata.to_json()).map_err(Box::new)
    })?;
    let stages_dir = join_path(path, "stages");
    for (index, (stage, stage_uid)) in stages.iter().zip(&uids).enumerate() {
        let stage_path = stage_path(stage_uid, index, stages.len(), &stages_dir);
        stage
            .call_method0("write")?
            .call_method1("session", (spark,))?
            .call_method1("save", (stage_path,))?;
    }
    Ok(())
}

/// Load the pipeline at `path`, as `PipelineReader.load`: `Pipeline(stages=stages)._resetUid(uid)`.
fn load_pipeline(
    py: Python<'_>,
    spark: &Bound<'_, PyAny>,
    session: &SparkSession,
    path: &str,
) -> PyResult<Py<PyAny>> {
    let (uid, stages) = load_with_stages(py, spark, session, path)?;
    let pipeline = PyPipeline {
        uid,
        stages: Some(stages.into_any().unbind()),
    };
    Ok(Py::new(py, pipeline)?.into_any())
}

/// Load the model at `path`, as `PipelineModelReader.load`:
/// `PipelineModel(stages=stages)._resetUid(uid)`.
fn load_pipeline_model(
    py: Python<'_>,
    spark: &Bound<'_, PyAny>,
    session: &SparkSession,
    path: &str,
) -> PyResult<Py<PyAny>> {
    let (uid, stages) = load_with_stages(py, spark, session, path)?;
    let model = PyPipelineModel {
        uid,
        stages: stages.into_any().unbind(),
    };
    Ok(Py::new(py, model)?.into_any())
}

/// The uid and the stages of the object saved at `path`, as `DefaultParamsReader.loadMetadata`
/// and `PipelineSharedReadWrite.load` read them: each stage of `stageUids`, by
/// [`load_params_instance`].
pub(super) fn load_with_stages<'py>(
    py: Python<'py>,
    spark: &Bound<'py, PyAny>,
    session: &SparkSession,
    path: &str,
) -> PyResult<(String, Bound<'py, PyList>)> {
    let metadata = load_metadata_dict(py, session, path)?;
    let stage_uids = metadata.get_item("paramMap")?.get_item("stageUids")?;
    let stages_dir = join_path(path, "stages");
    let stages = PyList::empty(py);
    for (index, stage_uid) in stage_uids.try_iter()?.enumerate() {
        let stage_uid = metadata_str(&stage_uid?)?;
        let stage_path = stage_path(&stage_uid, index, stage_uids.len()?, &stages_dir);
        stages.append(load_params_instance(py, spark, session, &stage_path)?)?;
    }
    let uid = metadata_str(&metadata.get_item("uid")?)?;
    Ok((uid, stages))
}

/// The metadata at `<path>/metadata`, as `DefaultParamsReader.loadMetadata` and `_parseMetaData`
/// read it: `json.loads` of its text, a dict of Python's values (of the floats NaN and Infinity and
/// the ints of any size that Python writes), whose `class` it reads (a KeyError if it has none).
pub(super) fn load_metadata_dict<'py>(
    py: Python<'py>,
    session: &SparkSession,
    path: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let text = interruptible(py, session, || {
        load_metadata_text(session, path).map_err(Box::new)
    })?;
    let metadata = py.import("json")?.call_method1("loads", (text,))?;
    metadata.get_item("class")?;
    Ok(metadata)
}

/// The string `value` of a metadata, of which Python makes a path or a uid; a TypeError of
/// Python's `+` of a string when it is none.
pub(super) fn metadata_str(value: &Bound<'_, PyAny>) -> PyResult<String> {
    value.extract::<String>().map_err(|_| {
        let name = value
            .get_type()
            .name()
            .map(|name| name.to_string())
            .unwrap_or_default();
        PyTypeError::new_err(format!("can only concatenate str (not \"{name}\") to str"))
    })
}

/// The object saved at `path`, as `DefaultParamsReader.loadParamsInstance`: of the class of its
/// metadata (a Python class of PySpark, or the PySpark class of a JVM class), `load` of `path`.
/// The class is the native one of `pyspark.ml.connect` of the same name where there is one that
/// loads, else PySpark's (or the user's).
pub(super) fn load_params_instance<'py>(
    py: Python<'py>,
    spark: &Bound<'py, PyAny>,
    session: &SparkSession,
    path: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let _ = spark;
    let metadata = load_metadata_dict(py, session, path)?;
    let class = metadata.get_item("class")?;
    // As `isPythonParamsInstance` and the `replace` of a JVM class, of the errors of a class that
    // is no string.
    let python_class: String = if class
        .call_method1("startswith", ("pyspark.ml.",))?
        .is_truthy()?
    {
        class.extract()?
    } else {
        class
            .call_method1("replace", ("org.apache.spark", "pyspark"))?
            .extract()?
    };
    class_of(py, &python_class)?.call_method1("load", (path,))
}

/// The Python class of the fully qualified name `name`, as `DefaultParamsReader.__get_class`, or
/// the native class of `pyspark.ml.connect` of the same module and name, if it loads.
fn class_of<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    // As `__get_class`, a name of no module imports the module "", which raises its ValueError.
    let (module, class) = name.rsplit_once('.').unwrap_or(("", name));
    if let Some(rest) = module.strip_prefix("pyspark.ml.") {
        if !rest.starts_with("connect") {
            let native = py
                .import(format!("pyspark.ml.connect.{rest}").as_str())
                .and_then(|module| module.getattr(class));
            if let Ok(native) = native {
                if native.hasattr("load")? {
                    return Ok(native);
                }
            }
        }
    }
    py.import(module)?.getattr(class)
}

/// Register the Pipeline and the PipelineModel.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPipeline>()?;
    m.add_class::<PyPipelineModel>()?;
    Ok(())
}
