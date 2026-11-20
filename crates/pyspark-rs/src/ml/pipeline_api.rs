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

/// The class of the metadata of `instance`, a Pipeline, a validator or a OneVsRest (or a model of
/// one), as PySpark's `DefaultParamsWriter` writes it (`instance.__module__ + "." +
/// instance.__class__.__name__`): `class`, PySpark's class of the native class `T`, which PySpark
/// loads, of an object of the native class or of its public class, and the module and the name of
/// the class of an object of a Python subclass of them (`__main__.MyPipeline`), as of a subclass of
/// PySpark's class.
pub(super) fn metadata_class<T: pyo3::PyTypeInfo>(
    instance: &Bound<'_, PyAny>,
    class: &str,
) -> PyResult<String> {
    let py = instance.py();
    let ty = instance.get_type();
    let public = crate::ml::object::public_class::<T>(py);
    if ty.is(py.get_type::<T>()) || public.is_some_and(|public| ty.is(&public)) {
        return Ok(class.to_string());
    }
    instance
        .getattr("__module__")?
        .add(".")?
        .add(instance.getattr("__class__")?.getattr("__name__")?)?
        .extract()
}

/// PySpark's `Pipeline`: its `stages` (a param of the converter `identity`, the object as it was
/// set), and its uid.
#[pyclass(subclass, name = "Pipeline", module = "pyspark.ml.connect.pipeline")]
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
        crate::ml::object::new_py(
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
    ) -> PyResult<Py<params_api::PyFitMultipleIterator>> {
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
    fn write(slf: &Bound<'_, Self>) -> PyResult<Py<object::PyMLWriter>> {
        object::PyMLWriter::new(object::WriteTarget::Python(save_pipeline))
            .object_of(slf.as_any())
    }

    /// Save to `path`, as `MLWritable.save`: `self.write().save(path)`.
    fn save(slf: &Bound<'_, Self>, path: &Bound<'_, PyAny>) -> PyResult<()> {
        slf.call_method0("write")?.call_method1("save", (path,))?;
        Ok(())
    }

    /// The reader of the class, as `Pipeline.read` over Spark Connect (`PipelineReader`).
    #[classmethod]
    fn read(cls: &Bound<'_, pyo3::types::PyType>) -> PyResult<Py<object::PyMLReader>> {
        let reader = object::PyMLReader::python(load_pipeline);
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
}

/// PySpark's `PipelineModel`: its `stages` (an attribute of the instance, as PySpark's
/// constructor sets it), of the transformers and the models of a fit, and its uid. It has no
/// params.
#[pyclass(
    subclass,
    name = "PipelineModel",
    module = "pyspark.ml.connect.pipeline"
)]
pub struct PyPipelineModel {
    uid: String,
    #[pyo3(get, set)]
    stages: Py<PyAny>,
}

impl PyPipelineModel {
    /// The model of the stages `stages`, of a new uid, as `PipelineModel(stages)`.
    fn of_stages(py: Python<'_>, stages: Py<PyAny>) -> PyResult<Self> {
        Self::of_class(py, "PipelineModel", stages)
    }

    /// The model of [`PyPipelineModel::of_stages`], of a uid of the class `class`, as
    /// `Identifiable._randomUID` names the uid of the class of the object (a Python subclass's).
    fn of_class(py: Python<'_>, class: &str, stages: Py<PyAny>) -> PyResult<Self> {
        Ok(Self {
            uid: params_api::random_uid(py, class)?,
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
        crate::ml::object::new_py(
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

    /// `PipelineModel(stages)`, of a uid of the class of the object, as `Identifiable._randomUID`
    /// names it (`MyPipelineModel_...` of a Python subclass).
    #[new]
    #[classmethod]
    fn new(cls: &Bound<'_, pyo3::types::PyType>, stages: Py<PyAny>) -> PyResult<Self> {
        Self::of_class(cls.py(), &cls.name()?.to_string(), stages)
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
        crate::ml::object::new_py(py, Self::of_stages(py, stages.into_any().unbind())?)
    }

    /// The writer of the model, as `PipelineModel.write` over Spark Connect (`RemoteMLWriter`,
    /// which saves it as `PipelineModelWriter` does).
    fn write(slf: &Bound<'_, Self>) -> PyResult<Py<object::PyMLWriter>> {
        object::PyMLWriter::new(object::WriteTarget::Python(save_pipeline_model))
            .object_of(slf.as_any())
    }

    fn save(slf: &Bound<'_, Self>, path: &Bound<'_, PyAny>) -> PyResult<()> {
        slf.call_method0("write")?.call_method1("save", (path,))?;
        Ok(())
    }

    #[classmethod]
    fn read(cls: &Bound<'_, pyo3::types::PyType>) -> PyResult<Py<object::PyMLReader>> {
        let reader = object::PyMLReader::python(load_pipeline_model);
        reader.object_of(cls.as_any())
    }

    #[classmethod]
    fn load(
        cls: &Bound<'_, pyo3::types::PyType>,
        path: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        Ok(cls.call_method0("read")?.call_method1("load", (path,))?.unbind())
    }
}

/// `extra`, or `dict()` of None, as `copy` takes it.
pub(super) fn extra_or_empty<'py>(
    py: Python<'py>,
    extra: Option<&Bound<'py, PyAny>>,
) -> Bound<'py, PyAny> {
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
    // As `Pipeline._fit`, which iterates the stages three times: to check their types, to find the
    // last estimator, and to fit them. So the stages of an iterator, such as a generator, which the
    // first consumes, give PySpark's PipelineModel of no stages.
    let stages = slf.call_method0("getStages")?;
    for stage in stages.try_iter()? {
        let stage = stage?;
        if !(stage.is_instance(&estimator)? || stage.is_instance(&transformer)?) {
            return Err(PyTypeError::new_err(format!(
                "Cannot recognize a pipeline stage of type {}.",
                stage.get_type().str()?
            )));
        }
    }
    let mut last_estimator = None;
    for (i, stage) in stages.try_iter()?.enumerate() {
        if stage?.is_instance(&estimator)? {
            last_estimator = Some(i);
        }
    }
    let mut dataset = dataset.clone();
    let transformers = PyList::empty(py);
    for (i, stage) in stages.try_iter()?.enumerate() {
        let stage = stage?;
        match last_estimator {
            Some(last) if i <= last => {
                if stage.is_instance(&transformer)? {
                    transformers.append(&stage)?;
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
    Ok(crate::ml::object::new_py(py, model)?.into_any())
}

/// Save the pipeline `instance`, as `PipelineWriter.saveImpl`: of its stages.
fn save_pipeline(
    instance: &Bound<'_, PyAny>,
    spark: &Bound<'_, PyAny>,
    session: &SparkSession,
    path: &str,
    _options: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let stages = instance.call_method0("getStages")?;
    let class = metadata_class::<PyPipeline>(instance, PIPELINE_CLASS)?;
    save_with_stages(instance, &class, &stages, spark, session, path)
}

/// Save the model `instance`, as `PipelineModelWriter.saveImpl`: of its stages.
fn save_pipeline_model(
    instance: &Bound<'_, PyAny>,
    spark: &Bound<'_, PyAny>,
    session: &SparkSession,
    path: &str,
    _options: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let stages = instance.getattr("stages")?;
    let class = metadata_class::<PyPipelineModel>(instance, PIPELINE_MODEL_CLASS)?;
    save_with_stages(instance, &class, &stages, spark, session, path)
}

/// Save `instance` of the class `class` and its stages `stages` to `path`, as
/// `PipelineSharedReadWrite.validateStages` and `saveImpl`: the metadata of the uids of the stages
/// (`stageUids`, of the language "Python") and of the defaults of `_defaultParamMap` (of the Params
/// of a Python subclass, which a Pipeline of PySpark's has none of), and each stage, of its own
/// writer and the session `spark`, at `stages/<index>_<uid>` (of `len(stages)`). As PySpark's, each
/// of the three iterates the stages: the check of `validateStages` consumes stages of a one-shot
/// iterator (a generator), so that the metadata has no uids and no stage is written.
pub(super) fn save_with_stages(
    instance: &Bound<'_, PyAny>,
    class: &str,
    stages: &Bound<'_, PyAny>,
    spark: &Bound<'_, PyAny>,
    session: &SparkSession,
    path: &str,
) -> PyResult<()> {
    let py = instance.py();
    for stage in stages.try_iter()? {
        let stage = stage?;
        // As `validateStages`.
        if !writable(&stage)? {
            return Err(PyValueError::new_err((
                "Pipeline write will fail on this pipeline because stage %s of type %s is not \
                 MLWritable",
                stage.getattr("uid")?.unbind(),
                stage.get_type().unbind(),
            )));
        }
    }
    let uids = stages
        .try_iter()?
        .map(|stage| stage?.getattr("uid")?.extract::<String>())
        .collect::<PyResult<Vec<_>>>()?;
    let uid: String = instance.getattr("uid")?.extract()?;
    // As PySpark's `{"stageUids": stageUids, "language": "Python"}`, of its order.
    let param_map = json_object(&[
        ("stageUids", serde_json::json!(uids).to_string()),
        ("language", serde_json::json!("Python").to_string()),
    ]);
    let default_map = default_params_json(instance)?;
    // Of strings.
    crate::interrupt::interruptible_without_parts(py, session, || {
        let version = session.version()?;
        let metadata = Metadata::new(class, &uid, &version, param_map, default_map);
        save_metadata(session, path, &metadata.to_json()).map_err(Box::new)
    })?;
    let stages_dir = join_path(path, "stages");
    for (index, stage) in stages.try_iter()?.enumerate() {
        let stage = stage?;
        let stage_uid: String = stage.getattr("uid")?.extract()?;
        let stage_path = stage_path(&stage_uid, index, stages.len()?, &stages_dir);
        stage
            .call_method0("write")?
            .call_method1("session", (spark,))?
            .call_method1("save", (stage_path,))?;
    }
    Ok(())
}

/// The `defaultParamMap` of the metadata of `instance`, as `DefaultParamsWriter` writes it of
/// `_defaultParamMap` (`json.dumps` of the defaults by the names of their Params): of an object of
/// PySpark's maps, the defaults of the Params of a Python subclass (`_setDefault(myParam=3)`), and
/// else none, as of PySpark's Pipeline, which has none.
fn default_params_json(instance: &Bound<'_, PyAny>) -> PyResult<String> {
    let py = instance.py();
    let has_maps = instance
        .getattr(pyo3::intern!(py, "__dict__"))
        .and_then(|attributes| attributes.contains("_defaultParamMap"))
        .unwrap_or(false);
    let defaults = PyDict::new(py);
    if has_maps {
        let map = instance.getattr(pyo3::intern!(py, "_defaultParamMap"))?;
        for (param, value) in map.cast_into::<PyDict>()?.iter() {
            defaults.set_item(param.getattr(pyo3::intern!(py, "name"))?, value)?;
        }
    }
    let kwargs = PyDict::new(py);
    kwargs.set_item("separators", (",", ":"))?;
    py.import("json")?
        .getattr("dumps")?
        .call((defaults,), Some(&kwargs))?
        .extract()
}

/// Whether `obj` writes, as PySpark's `isinstance(obj, MLWritable)`: an `MLWritable` of PySpark's,
/// or an object of a native class (or of a Python subclass of one), which has PySpark's `write`
/// but is no subclass of `MLWritable`, a plain class that no class can be registered with.
pub(super) fn writable(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = obj.py();
    let ml_writable = py.import("pyspark.ml.util")?.getattr("MLWritable")?;
    if obj.is_instance(&ml_writable)? {
        return Ok(true);
    }
    let natives = py.import("pyspark._pyspark")?;
    for class in obj.get_type().getattr("__mro__")?.try_iter()? {
        let class = class?;
        let name = class.getattr("__name__")?;
        let native = natives
            .getattr(name.cast::<pyo3::types::PyString>()?)
            .is_ok_and(|native| native.is(&class));
        if native {
            return obj.hasattr("write");
        }
    }
    Ok(false)
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
    Ok(crate::ml::object::new_py(py, pipeline)?.into_any())
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
    Ok(crate::ml::object::new_py(py, model)?.into_any())
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
    let text = crate::interrupt::interruptible_without_parts(py, session, || {
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
