// SPDX-License-Identifier: Apache-2.0
//! PySpark's `pyspark.ml.param.Params` API on the native ML classes, as official pyspark-client
//! 4.3.0 has it over Spark Connect: the Param objects of a class, and `params`, `getParam`,
//! `hasParam`, `isSet`, `isDefined`, `hasDefault`, `getOrDefault`, `explainParam(s)`,
//! `extractParamMap`, `set`, `clear` and `copy`, with PySpark's errors.
//!
//! The Param objects are those of PySpark's own class (`pyspark.ml.param.Param`), of the docs,
//! converters and defaults of each official class, from `ml_params.json` (written by
//! `scripts/capture_ml_api_golden.py`). The values of the set params are the ones that the native
//! class holds, which its setters convert as PySpark's `_set` does.

use super::*;
use pyo3::types::PyString;
use pyo3::PyTypeInfo;
use std::collections::HashMap;
use std::sync::OnceLock;

/// A param of an official `pyspark.ml` class: its name, its doc, the name of its `TypeConverters`
/// method, and its default, if any.
pub(super) struct ParamMeta {
    pub(super) name: String,
    pub(super) doc: String,
    pub(super) converter: String,
    pub(super) default: Option<serde_json::Value>,
}

/// The params of each official class, by its name.
fn all_params() -> &'static HashMap<String, Vec<ParamMeta>> {
    static PARAMS: OnceLock<HashMap<String, Vec<ParamMeta>>> = OnceLock::new();
    PARAMS.get_or_init(|| {
        let classes: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(include_str!("ml_params.json"))
                .expect("ml_params.json is valid JSON");
        classes
            .into_iter()
            .map(|(class, params)| {
                let params = params
                    .as_array()
                    .map(|params| {
                        params
                            .iter()
                            .map(|p| ParamMeta {
                                name: p["name"].as_str().unwrap_or_default().to_string(),
                                doc: p["doc"].as_str().unwrap_or_default().to_string(),
                                converter: p["converter"].as_str().unwrap_or_default().to_string(),
                                default: p.get("default").cloned(),
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                (class, params)
            })
            .collect()
    })
}

/// The params of the class of `slf`: of the first class of its MRO that is an official one, so
/// that a Python subclass of a native class has the params of it.
fn params_of(slf: &Bound<'_, PyAny>) -> PyResult<&'static [ParamMeta]> {
    for class in slf.get_type().mro().iter() {
        let name = class.getattr("__name__")?.extract::<String>()?;
        if let Some(params) = all_params().get(&name) {
            return Ok(params.as_slice());
        }
    }
    Ok(&[])
}

/// The kind of the values of a param of the `TypeConverters` method `converter`.
pub(super) fn kind_of_converter(converter: &str) -> Option<Kind> {
    Some(match converter {
        "toInt" => Kind::Int,
        "toFloat" => Kind::Float,
        "toString" => Kind::Str,
        "toBoolean" => Kind::Bool,
        "toListFloat" => Kind::FloatList,
        "toListInt" => Kind::IntList,
        "toListString" => Kind::StrList,
        "toListListFloat" => Kind::FloatListList,
        "toVector" => Kind::Vector,
        "toMatrix" => Kind::Matrix,
        "identity" => Kind::Identity,
        _ => return None,
    })
}

/// The Python value of a default of `ml_params.json` (the typed JSON of the capture script).
fn default_to_py(py: Python<'_>, value: &serde_json::Value) -> PyResult<Py<PyAny>> {
    use pyo3::IntoPyObjectExt;
    Ok(match value {
        serde_json::Value::Null => py.None(),
        serde_json::Value::Bool(b) => b.into_py_any(py)?,
        serde_json::Value::String(s) => s.into_py_any(py)?,
        serde_json::Value::Number(n) => match n.as_i64() {
            Some(i) => i.into_py_any(py)?,
            None => n.as_f64().unwrap_or_default().into_py_any(py)?,
        },
        serde_json::Value::Array(items) => PyList::new(
            py,
            items
                .iter()
                .map(|v| default_to_py(py, v))
                .collect::<PyResult<Vec<_>>>()?,
        )?
        .into_any()
        .unbind(),
        serde_json::Value::Object(map) => {
            if let Some(v) = map.get("int") {
                py.import("builtins")?
                    .getattr("int")?
                    .call1((v.to_string(),))?
                    .unbind()
            } else if let Some(v) = map.get("float") {
                let text = v.as_str().unwrap_or("0.0");
                py.import("builtins")?
                    .getattr("float")?
                    .call1((text,))?
                    .unbind()
            } else if let Some(serde_json::Value::Array(items)) = map.get("list") {
                PyList::new(
                    py,
                    items
                        .iter()
                        .map(|v| default_to_py(py, v))
                        .collect::<PyResult<Vec<_>>>()?,
                )?
                .into_any()
                .unbind()
            } else if let Some(values) = map.get("dense_vector") {
                let values = default_to_py(py, values)?;
                py.import("pyspark.ml.linalg")?
                    .getattr("DenseVector")?
                    .call1((values,))?
                    .unbind()
            } else {
                return Err(PyValueError::new_err(format!(
                    "unsupported default of ml_params.json: {value}"
                )));
            }
        }
    })
}

/// What the Params API needs of a native object: the Python values of its set params, as PySpark's
/// `_paramMap` holds them, the setting of a converted value, and the defaults that it holds itself.
pub(super) trait ParamsHolder: PyClass<Frozen = False> {
    /// The value of the set param `name`, as its setter converted it; None if it is not set.
    fn set_value(&self, py: Python<'_>, name: &str) -> PyResult<Option<Py<PyAny>>>;

    /// Set the param `name` to `value`, as the setters do.
    fn put(&mut self, py: Python<'_>, name: &'static str, value: Converted) -> PyResult<()>;

    /// The default of the param `name` that the object holds itself, where PySpark's
    /// `_defaultParamMap` has one that is not the default of its class (see [`OwnDefaults`]); None
    /// for the default of its class.
    fn own_default(&self, py: Python<'_>, name: &str) -> Option<Py<PyAny>> {
        let _ = (py, name);
        None
    }

    /// Hold `value` as the default of the param `name`; false of an object that holds none.
    fn put_own_default(&mut self, name: &'static str, value: Py<PyAny>) -> bool {
        let _ = (name, value);
        false
    }
}

/// The defaults that an object holds itself, by param, where PySpark's `_defaultParamMap` has one
/// that is not the default of its class: a fitted model has the defaults of its estimator, which
/// `_copyValues` copies to it (a seed of the name of the estimator's class, or the `coldStartStrategy`
/// that ALS has and ALSModel does not); and a loaded object has the defaults that it derived from
/// the uid that it was made of, which `_resetUid` keeps (an `outputCol` of that uid).
pub(super) type OwnDefaults = Vec<(&'static str, Py<PyAny>)>;

/// The default of the param `name` of `defaults`.
pub(super) fn own_default_of(
    py: Python<'_>,
    defaults: &OwnDefaults,
    name: &str,
) -> Option<Py<PyAny>> {
    defaults
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, value)| value.clone_ref(py))
}

/// Hold `value` as the default of the param `name` in `defaults`.
pub(super) fn put_own_default_in(defaults: &mut OwnDefaults, name: &'static str, value: Py<PyAny>) {
    defaults.retain(|(n, _)| *n != name);
    defaults.push((name, value));
}

/// A copy of `defaults`.
pub(super) fn clone_own_defaults(py: Python<'_>, defaults: &OwnDefaults) -> OwnDefaults {
    defaults
        .iter()
        .map(|(name, value)| (*name, value.clone_ref(py)))
        .collect()
}

/// A reserved key of the literal store that keeps the Python int (or list of ints) of an int
/// param whose value PySpark cannot send, so that its getter returns it as PySpark's does.
pub(super) fn unsendable_key(jvm: &str) -> String {
    format!("\u{0}unsendable:{jvm}")
}

/// The Python value of the literal of a set param of `params`, by its JVM name `jvm`.
pub(super) fn literal_value(
    py: Python<'_>,
    params: &Params,
    jvm: &str,
) -> PyResult<Option<Py<PyAny>>> {
    use spark_connect_proto::expression::ExprType;
    if let Some(expr) = params.get_param(&unsendable_key(jvm)) {
        if let Some(ExprType::Literal(literal)) = &expr.expr_type {
            if let Ok(spark_connect::ml::MlValue::String(text)) =
                spark_connect::ml::MlValue::from_literal(literal)
            {
                return Ok(Some(
                    py.import("ast")?
                        .getattr("literal_eval")?
                        .call1((text,))?
                        .unbind(),
                ));
            }
        }
    }
    let Some(expr) = params.get_param(jvm) else {
        return Ok(None);
    };
    let Some(ExprType::Literal(literal)) = &expr.expr_type else {
        return Ok(None);
    };
    let value = spark_connect::ml::MlValue::from_literal(literal).to_pyerr()?;
    Ok(Some(super::object::ml_value_to_py(py, &value)?))
}

impl<T: Native + PyClass<Frozen = False>> ParamsHolder for T {
    fn set_value(&self, py: Python<'_>, name: &str) -> PyResult<Option<Py<PyAny>>> {
        if let Some(raw) = self.raw_params() {
            if let Some((_, value)) = raw.iter().find(|(n, _)| *n == name) {
                return Ok(Some(value.clone_ref(py)));
            }
        }
        let Some(name) = class_param_name(name) else {
            return Ok(None);
        };
        literal_value(py, self.params(), T::jvm_name(name))
    }

    fn put(&mut self, _py: Python<'_>, name: &'static str, value: Converted) -> PyResult<()> {
        apply(self, name, value);
        Ok(())
    }

    fn own_default(&self, py: Python<'_>, name: &str) -> Option<Py<PyAny>> {
        own_default_of(py, self.own_defaults()?, name)
    }

    fn put_own_default(&mut self, name: &'static str, value: Py<PyAny>) -> bool {
        match self.own_defaults_mut() {
            Some(defaults) => {
                put_own_default_in(defaults, name, value);
                true
            }
            None => false,
        }
    }
}

/// The `&'static str` of the param name `name` of any official class.
fn class_param_name(name: &str) -> Option<&'static str> {
    all_params()
        .values()
        .flat_map(|params| params.iter())
        .find(|p| p.name == name)
        .map(|p| p.name.as_str())
}

/// The Param object of `meta` of `slf`, as PySpark's `Param(parent, name, doc, typeConverter)`
/// makes it of a parent of the uid of `slf`. (`Param` takes an `Identifiable` parent, of which it
/// keeps only the uid: it is made of PySpark's dummy parent, and given the uid of `slf`.)
fn param_object<'py>(slf: &Bound<'py, PyAny>, meta: &ParamMeta) -> PyResult<Bound<'py, PyAny>> {
    let py = slf.py();
    let module = py.import("pyspark.ml.param")?;
    let converter = module
        .getattr("TypeConverters")?
        .getattr(meta.converter.as_str())?;
    let dummy = module.getattr("Params")?.call_method0("_dummy")?;
    let param = module.getattr("Param")?.call1((
        dummy,
        meta.name.as_str(),
        meta.doc.as_str(),
        converter,
    ))?;
    param.setattr("parent", slf.getattr("uid")?)?;
    Ok(param)
}

/// The meta of the param `param` of `slf`, as PySpark's `_resolveParam`: of a Param of `slf`
/// (`_shouldOwn`), or of its name (`getParam`).
fn resolve<'py>(
    slf: &Bound<'py, PyAny>,
    param: &Bound<'py, PyAny>,
) -> PyResult<&'static ParamMeta> {
    let py = slf.py();
    let param_class = py.import("pyspark.ml.param")?.getattr("Param")?;
    if param.is_instance(&param_class)? {
        return should_own(slf, param);
    }
    if param.is_instance_of::<pyo3::types::PyString>() {
        return get_param_meta(slf, &param.extract::<String>()?);
    }
    Err(PyTypeError::new_err(format!(
        "Cannot resolve {} as a param.",
        param.repr()?
    )))
}

/// PySpark's `_shouldOwn`: the meta of the Param `param`, which must be one of `slf` (of its uid
/// and of a param that it has), else PySpark's ValueError.
fn should_own<'py>(
    slf: &Bound<'py, PyAny>,
    param: &Bound<'py, PyAny>,
) -> PyResult<&'static ParamMeta> {
    let parent: String = param.getattr("parent")?.extract()?;
    let name: String = param.getattr("name")?.extract()?;
    let uid: String = slf.getattr("uid")?.extract()?;
    let meta = params_of(slf)?.iter().find(|p| p.name == name);
    match meta {
        Some(meta) if parent == uid => Ok(meta),
        _ => Err(PyValueError::new_err(format!(
            "Param {} does not belong to {}.",
            param.repr()?,
            slf.repr()?
        ))),
    }
}

/// PySpark's `getParam` of a name (`getattr(self, paramName)`): the meta of the param of that name;
/// the AttributeError of an attribute that `slf` does not have, and PySpark's ValueError of one
/// that is not a Param.
fn get_param_meta(slf: &Bound<'_, PyAny>, name: &str) -> PyResult<&'static ParamMeta> {
    if let Some(meta) = params_of(slf)?.iter().find(|p| p.name == name) {
        return Ok(meta);
    }
    if !slf.hasattr(name)? {
        // As the AttributeError of `getattr` of a Python class, of its name.
        return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
            "'{}' object has no attribute '{name}'",
            slf.get_type().name()?
        )));
    }
    Err(PyValueError::new_err(format!(
        "Cannot find param with name {name}."
    )))
}

/// `params`: the Param objects of `slf`, in PySpark's order (by name).
pub(super) fn params<'py>(slf: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    params_of(slf)?
        .iter()
        .map(|meta| param_object(slf, meta))
        .collect()
}

/// The Param object named `name` of `slf` (the Param attribute of the class).
pub(super) fn param<'py>(slf: &Bound<'py, PyAny>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    param_object(slf, get_param_meta(slf, name)?)
}

/// `getParam(paramName)`.
pub(super) fn get_param<'py>(
    slf: &Bound<'py, PyAny>,
    name: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    // As `getattr(self, paramName)`, of a name only.
    let name: String = name.extract()?;
    param_object(slf, get_param_meta(slf, &name)?)
}

/// `hasParam(paramName)`, of a name only, as PySpark.
pub(super) fn has_param(slf: &Bound<'_, PyAny>, name: &Bound<'_, PyAny>) -> PyResult<bool> {
    if !name.is_instance_of::<pyo3::types::PyString>() {
        return Err(PyTypeError::new_err(
            "hasParam(): paramName must be a string",
        ));
    }
    let name: String = name.extract()?;
    Ok(params_of(slf)?.iter().any(|p| p.name == name))
}

fn set_value_of<T: ParamsHolder>(slf: &Bound<'_, T>, name: &str) -> PyResult<Option<Py<PyAny>>> {
    slf.try_borrow()?.set_value(slf.py(), name)
}

/// `isSet(param)`.
pub(super) fn is_set<T: ParamsHolder>(
    slf: &Bound<'_, T>,
    param: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let meta = resolve(slf.as_any(), param)?;
    Ok(set_value_of(slf, &meta.name)?.is_some())
}

/// The default of the class of `slf` of `default` (of `ml_params.json`), as PySpark derives it of
/// the instance where it does: `{"uid_suffix": s}` is the uid of `slf` and `s` (`HasOutputCol`), and
/// `{"class_name_hash": true}` the `hash` of the name of the class of `slf` (`HasSeed`).
fn class_default(slf: &Bound<'_, PyAny>, default: &serde_json::Value) -> PyResult<Py<PyAny>> {
    let py = slf.py();
    if let Some(suffix) = default
        .get("uid_suffix")
        .and_then(serde_json::Value::as_str)
    {
        let uid: String = slf.getattr("uid")?.extract()?;
        return Ok(PyString::new(py, &format!("{uid}{suffix}"))
            .into_any()
            .unbind());
    }
    if default.get("class_name_hash").is_some() {
        let name = slf.get_type().getattr("__name__")?;
        return Ok(py
            .import("builtins")?
            .getattr("hash")?
            .call1((name,))?
            .unbind());
    }
    default_to_py(py, default)
}

/// The default of the param `meta` of `slf`, as `_defaultParamMap` holds it: the one that `slf`
/// holds itself, else the one of its class, if any.
fn default_value<T: ParamsHolder>(
    slf: &Bound<'_, T>,
    meta: &ParamMeta,
) -> PyResult<Option<Py<PyAny>>> {
    if let Some(value) = slf.try_borrow()?.own_default(slf.py(), &meta.name) {
        return Ok(Some(value));
    }
    meta.default
        .as_ref()
        .map(|default| class_default(slf.as_any(), default))
        .transpose()
}

/// The defaults of `slf`, of its params that have one, as `_defaultParamMap`.
fn defaults_of<T: ParamsHolder>(slf: &Bound<'_, T>) -> PyResult<OwnDefaults> {
    let mut defaults = Vec::new();
    for meta in params_of(slf.as_any())? {
        if let Some(value) = default_value(slf, meta)? {
            defaults.push((meta.name.as_str(), value));
        }
    }
    Ok(defaults)
}

/// The defaults of the estimator `slf` that a fit copies to its model, as `_copyValues` copies them
/// (see [`copy_defaults`]).
pub(super) fn fit_defaults<T: ParamsHolder>(slf: &Bound<'_, T>) -> PyResult<OwnDefaults> {
    defaults_of(slf)
}

/// Give the model `model`, of the class `M`, the defaults `defaults` of its estimator of the params
/// that it has, as `_copyValues` copies them to the model of a fit.
pub(super) fn copy_defaults<M: ParamsHolder + PyTypeInfo>(
    py: Python<'_>,
    defaults: OwnDefaults,
    model: &mut M,
) -> PyResult<()> {
    let class = py.get_type::<M>().name()?.to_string();
    let params = all_params()
        .get(&class)
        .map(Vec::as_slice)
        .unwrap_or_default();
    for (name, value) in defaults {
        if params.iter().any(|p| p.name == name) {
            model.put_own_default(name, value);
        }
    }
    Ok(())
}

/// The defaults that an object of the class `class` derives from the uid that it is made of, as a
/// loaded one keeps them after `_resetUid`: of a uid of PySpark's `Identifiable._randomUID` (the
/// name of the class and the last 12 hex digits of a uuid4).
pub(super) fn uid_defaults(py: Python<'_>, class: &str) -> PyResult<OwnDefaults> {
    let Some(params) = all_params().get(class) else {
        return Ok(Vec::new());
    };
    let mut defaults = Vec::new();
    for meta in params {
        let Some(suffix) = meta
            .default
            .as_ref()
            .and_then(|d| d.get("uid_suffix"))
            .and_then(serde_json::Value::as_str)
        else {
            continue;
        };
        let hex: String = py
            .import("uuid")?
            .call_method0("uuid4")?
            .getattr("hex")?
            .extract()?;
        let uid = format!("{class}_{}", &hex[hex.len() - 12..]);
        defaults.push((
            meta.name.as_str(),
            PyString::new(py, &format!("{uid}{suffix}"))
                .into_any()
                .unbind(),
        ));
    }
    Ok(defaults)
}

/// `hasDefault(param)`.
pub(super) fn has_default<T: ParamsHolder>(
    slf: &Bound<'_, T>,
    param: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let meta = resolve(slf.as_any(), param)?;
    Ok(meta.default.is_some()
        || slf
            .try_borrow()?
            .own_default(slf.py(), &meta.name)
            .is_some())
}

/// `isDefined(param)`.
pub(super) fn is_defined<T: ParamsHolder>(
    slf: &Bound<'_, T>,
    param: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    Ok(has_default(slf, param)? || is_set(slf, param)?)
}

/// `getOrDefault(param)`: the set value, else the default, else PySpark's KeyError of the Param.
pub(super) fn get_or_default<T: ParamsHolder>(
    slf: &Bound<'_, T>,
    param: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let meta = resolve(slf.as_any(), param)?;
    get_or_default_meta(slf, meta)
}

fn get_or_default_meta<T: ParamsHolder>(
    slf: &Bound<'_, T>,
    meta: &ParamMeta,
) -> PyResult<Py<PyAny>> {
    if let Some(value) = set_value_of(slf, &meta.name)? {
        return Ok(value);
    }
    match default_value(slf, meta)? {
        Some(default) => Ok(default),
        None => Err(pyo3::exceptions::PyKeyError::new_err((param_object(
            slf.as_any(),
            meta,
        )?
        .unbind(),))),
    }
}

/// The getter of the param `name` (`getMaxIter`), as PySpark's `getOrDefault(self.maxIter)`.
pub(super) fn getter<T: ParamsHolder>(slf: &Bound<'_, T>, name: &str) -> PyResult<Py<PyAny>> {
    get_or_default_meta(slf, get_param_meta(slf.as_any(), name)?)
}

/// `explainParam(param)`, as PySpark formats it.
pub(super) fn explain_param<T: ParamsHolder>(
    slf: &Bound<'_, T>,
    param: &Bound<'_, PyAny>,
) -> PyResult<String> {
    let meta = resolve(slf.as_any(), param)?;
    explain(slf, meta)
}

fn explain<T: ParamsHolder>(slf: &Bound<'_, T>, meta: &ParamMeta) -> PyResult<String> {
    let py = slf.py();
    let set = set_value_of(slf, &meta.name)?;
    let default = default_value(slf, meta)?;
    let mut values = Vec::new();
    if default.is_some() || set.is_some() {
        if let Some(default) = default {
            values.push(format!("default: {}", default.bind(py).str()?));
        }
        if let Some(value) = set {
            values.push(format!("current: {}", value.bind(py).str()?));
        }
    } else {
        values.push("undefined".to_string());
    }
    Ok(format!(
        "{}: {} ({})",
        meta.name,
        meta.doc,
        values.join(", ")
    ))
}

/// `explainParams()`: the explanation of each param, by line.
pub(super) fn explain_params<T: ParamsHolder>(slf: &Bound<'_, T>) -> PyResult<String> {
    Ok(params_of(slf.as_any())?
        .iter()
        .map(|meta| explain(slf, meta))
        .collect::<PyResult<Vec<_>>>()?
        .join("\n"))
}

/// `extractParamMap(extra=None)`: the defaults, the set values, and `extra`, by Param.
pub(super) fn extract_param_map<'py, T: ParamsHolder>(
    slf: &Bound<'py, T>,
    extra: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let py = slf.py();
    let map = PyDict::new(py);
    for meta in params_of(slf.as_any())? {
        if let Some(default) = default_value(slf, meta)? {
            map.set_item(param_object(slf.as_any(), meta)?, default)?;
        }
    }
    for meta in params_of(slf.as_any())? {
        if let Some(value) = set_value_of(slf, &meta.name)? {
            map.set_item(param_object(slf.as_any(), meta)?, value)?;
        }
    }
    if let Some(extra) = extra {
        if !extra.is_none() {
            map.call_method1("update", (extra,))?;
        }
    }
    Ok(map)
}

/// `set(param, value)`, as PySpark's `Params.set`: a Param of `slf`, whose converter converts the
/// value, of which a ValueError becomes PySpark's ValueError of the param and a TypeError is raised
/// as it is.
pub(super) fn set<T: ParamsHolder>(
    slf: &Bound<'_, T>,
    param: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let py = slf.py();
    let meta = should_own(slf.as_any(), param)?;
    let converted = param
        .getattr("typeConverter")?
        .call1((value,))
        .map_err(|error| {
            if error.is_instance_of::<PyValueError>(py) {
                PyValueError::new_err(format!(
                    "Invalid param value given for param \"{}\". {}",
                    meta.name,
                    error.value(py)
                ))
            } else {
                error
            }
        })?;
    let kind = kind_of_converter(&meta.converter).unwrap_or(Kind::Identity);
    // Of a value that the converter has converted already, so that it does not fail again.
    let converted = convert(&meta.name, kind, &converted)?;
    slf.try_borrow_mut()?.put(py, meta.name.as_str(), converted)
}

/// Set the param `name` of `slf` to `value`, as PySpark's `_set(name=value)`: of the converter of
/// its class, whose TypeError is PySpark's of `_set` (see [`convert`]).
pub(super) fn set_by_name<T: ParamsHolder>(
    slf: &Bound<'_, T>,
    name: &str,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let meta = get_param_meta(slf.as_any(), name)?;
    let kind = kind_of_converter(&meta.converter).unwrap_or(Kind::Identity);
    let converted = convert(&meta.name, kind, value)?;
    slf.try_borrow_mut()?
        .put(slf.py(), meta.name.as_str(), converted)
}

/// The values of the params of `extra`, a param map of `slf` (`fit(dataset, params)`, `copy`),
/// by name, as PySpark's `_copyValues` sets the ones of the params that `slf` has.
pub(super) fn extra_values<'py>(
    slf: &Bound<'py, PyAny>,
    extra: &Bound<'py, PyAny>,
) -> PyResult<Vec<(&'static str, Bound<'py, PyAny>)>> {
    let py = slf.py();
    let param_class = py.import("pyspark.ml.param")?.getattr("Param")?;
    let uid: String = slf.getattr("uid")?.extract()?;
    let extra = extra.cast::<PyDict>().map_err(|_| {
        PyTypeError::new_err(format!(
            "Expecting a dict, but received an object of type {}.",
            extra
                .get_type()
                .repr()
                .map(|r| r.to_string())
                .unwrap_or_default()
        ))
    })?;
    let mut values = Vec::new();
    for (key, value) in extra.iter() {
        if !key.is_instance(&param_class)? {
            return Err(PyTypeError::new_err(format!(
                "Expecting a valid instance of Param, but received: {}",
                key.str()?
            )));
        }
        let parent: String = key.getattr("parent")?.extract()?;
        let name: String = key.getattr("name")?.extract()?;
        if parent != uid {
            continue;
        }
        if let Some(meta) = params_of(slf)?.iter().find(|p| p.name == name) {
            values.push((meta.name.as_str(), value));
        }
    }
    Ok(values)
}

/// What the Params API needs of a native class beyond its params: a duplicate of the same uid and
/// params (`copy`), and the setting of the uid that a `Read` returns.
pub(super) trait NativeInstance: PyClass + Sized {
    /// A new Python object of the same uid and params.
    fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>>;
    fn uid(&self) -> String;
    fn set_uid(&mut self, uid: &str);
}

/// `copy(extra=None)`, as PySpark's `Params.copy` and `_copyValues`: a new object of the same uid,
/// params and defaults, whose params of `extra` (a param map of `slf`) are set as `_set` sets them.
pub(super) fn copy<'py, T: ParamsHolder + NativeInstance>(
    slf: &Bound<'py, T>,
    extra: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, T>> {
    let py = slf.py();
    let values = match extra {
        Some(extra) if !extra.is_none() => extra_values(slf.as_any(), extra)?,
        _ => Vec::new(),
    };
    let that = slf.try_borrow()?.duplicate(py)?.into_bound(py);
    for (name, value) in values {
        set_by_name(&that, name, &value)?;
    }
    Ok(that)
}

/// The error that PySpark's serialization of the params of `slf` raises, if any, as `snapshot`
/// raises it: `VALUE_NOT_BETWEEN` of an int that PySpark cannot send, and
/// `CANNOT_INFER_ARRAY_ELEMENT_TYPE` of a list without elements.
fn serialization_error<T: Native>(py: Python<'_>, this: &T) -> Option<PyErr> {
    if !this.unsendable().is_empty() {
        return Some(crate::functions::value_not_between(py));
    }
    if this.params().has_empty_array() {
        return Some(crate::column::raise_pyspark(
            py,
            "PySparkTypeError",
            "CANNOT_INFER_ARRAY_ELEMENT_TYPE",
            &[],
        ));
    }
    None
}

/// The writer of the operator `slf` of the JVM class `jvm` and type `op_type`, as
/// `RemoteMLWriter` writes an estimator, transformer or evaluator: of its uid and params, whose
/// serialization error, if any, its `save` raises, as PySpark's `serialize_ml_params` does then.
pub(super) fn operator_writer<T: Native + PyClass<Frozen = False>>(
    slf: &Bound<'_, T>,
    jvm: &str,
    op_type: spark_connect::ml::OperatorType,
) -> PyResult<super::object::PyMLWriter> {
    let py = slf.py();
    let uid: String = slf.as_any().getattr("uid")?.extract()?;
    let this = slf.try_borrow()?;
    let operator = spark_connect::ml::MlOperator::with_uid(jvm, &uid, op_type);
    Ok(
        super::object::PyMLWriter::new(super::object::WriteTarget::Operator(
            operator,
            this.params().clone(),
        ))
        .with_error(serialization_error(py, &*this)),
    )
}

/// The instance of `T` of what a `Read` returns, as `RemoteMLReader.loadInstance`: the default
/// instance of the class (`py_type()`), of the uid and of the params that the server read, which it
/// sets as `_set` does.
pub(super) fn from_read<T: ParamsHolder + NativeInstance>(
    py: Python<'_>,
    read: spark_connect::ml::ReadResult,
) -> PyResult<Py<PyAny>> {
    let instance = py.get_type::<T>().call0()?.cast_into::<T>()?;
    // The defaults of the uid that the instance was made of, which `_resetUid` keeps.
    for meta in params_of(instance.as_any())? {
        if meta
            .default
            .as_ref()
            .is_some_and(|d| d.get("uid_suffix").is_some())
        {
            if let Some(value) = default_value(&instance, meta)? {
                instance
                    .try_borrow_mut()?
                    .put_own_default(meta.name.as_str(), value);
            }
        }
    }
    instance.try_borrow_mut()?.set_uid(&read.uid);
    let proto = read.params.to_proto();
    let mut names: Vec<&String> = proto.params.keys().collect();
    names.sort();
    for name in names {
        let value = spark_connect::ml::MlValue::from_literal(&proto.params[name]).to_pyerr()?;
        let value = super::object::ml_value_to_py(py, &value)?;
        if params_of(instance.as_any())?
            .iter()
            .any(|p| &p.name == name)
        {
            set_by_name(&instance, name, value.bind(py))?;
        }
    }
    Ok(instance.into_any().unbind())
}

/// PySpark's `_FitMultipleIterator` of `fitMultiple(dataset, paramMaps)`: each `next` fits the
/// copy of the estimator of the next param map, and returns its index and model, until none are
/// left (`StopIteration("No models remaining.")`). It may be shared by threads, as PySpark's.
#[pyclass(name = "_FitMultipleIterator", module = "pyspark.ml.base")]
pub struct PyFitMultipleIterator {
    fit: Py<PyAny>,
    dataset: Py<PyAny>,
    maps: Vec<Py<PyAny>>,
    counter: std::sync::Mutex<usize>,
}

#[pymethods]
impl PyFitMultipleIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<(usize, Py<PyAny>)> {
        let index = {
            let mut counter = self
                .counter
                .lock()
                .map_err(|_| PyValueError::new_err("the iterator is poisoned"))?;
            let index = *counter;
            *counter += 1;
            index
        };
        if index >= self.maps.len() {
            return Err(pyo3::exceptions::PyStopIteration::new_err(
                "No models remaining.",
            ));
        }
        let model = self
            .fit
            .bind(py)
            .call1((self.dataset.bind(py), self.maps[index].bind(py)))?;
        Ok((index, model.unbind()))
    }
}

/// `fitMultiple(dataset, paramMaps)`, as PySpark's `Estimator.fitMultiple`: the iterator of the
/// fits of a copy of the estimator, one per param map.
pub(super) fn fit_multiple<'py>(
    slf: &Bound<'py, PyAny>,
    dataset: &Bound<'py, PyAny>,
    param_maps: &Bound<'py, PyAny>,
) -> PyResult<PyFitMultipleIterator> {
    let estimator = slf.call_method0("copy")?;
    let maps = param_maps
        .try_iter()?
        .map(|m| m.map(Bound::unbind))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyFitMultipleIterator {
        fit: estimator.getattr("fit")?.unbind(),
        dataset: dataset.clone().unbind(),
        maps,
        counter: std::sync::Mutex::new(0),
    })
}
