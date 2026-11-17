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
    params_of_type(slf.get_type().as_any())
}

/// The AttributeError of the attribute `name` that `obj` does not have, as CPython raises it of
/// an object of a Python class: of the name of the class (CPython names a native type by its
/// module path too, its `tp_name`), and of the name and the object, of which a traceback suggests a
/// close name.
pub(crate) fn no_attribute(obj: &Bound<'_, PyAny>, name: &Bound<'_, PyAny>) -> PyErr {
    let py = obj.py();
    let make = || -> PyResult<PyErr> {
        let class = obj.get_type().name()?;
        let message = format!("'{class}' object has no attribute '{}'", name.str()?);
        let kwargs = PyDict::new(py);
        kwargs.set_item("name", name)?;
        kwargs.set_item("obj", obj)?;
        let error = py
            .import("builtins")?
            .getattr("AttributeError")?
            .call((message,), Some(&kwargs))?;
        Ok(PyErr::from_value(error))
    };
    make().unwrap_or_else(|e| e)
}

/// The params of the class `cls`, as [`params_of`] of its instances.
fn params_of_type(cls: &Bound<'_, PyAny>) -> PyResult<&'static [ParamMeta]> {
    for class in cls.getattr("__mro__")?.try_iter()? {
        let name = class?.getattr("__name__")?.extract::<String>()?;
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

    /// Unset the param `name`, as `Params.clear` of a Python class of PySpark does (the classes
    /// of the server's objects do not, as `JavaParams.clear` over Spark Connect).
    #[allow(dead_code)] // The Python classes of PySpark (a Pipeline, the validators) unset.
    fn unset(&mut self, name: &'static str) -> PyResult<()> {
        let _ = name;
        Ok(())
    }

    /// The `params` walk of PySpark's `Params.params`, which `explainParams` and `copy` run too:
    /// it reads every attribute of the object but its properties, so the members of
    /// `functools.cached_property` of a model (`trees`), which a model of no server-side object
    /// raises its AssertionError of.
    fn walk(&self, py: Python<'_>) -> PyResult<()> {
        let _ = py;
        Ok(())
    }
}

/// The defaults that an object holds itself, by param, where PySpark's `_defaultParamMap` has one
/// that is not the default of its class: a fitted model has the defaults of its estimator, which
/// `_copyValues` copies to it (a seed of the name of the estimator's class, or the
/// `coldStartStrategy` that ALS has and ALSModel does not); and a loaded object has the defaults
/// that it derived from the uid that it was made of, which `_resetUid` keeps (an `outputCol` of
/// that uid).
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

/// The error that PySpark's serialization of the params raises of the value that `params` keeps
/// of the param `jvm`, which it cannot send (see [`unsendable_key`]): the error of
/// `LiteralExpression._from_value` of the value, as the values of the ML fetches raise it
/// (`functions::to_value_column`). Of an int beyond a long, it is `VALUE_NOT_BETWEEN`; of a list of
/// ints, the bare `AssertionError` of an element out of the range of the type of the first, the
/// `VALUE_NOT_BETWEEN` of a first element beyond a long, or the `CANNOT_INFER_ARRAY_ELEMENT_TYPE`
/// of no element.
pub(super) fn unsendable_error(py: Python<'_>, params: &Params, jvm: &str) -> PyErr {
    let error = || -> PyResult<PyErr> {
        let Some(value) = literal_value(py, params, jvm)? else {
            return Ok(crate::functions::value_not_between(py));
        };
        Ok(match crate::functions::to_value_column(value.bind(py)) {
            Err(error) => error,
            Ok(_) => crate::functions::value_not_between(py),
        })
    };
    error().unwrap_or_else(|error| error)
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

    /// Unset the param `name`, as `del self._paramMap[param]` of PySpark's `_clear` does.
    fn unset(&mut self, name: &'static str) -> PyResult<()> {
        self.unsendable_mut().retain(|unsent| *unsent != name);
        if let Some(raw) = self.raw_params_mut() {
            raw.retain(|(key, _)| *key != name);
        }
        let jvm = T::jvm_name(name);
        let params = std::mem::take(self.params_mut())
            .clear_param(&unsendable_key(jvm))
            .clear_param(jvm);
        *self.params_mut() = params;
        Ok(())
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
    let param = class_param_object(slf.py(), meta)?;
    param.setattr("parent", slf.getattr("uid")?)?;
    Ok(param)
}

/// The Param of `meta` of a class, as PySpark's class attribute is:
/// `Param(Params._dummy(), name, doc, typeConverter)`, of the parent "undefined".
fn class_param_object<'py>(py: Python<'py>, meta: &ParamMeta) -> PyResult<Bound<'py, PyAny>> {
    let module = py.import("pyspark.ml.param")?;
    let converter = module
        .getattr("TypeConverters")?
        .getattr(meta.converter.as_str())?;
    let dummy = module.getattr("Params")?.call_method0("_dummy")?;
    module
        .getattr("Param")?
        .call1((dummy, meta.name.as_str(), meta.doc.as_str(), converter))
}

/// The Param attribute of a native class, as the attribute of PySpark's class is: of the class
/// (`LinearSVC.maxIter`), the Param of the parent "undefined" of the class's params, the same
/// object on every access; of an instance, the Param of its uid (see [`param`]).
#[pyclass(name = "_ParamAttribute", module = "pyspark.ml.connect", frozen)]
pub struct PyParamAttribute {
    name: &'static str,
    class_param: OnceLock<Py<PyAny>>,
}

impl PyParamAttribute {
    pub(super) fn new(name: &'static str) -> Self {
        Self {
            name,
            class_param: OnceLock::new(),
        }
    }
}

#[pymethods]
impl PyParamAttribute {
    fn __get__(
        &self,
        py: Python<'_>,
        obj: Option<&Bound<'_, PyAny>>,
        owner: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        if let Some(obj) = obj.filter(|obj| !obj.is_none()) {
            return Ok(param(obj, self.name)?.unbind());
        }
        if let Some(param) = self.class_param.get() {
            return Ok(param.clone_ref(py));
        }
        let owner = owner.ok_or_else(|| {
            PyTypeError::new_err("the Param attribute of a class is read of the class")
        })?;
        let meta = params_of_type(owner)?
            .iter()
            .find(|p| p.name == self.name)
            .ok_or_else(|| {
                PyValueError::new_err(format!("Cannot find param with name {}.", self.name))
            })?;
        let param = class_param_object(py, meta)?.unbind();
        Ok(self.class_param.get_or_init(|| param).clone_ref(py))
    }
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
    // As `getattr(self, paramName)`, of a name only, and of its TypeError of another object.
    if !name.is_instance_of::<pyo3::types::PyString>() {
        return Err(PyTypeError::new_err(format!(
            "attribute name must be string, not '{}'",
            name.get_type().name()?
        )));
    }
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
/// the instance where it does: `{"uid_suffix": s}` is the uid of `slf` and `s` (`HasOutputCol`),
/// and `{"class_name_hash": true}` the `hash` of the name of the class of `slf` (`HasSeed`).
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
    if let Some(fetch) = default.get("server_default") {
        // A default of the server that the object does not hold (PySpark's constructor fetches
        // it, which sets it in the param maps that the native object takes): fetched now.
        return server_default(py, fetch);
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

/// The value of the ML helper's attribute of `fetch` (`[attr, args...]` of a
/// `{"server_default": ...}` of ml_params.json), of the active session, as PySpark's
/// `invoke_helper_attr` reads it: its `assert` of an active session raises an AssertionError
/// without one.
fn server_default(py: Python<'_>, fetch: &serde_json::Value) -> PyResult<Py<PyAny>> {
    let parts = fetch.as_array().map(Vec::as_slice).unwrap_or_default();
    let Some((attr, args)) = parts
        .split_first()
        .and_then(|(attr, args)| Some((attr.as_str()?, args)))
    else {
        return Err(PyValueError::new_err(format!(
            "unsupported server default of ml_params.json: {fetch}"
        )));
    };
    let session = super::object::session_of(py, None)?;
    let args = args
        .iter()
        .map(|arg| {
            arg.as_str()
                .map(|arg| {
                    spark_connect::ml::FetchArg::Value(spark_connect::ml::MlValue::String(
                        arg.to_string(),
                    ))
                })
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "unsupported server default of ml_params.json: {fetch}"
                    ))
                })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let target = spark_connect::ml::MlTarget::helper(&session);
    let output = interruptible(py, &session, || {
        spark_connect::ml::fetch(&target, attr, args).map_err(Box::new)
    })?;
    super::object::fetch_output_to_py(py, output)
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

/// The default of the param `name` of `slf`, as `_defaultParamMap[param]`: the one that `slf`
/// holds itself, else the one of its class, if any.
pub(super) fn default_of<T: ParamsHolder>(
    slf: &Bound<'_, T>,
    name: &str,
) -> PyResult<Option<Py<PyAny>>> {
    default_value(slf, get_param_meta(slf.as_any(), name)?)
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

/// A uid of the class `class`, as PySpark's `Identifiable._randomUID`: the name of the class and
/// the last 12 hex digits of a uuid4.
pub(super) fn random_uid(py: Python<'_>, class: &str) -> PyResult<String> {
    let hex: String = py
        .import("uuid")?
        .call_method0("uuid4")?
        .getattr("hex")?
        .extract()?;
    Ok(format!("{class}_{}", &hex[hex.len() - 12..]))
}

/// The defaults that an object of the class `class` derives from the uid that it is made of, as a
/// loaded one keeps them after `_resetUid`: of a uid of [`random_uid`].
pub(super) fn uid_defaults(py: Python<'_>, class: &str) -> PyResult<OwnDefaults> {
    uid_defaults_with(py, class, None)
}

/// The defaults that an object of the class `class` of the uid `uid` derives from it, as PySpark's
/// constructor sets them (`outputCol` of `HasOutputCol.__init__`, the uid and `__output`): of a
/// model of no server-side object, whose uid is the one that it is made of.
pub(super) fn uid_defaults_of(py: Python<'_>, class: &str, uid: &str) -> PyResult<OwnDefaults> {
    uid_defaults_with(py, class, Some(uid))
}

/// The defaults of [`uid_defaults`] of a uid of [`random_uid`] for each, else of [`uid_defaults_of`].
fn uid_defaults_with(py: Python<'_>, class: &str, uid: Option<&str>) -> PyResult<OwnDefaults> {
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
        let uid = match uid {
            Some(uid) => uid.to_string(),
            None => random_uid(py, class)?,
        };
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

/// `clear(param)`, as `Params.clear` of a Python class of PySpark: the param of `param` (a Param
/// of `slf` or its name), if set, is unset.
#[allow(dead_code)] // The Python classes of PySpark (a Pipeline, the validators) clear.
pub(super) fn clear<T: ParamsHolder>(slf: &Bound<'_, T>, param: &Bound<'_, PyAny>) -> PyResult<()> {
    let meta = resolve(slf.as_any(), param)?;
    if set_value_of(slf, &meta.name)?.is_some() {
        slf.try_borrow_mut()?.unset(meta.name.as_str())?;
    }
    Ok(())
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
    slf.try_borrow()?.walk(slf.py())?;
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

    /// Whether `copy.deepcopy` and `pickle` take the object's state: the error of an object of a
    /// server-side object (a fitted or loaded model), whose `RemoteModelRef` PySpark's cannot copy.
    fn check_picklable(&self) -> PyResult<()> {
        Ok(())
    }
}

/// The TypeError of `copy.deepcopy` and `pickle` of PySpark's object of a server-side object, of
/// the lock of its `RemoteModelRef`.
pub(super) fn lock_error() -> PyErr {
    PyTypeError::new_err("cannot pickle '_thread.lock' object")
}

/// Give `that`, a copy of `this`, the class of `this` (a Python subclass of the public class), if
/// their layouts allow it, and the attributes of `this`, shared, as `copy.copy` copies them.
pub(super) fn copy_class_and_attributes(
    this: &Bound<'_, PyAny>,
    that: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let class = this.get_type();
    if !that.get_type().is(&class) {
        // A subclass of `__slots__`, of another layout, keeps the class of the copy.
        let _ = that.setattr("__class__", &class);
    }
    if let (Ok(from), Ok(to)) = (this.getattr("__dict__"), that.getattr("__dict__")) {
        to.call_method1("update", (from,))?;
    }
    Ok(())
}

/// `copy.copy` of `slf`, as of PySpark's object: of the same uid, params and defaults (and of a
/// model, of the same server-side object, summary and cached values, see `ModelCore::duplicate`),
/// of the class of `slf` and of its attributes, shared.
pub(super) fn shallow_copy<'py, T: NativeInstance>(
    slf: &Bound<'py, T>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = slf.py();
    let that = slf.try_borrow()?.duplicate(py)?.into_bound(py).into_any();
    copy_class_and_attributes(slf.as_any(), &that)?;
    Ok(that)
}

/// The values of the params of `slf` by name, in the order of the class's params: its set values,
/// and its defaults (its own, else the class's), of which the public class makes PySpark's
/// `_paramMap` and `_defaultParamMap` of an object that this client makes (a fit, a load).
pub(super) fn native_params<'py, T: ParamsHolder>(
    slf: &Bound<'py, T>,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    let py = slf.py();
    let (set, defaults) = (PyDict::new(py), PyDict::new(py));
    for meta in params_of(slf.as_any())? {
        if let Some(value) = set_value_of(slf, &meta.name)? {
            set.set_item(&meta.name, value)?;
        }
        if let Some(value) = default_value(slf, meta)? {
            defaults.set_item(&meta.name, value)?;
        }
    }
    Ok((set, defaults))
}

/// Make the params of `slf` those of PySpark's `_paramMap` and `_defaultParamMap` of it, of the
/// values `set` and `defaults` by name, before a request that sends them: the values of `set`, the
/// others unset, and the defaults of `defaults` that the object holds of its own. The values of
/// `set` are set in its order, of `_paramMap`, in which PySpark's serialization of the params reads
/// them: the error of the first that it cannot send is the first of `_paramMap`'s (see
/// `snapshot`).
pub(super) fn sync_params<T: ParamsHolder>(
    slf: &Bound<'_, T>,
    set: &Bound<'_, PyDict>,
    defaults: &Bound<'_, PyDict>,
) -> PyResult<()> {
    let params = params_of(slf.as_any())?;
    for meta in params {
        if !set.contains(&meta.name)? {
            slf.try_borrow_mut()?.unset(meta.name.as_str())?;
        }
        if let Some(value) = defaults.get_item(&meta.name)? {
            slf.try_borrow_mut()?
                .put_own_default(meta.name.as_str(), value.unbind());
        }
    }
    for (name, value) in set.iter() {
        let name = name.extract::<String>()?;
        if params.iter().any(|meta| meta.name == name) {
            set_by_name(slf, &name, &value)?;
        }
    }
    Ok(())
}

/// The state of `slf` that `copy.deepcopy` and `pickle` take (`__getstate__`): its uid, its set
/// params and the defaults that it holds itself, by name, and its attributes.
pub(super) fn get_state<'py, T: ParamsHolder + NativeInstance>(
    slf: &Bound<'py, T>,
) -> PyResult<Bound<'py, PyDict>> {
    let py = slf.py();
    slf.try_borrow()?.check_picklable()?;
    let (params, defaults) = (PyDict::new(py), PyDict::new(py));
    {
        let this = slf.try_borrow()?;
        for meta in params_of(slf.as_any())? {
            if let Some(value) = this.set_value(py, &meta.name)? {
                params.set_item(&meta.name, value)?;
            }
            if let Some(value) = this.own_default(py, &meta.name) {
                defaults.set_item(&meta.name, value)?;
            }
        }
    }
    let state = PyDict::new(py);
    state.set_item("uid", slf.try_borrow()?.uid())?;
    state.set_item("params", params)?;
    state.set_item("defaults", defaults)?;
    if let Ok(attributes) = slf.as_any().getattr("__dict__") {
        state.set_item("__dict__", attributes.call_method0("copy")?)?;
    }
    Ok(state)
}

/// `__setstate__(state)` of `slf`, of a state of [`get_state`]: of an object of the class's
/// constructor of no arguments, as `copyreg.__newobj__` makes it.
pub(super) fn set_state<T: ParamsHolder + NativeInstance>(
    slf: &Bound<'_, T>,
    state: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let state = state.cast::<PyDict>()?;
    if let Some(uid) = state.get_item("uid")? {
        slf.try_borrow_mut()?.set_uid(&uid.extract::<String>()?);
    }
    if let Some(params) = state.get_item("params")? {
        for (name, value) in params.cast::<PyDict>()?.iter() {
            set_by_name(slf, &name.extract::<String>()?, &value)?;
        }
    }
    if let Some(defaults) = state.get_item("defaults")? {
        for (name, value) in defaults.cast::<PyDict>()?.iter() {
            let meta = get_param_meta(slf.as_any(), &name.extract::<String>()?)?;
            slf.try_borrow_mut()?
                .put_own_default(meta.name.as_str(), value.unbind());
        }
    }
    if let Some(attributes) = state.get_item("__dict__")? {
        slf.as_any()
            .getattr("__dict__")?
            .call_method1("update", (attributes,))?;
    }
    Ok(())
}

/// `copy(extra=None)`, as PySpark's `Params.copy` and `_copyValues`: a new object of the same uid,
/// params and defaults, whose params of `extra` (a param map of `slf`) are set as `_set` sets them.
pub(super) fn copy<'py, T: ParamsHolder + NativeInstance>(
    slf: &Bound<'py, T>,
    extra: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, T>> {
    let py = slf.py();
    // `_copyValues` checks `extra` before its `params` walk, which raises the AssertionError of a
    // model of no server-side object of `functools.cached_property` members (`trees`).
    let values = match extra {
        Some(extra) if !extra.is_none() => extra_values(slf.as_any(), extra)?,
        _ => Vec::new(),
    };
    slf.try_borrow()?.walk(py)?;
    let that = slf.try_borrow()?.duplicate(py)?.into_bound(py);
    for (name, value) in values {
        set_by_name(&that, name, &value)?;
    }
    Ok(that)
}

/// The error that PySpark's serialization of the params `params` raises, if any, as `snapshot`
/// raises it: of the first param of a value that PySpark cannot send, of the JVM name
/// `unsendable` ([`unsendable_error`]), and `CANNOT_INFER_ARRAY_ELEMENT_TYPE` of a list without
/// elements. It runs Python code: give it a copy of the params of an object that is not borrowed,
/// so that a setter that another thread calls meanwhile does not fail.
pub(super) fn params_error(
    py: Python<'_>,
    params: &Params,
    unsendable: Option<&str>,
) -> Option<PyErr> {
    if let Some(jvm) = unsendable {
        return Some(unsendable_error(py, params, jvm));
    }
    if params.has_empty_array() {
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
    let (params, unsendable) = {
        let this = slf.try_borrow()?;
        let unsendable = this.unsendable().first().map(|name| T::jvm_name(name));
        (this.params().clone(), unsendable)
    };
    let error = params_error(py, &params, unsendable);
    let operator = spark_connect::ml::MlOperator::with_uid(jvm, &uid, op_type);
    Ok(
        super::object::PyMLWriter::new(super::object::WriteTarget::Operator(operator, params))
            .with_error(error),
    )
}

/// The instance of `T` of what a `Read` returns, as `RemoteMLReader.loadInstance`: the default
/// instance of the class (`py_type()`), of the uid and of the params that the server read, which it
/// sets as `_set` does.
pub(super) fn from_read<T: ParamsHolder + NativeInstance>(
    py: Python<'_>,
    read: spark_connect::ml::ReadResult,
) -> PyResult<Py<PyAny>> {
    // Of the public class of `T`, as a constructor of the class that `load` is called of.
    let class = super::object::public_class::<T>(py).unwrap_or_else(|| py.get_type::<T>());
    let instance = class.call0()?.cast_into::<T>()?;
    let proto = read.params.to_proto();
    let mut names: Vec<&String> = proto.params.keys().collect();
    names.sort();
    if instance.as_any().hasattr("_resetUid")? {
        // As `RemoteMLReader.loadInstance` of PySpark's param maps of the object: its
        // `_resetUid(uid)`, which keeps the defaults of the uid that it was made of, and its
        // `_set(**params)` of the params that the server read.
        instance
            .as_any()
            .call_method1("_resetUid", (read.uid.as_str(),))?;
        let params = PyDict::new(py);
        for name in names {
            if params_of(instance.as_any())?
                .iter()
                .any(|p| &p.name == name)
            {
                let value =
                    spark_connect::ml::MlValue::from_literal(&proto.params[name]).to_pyerr()?;
                params.set_item(name, super::object::ml_value_to_py(py, &value)?)?;
            }
        }
        instance.as_any().call_method("_set", (), Some(&params))?;
        return Ok(instance.into_any().unbind());
    }
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
#[pyclass(subclass, name = "_FitMultipleIterator", module = "pyspark.ml.base")]
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
) -> PyResult<Py<PyFitMultipleIterator>> {
    let estimator = slf.call_method0("copy")?;
    let maps = param_maps
        .try_iter()?
        .map(|m| m.map(Bound::unbind))
        .collect::<PyResult<Vec<_>>>()?;
    super::object::new_py(
        slf.py(),
        PyFitMultipleIterator {
            fit: estimator.getattr("fit")?.unbind(),
            dataset: dataset.clone().unbind(),
            maps,
            counter: std::sync::Mutex::new(0),
        },
    )
}
