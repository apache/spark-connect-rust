//! PyO3 wrappers for `spark_connect::ml` (mirrors `pyspark.ml.connect`).
//!
//! Each wrapper holds a core estimator, transformer or evaluator, whose params are those that
//! Python set, as PySpark's `_paramMap`: a fit, transform or evaluate sends only them, and the
//! server uses its defaults for the others, as with PySpark. The constructors are keyword-only,
//! as PySpark's `@keyword_only` ones, and they and the setters convert each value with the
//! `TypeConverters` of the param in PySpark, and send it as PySpark sends a Python value. A
//! setter sets the param of the wrapper itself and returns the wrapper, as PySpark's
//! `Params._set` does. `fit`, `transform` and `evaluate` clone the core object under a short
//! borrow, so that another thread may call a setter while they wait for the server. They take
//! a Spark DataFrame, matching the dataset contract of standard `pyspark.ml` over Spark Connect.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pyclass::boolean_struct::False;
use pyo3::types::{PyDict, PyList, PyTuple, PyType};
use pyo3::{IntoPyObjectExt, PyClass};
use spark_connect::ml::{
    AFTSurvivalRegression, BisectingKMeans, Classifier, DecisionTreeClassifier,
    DecisionTreeRegressor, Estimator, Evaluator, FMClassifier, FMRegressor, GBTClassifier,
    GBTRegressor, GaussianMixture, GeneralizedLinearRegression, IsotonicRegression, KMeans,
    LinearRegression, LinearSVC, LogisticRegression, MlMatrix, MlVector, Model,
    MultilayerPerceptronClassifier, NaiveBayes, OneVsRest, Params, PowerIterationClustering,
    RandomForestClassifier, RandomForestRegressor, Transformer, LDA,
};
use spark_connect::DataFrame;

pub(crate) mod stat;

#[macro_use]
pub(crate) mod macros;
pub(crate) mod object;
pub(crate) mod params_api;

// The typed models and summaries of each family, in the files of the areas that complete them.
pub(crate) mod classification_api;
pub(crate) mod clustering_api;
pub(crate) mod pipeline_api;
pub(crate) mod regression_api;
pub(crate) mod tuning_api;

pub(crate) mod evaluation;
pub use evaluation::{
    PyBinaryClassificationEvaluator, PyClusteringEvaluator, PyMulticlassClassificationEvaluator,
    PyMultilabelClassificationEvaluator, PyRankingEvaluator, PyRegressionEvaluator,
};

pub(crate) mod feature;
pub(crate) mod feature_api;
pub(crate) mod fpm;

pub(crate) mod recommendation;

use crate::dataframe::PyDataFrame;
use crate::errors::ResultExt;
use crate::interrupt::interruptible;

/// The kind of a param: the method of PySpark's `TypeConverters` that converts its value, and
/// the literal that PySpark sends of the converted value.
#[derive(Clone, Copy)]
enum Kind {
    /// `toInt`: an `Integer` literal when the int fits 32 bits, else a `Long`.
    Int,
    /// `toInt` of a param that PySpark compares but never sends, OneVsRest's `parallelism`: an
    /// int beyond 64 bits becomes the bound of its sign, rather than an int that PySpark cannot
    /// send.
    ClampedInt,
    /// `toFloat`: a `Double` literal.
    Float,
    /// `toString`: a `String` literal.
    Str,
    /// `toBoolean`: a `Boolean` literal.
    Bool,
    /// `toListString`: an array literal of strings.
    StrList,
    /// `toListFloat`: an array literal of doubles.
    FloatList,
    IntList,
    FloatListList,
    Identity,
    Vector,
    /// `toMatrix`: the MatrixUDT struct literal of a `DenseMatrix` or a `SparseMatrix`, as the
    /// `serialize_param` of `pyspark.ml.connect.serialize` sends it (see [`MlMatrix`]).
    Matrix,
}

impl Kind {
    fn converter(self) -> &'static str {
        match self {
            Kind::Int | Kind::ClampedInt => "toInt",
            Kind::Float => "toFloat",
            Kind::Str => "toString",
            Kind::Bool => "toBoolean",
            Kind::StrList => "toListString",
            Kind::FloatList => "toListFloat",
            Kind::IntList => "toListInt",
            Kind::Vector => "toVector",
            Kind::FloatListList => "toListListFloat",
            Kind::Identity => "identity",
            Kind::Matrix => "toMatrix",
        }
    }
}

/// `pyspark.ml.param.TypeConverters`, which converts the values of the params of PySpark.
static TYPE_CONVERTERS: pyo3::sync::PyOnceLock<Py<PyType>> = pyo3::sync::PyOnceLock::new();

/// The value of a param, as PySpark's `Params._set` converts it.
enum Converted {
    /// None, which PySpark sends as a null literal: the server rejects a null number (a
    /// NullPointerException), and reads a null bootstrap as false.
    Null,
    Int(i64),
    /// An int beyond a long (or a list of ints with one), which PySpark's `LiteralExpression`
    /// cannot send, of the `repr` of the converted value, which its getter returns.
    Unsendable(String),
    Float(f64),
    Str(String),
    Bool(bool),
    StrList(Vec<String>),
    FloatList(Vec<f64>),
    IntList(Vec<i64>),
    Vector(MlVector),
    FloatListList(Vec<Vec<f64>>),
    Raw(Py<PyAny>),
    Matrix(MlMatrix),
}

/// Convert `value` of the param `name` with its `TypeConverters` method, raising PySpark's
/// TypeError of `Params._set` for a value that it cannot convert.
fn convert(name: &str, kind: Kind, value: &Bound<'_, PyAny>) -> PyResult<Converted> {
    if value.is_none() {
        return Ok(Converted::Null);
    }
    let py = value.py();
    let converters = TYPE_CONVERTERS.import(py, "pyspark.ml.param", "TypeConverters")?;
    let converted = converters
        .call_method1(kind.converter(), (value,))
        .map_err(|error| {
            if error.is_instance_of::<PyTypeError>(py) {
                PyTypeError::new_err(format!(
                    "Invalid param value given for param \"{name}\". {}",
                    error.value(py)
                ))
            } else {
                error
            }
        })?;
    Ok(match kind {
        Kind::Int => match converted.extract::<i64>() {
            Ok(value) => Converted::Int(value),
            Err(_) => Converted::Unsendable(converted.repr()?.to_string()),
        },
        Kind::ClampedInt => Converted::Int(match converted.extract::<i64>() {
            Ok(value) => value,
            Err(_) if converted.gt(0)? => i64::MAX,
            Err(_) => i64::MIN,
        }),
        Kind::Float => Converted::Float(converted.extract()?),
        Kind::Str => Converted::Str(converted.extract()?),
        Kind::Bool => Converted::Bool(converted.extract()?),
        Kind::StrList => Converted::StrList(converted.extract()?),
        Kind::FloatList => Converted::FloatList(converted.extract()?),
        Kind::IntList => match converted.extract::<Vec<i64>>() {
            Ok(value) => Converted::IntList(value),
            Err(_) => Converted::Unsendable(converted.repr()?.to_string()),
        },
        Kind::FloatListList => Converted::FloatListList(converted.extract()?),
        Kind::Identity => Converted::Raw(converted.unbind()),
        Kind::Vector => {
            let linalg = py.import("pyspark.ml.linalg")?;
            let values = converted
                .getattr("values")?
                .call_method0("tolist")?
                .extract()?;
            if converted.is_instance(&linalg.getattr("SparseVector")?)? {
                Converted::Vector(MlVector::Sparse {
                    size: converted.getattr("size")?.extract()?,
                    indices: converted
                        .getattr("indices")?
                        .call_method0("tolist")?
                        .extract()?,
                    values,
                })
            } else {
                Converted::Vector(MlVector::Dense(values))
            }
        }
        Kind::Matrix => {
            let linalg = py.import("pyspark.ml.linalg")?;
            let num_rows = converted.getattr("numRows")?.extract()?;
            let num_cols = converted.getattr("numCols")?.extract()?;
            let values = converted
                .getattr("values")?
                .call_method0("tolist")?
                .extract()?;
            let is_transposed = converted.getattr("isTransposed")?.extract()?;
            let array = |name| -> PyResult<Vec<i32>> {
                converted.getattr(name)?.call_method0("tolist")?.extract()
            };
            if converted.is_instance(&linalg.getattr("SparseMatrix")?)? {
                Converted::Matrix(MlMatrix::Sparse {
                    num_rows,
                    num_cols,
                    col_ptrs: array("colPtrs")?,
                    row_indices: array("rowIndices")?,
                    values,
                    is_transposed,
                })
            } else {
                Converted::Matrix(MlMatrix::Dense {
                    num_rows,
                    num_cols,
                    values,
                    is_transposed,
                })
            }
        }
    })
}

/// A wrapper of a core ML object, whose params Python sets.
trait Native {
    type Core: Clone;
    fn inner(&self) -> &Self::Core;
    fn snapshot_core(&self, _py: Python<'_>) -> PyResult<Self::Core> {
        Ok(self.inner().clone())
    }
    fn raw_params_mut(&mut self) -> Option<&mut Vec<(&'static str, Py<PyAny>)>> {
        None
    }
    /// The Python values of the params of the converter `identity`, which are kept as they are.
    fn raw_params(&self) -> Option<&[(&'static str, Py<PyAny>)]> {
        None
    }
    fn params_mut(&mut self) -> &mut Params;
    /// The params whose value PySpark cannot send: a fit, transform or evaluate raises PySpark's
    /// error of its serialization of the params, before any request.
    fn params(&self) -> &Params;
    fn unsendable(&self) -> &[&'static str];
    fn unsendable_mut(&mut self) -> &mut Vec<&'static str>;
    /// The name of the param `name` in the JVM class, where it is not the name of PySpark's.
    fn jvm_name(name: &'static str) -> &'static str {
        name
    }
    /// The defaults that the object holds itself (see [`params_api::OwnDefaults`]), of a class
    /// whose objects hold them: of one with a default of its uid (`outputCol`), which a load
    /// keeps of the uid that the object was made of.
    fn own_defaults(&self) -> Option<&params_api::OwnDefaults> {
        None
    }
    fn own_defaults_mut(&mut self) -> Option<&mut params_api::OwnDefaults> {
        None
    }
}

macro_rules! native {
    ($($wrapper:ty => $core:ty),* $(,)?) => {$(
        impl Native for $wrapper {
            type Core = $core;
            fn inner(&self) -> &$core {
                &self.inner
            }
            fn params_mut(&mut self) -> &mut Params {
                self.inner.params_mut()
            }
            fn params(&self) -> &Params {
                self.inner.params()
            }
            fn unsendable(&self) -> &[&'static str] {
                &self.unsendable
            }
            fn unsendable_mut(&mut self) -> &mut Vec<&'static str> {
                &mut self.unsendable
            }
        }
    )*};
}

/// Set the param `name` of `target` to `value`.
fn apply<T: Native>(target: &mut T, name: &'static str, value: Converted) {
    target.unsendable_mut().retain(|unsent| *unsent != name);
    if let Some(raw) = target.raw_params_mut() {
        raw.retain(|(key, _)| *key != name);
    }
    let jvm = T::jvm_name(name);
    let params = std::mem::take(target.params_mut()).clear_param(&params_api::unsendable_key(jvm));
    let params = match value {
        Converted::Null => params.set_param_null(jvm),
        Converted::Unsendable(text) => {
            target.unsendable_mut().push(name);
            // Kept under a reserved key, which no request sends: a fit, transform or evaluate
            // raises PySpark's error of the value first (see `snapshot`).
            params
                .clear_param(jvm)
                .set_param_string(&params_api::unsendable_key(jvm), &text)
        }
        Converted::Int(value) => params.set_param_int_narrow(jvm, value),
        Converted::Float(value) => params.set_param_double(jvm, value),
        Converted::Str(value) => params.set_param_string(jvm, &value),
        Converted::Bool(value) => params.set_param_bool(jvm, value),
        Converted::StrList(values) => params.set_param_string_array(jvm, values),
        Converted::FloatList(values) => params.set_param_double_array(jvm, values),
        Converted::IntList(values) => params.set_param_int_array(jvm, values),
        Converted::Vector(value) => params.set_param_vector(jvm, value),
        Converted::FloatListList(value) => params.set_param_double_arrays(jvm, value),
        Converted::Raw(value) => {
            target
                .raw_params_mut()
                .expect("identity parameters retain their Python values")
                .push((name, value));
            params.clear_param(jvm)
        }
        Converted::Matrix(value) => params.set_param_matrix(jvm, value),
    };
    *target.params_mut() = params;
}

/// A setter: set the param `name` of `slf` to `value` and return `slf`, as PySpark's `_set`.
fn set_param<'py, T>(
    slf: Bound<'py, T>,
    name: &'static str,
    kind: Kind,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, T>>
where
    T: Native + PyClass<Frozen = False>,
{
    // Converted before the borrow: the conversion runs Python code.
    let converted = convert(name, kind, value)?;
    apply(&mut *slf.try_borrow_mut()?, name, converted);
    Ok(slf)
}

/// Bind the arguments of a call of `method` (`__init__` or `setParams`) of `class` to the params
/// that it takes, as Python binds them: `positional` names the params taken by position, which
/// are none for a keyword-only method, so that a positional argument raises the TypeError of
/// PySpark's `@keyword_only`. `params` are the params that it takes. The values are returned
/// with their kinds, unconverted, in the order that they are set.
fn bind_params<'py>(
    method: &str,
    class: &str,
    positional: &[&'static str],
    params: &[(&'static str, Kind)],
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Vec<(&'static str, Kind, Bound<'py, PyAny>)>> {
    if positional.is_empty() && !args.is_empty() {
        return Err(PyTypeError::new_err(format!(
            "Method {method} forces keyword arguments."
        )));
    }
    if args.len() > positional.len() {
        return Err(PyTypeError::new_err(format!(
            "{class}.{method}() takes from 1 to {} positional arguments but {} were given",
            positional.len() + 1,
            args.len() + 1
        )));
    }
    let mut given: Vec<(&'static str, Bound<'py, PyAny>)> =
        positional.iter().copied().zip(args.iter()).collect();
    if let Some(kwargs) = kwargs {
        for (key, value) in kwargs.iter() {
            let key = key.extract::<String>()?;
            let Some(&(name, _)) = params.iter().find(|(name, _)| *name == key) else {
                return Err(PyTypeError::new_err(format!(
                    "{class}.{method}() got an unexpected keyword argument '{key}'"
                )));
            };
            if given.iter().any(|(set, _)| *set == name) {
                return Err(PyTypeError::new_err(format!(
                    "{class}.{method}() got multiple values for argument '{name}'"
                )));
            }
            given.push((name, value));
        }
    }
    Ok(given
        .into_iter()
        .map(|(name, value)| {
            let kind = params
                .iter()
                .find(|(param, _)| *param == name)
                .map(|(_, kind)| *kind)
                .expect("a bound param is a param");
            (name, kind, value)
        })
        .collect())
}

/// Set the params that a constructor of `class` is given, as PySpark's `__init__` binds and
/// `_set`s them (see [`bind_params`]). `positional` names the params taken by position, and
/// `params` are the params that it takes.
fn init_params<T: Native>(
    target: &mut T,
    class: &str,
    positional: &[&'static str],
    params: &[(&'static str, Kind)],
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    for (name, kind, value) in bind_params("__init__", class, positional, params, args, kwargs)? {
        let converted = convert(name, kind, &value)?;
        apply(target, name, converted);
    }
    Ok(())
}

/// `setParams` of `class`: as PySpark's `@keyword_only` binds it to `_set` (see [`bind_params`]),
/// it sets the params that it is given, in order, and returns `slf`. As `_set`, a value that its
/// converter rejects raises the TypeError after the params before it are set.
fn set_params_of<'py, T>(
    slf: Bound<'py, T>,
    class: &str,
    params: &[(&'static str, Kind)],
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, T>>
where
    T: Native + PyClass<Frozen = False>,
{
    for (name, kind, value) in bind_params("setParams", class, &[], params, args, kwargs)? {
        // Converted before the borrow: the conversion runs Python code.
        let converted = convert(name, kind, &value)?;
        apply(&mut *slf.try_borrow_mut()?, name, converted);
    }
    Ok(slf)
}

/// The message of the TypeError of a fit, transform or evaluate whose `params` is a param map that
/// the native classes cannot take. PySpark applies a non-empty map by copying the object with it,
/// keyed by its Param objects, and the native classes have none.
const NO_PARAM_MAP: &str = "The native ML classes take no param map: set their params with \
     the constructor or setParams instead.";

/// Whether the `params` of a fit, transform or evaluate overrides no param, as PySpark's
/// `params is None` and `not params` of a dict do: None and an empty dict do. A non-empty dict
/// raises the TypeError of [`NO_PARAM_MAP`].
fn overrides_nothing(params: &Bound<'_, PyAny>) -> PyResult<bool> {
    if params.is_none() {
        return Ok(true);
    }
    match params.cast::<PyDict>() {
        Ok(dict) if dict.is_empty() => Ok(true),
        Ok(_) => Err(PyTypeError::new_err(NO_PARAM_MAP)),
        Err(_) => Ok(false),
    }
}

/// Check the `params` of a fit, as the `Estimator.fit` of `pyspark.ml` checks it, before the fit:
/// None and an empty dict pass, a non-empty dict raises [`NO_PARAM_MAP`], and any other value
/// raises PySpark's TypeError. A list or tuple is not a param map, and [`fit_with`] fits it before
/// this check.
fn check_fit_params(params: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    let Some(params) = params else {
        return Ok(());
    };
    if overrides_nothing(params)? {
        return Ok(());
    }
    Err(PyTypeError::new_err(format!(
        "Params must be either a param map or a list/tuple of param maps, but got {}.",
        params.get_type()
    )))
}

/// The result of a fit: the model of a fit of one param map, or the list of the models of a fit of
/// a list or tuple of param maps, as `Estimator.fit` of `pyspark.ml` returns them.
pub(crate) enum Fitted<M> {
    One(M),
    Many(Py<PyList>),
    /// The model of the fit of a copy of the estimator, of a param map.
    Object(Py<PyAny>),
}

impl<'py, M: IntoPyObject<'py>> IntoPyObject<'py> for Fitted<M> {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            Fitted::One(model) => model.into_bound_py_any(py),
            Fitted::Many(models) => Ok(models.into_bound(py).into_any()),
            Fitted::Object(model) => Ok(model.into_bound(py)),
        }
    }
}

/// Fit a native estimator with the `params` of its `fit`, as `Estimator.fit` of `pyspark.ml` fits
/// it. A list or tuple of param maps fits the estimator once per map, in order, by calling its
/// `fit` with each map, so that each map gets the checks and the errors of a fit with that map
/// alone, and the models are returned as a list. Any other `params` is checked by
/// [`check_fit_params`], and then `fit` fits the estimator.
pub(crate) fn fit_with<'py, T: PyClass, M>(
    slf: &Bound<'py, T>,
    dataset: &Bound<'py, PyAny>,
    params: Option<&Bound<'py, PyAny>>,
    fit: impl FnOnce() -> PyResult<M>,
) -> PyResult<Fitted<M>> {
    if let Some(maps) = params
        .filter(|params| params.is_instance_of::<PyList>() || params.is_instance_of::<PyTuple>())
    {
        let models = PyList::empty(slf.py());
        for map in maps.try_iter()? {
            models.append(slf.as_any().call_method1("fit", (dataset, map?))?)?;
        }
        return Ok(Fitted::Many(models.unbind()));
    }
    // A param map of the Params API applies, as `Estimator.fit` applies it: the fit of a copy of
    // the estimator with it (`self.copy(params)._fit(dataset)`).
    if let Some(map) = params.filter(|p| p.cast::<PyDict>().is_ok_and(|d| !d.is_empty())) {
        if slf.as_any().hasattr("copy")? {
            let copy = slf.as_any().call_method1("copy", (map,))?;
            return Ok(Fitted::Object(
                copy.call_method1("fit", (dataset,))?.unbind(),
            ));
        }
    }
    check_fit_params(params)?;
    fit().map(Fitted::One)
}

/// Check the `params` of a transform or an evaluate, as the `Transformer.transform` and
/// `Evaluator.evaluate` of `pyspark.ml` check them, before the call: any value that is not a param
/// map raises PySpark's TypeError.
fn check_params(params: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    let Some(params) = params else {
        return Ok(());
    };
    if overrides_nothing(params)? {
        return Ok(());
    }
    Err(PyTypeError::new_err(format!(
        "Params must be a param map but got {}.",
        params.get_type()
    )))
}

/// The core object of `slf`, cloned under a short borrow, so that a setter that another
/// thread calls while a fit, transform or evaluate waits for the server does not fail. As
/// PySpark's serialization of the params, it first raises `VALUE_NOT_BETWEEN` of an int
/// that PySpark cannot send, and `CANNOT_INFER_ARRAY_ELEMENT_TYPE` of a list param without
/// elements, whose element type PySpark cannot infer.
fn snapshot<T: Native + PyClass>(slf: &Bound<'_, T>) -> PyResult<T::Core> {
    let this = slf.try_borrow()?;
    if !this.unsendable().is_empty() {
        return Err(crate::functions::value_not_between(slf.py()));
    }
    if this.params().has_empty_array() {
        return Err(crate::column::raise_pyspark(
            slf.py(),
            "PySparkTypeError",
            "CANNOT_INFER_ARRAY_ELEMENT_TYPE",
            &[],
        ));
    }
    this.snapshot_core(slf.py())
}

/// The dataset of `method`, a Spark DataFrame. Another object raises the `AttributeError` of
/// `sparkSession`, the attribute that `pyspark.ml` reads of the dataset first (`try_remote_fit`,
/// `try_remote_transform_relation` and `try_remote_evaluate` of `pyspark.ml.util`), so a pandas
/// DataFrame raises pandas' own AttributeError too, and an object that has it raises a TypeError.
fn dataset_of(dataset: &Bound<'_, PyAny>, method: &str) -> PyResult<DataFrame> {
    if let Ok(df) = dataset.extract::<PyRef<'_, PyDataFrame>>() {
        return Ok(df.dataframe.clone());
    }
    dataset.getattr("sparkSession")?;
    Err(PyTypeError::new_err(format!(
        "{method}() takes a Spark DataFrame, not '{}'",
        dataset.get_type().name()?
    )))
}

/// Fit a native estimator on `dataset`, as `try_remote_fit` does, and make its typed model of the core
/// model and the training DataFrame with `model` (`PyLinearSVCModel::fitted`), which has the
/// defaults of the estimator, as `_copyValues` copies them to the model of a fit.
fn fit_native_model<T, M>(
    slf: &Bound<'_, T>,
    dataset: &Bound<'_, PyAny>,
    model: impl FnOnce(Python<'_>, Box<dyn Model>, &DataFrame) -> PyResult<M>,
) -> PyResult<Py<M>>
where
    T: Native + PyClass<Frozen = False>,
    T::Core: Estimator,
    M: params_api::ParamsHolder + pyo3::PyTypeInfo + object::ModelObject,
{
    let py = slf.py();
    let df = dataset_of(dataset, "fit")?;
    let mut estimator = snapshot(slf)?;
    // The defaults of the estimator, which `_copyValues` gives the model of the fit.
    let defaults = params_api::fit_defaults(slf)?;
    let session = df.spark_session();
    let fitted = interruptible(py, &session, || estimator.fit(&df).map_err(Box::new))?;
    let mut model = model(py, fitted, &df)?;
    params_api::copy_defaults(py, defaults, &mut model)?;
    // The object of the model's class, as `Fitted` returns it.
    object::ModelObject::into_object(model, py)
}

/// A fitted ML model (result of `Estimator.fit`), wrapping a boxed core `Model`.
#[pyclass(subclass, name = "MLModel", module = "pyspark.ml.connect")]
pub struct PyMLModel {
    model: Box<dyn Model>,
    /// The session of the DataFrame that the model was fitted on.
    session_id: String,
}

#[pymethods]
impl PyMLModel {
    /// Apply the model to `dataset`, a Spark DataFrame, as `try_remote_transform_relation` of
    /// `pyspark.ml` sends it, with the output columns of the JVM model: it raises its own error
    /// of an input column that it cannot read, or of an output column that the dataset has
    /// already, when the transformed DataFrame is analyzed or run.
    #[pyo3(signature = (dataset, params=None))]
    fn transform(
        &self,
        py: Python<'_>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        check_params(params)?;
        let input = dataset_of(dataset, "transform")?;
        Ok(PyDataFrame::new(self.transform_dataframe(py, &input)?))
    }
}

impl PyMLModel {
    /// The transform of the DataFrame `df` by the JVM model, of `df` as it is.
    fn transform_dataframe(&self, py: Python<'_>, df: &DataFrame) -> PyResult<DataFrame> {
        let mut model = self.model.clone_box();
        let session = df.spark_session();
        if session.session_id() != self.session_id {
            // As PySpark, the transform of a DataFrame of another session refers to the model
            // on the server of that session, which raises CONNECT_ML.CACHE_INVALID when the
            // DataFrame is analyzed or run, as it caches no such model. (The Rust API rejects
            // the transform itself.) Its input is the placeholder of the DataFrame, as in the
            // plans of `pyspark.ml`: the core resolves it, counting its levels for the depth
            // bound, where an input put in whole would count as one level.
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
}

/// LogisticRegression estimator, of the params of `pyspark.ml.classification.LogisticRegression`.
#[pyclass(
    subclass,
    name = "LogisticRegression",
    module = "pyspark.ml.connect.classification"
)]
pub struct PyLogisticRegression {
    inner: LogisticRegression,
    unsendable: Vec<&'static str>,
}

/// The params of `pyspark.ml.classification.LogisticRegression`, in the order of its constructor,
/// with the `TypeConverters` of each: its bounds on the coefficients are matrices, and its bounds on
/// the intercepts vectors.
const LOGISTIC_REGRESSION_PARAMS: &[(&str, Kind)] = &[
    ("featuresCol", Kind::Str),
    ("labelCol", Kind::Str),
    ("predictionCol", Kind::Str),
    ("maxIter", Kind::Int),
    ("regParam", Kind::Float),
    ("elasticNetParam", Kind::Float),
    ("tol", Kind::Float),
    ("fitIntercept", Kind::Bool),
    ("threshold", Kind::Float),
    ("thresholds", Kind::FloatList),
    ("probabilityCol", Kind::Str),
    ("rawPredictionCol", Kind::Str),
    ("standardization", Kind::Bool),
    ("weightCol", Kind::Str),
    ("aggregationDepth", Kind::Int),
    ("family", Kind::Str),
    ("lowerBoundsOnCoefficients", Kind::Matrix),
    ("upperBoundsOnCoefficients", Kind::Matrix),
    ("lowerBoundsOnIntercepts", Kind::Vector),
    ("upperBoundsOnIntercepts", Kind::Vector),
    ("maxBlockSizeInMB", Kind::Float),
];

native_pymethods! {
    PyLogisticRegression;
    jvm: "org.apache.spark.ml.classification.LogisticRegression", op: Estimator;
    params: [
        "aggregationDepth" => (
            "getAggregationDepth",
            get_aggregation_depth,
            param_aggregation_depth
        ),
        "elasticNetParam" => ("getElasticNetParam", get_elastic_net_param, param_elastic_net_param),
        "family" => ("getFamily", get_family, param_family),
        "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
        "fitIntercept" => ("getFitIntercept", get_fit_intercept, param_fit_intercept),
        "labelCol" => ("getLabelCol", get_label_col, param_label_col),
        "lowerBoundsOnCoefficients" => (
            "getLowerBoundsOnCoefficients",
            get_lower_bounds_on_coefficients,
            param_lower_bounds_on_coefficients
        ),
        "lowerBoundsOnIntercepts" => (
            "getLowerBoundsOnIntercepts",
            get_lower_bounds_on_intercepts,
            param_lower_bounds_on_intercepts
        ),
        "maxBlockSizeInMB" => (
            "getMaxBlockSizeInMB",
            get_max_block_size_in_mb,
            param_max_block_size_in_mb
        ),
        "maxIter" => ("getMaxIter", get_max_iter, param_max_iter),
        "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
        "probabilityCol" => ("getProbabilityCol", get_probability_col, param_probability_col),
        "rawPredictionCol" => (
            "getRawPredictionCol",
            get_raw_prediction_col,
            param_raw_prediction_col
        ),
        "regParam" => ("getRegParam", get_reg_param, param_reg_param),
        "standardization" => ("getStandardization", get_standardization, param_standardization),
        "tol" => ("getTol", get_tol, param_tol),
        "upperBoundsOnCoefficients" => (
            "getUpperBoundsOnCoefficients",
            get_upper_bounds_on_coefficients,
            param_upper_bounds_on_coefficients
        ),
        "upperBoundsOnIntercepts" => (
            "getUpperBoundsOnIntercepts",
            get_upper_bounds_on_intercepts,
            param_upper_bounds_on_intercepts
        ),
        "weightCol" => ("getWeightCol", get_weight_col, param_weight_col),
    ];
    #[new]
    #[pyo3(
        signature = (*args, **kwargs),
        text_signature = "(*, featuresCol='features', labelCol='label', predictionCol='prediction', maxIter=100, regParam=0.0, elasticNetParam=0.0, tol=1e-06, fitIntercept=True, threshold=0.5, thresholds=None, probabilityCol='probability', rawPredictionCol='rawPrediction', standardization=True, weightCol=None, aggregationDepth=2, family='auto', lowerBoundsOnCoefficients=None, upperBoundsOnCoefficients=None, lowerBoundsOnIntercepts=None, upperBoundsOnIntercepts=None, maxBlockSizeInMB=0.0)"
    )]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: LogisticRegression::new(),
            unsendable: Vec::new(),
        };
        init_params(
            &mut this,
            "LogisticRegression",
            &[],
            LOGISTIC_REGRESSION_PARAMS,
            args,
            kwargs,
        )?;
        classification_api::check_threshold_consistency(args.py(), &this)?;
        Ok(this)
    }

    /// `setParams` of PySpark's LogisticRegression: as its constructor, it sets the params that it
    /// is given (see [`set_params_of`]), then checks `threshold` and `thresholds`, whichever call
    /// set them (see [`classification_api::check_threshold_consistency`]).
    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, featuresCol='features', labelCol='label', predictionCol='prediction', maxIter=100, regParam=0.0, elasticNetParam=0.0, tol=1e-06, fitIntercept=True, threshold=0.5, thresholds=None, probabilityCol='probability', rawPredictionCol='rawPrediction', standardization=True, weightCol=None, aggregationDepth=2, family='auto', lowerBoundsOnCoefficients=None, upperBoundsOnCoefficients=None, lowerBoundsOnIntercepts=None, upperBoundsOnIntercepts=None, maxBlockSizeInMB=0.0)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        let slf = set_params_of(
            slf,
            "LogisticRegression",
            LOGISTIC_REGRESSION_PARAMS,
            args,
            kwargs,
        )?;
        classification_api::check_threshold_consistency(slf.py(), &*slf.try_borrow()?)?;
        Ok(slf)
    }

    #[pyo3(name = "setFeaturesCol")]
    fn set_features_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "featuresCol", Kind::Str, value)
    }
    #[pyo3(name = "setLabelCol")]
    fn set_label_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "labelCol", Kind::Str, value)
    }
    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "predictionCol", Kind::Str, value)
    }
    #[pyo3(name = "setProbabilityCol")]
    fn set_probability_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "probabilityCol", Kind::Str, value)
    }
    #[pyo3(name = "setMaxIter")]
    fn set_max_iter<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxIter", Kind::Int, value)
    }
    #[pyo3(name = "setRegParam")]
    fn set_reg_param<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "regParam", Kind::Float, value)
    }
    #[pyo3(name = "setElasticNetParam")]
    fn set_elastic_net_param<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "elasticNetParam", Kind::Float, value)
    }
    #[pyo3(name = "setFitIntercept")]
    fn set_fit_intercept<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "fitIntercept", Kind::Bool, value)
    }
    /// Set `threshold`, and leave `thresholds` as it is, as PySpark's does over Spark Connect,
    /// where the `clear` of `JavaParams` does nothing: a fit sends both, which the JVM checks.
    #[pyo3(name = "setThreshold")]
    fn set_threshold<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "threshold", Kind::Float, value)
    }
    /// Set `thresholds`, and leave `threshold` as it is, as `setThresholds` of PySpark's
    /// `ProbabilisticClassifier`, which LogisticRegression's is: a fit sends both, which the JVM
    /// checks.
    #[pyo3(name = "setThresholds")]
    fn set_thresholds<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "thresholds", Kind::FloatList, value)
    }
    #[pyo3(name = "setRawPredictionCol")]
    fn set_raw_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "rawPredictionCol", Kind::Str, value)
    }
    #[pyo3(name = "setStandardization")]
    fn set_standardization<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "standardization", Kind::Bool, value)
    }
    #[pyo3(name = "setWeightCol")]
    fn set_weight_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "weightCol", Kind::Str, value)
    }
    #[pyo3(name = "setAggregationDepth")]
    fn set_aggregation_depth<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "aggregationDepth", Kind::Int, value)
    }
    #[pyo3(name = "setFamily")]
    fn set_family<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "family", Kind::Str, value)
    }
    #[pyo3(name = "setLowerBoundsOnCoefficients")]
    fn set_lower_bounds_on_coefficients<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "lowerBoundsOnCoefficients", Kind::Matrix, value)
    }
    #[pyo3(name = "setUpperBoundsOnCoefficients")]
    fn set_upper_bounds_on_coefficients<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "upperBoundsOnCoefficients", Kind::Matrix, value)
    }
    #[pyo3(name = "setLowerBoundsOnIntercepts")]
    fn set_lower_bounds_on_intercepts<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "lowerBoundsOnIntercepts", Kind::Vector, value)
    }
    #[pyo3(name = "setUpperBoundsOnIntercepts")]
    fn set_upper_bounds_on_intercepts<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "upperBoundsOnIntercepts", Kind::Vector, value)
    }
    #[pyo3(name = "setMaxBlockSizeInMB")]
    fn set_max_block_size_in_mb<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxBlockSizeInMB", Kind::Float, value)
    }
    #[pyo3(name = "setTol")]
    fn set_tol<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "tol", Kind::Float, value)
    }
    /// `getThreshold()`, as PySpark's LogisticRegression (see
    /// [`classification_api::threshold_of`]).
    #[pyo3(name = "getThreshold")]
    fn get_threshold(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        classification_api::threshold_of(slf)
    }

    /// `getThresholds()`, as PySpark's LogisticRegression (see
    /// [`classification_api::thresholds_of`]).
    #[pyo3(name = "getThresholds")]
    fn get_thresholds(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        classification_api::thresholds_of(slf)
    }

    /// The Param attribute `threshold` of the class, as `params_only` of the macros makes it (the
    /// class's Param, and of an instance, its Param of the uid).
    #[classattr]
    #[pyo3(name = "threshold")]
    fn param_threshold() -> params_api::PyParamAttribute {
        params_api::PyParamAttribute::new("threshold")
    }

    /// The Param attribute `thresholds` of the class (see `threshold`).
    #[classattr]
    #[pyo3(name = "thresholds")]
    fn param_thresholds() -> params_api::PyParamAttribute {
        params_api::PyParamAttribute::new("thresholds")
    }

    /// Fit `dataset`, a Spark DataFrame, as PySpark's `pyspark.ml` fits it:
    /// its model outputs the JVM model's `rawPrediction` and `probability` vectors and `double`
    /// prediction, and its training summary is of the training dataset.
    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<Py<classification_api::PyLogisticRegressionModel>>> {
        fit_with(slf, dataset, params, || {
            fit_native_model(slf, dataset, |py, model, df| {
                classification_api::PyLogisticRegressionModel::fitted(
                    py,
                    model,
                    df,
                    object::SummaryDataset::Train,
                )
            })
        })
    }

    /// The fits of a copy of the estimator, one per param map, as `Estimator.fitMultiple`, of
    /// PySpark's keywords.
    #[pyo3(name = "fitMultiple")]
    #[allow(non_snake_case)]
    fn fit_multiple<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        paramMaps: &Bound<'py, PyAny>,
    ) -> PyResult<Py<params_api::PyFitMultipleIterator>> {
        params_api::fit_multiple(slf.as_any(), dataset, paramMaps)
    }
}

impl params_api::NativeInstance for PyLogisticRegression {
    fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        object::new_py(
            py,
            Self {
                inner: self.inner.clone(),
                unsendable: self.unsendable.clone(),
            },
        )
    }

    fn uid(&self) -> String {
        Estimator::operator(&self.inner).uid.clone()
    }

    fn set_uid(&mut self, uid: &str) {
        Estimator::operator_mut(&mut self.inner).uid = uid.to_string();
    }
}

/// Linear support vector classifier backed by Spark Connect's JVM ML API.
#[pyclass(
    subclass,
    name = "LinearSVC",
    module = "pyspark.ml.connect.classification"
)]
pub struct PyLinearSVC {
    inner: LinearSVC,
    unsendable: Vec<&'static str>,
}

const LINEAR_SVC_PARAMS: &[(&str, Kind)] = &[
    ("featuresCol", Kind::Str),
    ("labelCol", Kind::Str),
    ("predictionCol", Kind::Str),
    ("maxIter", Kind::Int),
    ("regParam", Kind::Float),
    ("tol", Kind::Float),
    ("rawPredictionCol", Kind::Str),
    ("fitIntercept", Kind::Bool),
    ("standardization", Kind::Bool),
    ("threshold", Kind::Float),
    ("weightCol", Kind::Str),
    ("aggregationDepth", Kind::Int),
    ("maxBlockSizeInMB", Kind::Float),
];

native_pymethods! {
    PyLinearSVC;
    jvm: "org.apache.spark.ml.classification.LinearSVC", op: Estimator;
    params: [
        "aggregationDepth" => (
            "getAggregationDepth",
            get_aggregation_depth,
            param_aggregation_depth
        ),
        "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
        "fitIntercept" => ("getFitIntercept", get_fit_intercept, param_fit_intercept),
        "labelCol" => ("getLabelCol", get_label_col, param_label_col),
        "maxBlockSizeInMB" => (
            "getMaxBlockSizeInMB",
            get_max_block_size_in_mb,
            param_max_block_size_in_mb
        ),
        "maxIter" => ("getMaxIter", get_max_iter, param_max_iter),
        "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
        "rawPredictionCol" => (
            "getRawPredictionCol",
            get_raw_prediction_col,
            param_raw_prediction_col
        ),
        "regParam" => ("getRegParam", get_reg_param, param_reg_param),
        "standardization" => ("getStandardization", get_standardization, param_standardization),
        "threshold" => ("getThreshold", get_threshold, param_threshold),
        "tol" => ("getTol", get_tol, param_tol),
        "weightCol" => ("getWeightCol", get_weight_col, param_weight_col),
    ];
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, featuresCol='features', labelCol='label', predictionCol='prediction', maxIter=100, regParam=0.0, tol=1e-06, rawPredictionCol='rawPrediction', fitIntercept=True, standardization=True, threshold=0.0, weightCol=None, aggregationDepth=2, maxBlockSizeInMB=0.0)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: LinearSVC::new(),
            unsendable: Vec::new(),
        };
        init_params(&mut this, "LinearSVC", &[], LINEAR_SVC_PARAMS, args, kwargs)?;
        Ok(this)
    }
    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, featuresCol='features', labelCol='label', predictionCol='prediction', maxIter=100, regParam=0.0, tol=1e-06, rawPredictionCol='rawPrediction', fitIntercept=True, standardization=True, threshold=0.0, weightCol=None, aggregationDepth=2, maxBlockSizeInMB=0.0)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(slf, "LinearSVC", LINEAR_SVC_PARAMS, args, kwargs)
    }

    #[pyo3(name = "setFeaturesCol")]
    fn set_feature_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "featuresCol", Kind::Str, value)
    }

    #[pyo3(name = "setLabelCol")]
    fn set_label_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "labelCol", Kind::Str, value)
    }

    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "predictionCol", Kind::Str, value)
    }

    #[pyo3(name = "setMaxIter")]
    fn set_max_iter<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxIter", Kind::Int, value)
    }

    #[pyo3(name = "setRegParam")]
    fn set_reg_param<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "regParam", Kind::Float, value)
    }

    #[pyo3(name = "setTol")]
    fn set_tol<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "tol", Kind::Float, value)
    }

    #[pyo3(name = "setRawPredictionCol")]
    fn set_raw_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "rawPredictionCol", Kind::Str, value)
    }

    #[pyo3(name = "setFitIntercept")]
    fn set_fit_intercept<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "fitIntercept", Kind::Bool, value)
    }

    #[pyo3(name = "setStandardization")]
    fn set_standardization<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "standardization", Kind::Bool, value)
    }

    #[pyo3(name = "setThreshold")]
    fn set_threshold<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "threshold", Kind::Float, value)
    }

    #[pyo3(name = "setWeightCol")]
    fn set_weight_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "weightCol", Kind::Str, value)
    }

    #[pyo3(name = "setAggregationDepth")]
    fn set_aggregation_depth<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "aggregationDepth", Kind::Int, value)
    }

    #[pyo3(name = "setMaxBlockSizeInMB")]
    fn set_max_block_size_in_mb<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxBlockSizeInMB", Kind::Float, value)
    }

    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<Py<classification_api::PyLinearSVCModel>>> {
        fit_with(slf, dataset, params, || {
            fit_native_model(slf, dataset, |py, model, df| {
                classification_api::PyLinearSVCModel::fitted(
                    py,
                    model,
                    df,
                    object::SummaryDataset::Train,
                )
            })
        })
    }

    /// The fits of a copy of the estimator, one per param map, as `Estimator.fitMultiple`, of
    /// PySpark's keywords.
    #[pyo3(name = "fitMultiple")]
    #[allow(non_snake_case)]
    fn fit_multiple<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        paramMaps: &Bound<'py, PyAny>,
    ) -> PyResult<Py<params_api::PyFitMultipleIterator>> {
        params_api::fit_multiple(slf.as_any(), dataset, paramMaps)
    }
}

impl params_api::NativeInstance for PyLinearSVC {
    fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        object::new_py(
            py,
            Self {
                inner: self.inner.clone(),
                unsendable: self.unsendable.clone(),
            },
        )
    }

    fn uid(&self) -> String {
        Estimator::operator(&self.inner).uid.clone()
    }

    fn set_uid(&mut self, uid: &str) {
        Estimator::operator_mut(&mut self.inner).uid = uid.to_string();
    }
}

/// Native FMClassifier.
#[pyclass(
    subclass,
    name = "FMClassifier",
    module = "pyspark.ml.connect.classification"
)]
pub struct PyFMClassifier {
    inner: FMClassifier,
    unsendable: Vec<&'static str>,
}

const FMCLASSIFIER_PARAMS: &[(&str, Kind)] = &[
    ("featuresCol", Kind::Str),
    ("labelCol", Kind::Str),
    ("predictionCol", Kind::Str),
    ("probabilityCol", Kind::Str),
    ("rawPredictionCol", Kind::Str),
    ("factorSize", Kind::Int),
    ("fitIntercept", Kind::Bool),
    ("fitLinear", Kind::Bool),
    ("regParam", Kind::Float),
    ("miniBatchFraction", Kind::Float),
    ("initStd", Kind::Float),
    ("maxIter", Kind::Int),
    ("stepSize", Kind::Float),
    ("tol", Kind::Float),
    ("solver", Kind::Str),
    ("thresholds", Kind::FloatList),
    ("seed", Kind::Int),
];

native_pymethods! {
    PyFMClassifier;
    jvm: "org.apache.spark.ml.classification.FMClassifier", op: Estimator;
    params: [
        "factorSize" => ("getFactorSize", get_factor_size, param_factor_size),
        "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
        "fitIntercept" => ("getFitIntercept", get_fit_intercept, param_fit_intercept),
        "fitLinear" => ("getFitLinear", get_fit_linear, param_fit_linear),
        "initStd" => ("getInitStd", get_init_std, param_init_std),
        "labelCol" => ("getLabelCol", get_label_col, param_label_col),
        "maxIter" => ("getMaxIter", get_max_iter, param_max_iter),
        "miniBatchFraction" => (
            "getMiniBatchFraction",
            get_mini_batch_fraction,
            param_mini_batch_fraction
        ),
        "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
        "probabilityCol" => ("getProbabilityCol", get_probability_col, param_probability_col),
        "rawPredictionCol" => (
            "getRawPredictionCol",
            get_raw_prediction_col,
            param_raw_prediction_col
        ),
        "regParam" => ("getRegParam", get_reg_param, param_reg_param),
        "seed" => ("getSeed", get_seed, param_seed),
        "solver" => ("getSolver", get_solver, param_solver),
        "stepSize" => ("getStepSize", get_step_size, param_step_size),
        "thresholds" => ("getThresholds", get_thresholds, param_thresholds),
        "tol" => ("getTol", get_tol, param_tol),
        "weightCol" => ("getWeightCol", get_weight_col, param_weight_col),
    ];
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, featuresCol='features', labelCol='label', predictionCol='prediction', probabilityCol='probability', rawPredictionCol='rawPrediction', factorSize=8, fitIntercept=True, fitLinear=True, regParam=0.0, miniBatchFraction=1.0, initStd=0.01, maxIter=100, stepSize=1.0, tol=1e-06, solver='adamW', thresholds=None, seed=None)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: FMClassifier::new(),
            unsendable: Vec::new(),
        };
        init_params(&mut this, "FMClassifier", &[], FMCLASSIFIER_PARAMS, args, kwargs)?;
        Ok(this)
    }
    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, featuresCol='features', labelCol='label', predictionCol='prediction', probabilityCol='probability', rawPredictionCol='rawPrediction', factorSize=8, fitIntercept=True, fitLinear=True, regParam=0.0, miniBatchFraction=1.0, initStd=0.01, maxIter=100, stepSize=1.0, tol=1e-06, solver='adamW', thresholds=None, seed=None)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(slf, "FMClassifier", FMCLASSIFIER_PARAMS, args, kwargs)
    }

    #[pyo3(name = "setFactorSize")]
    fn set_factor_size<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "factorSize", Kind::Int, value)
    }

    #[pyo3(name = "setFeaturesCol")]
    fn set_features_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "featuresCol", Kind::Str, value)
    }

    #[pyo3(name = "setFitIntercept")]
    fn set_fit_intercept<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "fitIntercept", Kind::Bool, value)
    }

    #[pyo3(name = "setFitLinear")]
    fn set_fit_linear<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "fitLinear", Kind::Bool, value)
    }

    #[pyo3(name = "setRegParam")]
    fn set_reg_param<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "regParam", Kind::Float, value)
    }

    #[pyo3(name = "setMiniBatchFraction")]
    fn set_mini_batch_fraction<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "miniBatchFraction", Kind::Float, value)
    }

    #[pyo3(name = "setInitStd")]
    fn set_init_std<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "initStd", Kind::Float, value)
    }

    #[pyo3(name = "setMaxIter")]
    fn set_max_iter<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxIter", Kind::Int, value)
    }

    #[pyo3(name = "setStepSize")]
    fn set_step_size<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "stepSize", Kind::Float, value)
    }

    #[pyo3(name = "setTol")]
    fn set_tol<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "tol", Kind::Float, value)
    }

    #[pyo3(name = "setSolver")]
    fn set_solver<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "solver", Kind::Str, value)
    }

    #[pyo3(name = "setThresholds")]
    fn set_thresholds<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "thresholds", Kind::FloatList, value)
    }

    #[pyo3(name = "setSeed")]
    fn set_seed<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "seed", Kind::Int, value)
    }

    #[pyo3(name = "setLabelCol")]
    fn set_label_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "labelCol", Kind::Str, value)
    }

    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "predictionCol", Kind::Str, value)
    }

    #[pyo3(name = "setProbabilityCol")]
    fn set_probability_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "probabilityCol", Kind::Str, value)
    }

    #[pyo3(name = "setRawPredictionCol")]
    fn set_raw_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "rawPredictionCol", Kind::Str, value)
    }


    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<Py<classification_api::PyFMClassificationModel>>> {
        fit_with(slf, dataset, params, || {
            fit_native_model(slf, dataset, |py, model, df| {
                classification_api::PyFMClassificationModel::fitted(
                    py,
                    model,
                    df,
                    object::SummaryDataset::Train,
                )
            })
        })
    }

    /// The fits of a copy of the estimator, one per param map, as `Estimator.fitMultiple`, of
    /// PySpark's keywords.
    #[pyo3(name = "fitMultiple")]
    #[allow(non_snake_case)]
    fn fit_multiple<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        paramMaps: &Bound<'py, PyAny>,
    ) -> PyResult<Py<params_api::PyFitMultipleIterator>> {
        params_api::fit_multiple(slf.as_any(), dataset, paramMaps)
    }

}

impl params_api::NativeInstance for PyFMClassifier {
    fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        object::new_py(
            py,
            Self {
                inner: self.inner.clone(),
                unsendable: self.unsendable.clone(),
            },
        )
    }

    fn uid(&self) -> String {
        Estimator::operator(&self.inner).uid.clone()
    }

    fn set_uid(&mut self, uid: &str) {
        Estimator::operator_mut(&mut self.inner).uid = uid.to_string();
    }
}

/// Native MultilayerPerceptronClassifier.
#[pyclass(
    subclass,
    name = "MultilayerPerceptronClassifier",
    module = "pyspark.ml.connect.classification"
)]
pub struct PyMultilayerPerceptronClassifier {
    inner: MultilayerPerceptronClassifier,
    unsendable: Vec<&'static str>,
}

const MULTILAYER_PERCEPTRON_CLASSIFIER_PARAMS: &[(&str, Kind)] = &[
    ("featuresCol", Kind::Str),
    ("labelCol", Kind::Str),
    ("predictionCol", Kind::Str),
    ("maxIter", Kind::Int),
    ("tol", Kind::Float),
    ("seed", Kind::Int),
    ("layers", Kind::IntList),
    ("blockSize", Kind::Int),
    ("stepSize", Kind::Float),
    ("solver", Kind::Str),
    ("initialWeights", Kind::Vector),
    ("probabilityCol", Kind::Str),
    ("rawPredictionCol", Kind::Str),
];

native_pymethods! {
    PyMultilayerPerceptronClassifier;
    jvm: "org.apache.spark.ml.classification.MultilayerPerceptronClassifier", op: Estimator;
    params: [
        "blockSize" => ("getBlockSize", get_block_size, param_block_size),
        "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
        "initialWeights" => ("getInitialWeights", get_initial_weights, param_initial_weights),
        "labelCol" => ("getLabelCol", get_label_col, param_label_col),
        "layers" => ("getLayers", get_layers, param_layers),
        "maxIter" => ("getMaxIter", get_max_iter, param_max_iter),
        "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
        "probabilityCol" => ("getProbabilityCol", get_probability_col, param_probability_col),
        "rawPredictionCol" => (
            "getRawPredictionCol",
            get_raw_prediction_col,
            param_raw_prediction_col
        ),
        "seed" => ("getSeed", get_seed, param_seed),
        "solver" => ("getSolver", get_solver, param_solver),
        "stepSize" => ("getStepSize", get_step_size, param_step_size),
        "thresholds" => ("getThresholds", get_thresholds, param_thresholds),
        "tol" => ("getTol", get_tol, param_tol),
    ];
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, featuresCol='features', labelCol='label', predictionCol='prediction', maxIter=100, tol=1e-06, seed=None, layers=None, blockSize=128, stepSize=0.03, solver='l-bfgs', initialWeights=None, probabilityCol='probability', rawPredictionCol='rawPrediction')")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: MultilayerPerceptronClassifier::new(),
            unsendable: Vec::new(),
        };
        init_params(
            &mut this,
            "MultilayerPerceptronClassifier",
            &[],
            MULTILAYER_PERCEPTRON_CLASSIFIER_PARAMS,
            args,
            kwargs,
        )?;
        Ok(this)
    }
    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, featuresCol='features', labelCol='label', predictionCol='prediction', maxIter=100, tol=1e-06, seed=None, layers=None, blockSize=128, stepSize=0.03, solver='l-bfgs', initialWeights=None, probabilityCol='probability', rawPredictionCol='rawPrediction')"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(
            slf,
            "MultilayerPerceptronClassifier",
            MULTILAYER_PERCEPTRON_CLASSIFIER_PARAMS,
            args,
            kwargs,
        )
    }

    #[pyo3(name = "setBlockSize")]
    fn set_block_size<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "blockSize", Kind::Int, value)
    }

    #[pyo3(name = "setFeaturesCol")]
    fn set_features_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "featuresCol", Kind::Str, value)
    }

    #[pyo3(name = "setInitialWeights")]
    fn set_initial_weights<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "initialWeights", Kind::Vector, value)
    }

    #[pyo3(name = "setLabelCol")]
    fn set_label_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "labelCol", Kind::Str, value)
    }

    #[pyo3(name = "setLayers")]
    fn set_layers<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "layers", Kind::IntList, value)
    }

    #[pyo3(name = "setMaxIter")]
    fn set_max_iter<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxIter", Kind::Int, value)
    }

    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "predictionCol", Kind::Str, value)
    }

    #[pyo3(name = "setProbabilityCol")]
    fn set_probability_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "probabilityCol", Kind::Str, value)
    }

    #[pyo3(name = "setRawPredictionCol")]
    fn set_raw_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "rawPredictionCol", Kind::Str, value)
    }

    #[pyo3(name = "setSeed")]
    fn set_seed<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "seed", Kind::Int, value)
    }

    #[pyo3(name = "setSolver")]
    fn set_solver<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "solver", Kind::Str, value)
    }

    #[pyo3(name = "setStepSize")]
    fn set_step_size<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "stepSize", Kind::Float, value)
    }

    #[pyo3(name = "setThresholds")]
    fn set_thresholds<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "thresholds", Kind::FloatList, value)
    }

    #[pyo3(name = "setTol")]
    fn set_tol<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "tol", Kind::Float, value)
    }


    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<Py<classification_api::PyMultilayerPerceptronClassificationModel>>> {
        fit_with(slf, dataset, params, || {
            fit_native_model(slf, dataset, |py, model, df| {
                classification_api::PyMultilayerPerceptronClassificationModel::fitted(
                    py,
                    model,
                    df,
                    object::SummaryDataset::Train,
                )
            })
        })
    }

    /// The fits of a copy of the estimator, one per param map, as `Estimator.fitMultiple`, of
    /// PySpark's keywords.
    #[pyo3(name = "fitMultiple")]
    #[allow(non_snake_case)]
    fn fit_multiple<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        paramMaps: &Bound<'py, PyAny>,
    ) -> PyResult<Py<params_api::PyFitMultipleIterator>> {
        params_api::fit_multiple(slf.as_any(), dataset, paramMaps)
    }

}

impl params_api::NativeInstance for PyMultilayerPerceptronClassifier {
    fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        object::new_py(
            py,
            Self {
                inner: self.inner.clone(),
                unsendable: self.unsendable.clone(),
            },
        )
    }

    fn uid(&self) -> String {
        Estimator::operator(&self.inner).uid.clone()
    }

    fn set_uid(&mut self, uid: &str) {
        Estimator::operator_mut(&mut self.inner).uid = uid.to_string();
    }
}

/// Native OneVsRest.
#[pyclass(
    subclass,
    name = "OneVsRest",
    module = "pyspark.ml.connect.classification"
)]
pub struct PyOneVsRest {
    inner: OneVsRest,
    /// The Python values of the identity params, which the core does not hold: the classifier.
    raw: Vec<(&'static str, Py<PyAny>)>,
}

const ONE_VS_REST_PARAMS: &[(&str, Kind)] = &[
    ("classifier", Kind::Identity),
    ("featuresCol", Kind::Str),
    ("labelCol", Kind::Str),
    ("predictionCol", Kind::Str),
    ("rawPredictionCol", Kind::Str),
    ("weightCol", Kind::Str),
    ("parallelism", Kind::ClampedInt),
];

impl params_api::ParamsHolder for PyOneVsRest {
    fn param_kind(&self, name: &str) -> Option<Kind> {
        ONE_VS_REST_PARAMS
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, kind)| *kind)
    }

    fn set_value(&self, py: Python<'_>, name: &str) -> PyResult<Option<Py<PyAny>>> {
        if let Some((_, value)) = self.raw.iter().find(|(n, _)| *n == name) {
            return Ok(Some(value.clone_ref(py)));
        }
        params_api::literal_value(py, self.inner.params(), name)
    }

    fn put(&mut self, py: Python<'_>, name: &'static str, value: Converted) -> PyResult<()> {
        self.raw.retain(|(n, _)| *n != name);
        let params = std::mem::take(self.inner.params_mut());
        *self.inner.params_mut() = match value {
            Converted::Raw(value) => {
                self.raw.push((name, value));
                params.clear_param(name)
            }
            value => object::put_model_param(py, params, name, value)?,
        };
        Ok(())
    }

    fn unset(&mut self, name: &'static str) -> PyResult<()> {
        self.raw.retain(|(n, _)| *n != name);
        let params = std::mem::take(self.inner.params_mut()).clear_param(name);
        *self.inner.params_mut() = params;
        Ok(())
    }
}

native_pymethods! {
    PyOneVsRest;
    persistence: python;
    params: [
        "classifier" => ("getClassifier", get_classifier, param_classifier),
        "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
        "labelCol" => ("getLabelCol", get_label_col, param_label_col),
        "parallelism" => ("getParallelism", get_parallelism, param_parallelism),
        "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
        "rawPredictionCol" => (
            "getRawPredictionCol",
            get_raw_prediction_col,
            param_raw_prediction_col
        ),
        "weightCol" => ("getWeightCol", get_weight_col, param_weight_col),
    ];
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, featuresCol='features', labelCol='label', predictionCol='prediction', rawPredictionCol='rawPrediction', classifier=None, weightCol=None, parallelism=1)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: OneVsRest::new(),
            raw: Vec::new(),
        };
        check_classifier_kwarg(kwargs)?;
        set_ovr_params(&mut this, args.py(), "__init__", args, kwargs)?;
        Ok(this)
    }

    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, featuresCol='features', labelCol='label', predictionCol='prediction', rawPredictionCol='rawPrediction', classifier=None, weightCol=None, parallelism=1)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        check_classifier_kwarg(kwargs)?;
        let py = slf.py();
        for (name, kind, value) in
            bind_params("setParams", "OneVsRest", &[], ONE_VS_REST_PARAMS, args, kwargs)?
        {
            let converted = convert(name, kind, &value)?;
            params_api::ParamsHolder::put(&mut *slf.try_borrow_mut()?, py, name, converted)?;
        }
        Ok(slf)
    }

    #[pyo3(name = "setFeaturesCol")]
    fn set_feature_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_ovr(slf, "featuresCol", Kind::Str, value)
    }

    #[pyo3(name = "setLabelCol")]
    fn set_label_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_ovr(slf, "labelCol", Kind::Str, value)
    }

    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_ovr(slf, "predictionCol", Kind::Str, value)
    }

    #[pyo3(name = "setRawPredictionCol")]
    fn set_raw_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_ovr(slf, "rawPredictionCol", Kind::Str, value)
    }

    #[pyo3(name = "setWeightCol")]
    fn set_weight_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_ovr(slf, "weightCol", Kind::Str, value)
    }

    #[pyo3(name = "setParallelism")]
    fn set_parallelism<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_ovr(slf, "parallelism", Kind::ClampedInt, value)
    }

    /// The classifier, as `OneVsRest.setClassifier`: a native classification estimator, or None.
    #[pyo3(name = "setClassifier")]
    fn set_classifier<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        if !value.is_none() {
            native_classifier(value, false)?;
        }
        set_ovr(slf, "classifier", Kind::Identity, value)
    }

    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<Py<PyAny>>> {
        // A list of param maps fits once per map (`fit_with`), each in PySpark's order below.
        fit_with(slf, dataset, params, || -> PyResult<_> {
            let py = slf.py();
            // PySpark's `_fit` reads the classifier of the param, which raises its KeyError when
            // it is unset.
            let classifier = slf.call_method0("getClassifier")?.unbind();
            // The native fit is that of a native classifier on a Spark DataFrame. PySpark's own
            // `OneVsRest._fit` fits any other over this object's params, as PySpark does: a
            // classifier that is not a native one (PySpark's own classes, or a Python subclass of
            // them), whose model is PySpark's `OneVsRestModel` of the classifier's models; a None
            // classifier, which fails at its first use there; and a dataset that is not a Spark
            // DataFrame, such as a pandas DataFrame, which fails at the read of its max label.
            let spark_df = dataset
                .extract::<PyRef<'_, PyDataFrame>>()
                .ok()
                .map(|df| df.dataframe.clone());
            let df = match spark_df {
                Some(df) if is_native_classifier(classifier.bind(py)) => df,
                _ => {
                    return Ok(py
                        .import("pyspark.ml.classification")?
                        .getattr("OneVsRest")?
                        .getattr("_fit")?
                        .call1((slf, dataset))?
                        .unbind());
                }
            };
            let defaults = params_api::fit_defaults(slf)?;
            let mut core = slf.try_borrow()?.inner.clone();
            core.check_required_params().to_pyerr()?;
            let session = df.spark_session();
            let max = interruptible(py, &session, || core.max_label(&df).map_err(Box::new))?;
            // `numClasses = int(max) + 1`, by Python's own `int` and `min` below, so that a null, NaN
            // or infinite max, and a null parallelism, raise their errors of this Python.
            let builtins = py.import("builtins")?;
            let num_classes = builtins.getattr("int")?.call1((max,))?.add(1)?;
            // PySpark passes the weight to the classifiers that are a `HasWeightCol`, and warns of
            // the others by the classifier's `str` (its uid), as `"...{}...".format(classifier)`.
            let supports_weight = native_classifier(classifier.bind(py), false)?.supports_weight();
            if !supports_weight && core.weight_col().is_some_and(|s| !s.is_empty()) {
                let text = format!(
                    "weightCol is ignored, as it is not supported by {} now.",
                    classifier.bind(py).str()?
                );
                py.import("warnings")?.call_method1("warn", (text,))?;
            }
            // `ThreadPool(processes=min(parallelism, numClasses))`, which raises this ValueError for
            // fewer than one process.
            let threads = builtins
                .getattr("min")?
                .call1((core.parallelism_i64(), &num_classes))?;
            // At most `parallelism`, a 64-bit int, so an int that does not fit 64 bits is below it.
            let threads = threads.extract::<i64>().unwrap_or(i64::MIN);
            if threads < 1 {
                return Err(PyValueError::new_err(
                    "Number of processes must be at least 1",
                ));
            }
            // PySpark would fit that many classifiers; refuse a label that no class index fits.
            let num_classes = match num_classes.extract::<i32>() {
                Ok(classes) => classes as usize,
                Err(_) => {
                    return Err(PyValueError::new_err(
                        "OneVsRest requires a finite nonnegative maximum class label",
                    ))
                }
            };
            // The classifier's params are serialized by its fits, which run in that pool.
            core = core.set_classifier(native_classifier(classifier.bind(py), true)?);
            core.ensure_operator();
            // The columns of the fit, which the submodels of the model are made of.
            let fit = classification_api::OneVsRestFit {
                train: df.clone(),
                label: core.label_col().to_string(),
                features: core.features_col().to_string(),
                weight: core
                    .weight_col()
                    .filter(|name| !name.is_empty() && supports_weight)
                    .map(str::to_string),
            };
            let model = interruptible(py, &session, || {
                core.fit_classes(&df, num_classes, threads as usize)
                    .map_err(Box::new)
            })?;
            // The model has the defaults of the estimator, as `_copyValues` gives it.
            let mut fitted = classification_api::PyOneVsRestModel::fitted(
                py,
                Box::new(model),
                fit,
                classifier,
            )?;
            params_api::copy_defaults(py, defaults, &mut fitted)?;
            Ok(object::new_py(py, fitted)?.into_any())
        })
    }

    /// The fits of a copy of the estimator, one per param map, as `Estimator.fitMultiple`, of
    /// PySpark's keywords.
    #[pyo3(name = "fitMultiple")]
    #[allow(non_snake_case)]
    fn fit_multiple<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        paramMaps: &Bound<'py, PyAny>,
    ) -> PyResult<Py<params_api::PyFitMultipleIterator>> {
        params_api::fit_multiple(slf.as_any(), dataset, paramMaps)
    }

    /// As `OneVsRest.copy`: a copy of the same uid and params, with `extra`, and of the classifier
    /// too, a copy of it with `extra` (`self.getClassifier().copy(extra)`).
    #[pyo3(name = "copy", signature = (extra=None))]
    fn __params_copy<'py>(
        slf: &Bound<'py, Self>,
        extra: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let that = params_api::copy(slf, extra)?;
        let classifier =
            params_api::ParamsHolder::set_value(&*slf.try_borrow()?, py, "classifier")?;
        if let Some(classifier) = classifier.filter(|c| !c.bind(py).is_none()) {
            let extra = pipeline_api::extra_or_empty(py, extra);
            let copy = classifier.bind(py).call_method1("copy", (extra,))?;
            params_api::ParamsHolder::put(
                &mut *that.try_borrow_mut()?,
                py,
                "classifier",
                Converted::Raw(copy.unbind()),
            )?;
        }
        Ok(that)
    }

    /// The writer of the estimator, as `OneVsRest.write` over Spark Connect (`RemoteMLWriter`,
    /// which saves it as `OneVsRestWriter` does).
    fn write(slf: &Bound<'_, Self>) -> PyResult<Py<object::PyMLWriter>> {
        let writer = object::PyMLWriter::new(object::WriteTarget::Python(
            slf.clone().into_any().unbind(),
            classification_api::save_one_vs_rest,
        ));
        object::new_py(slf.py(), writer)
    }

    /// Save to `path`, as `MLWritable.save`: `self.write().save(path)`.
    fn save(slf: &Bound<'_, Self>, path: &Bound<'_, PyAny>) -> PyResult<()> {
        slf.call_method0("write")?.call_method1("save", (path,))?;
        Ok(())
    }

    /// The reader of the class, as `OneVsRest.read` over Spark Connect (`RemoteMLReader`, which
    /// loads it as `OneVsRestReader` does).
    #[classmethod]
    fn read(cls: &Bound<'_, pyo3::types::PyType>) -> PyResult<Py<object::PyMLReader>> {
        let reader = object::PyMLReader::python(classification_api::load_one_vs_rest);
        object::new_py(cls.py(), reader)
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

/// Set the param `name` of a OneVsRest of the converter kind `kind`, and return `slf`, as `_set`.
fn set_ovr<'py>(
    slf: Bound<'py, PyOneVsRest>,
    name: &'static str,
    kind: Kind,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyOneVsRest>> {
    let py = slf.py();
    // Converted before the borrow: the conversion runs Python code.
    let converted = convert(name, kind, value)?;
    params_api::ParamsHolder::put(&mut *slf.try_borrow_mut()?, py, name, converted)?;
    Ok(slf)
}

/// Set the params of a constructor (`__init__`) or `setParams` of a OneVsRest, as PySpark's
/// `@keyword_only` binds them to `_set`.
fn set_ovr_params(
    target: &mut PyOneVsRest,
    py: Python<'_>,
    method: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    for (name, kind, value) in
        bind_params(method, "OneVsRest", &[], ONE_VS_REST_PARAMS, args, kwargs)?
    {
        let converted = convert(name, kind, &value)?;
        params_api::ParamsHolder::put(target, py, name, converted)?;
    }
    Ok(())
}

impl params_api::NativeInstance for PyOneVsRest {
    /// As PySpark's `copy`, which copies the classifier too (`self.getClassifier().copy(extra)`).
    fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        let raw = self
            .raw
            .iter()
            .map(|(name, value)| {
                let value = if *name == "classifier" {
                    value.bind(py).call_method0("copy")?.unbind()
                } else {
                    value.clone_ref(py)
                };
                Ok((*name, value))
            })
            .collect::<PyResult<Vec<_>>>()?;
        object::new_py(
            py,
            Self {
                inner: self.inner.clone(),
                raw,
            },
        )
    }

    fn uid(&self) -> String {
        Estimator::operator(&self.inner).uid.clone()
    }

    fn set_uid(&mut self, uid: &str) {
        Estimator::operator_mut(&mut self.inner).uid = uid.to_string();
    }
}

/// The native classifier of a `classifier` keyword of a constructor or `setParams`, which PySpark
/// takes as any object: checked here, as `setClassifier` checks it.
fn check_classifier_kwarg(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
    if let Some(classifier) = kwargs.and_then(|kw| kw.get_item("classifier").ok().flatten()) {
        if !classifier.is_none() {
            native_classifier(&classifier, false)?;
        }
    }
    Ok(())
}

/// Whether `value` is of one of the native classification estimators that
/// [`native_classifier`] takes, or of a subclass of one.
fn is_native_classifier(value: &Bound<'_, PyAny>) -> bool {
    value.is_instance_of::<PyLogisticRegression>()
        || value.is_instance_of::<PyLinearSVC>()
        || value.is_instance_of::<PyDecisionTreeClassifier>()
        || value.is_instance_of::<PyRandomForestClassifier>()
        || value.is_instance_of::<PyGBTClassifier>()
        || value.is_instance_of::<PyNaiveBayes>()
        || value.is_instance_of::<PyFMClassifier>()
        || value.is_instance_of::<PyMultilayerPerceptronClassifier>()
}

// Snapshot at fit time, retaining PySpark's reference semantics for a classifier
// mutated after it was assigned to OneVsRest.
fn native_classifier(value: &Bound<'_, PyAny>, validate_params: bool) -> PyResult<Classifier> {
    if let Ok(value) = value.cast::<PyLogisticRegression>() {
        return Ok(if validate_params {
            snapshot(value)?
        } else {
            value.try_borrow()?.inner.clone()
        }
        .into());
    }
    if let Ok(value) = value.cast::<PyLinearSVC>() {
        return Ok(if validate_params {
            snapshot(value)?
        } else {
            value.try_borrow()?.inner.clone()
        }
        .into());
    }
    if let Ok(value) = value.cast::<PyDecisionTreeClassifier>() {
        return Ok(if validate_params {
            snapshot(value)?
        } else {
            value.try_borrow()?.inner.clone()
        }
        .into());
    }
    if let Ok(value) = value.cast::<PyRandomForestClassifier>() {
        return Ok(if validate_params {
            snapshot(value)?
        } else {
            value.try_borrow()?.inner.clone()
        }
        .into());
    }
    if let Ok(value) = value.cast::<PyGBTClassifier>() {
        return Ok(if validate_params {
            snapshot(value)?
        } else {
            value.try_borrow()?.inner.clone()
        }
        .into());
    }
    if let Ok(value) = value.cast::<PyNaiveBayes>() {
        return Ok(if validate_params {
            snapshot(value)?
        } else {
            value.try_borrow()?.inner.clone()
        }
        .into());
    }
    if let Ok(value) = value.cast::<PyFMClassifier>() {
        return Ok(if validate_params {
            snapshot(value)?
        } else {
            value.try_borrow()?.inner.clone()
        }
        .into());
    }
    if let Ok(value) = value.cast::<PyMultilayerPerceptronClassifier>() {
        return Ok(if validate_params {
            snapshot(value)?
        } else {
            value.try_borrow()?.inner.clone()
        }
        .into());
    }
    Err(PyTypeError::new_err("classifier must be a native classification estimator from pyspark.ml.connect.classification"))
}

/// Linear regression backed by Spark Connect's JVM ML API.
#[pyclass(
    subclass,
    name = "LinearRegression",
    module = "pyspark.ml.connect.regression"
)]
pub struct PyLinearRegression {
    inner: LinearRegression,
    unsendable: Vec<&'static str>,
}

const LINEAR_REGRESSION_PARAMS: &[(&str, Kind)] = &[
    ("featuresCol", Kind::Str),
    ("labelCol", Kind::Str),
    ("predictionCol", Kind::Str),
    ("maxIter", Kind::Int),
    ("regParam", Kind::Float),
    ("elasticNetParam", Kind::Float),
    ("tol", Kind::Float),
    ("fitIntercept", Kind::Bool),
    ("standardization", Kind::Bool),
    ("solver", Kind::Str),
    ("weightCol", Kind::Str),
    ("aggregationDepth", Kind::Int),
    ("loss", Kind::Str),
    ("epsilon", Kind::Float),
    ("maxBlockSizeInMB", Kind::Float),
];

native_pymethods! {
    PyLinearRegression;
    jvm: "org.apache.spark.ml.regression.LinearRegression", op: Estimator;
    params: [
    "aggregationDepth" => ("getAggregationDepth", get_aggregation_depth, param_aggregation_depth),
    "elasticNetParam" => ("getElasticNetParam", get_elastic_net_param, param_elastic_net_param),
    "epsilon" => ("getEpsilon", get_epsilon, param_epsilon),
    "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
    "fitIntercept" => ("getFitIntercept", get_fit_intercept, param_fit_intercept),
    "labelCol" => ("getLabelCol", get_label_col, param_label_col),
    "loss" => ("getLoss", get_loss, param_loss),
    "maxBlockSizeInMB" => (
        "getMaxBlockSizeInMB",
        get_max_block_size_in_mb,
        param_max_block_size_in_mb
    ),
    "maxIter" => ("getMaxIter", get_max_iter, param_max_iter),
    "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
    "regParam" => ("getRegParam", get_reg_param, param_reg_param),
    "solver" => ("getSolver", get_solver, param_solver),
    "standardization" => ("getStandardization", get_standardization, param_standardization),
    "tol" => ("getTol", get_tol, param_tol),
    "weightCol" => ("getWeightCol", get_weight_col, param_weight_col),
    ];
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, featuresCol='features', labelCol='label', predictionCol='prediction', maxIter=100, regParam=0.0, elasticNetParam=0.0, tol=1e-06, fitIntercept=True, standardization=True, solver='auto', weightCol=None, aggregationDepth=2, loss='squaredError', epsilon=1.35, maxBlockSizeInMB=0.0)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: LinearRegression::new(),
            unsendable: Vec::new(),
        };
        init_params(
            &mut this,
            "LinearRegression",
            &[],
            LINEAR_REGRESSION_PARAMS,
            args,
            kwargs,
        )?;
        Ok(this)
    }
    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, featuresCol='features', labelCol='label', predictionCol='prediction', maxIter=100, regParam=0.0, elasticNetParam=0.0, tol=1e-06, fitIntercept=True, standardization=True, solver='auto', weightCol=None, aggregationDepth=2, loss='squaredError', epsilon=1.35, maxBlockSizeInMB=0.0)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(
            slf,
            "LinearRegression",
            LINEAR_REGRESSION_PARAMS,
            args,
            kwargs,
        )
    }

    #[pyo3(name = "setFeaturesCol")]
    fn set_feature_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "featuresCol", Kind::Str, value)
    }

    #[pyo3(name = "setLabelCol")]
    fn set_label_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "labelCol", Kind::Str, value)
    }

    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "predictionCol", Kind::Str, value)
    }

    #[pyo3(name = "setMaxIter")]
    fn set_max_iter<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxIter", Kind::Int, value)
    }

    #[pyo3(name = "setRegParam")]
    fn set_reg_param<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "regParam", Kind::Float, value)
    }

    #[pyo3(name = "setElasticNetParam")]
    fn set_elastic_net_param<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "elasticNetParam", Kind::Float, value)
    }

    #[pyo3(name = "setTol")]
    fn set_tol<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "tol", Kind::Float, value)
    }

    #[pyo3(name = "setFitIntercept")]
    fn set_fit_intercept<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "fitIntercept", Kind::Bool, value)
    }

    #[pyo3(name = "setStandardization")]
    fn set_standardization<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "standardization", Kind::Bool, value)
    }

    #[pyo3(name = "setSolver")]
    fn set_solver<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "solver", Kind::Str, value)
    }

    #[pyo3(name = "setWeightCol")]
    fn set_weight_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "weightCol", Kind::Str, value)
    }

    #[pyo3(name = "setAggregationDepth")]
    fn set_aggregation_depth<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "aggregationDepth", Kind::Int, value)
    }

    #[pyo3(name = "setLoss")]
    fn set_loss<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "loss", Kind::Str, value)
    }

    #[pyo3(name = "setEpsilon")]
    fn set_epsilon<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "epsilon", Kind::Float, value)
    }

    #[pyo3(name = "setMaxBlockSizeInMB")]
    fn set_max_block_size_in_mb<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxBlockSizeInMB", Kind::Float, value)
    }

    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<Py<regression_api::PyLinearRegressionModel>>> {
        fit_with(slf, dataset, params, || {
            fit_native_model(slf, dataset, |py, model, df| {
                regression_api::PyLinearRegressionModel::fitted(
                    py,
                    model,
                    df,
                    object::SummaryDataset::Train,
                )
            })
        })
    }

    /// The fits of a copy of the estimator, one per param map, as `Estimator.fitMultiple`, of
    /// PySpark's keywords.
    #[pyo3(name = "fitMultiple")]
    #[allow(non_snake_case)]
    fn fit_multiple<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        paramMaps: &Bound<'py, PyAny>,
    ) -> PyResult<Py<params_api::PyFitMultipleIterator>> {
        params_api::fit_multiple(slf.as_any(), dataset, paramMaps)
    }
}

impl params_api::NativeInstance for PyLinearRegression {
    fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        object::new_py(
            py,
            Self {
                inner: self.inner.clone(),
                unsendable: self.unsendable.clone(),
            },
        )
    }

    fn uid(&self) -> String {
        Estimator::operator(&self.inner).uid.clone()
    }

    fn set_uid(&mut self, uid: &str) {
        Estimator::operator_mut(&mut self.inner).uid = uid.to_string();
    }
}

/// Generalized linear regression fitted and cached by Spark Connect.
#[pyclass(
    subclass,
    name = "GeneralizedLinearRegression",
    module = "pyspark.ml.connect.regression"
)]
pub struct PyGeneralizedLinearRegression {
    inner: GeneralizedLinearRegression,
    unsendable: Vec<&'static str>,
}

const GENERALIZED_LINEAR_REGRESSION_PARAMS: &[(&str, Kind)] = &[
    ("labelCol", Kind::Str),
    ("featuresCol", Kind::Str),
    ("predictionCol", Kind::Str),
    ("family", Kind::Str),
    ("link", Kind::Str),
    ("fitIntercept", Kind::Bool),
    ("maxIter", Kind::Int),
    ("tol", Kind::Float),
    ("regParam", Kind::Float),
    ("weightCol", Kind::Str),
    ("solver", Kind::Str),
    ("linkPredictionCol", Kind::Str),
    ("variancePower", Kind::Float),
    ("linkPower", Kind::Float),
    ("offsetCol", Kind::Str),
    ("aggregationDepth", Kind::Int),
];

native_pymethods! {
    PyGeneralizedLinearRegression;
    jvm: "org.apache.spark.ml.regression.GeneralizedLinearRegression", op: Estimator;
    params: [
    "aggregationDepth" => ("getAggregationDepth", get_aggregation_depth, param_aggregation_depth),
    "family" => ("getFamily", get_family, param_family),
    "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
    "fitIntercept" => ("getFitIntercept", get_fit_intercept, param_fit_intercept),
    "labelCol" => ("getLabelCol", get_label_col, param_label_col),
    "link" => ("getLink", get_link, param_link),
    "linkPower" => ("getLinkPower", get_link_power, param_link_power),
    "linkPredictionCol" => (
        "getLinkPredictionCol",
        get_link_prediction_col,
        param_link_prediction_col
    ),
    "maxIter" => ("getMaxIter", get_max_iter, param_max_iter),
    "offsetCol" => ("getOffsetCol", get_offset_col, param_offset_col),
    "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
    "regParam" => ("getRegParam", get_reg_param, param_reg_param),
    "solver" => ("getSolver", get_solver, param_solver),
    "tol" => ("getTol", get_tol, param_tol),
    "variancePower" => ("getVariancePower", get_variance_power, param_variance_power),
    "weightCol" => ("getWeightCol", get_weight_col, param_weight_col),
    ];
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, labelCol='label', featuresCol='features', predictionCol='prediction', family='gaussian', link=None, fitIntercept=True, maxIter=25, tol=1e-06, regParam=0.0, weightCol=None, solver='irls', linkPredictionCol=None, variancePower=0.0, linkPower=None, offsetCol=None, aggregationDepth=2)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: GeneralizedLinearRegression::new(),
            unsendable: Vec::new(),
        };
        init_params(
            &mut this,
            "GeneralizedLinearRegression",
            &[],
            GENERALIZED_LINEAR_REGRESSION_PARAMS,
            args,
            kwargs,
        )?;
        Ok(this)
    }
    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, labelCol='label', featuresCol='features', predictionCol='prediction', family='gaussian', link=None, fitIntercept=True, maxIter=25, tol=1e-06, regParam=0.0, weightCol=None, solver='irls', linkPredictionCol=None, variancePower=0.0, linkPower=None, offsetCol=None, aggregationDepth=2)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(
            slf,
            "GeneralizedLinearRegression",
            GENERALIZED_LINEAR_REGRESSION_PARAMS,
            args,
            kwargs,
        )
    }

    #[pyo3(name = "setLabelCol")]
    fn set_label_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "labelCol", Kind::Str, value)
    }

    #[pyo3(name = "setFeaturesCol")]
    fn set_feature_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "featuresCol", Kind::Str, value)
    }

    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "predictionCol", Kind::Str, value)
    }

    #[pyo3(name = "setFamily")]
    fn set_family<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "family", Kind::Str, value)
    }

    #[pyo3(name = "setLink")]
    fn set_link<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "link", Kind::Str, value)
    }

    #[pyo3(name = "setFitIntercept")]
    fn set_fit_intercept<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "fitIntercept", Kind::Bool, value)
    }

    #[pyo3(name = "setMaxIter")]
    fn set_max_iter<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxIter", Kind::Int, value)
    }

    #[pyo3(name = "setTol")]
    fn set_tol<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "tol", Kind::Float, value)
    }

    #[pyo3(name = "setRegParam")]
    fn set_reg_param<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "regParam", Kind::Float, value)
    }

    #[pyo3(name = "setWeightCol")]
    fn set_weight_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "weightCol", Kind::Str, value)
    }

    #[pyo3(name = "setSolver")]
    fn set_solver<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "solver", Kind::Str, value)
    }

    #[pyo3(name = "setLinkPredictionCol")]
    fn set_link_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "linkPredictionCol", Kind::Str, value)
    }

    #[pyo3(name = "setVariancePower")]
    fn set_variance_power<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "variancePower", Kind::Float, value)
    }

    #[pyo3(name = "setLinkPower")]
    fn set_link_power<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "linkPower", Kind::Float, value)
    }

    #[pyo3(name = "setOffsetCol")]
    fn set_offset_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "offsetCol", Kind::Str, value)
    }

    #[pyo3(name = "setAggregationDepth")]
    fn set_aggregation_depth<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "aggregationDepth", Kind::Int, value)
    }

    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<Py<regression_api::PyGeneralizedLinearRegressionModel>>> {
        fit_with(slf, dataset, params, || {
            fit_native_model(slf, dataset, |py, model, df| {
                regression_api::PyGeneralizedLinearRegressionModel::fitted(
                    py,
                    model,
                    df,
                    object::SummaryDataset::Train,
                )
            })
        })
    }

    /// The fits of a copy of the estimator, one per param map, as `Estimator.fitMultiple`, of
    /// PySpark's keywords.
    #[pyo3(name = "fitMultiple")]
    #[allow(non_snake_case)]
    fn fit_multiple<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        paramMaps: &Bound<'py, PyAny>,
    ) -> PyResult<Py<params_api::PyFitMultipleIterator>> {
        params_api::fit_multiple(slf.as_any(), dataset, paramMaps)
    }
}

impl params_api::NativeInstance for PyGeneralizedLinearRegression {
    fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        object::new_py(
            py,
            Self {
                inner: self.inner.clone(),
                unsendable: self.unsendable.clone(),
            },
        )
    }

    fn uid(&self) -> String {
        Estimator::operator(&self.inner).uid.clone()
    }

    fn set_uid(&mut self, uid: &str) {
        Estimator::operator_mut(&mut self.inner).uid = uid.to_string();
    }
}

/// Weibull accelerated failure time regression using Spark's JVM implementation.
#[pyclass(
    subclass,
    name = "AFTSurvivalRegression",
    module = "pyspark.ml.connect.regression"
)]
pub struct PyAFTSurvivalRegression {
    inner: AFTSurvivalRegression,
    unsendable: Vec<&'static str>,
}

const AFTSURVIVAL_REGRESSION_PARAMS: &[(&str, Kind)] = &[
    ("featuresCol", Kind::Str),
    ("labelCol", Kind::Str),
    ("predictionCol", Kind::Str),
    ("fitIntercept", Kind::Bool),
    ("maxIter", Kind::Int),
    ("tol", Kind::Float),
    ("censorCol", Kind::Str),
    ("quantileProbabilities", Kind::FloatList),
    ("quantilesCol", Kind::Str),
    ("aggregationDepth", Kind::Int),
    ("maxBlockSizeInMB", Kind::Float),
];

native_pymethods! {
    PyAFTSurvivalRegression;
    jvm: "org.apache.spark.ml.regression.AFTSurvivalRegression", op: Estimator;
    params: [
    "aggregationDepth" => ("getAggregationDepth", get_aggregation_depth, param_aggregation_depth),
    "censorCol" => ("getCensorCol", get_censor_col, param_censor_col),
    "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
    "fitIntercept" => ("getFitIntercept", get_fit_intercept, param_fit_intercept),
    "labelCol" => ("getLabelCol", get_label_col, param_label_col),
    "maxBlockSizeInMB" => (
        "getMaxBlockSizeInMB",
        get_max_block_size_in_mb,
        param_max_block_size_in_mb
    ),
    "maxIter" => ("getMaxIter", get_max_iter, param_max_iter),
    "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
    "quantileProbabilities" => (
        "getQuantileProbabilities",
        get_quantile_probabilities,
        param_quantile_probabilities
    ),
    "quantilesCol" => ("getQuantilesCol", get_quantiles_col, param_quantiles_col),
    "tol" => ("getTol", get_tol, param_tol),
    ];
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, featuresCol='features', labelCol='label', predictionCol='prediction', fitIntercept=True, maxIter=100, tol=1e-06, censorCol='censor', quantileProbabilities=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99], quantilesCol=None, aggregationDepth=2, maxBlockSizeInMB=0.0)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: AFTSurvivalRegression::new(),
            unsendable: Vec::new(),
        };
        init_params(
            &mut this,
            "AFTSurvivalRegression",
            &[],
            AFTSURVIVAL_REGRESSION_PARAMS,
            args,
            kwargs,
        )?;
        Ok(this)
    }

    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, featuresCol='features', labelCol='label', predictionCol='prediction', fitIntercept=True, maxIter=100, tol=1e-06, censorCol='censor', quantileProbabilities=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99], quantilesCol=None, aggregationDepth=2, maxBlockSizeInMB=0.0)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(
            slf,
            "AFTSurvivalRegression",
            AFTSURVIVAL_REGRESSION_PARAMS,
            args,
            kwargs,
        )
    }

    #[pyo3(name = "setFeaturesCol")]
    fn set_feature_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "featuresCol", Kind::Str, value)
    }

    #[pyo3(name = "setLabelCol")]
    fn set_label_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "labelCol", Kind::Str, value)
    }

    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "predictionCol", Kind::Str, value)
    }

    #[pyo3(name = "setFitIntercept")]
    fn set_fit_intercept<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "fitIntercept", Kind::Bool, value)
    }

    #[pyo3(name = "setMaxIter")]
    fn set_max_iter<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxIter", Kind::Int, value)
    }

    #[pyo3(name = "setTol")]
    fn set_tol<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "tol", Kind::Float, value)
    }

    #[pyo3(name = "setCensorCol")]
    fn set_censor_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "censorCol", Kind::Str, value)
    }

    #[pyo3(name = "setQuantileProbabilities")]
    fn set_quantile_probabilities<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "quantileProbabilities", Kind::FloatList, value)
    }

    #[pyo3(name = "setQuantilesCol")]
    fn set_quantiles_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "quantilesCol", Kind::Str, value)
    }

    #[pyo3(name = "setAggregationDepth")]
    fn set_aggregation_depth<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "aggregationDepth", Kind::Int, value)
    }

    #[pyo3(name = "setMaxBlockSizeInMB")]
    fn set_max_block_size_in_mb<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxBlockSizeInMB", Kind::Float, value)
    }

    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<Py<regression_api::PyAFTSurvivalRegressionModel>>> {
        fit_with(slf, dataset, params, || {
            fit_native_model(slf, dataset, |py, model, df| {
                regression_api::PyAFTSurvivalRegressionModel::fitted(
                    py,
                    model,
                    df,
                    object::SummaryDataset::None,
                )
            })
        })
    }

    /// The fits of a copy of the estimator, one per param map, as `Estimator.fitMultiple`, of
    /// PySpark's keywords.
    #[pyo3(name = "fitMultiple")]
    #[allow(non_snake_case)]
    fn fit_multiple<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        paramMaps: &Bound<'py, PyAny>,
    ) -> PyResult<Py<params_api::PyFitMultipleIterator>> {
        params_api::fit_multiple(slf.as_any(), dataset, paramMaps)
    }
}

impl params_api::NativeInstance for PyAFTSurvivalRegression {
    fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        object::new_py(
            py,
            Self {
                inner: self.inner.clone(),
                unsendable: self.unsendable.clone(),
            },
        )
    }

    fn uid(&self) -> String {
        Estimator::operator(&self.inner).uid.clone()
    }

    fn set_uid(&mut self, uid: &str) {
        Estimator::operator_mut(&mut self.inner).uid = uid.to_string();
    }
}

/// Univariate isotonic or antitonic regression using Spark's JVM implementation.
#[pyclass(
    subclass,
    name = "IsotonicRegression",
    module = "pyspark.ml.connect.regression"
)]
pub struct PyIsotonicRegression {
    inner: IsotonicRegression,
    unsendable: Vec<&'static str>,
}

const ISOTONIC_REGRESSION_PARAMS: &[(&str, Kind)] = &[
    ("featuresCol", Kind::Str),
    ("labelCol", Kind::Str),
    ("predictionCol", Kind::Str),
    ("weightCol", Kind::Str),
    ("isotonic", Kind::Bool),
    ("featureIndex", Kind::Int),
];

native_pymethods! {
    PyIsotonicRegression;
    jvm: "org.apache.spark.ml.regression.IsotonicRegression", op: Estimator;
    params: [
    "featureIndex" => ("getFeatureIndex", get_feature_index, param_feature_index),
    "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
    "isotonic" => ("getIsotonic", get_isotonic, param_isotonic),
    "labelCol" => ("getLabelCol", get_label_col, param_label_col),
    "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
    "weightCol" => ("getWeightCol", get_weight_col, param_weight_col),
    ];
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, featuresCol='features', labelCol='label', predictionCol='prediction', weightCol=None, isotonic=True, featureIndex=0)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: IsotonicRegression::new(),
            unsendable: Vec::new(),
        };
        init_params(
            &mut this,
            "IsotonicRegression",
            &[],
            ISOTONIC_REGRESSION_PARAMS,
            args,
            kwargs,
        )?;
        Ok(this)
    }

    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, featuresCol='features', labelCol='label', predictionCol='prediction', weightCol=None, isotonic=True, featureIndex=0)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(
            slf,
            "IsotonicRegression",
            ISOTONIC_REGRESSION_PARAMS,
            args,
            kwargs,
        )
    }

    #[pyo3(name = "setFeaturesCol")]
    fn set_feature_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "featuresCol", Kind::Str, value)
    }

    #[pyo3(name = "setLabelCol")]
    fn set_label_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "labelCol", Kind::Str, value)
    }

    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "predictionCol", Kind::Str, value)
    }

    #[pyo3(name = "setWeightCol")]
    fn set_weight_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "weightCol", Kind::Str, value)
    }

    #[pyo3(name = "setIsotonic")]
    fn set_isotonic<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "isotonic", Kind::Bool, value)
    }

    #[pyo3(name = "setFeatureIndex")]
    fn set_feature_index<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "featureIndex", Kind::Int, value)
    }

    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<Py<regression_api::PyIsotonicRegressionModel>>> {
        fit_with(slf, dataset, params, || {
            fit_native_model(slf, dataset, |py, model, df| {
                regression_api::PyIsotonicRegressionModel::fitted(
                    py,
                    model,
                    df,
                    object::SummaryDataset::None,
                )
            })
        })
    }

    /// The fits of a copy of the estimator, one per param map, as `Estimator.fitMultiple`, of
    /// PySpark's keywords.
    #[pyo3(name = "fitMultiple")]
    #[allow(non_snake_case)]
    fn fit_multiple<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        paramMaps: &Bound<'py, PyAny>,
    ) -> PyResult<Py<params_api::PyFitMultipleIterator>> {
        params_api::fit_multiple(slf.as_any(), dataset, paramMaps)
    }
}

impl params_api::NativeInstance for PyIsotonicRegression {
    fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        object::new_py(
            py,
            Self {
                inner: self.inner.clone(),
                unsendable: self.unsendable.clone(),
            },
        )
    }

    fn uid(&self) -> String {
        Estimator::operator(&self.inner).uid.clone()
    }

    fn set_uid(&mut self, uid: &str) {
        Estimator::operator_mut(&mut self.inner).uid = uid.to_string();
    }
}

/// Factorization machines regression using Spark's JVM implementation.
#[pyclass(
    subclass,
    name = "FMRegressor",
    module = "pyspark.ml.connect.regression"
)]
pub struct PyFMRegressor {
    inner: FMRegressor,
    unsendable: Vec<&'static str>,
}

const FMREGRESSOR_PARAMS: &[(&str, Kind)] = &[
    ("featuresCol", Kind::Str),
    ("labelCol", Kind::Str),
    ("predictionCol", Kind::Str),
    ("factorSize", Kind::Int),
    ("fitIntercept", Kind::Bool),
    ("fitLinear", Kind::Bool),
    ("regParam", Kind::Float),
    ("miniBatchFraction", Kind::Float),
    ("initStd", Kind::Float),
    ("maxIter", Kind::Int),
    ("stepSize", Kind::Float),
    ("tol", Kind::Float),
    ("solver", Kind::Str),
    ("seed", Kind::Int),
];

native_pymethods! {
    PyFMRegressor;
    jvm: "org.apache.spark.ml.regression.FMRegressor", op: Estimator;
    params: [
    "factorSize" => ("getFactorSize", get_factor_size, param_factor_size),
    "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
    "fitIntercept" => ("getFitIntercept", get_fit_intercept, param_fit_intercept),
    "fitLinear" => ("getFitLinear", get_fit_linear, param_fit_linear),
    "initStd" => ("getInitStd", get_init_std, param_init_std),
    "labelCol" => ("getLabelCol", get_label_col, param_label_col),
    "maxIter" => ("getMaxIter", get_max_iter, param_max_iter),
    "miniBatchFraction" => (
        "getMiniBatchFraction",
        get_mini_batch_fraction,
        param_mini_batch_fraction
    ),
    "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
    "regParam" => ("getRegParam", get_reg_param, param_reg_param),
    "seed" => ("getSeed", get_seed, param_seed),
    "solver" => ("getSolver", get_solver, param_solver),
    "stepSize" => ("getStepSize", get_step_size, param_step_size),
    "tol" => ("getTol", get_tol, param_tol),
    "weightCol" => ("getWeightCol", get_weight_col, param_weight_col),
    ];
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, featuresCol='features', labelCol='label', predictionCol='prediction', factorSize=8, fitIntercept=True, fitLinear=True, regParam=0.0, miniBatchFraction=1.0, initStd=0.01, maxIter=100, stepSize=1.0, tol=1e-06, solver='adamW', seed=None)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: FMRegressor::new(),
            unsendable: Vec::new(),
        };
        init_params(
            &mut this,
            "FMRegressor",
            &[],
            FMREGRESSOR_PARAMS,
            args,
            kwargs,
        )?;
        Ok(this)
    }

    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, featuresCol='features', labelCol='label', predictionCol='prediction', factorSize=8, fitIntercept=True, fitLinear=True, regParam=0.0, miniBatchFraction=1.0, initStd=0.01, maxIter=100, stepSize=1.0, tol=1e-06, solver='adamW', seed=None)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(slf, "FMRegressor", FMREGRESSOR_PARAMS, args, kwargs)
    }

    #[pyo3(name = "setFeaturesCol")]
    fn set_feature_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "featuresCol", Kind::Str, value)
    }

    #[pyo3(name = "setLabelCol")]
    fn set_label_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "labelCol", Kind::Str, value)
    }

    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "predictionCol", Kind::Str, value)
    }

    #[pyo3(name = "setFactorSize")]
    fn set_factor_size<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "factorSize", Kind::Int, value)
    }

    #[pyo3(name = "setFitIntercept")]
    fn set_fit_intercept<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "fitIntercept", Kind::Bool, value)
    }

    #[pyo3(name = "setFitLinear")]
    fn set_fit_linear<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "fitLinear", Kind::Bool, value)
    }

    #[pyo3(name = "setRegParam")]
    fn set_reg_param<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "regParam", Kind::Float, value)
    }

    #[pyo3(name = "setMiniBatchFraction")]
    fn set_mini_batch_fraction<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "miniBatchFraction", Kind::Float, value)
    }

    #[pyo3(name = "setInitStd")]
    fn set_init_std<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "initStd", Kind::Float, value)
    }

    #[pyo3(name = "setMaxIter")]
    fn set_max_iter<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxIter", Kind::Int, value)
    }

    #[pyo3(name = "setStepSize")]
    fn set_step_size<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "stepSize", Kind::Float, value)
    }

    #[pyo3(name = "setTol")]
    fn set_tol<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "tol", Kind::Float, value)
    }

    #[pyo3(name = "setSolver")]
    fn set_solver<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "solver", Kind::Str, value)
    }

    #[pyo3(name = "setSeed")]
    fn set_seed<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "seed", Kind::Int, value)
    }

    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<Py<regression_api::PyFMRegressionModel>>> {
        fit_with(slf, dataset, params, || {
            fit_native_model(slf, dataset, |py, model, df| {
                regression_api::PyFMRegressionModel::fitted(
                    py,
                    model,
                    df,
                    object::SummaryDataset::None,
                )
            })
        })
    }

    /// The fits of a copy of the estimator, one per param map, as `Estimator.fitMultiple`, of
    /// PySpark's keywords.
    #[pyo3(name = "fitMultiple")]
    #[allow(non_snake_case)]
    fn fit_multiple<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        paramMaps: &Bound<'py, PyAny>,
    ) -> PyResult<Py<params_api::PyFitMultipleIterator>> {
        params_api::fit_multiple(slf.as_any(), dataset, paramMaps)
    }
}

impl params_api::NativeInstance for PyFMRegressor {
    fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        object::new_py(
            py,
            Self {
                inner: self.inner.clone(),
                unsendable: self.unsendable.clone(),
            },
        )
    }

    fn uid(&self) -> String {
        Estimator::operator(&self.inner).uid.clone()
    }

    fn set_uid(&mut self, uid: &str) {
        Estimator::operator_mut(&mut self.inner).uid = uid.to_string();
    }
}

/// Decision tree classifier fitted and cached by Spark Connect.
#[pyclass(
    subclass,
    name = "DecisionTreeClassifier",
    module = "pyspark.ml.connect.classification"
)]
pub struct PyDecisionTreeClassifier {
    inner: DecisionTreeClassifier,
    unsendable: Vec<&'static str>,
}

const DECISION_TREE_CLASSIFIER_PARAMS: &[(&str, Kind)] = &[
    ("featuresCol", Kind::Str),
    ("labelCol", Kind::Str),
    ("predictionCol", Kind::Str),
    ("probabilityCol", Kind::Str),
    ("rawPredictionCol", Kind::Str),
    ("maxDepth", Kind::Int),
    ("maxBins", Kind::Int),
    ("minInstancesPerNode", Kind::Int),
    ("minInfoGain", Kind::Float),
    ("pruneTree", Kind::Bool),
    ("maxMemoryInMB", Kind::Int),
    ("cacheNodeIds", Kind::Bool),
    ("checkpointInterval", Kind::Int),
    ("impurity", Kind::Str),
    ("seed", Kind::Int),
    ("leafCol", Kind::Str),
    ("minWeightFractionPerNode", Kind::Float),
    ("weightCol", Kind::Str),
];

native_pymethods! {
    PyDecisionTreeClassifier;
    jvm: "org.apache.spark.ml.classification.DecisionTreeClassifier", op: Estimator;
    params: [
        "cacheNodeIds" => ("getCacheNodeIds", get_cache_node_ids, param_cache_node_ids),
        "checkpointInterval" => (
            "getCheckpointInterval",
            get_checkpoint_interval,
            param_checkpoint_interval
        ),
        "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
        "impurity" => ("getImpurity", get_impurity, param_impurity),
        "labelCol" => ("getLabelCol", get_label_col, param_label_col),
        "leafCol" => ("getLeafCol", get_leaf_col, param_leaf_col),
        "maxBins" => ("getMaxBins", get_max_bins, param_max_bins),
        "maxDepth" => ("getMaxDepth", get_max_depth, param_max_depth),
        "maxMemoryInMB" => ("getMaxMemoryInMB", get_max_memory_in_mb, param_max_memory_in_mb),
        "minInfoGain" => ("getMinInfoGain", get_min_info_gain, param_min_info_gain),
        "minInstancesPerNode" => (
            "getMinInstancesPerNode",
            get_min_instances_per_node,
            param_min_instances_per_node
        ),
        "minWeightFractionPerNode" => (
            "getMinWeightFractionPerNode",
            get_min_weight_fraction_per_node,
            param_min_weight_fraction_per_node
        ),
        "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
        "probabilityCol" => ("getProbabilityCol", get_probability_col, param_probability_col),
        "pruneTree" => ("getPruneTree", get_prune_tree, param_prune_tree),
        "rawPredictionCol" => (
            "getRawPredictionCol",
            get_raw_prediction_col,
            param_raw_prediction_col
        ),
        "seed" => ("getSeed", get_seed, param_seed),
        "thresholds" => ("getThresholds", get_thresholds, param_thresholds),
        "weightCol" => ("getWeightCol", get_weight_col, param_weight_col),
    ];
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, featuresCol='features', labelCol='label', predictionCol='prediction', probabilityCol='probability', rawPredictionCol='rawPrediction', maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, pruneTree=True, maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10, impurity='gini', seed=None, weightCol=None, leafCol='', minWeightFractionPerNode=0.0)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: DecisionTreeClassifier::new(),
            unsendable: Vec::new(),
        };
        init_params(
            &mut this,
            "DecisionTreeClassifier",
            &[],
            DECISION_TREE_CLASSIFIER_PARAMS,
            args,
            kwargs,
        )?;
        Ok(this)
    }
    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, featuresCol='features', labelCol='label', predictionCol='prediction', probabilityCol='probability', rawPredictionCol='rawPrediction', maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, pruneTree=True, maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10, impurity='gini', seed=None, weightCol=None, leafCol='', minWeightFractionPerNode=0.0)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(
            slf,
            "DecisionTreeClassifier",
            DECISION_TREE_CLASSIFIER_PARAMS,
            args,
            kwargs,
        )
    }

    #[pyo3(name = "setFeaturesCol")]
    fn set_feature_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "featuresCol", Kind::Str, value)
    }

    #[pyo3(name = "setLabelCol")]
    fn set_label_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "labelCol", Kind::Str, value)
    }

    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "predictionCol", Kind::Str, value)
    }

    #[pyo3(name = "setProbabilityCol")]
    fn set_probability_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "probabilityCol", Kind::Str, value)
    }

    #[pyo3(name = "setRawPredictionCol")]
    fn set_raw_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "rawPredictionCol", Kind::Str, value)
    }

    #[pyo3(name = "setMaxDepth")]
    fn set_max_depth<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxDepth", Kind::Int, value)
    }

    #[pyo3(name = "setMaxBins")]
    fn set_max_bins<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxBins", Kind::Int, value)
    }

    #[pyo3(name = "setMinInstancesPerNode")]
    fn set_min_instances_per_node<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "minInstancesPerNode", Kind::Int, value)
    }

    #[pyo3(name = "setMinInfoGain")]
    fn set_min_info_gain<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "minInfoGain", Kind::Float, value)
    }

    #[pyo3(name = "setPruneTree")]
    fn set_prune_tree<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "pruneTree", Kind::Bool, value)
    }

    #[pyo3(name = "setMaxMemoryInMB")]
    fn set_max_memory_in_mb<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxMemoryInMB", Kind::Int, value)
    }

    #[pyo3(name = "setCacheNodeIds")]
    fn set_cache_node_ids<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "cacheNodeIds", Kind::Bool, value)
    }

    #[pyo3(name = "setCheckpointInterval")]
    fn set_checkpoint_interval<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "checkpointInterval", Kind::Int, value)
    }

    #[pyo3(name = "setImpurity")]
    fn set_impurity<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "impurity", Kind::Str, value)
    }

    #[pyo3(name = "setSeed")]
    fn set_seed<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "seed", Kind::Int, value)
    }

    #[pyo3(name = "setLeafCol")]
    fn set_leaf_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "leafCol", Kind::Str, value)
    }

    #[pyo3(name = "setMinWeightFractionPerNode")]
    fn set_min_weight_fraction_per_node<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "minWeightFractionPerNode", Kind::Float, value)
    }

    #[pyo3(name = "setWeightCol")]
    fn set_weight_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "weightCol", Kind::Str, value)
    }

    #[pyo3(name = "setThresholds")]
    fn set_thresholds<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "thresholds", Kind::FloatList, value)
    }

    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<Py<classification_api::PyDecisionTreeClassificationModel>>> {
        fit_with(slf, dataset, params, || {
            fit_native_model(slf, dataset, |py, model, df| {
                classification_api::PyDecisionTreeClassificationModel::fitted(
                    py,
                    model,
                    df,
                    object::SummaryDataset::None,
                )
            })
        })
    }

    /// The fits of a copy of the estimator, one per param map, as `Estimator.fitMultiple`, of
    /// PySpark's keywords.
    #[pyo3(name = "fitMultiple")]
    #[allow(non_snake_case)]
    fn fit_multiple<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        paramMaps: &Bound<'py, PyAny>,
    ) -> PyResult<Py<params_api::PyFitMultipleIterator>> {
        params_api::fit_multiple(slf.as_any(), dataset, paramMaps)
    }

    /// The supportedImpurities of the class, as PySpark's class attribute.
    #[classattr]
    #[allow(non_snake_case)]
    fn supportedImpurities() -> Vec<&'static str> {
        vec!["entropy", "gini"]
    }
}

impl params_api::NativeInstance for PyDecisionTreeClassifier {
    fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        object::new_py(
            py,
            Self {
                inner: self.inner.clone(),
                unsendable: self.unsendable.clone(),
            },
        )
    }

    fn uid(&self) -> String {
        Estimator::operator(&self.inner).uid.clone()
    }

    fn set_uid(&mut self, uid: &str) {
        Estimator::operator_mut(&mut self.inner).uid = uid.to_string();
    }
}

/// Naive Bayes classifier fitted and cached by Spark Connect.
#[pyclass(
    subclass,
    name = "NaiveBayes",
    module = "pyspark.ml.connect.classification"
)]
pub struct PyNaiveBayes {
    inner: NaiveBayes,
    unsendable: Vec<&'static str>,
}

const NAIVE_BAYES_PARAMS: &[(&str, Kind)] = &[
    ("featuresCol", Kind::Str),
    ("labelCol", Kind::Str),
    ("predictionCol", Kind::Str),
    ("probabilityCol", Kind::Str),
    ("rawPredictionCol", Kind::Str),
    ("smoothing", Kind::Float),
    ("modelType", Kind::Str),
    ("thresholds", Kind::FloatList),
    ("weightCol", Kind::Str),
];

native_pymethods! {
    PyNaiveBayes;
    jvm: "org.apache.spark.ml.classification.NaiveBayes", op: Estimator;
    params: [
        "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
        "labelCol" => ("getLabelCol", get_label_col, param_label_col),
        "modelType" => ("getModelType", get_model_type, param_model_type),
        "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
        "probabilityCol" => ("getProbabilityCol", get_probability_col, param_probability_col),
        "rawPredictionCol" => (
            "getRawPredictionCol",
            get_raw_prediction_col,
            param_raw_prediction_col
        ),
        "smoothing" => ("getSmoothing", get_smoothing, param_smoothing),
        "thresholds" => ("getThresholds", get_thresholds, param_thresholds),
        "weightCol" => ("getWeightCol", get_weight_col, param_weight_col),
    ];
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, featuresCol='features', labelCol='label', predictionCol='prediction', probabilityCol='probability', rawPredictionCol='rawPrediction', smoothing=1.0, modelType='multinomial', thresholds=None, weightCol=None)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: NaiveBayes::new(),
            unsendable: Vec::new(),
        };
        init_params(
            &mut this,
            "NaiveBayes",
            &[],
            NAIVE_BAYES_PARAMS,
            args,
            kwargs,
        )?;
        Ok(this)
    }
    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, featuresCol='features', labelCol='label', predictionCol='prediction', probabilityCol='probability', rawPredictionCol='rawPrediction', smoothing=1.0, modelType='multinomial', thresholds=None, weightCol=None)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(slf, "NaiveBayes", NAIVE_BAYES_PARAMS, args, kwargs)
    }

    #[pyo3(name = "setFeaturesCol")]
    fn set_feature_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "featuresCol", Kind::Str, value)
    }

    #[pyo3(name = "setLabelCol")]
    fn set_label_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "labelCol", Kind::Str, value)
    }

    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "predictionCol", Kind::Str, value)
    }

    #[pyo3(name = "setProbabilityCol")]
    fn set_probability_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "probabilityCol", Kind::Str, value)
    }

    #[pyo3(name = "setRawPredictionCol")]
    fn set_raw_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "rawPredictionCol", Kind::Str, value)
    }

    #[pyo3(name = "setSmoothing")]
    fn set_smoothing<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "smoothing", Kind::Float, value)
    }

    #[pyo3(name = "setModelType")]
    fn set_model_type<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "modelType", Kind::Str, value)
    }

    #[pyo3(name = "setThresholds")]
    fn set_thresholds<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "thresholds", Kind::FloatList, value)
    }

    #[pyo3(name = "setWeightCol")]
    fn set_weight_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "weightCol", Kind::Str, value)
    }

    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<Py<classification_api::PyNaiveBayesModel>>> {
        fit_with(slf, dataset, params, || {
            fit_native_model(slf, dataset, |py, model, df| {
                classification_api::PyNaiveBayesModel::fitted(
                    py,
                    model,
                    df,
                    object::SummaryDataset::None,
                )
            })
        })
    }

    /// The fits of a copy of the estimator, one per param map, as `Estimator.fitMultiple`, of
    /// PySpark's keywords.
    #[pyo3(name = "fitMultiple")]
    #[allow(non_snake_case)]
    fn fit_multiple<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        paramMaps: &Bound<'py, PyAny>,
    ) -> PyResult<Py<params_api::PyFitMultipleIterator>> {
        params_api::fit_multiple(slf.as_any(), dataset, paramMaps)
    }
}

impl params_api::NativeInstance for PyNaiveBayes {
    fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        object::new_py(
            py,
            Self {
                inner: self.inner.clone(),
                unsendable: self.unsendable.clone(),
            },
        )
    }

    fn uid(&self) -> String {
        Estimator::operator(&self.inner).uid.clone()
    }

    fn set_uid(&mut self, uid: &str) {
        Estimator::operator_mut(&mut self.inner).uid = uid.to_string();
    }
}

/// Gradient-boosted binary classifier fitted and cached by Spark Connect.
#[pyclass(
    subclass,
    name = "GBTClassifier",
    module = "pyspark.ml.connect.classification"
)]
pub struct PyGBTClassifier {
    inner: GBTClassifier,
    unsendable: Vec<&'static str>,
}

const GBT_CLASSIFIER_PARAMS: &[(&str, Kind)] = &[
    ("featuresCol", Kind::Str),
    ("labelCol", Kind::Str),
    ("predictionCol", Kind::Str),
    ("maxDepth", Kind::Int),
    ("maxBins", Kind::Int),
    ("minInstancesPerNode", Kind::Int),
    ("minInfoGain", Kind::Float),
    ("maxMemoryInMB", Kind::Int),
    ("cacheNodeIds", Kind::Bool),
    ("checkpointInterval", Kind::Int),
    ("lossType", Kind::Str),
    ("maxIter", Kind::Int),
    ("stepSize", Kind::Float),
    ("seed", Kind::Int),
    ("subsamplingRate", Kind::Float),
    ("impurity", Kind::Str),
    ("featureSubsetStrategy", Kind::Str),
    ("validationTol", Kind::Float),
    ("validationIndicatorCol", Kind::Str),
    ("leafCol", Kind::Str),
    ("minWeightFractionPerNode", Kind::Float),
    ("weightCol", Kind::Str),
];

native_pymethods! {
    PyGBTClassifier;
    jvm: "org.apache.spark.ml.classification.GBTClassifier", op: Estimator;
    params: [
        "cacheNodeIds" => ("getCacheNodeIds", get_cache_node_ids, param_cache_node_ids),
        "checkpointInterval" => (
            "getCheckpointInterval",
            get_checkpoint_interval,
            param_checkpoint_interval
        ),
        "featureSubsetStrategy" => (
            "getFeatureSubsetStrategy",
            get_feature_subset_strategy,
            param_feature_subset_strategy
        ),
        "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
        "impurity" => ("getImpurity", get_impurity, param_impurity),
        "labelCol" => ("getLabelCol", get_label_col, param_label_col),
        "leafCol" => ("getLeafCol", get_leaf_col, param_leaf_col),
        "lossType" => ("getLossType", get_loss_type, param_loss_type),
        "maxBins" => ("getMaxBins", get_max_bins, param_max_bins),
        "maxDepth" => ("getMaxDepth", get_max_depth, param_max_depth),
        "maxIter" => ("getMaxIter", get_max_iter, param_max_iter),
        "maxMemoryInMB" => ("getMaxMemoryInMB", get_max_memory_in_mb, param_max_memory_in_mb),
        "minInfoGain" => ("getMinInfoGain", get_min_info_gain, param_min_info_gain),
        "minInstancesPerNode" => (
            "getMinInstancesPerNode",
            get_min_instances_per_node,
            param_min_instances_per_node
        ),
        "minWeightFractionPerNode" => (
            "getMinWeightFractionPerNode",
            get_min_weight_fraction_per_node,
            param_min_weight_fraction_per_node
        ),
        "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
        "probabilityCol" => ("getProbabilityCol", get_probability_col, param_probability_col),
        "rawPredictionCol" => (
            "getRawPredictionCol",
            get_raw_prediction_col,
            param_raw_prediction_col
        ),
        "seed" => ("getSeed", get_seed, param_seed),
        "stepSize" => ("getStepSize", get_step_size, param_step_size),
        "subsamplingRate" => ("getSubsamplingRate", get_subsampling_rate, param_subsampling_rate),
        "thresholds" => ("getThresholds", get_thresholds, param_thresholds),
        "validationIndicatorCol" => (
            "getValidationIndicatorCol",
            get_validation_indicator_col,
            param_validation_indicator_col
        ),
        "validationTol" => ("getValidationTol", get_validation_tol, param_validation_tol),
        "weightCol" => ("getWeightCol", get_weight_col, param_weight_col),
    ];
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, featuresCol='features', labelCol='label', predictionCol='prediction', maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10, lossType='logistic', maxIter=20, stepSize=0.1, seed=None, subsamplingRate=1.0, impurity='variance', featureSubsetStrategy='all', validationTol=0.01, validationIndicatorCol=None, leafCol='', minWeightFractionPerNode=0.0, weightCol=None)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: GBTClassifier::new(),
            unsendable: Vec::new(),
        };
        init_params(
            &mut this,
            "GBTClassifier",
            &[],
            GBT_CLASSIFIER_PARAMS,
            args,
            kwargs,
        )?;
        Ok(this)
    }
    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, featuresCol='features', labelCol='label', predictionCol='prediction', maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10, lossType='logistic', maxIter=20, stepSize=0.1, seed=None, subsamplingRate=1.0, impurity='variance', featureSubsetStrategy='all', validationTol=0.01, validationIndicatorCol=None, leafCol='', minWeightFractionPerNode=0.0, weightCol=None)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(slf, "GBTClassifier", GBT_CLASSIFIER_PARAMS, args, kwargs)
    }

    #[pyo3(name = "setFeaturesCol")]
    fn set_feature_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "featuresCol", Kind::Str, value)
    }

    #[pyo3(name = "setLabelCol")]
    fn set_label_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "labelCol", Kind::Str, value)
    }

    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "predictionCol", Kind::Str, value)
    }

    #[pyo3(name = "setProbabilityCol")]
    fn set_probability_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "probabilityCol", Kind::Str, value)
    }

    #[pyo3(name = "setRawPredictionCol")]
    fn set_raw_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "rawPredictionCol", Kind::Str, value)
    }

    #[pyo3(name = "setMaxDepth")]
    fn set_max_depth<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxDepth", Kind::Int, value)
    }

    #[pyo3(name = "setMaxBins")]
    fn set_max_bins<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxBins", Kind::Int, value)
    }

    #[pyo3(name = "setMinInstancesPerNode")]
    fn set_min_instances_per_node<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "minInstancesPerNode", Kind::Int, value)
    }

    #[pyo3(name = "setMinInfoGain")]
    fn set_min_info_gain<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "minInfoGain", Kind::Float, value)
    }

    #[pyo3(name = "setMaxMemoryInMB")]
    fn set_max_memory_in_mb<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxMemoryInMB", Kind::Int, value)
    }

    #[pyo3(name = "setCacheNodeIds")]
    fn set_cache_node_ids<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "cacheNodeIds", Kind::Bool, value)
    }

    #[pyo3(name = "setCheckpointInterval")]
    fn set_checkpoint_interval<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "checkpointInterval", Kind::Int, value)
    }

    #[pyo3(name = "setImpurity")]
    fn set_impurity<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "impurity", Kind::Str, value)
    }

    #[pyo3(name = "setFeatureSubsetStrategy")]
    fn set_feature_subset_strategy<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "featureSubsetStrategy", Kind::Str, value)
    }

    #[pyo3(name = "setSeed")]
    fn set_seed<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "seed", Kind::Int, value)
    }

    #[pyo3(name = "setSubsamplingRate")]
    fn set_subsampling_rate<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "subsamplingRate", Kind::Float, value)
    }

    #[pyo3(name = "setLeafCol")]
    fn set_leaf_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "leafCol", Kind::Str, value)
    }

    #[pyo3(name = "setMinWeightFractionPerNode")]
    fn set_min_weight_fraction_per_node<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "minWeightFractionPerNode", Kind::Float, value)
    }

    #[pyo3(name = "setWeightCol")]
    fn set_weight_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "weightCol", Kind::Str, value)
    }

    #[pyo3(name = "setThresholds")]
    fn set_thresholds<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "thresholds", Kind::FloatList, value)
    }

    #[pyo3(name = "setLossType")]
    fn set_loss_type<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "lossType", Kind::Str, value)
    }

    #[pyo3(name = "setMaxIter")]
    fn set_max_iter<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxIter", Kind::Int, value)
    }

    #[pyo3(name = "setStepSize")]
    fn set_step_size<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "stepSize", Kind::Float, value)
    }

    #[pyo3(name = "setValidationIndicatorCol")]
    fn set_validation_indicator_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "validationIndicatorCol", Kind::Str, value)
    }

    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<Py<classification_api::PyGBTClassificationModel>>> {
        fit_with(slf, dataset, params, || {
            fit_native_model(slf, dataset, |py, model, df| {
                classification_api::PyGBTClassificationModel::fitted(
                    py,
                    model,
                    df,
                    object::SummaryDataset::None,
                )
            })
        })
    }

    /// The fits of a copy of the estimator, one per param map, as `Estimator.fitMultiple`, of
    /// PySpark's keywords.
    #[pyo3(name = "fitMultiple")]
    #[allow(non_snake_case)]
    fn fit_multiple<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        paramMaps: &Bound<'py, PyAny>,
    ) -> PyResult<Py<params_api::PyFitMultipleIterator>> {
        params_api::fit_multiple(slf.as_any(), dataset, paramMaps)
    }

    /// The supportedFeatureSubsetStrategies of the class, as PySpark's class attribute.
    #[classattr]
    #[allow(non_snake_case)]
    fn supportedFeatureSubsetStrategies() -> Vec<&'static str> {
        vec!["auto", "all", "onethird", "sqrt", "log2"]
    }

    /// The supportedImpurities of the class, as PySpark's class attribute.
    #[classattr]
    #[allow(non_snake_case)]
    fn supportedImpurities() -> Vec<&'static str> {
        vec!["variance"]
    }

    /// The supportedLossTypes of the class, as PySpark's class attribute.
    #[classattr]
    #[allow(non_snake_case)]
    fn supportedLossTypes() -> Vec<&'static str> {
        vec!["logistic"]
    }
}

impl params_api::NativeInstance for PyGBTClassifier {
    fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        object::new_py(
            py,
            Self {
                inner: self.inner.clone(),
                unsendable: self.unsendable.clone(),
            },
        )
    }

    fn uid(&self) -> String {
        Estimator::operator(&self.inner).uid.clone()
    }

    fn set_uid(&mut self, uid: &str) {
        Estimator::operator_mut(&mut self.inner).uid = uid.to_string();
    }
}

/// Random forest classifier fitted and cached by Spark Connect.
#[pyclass(
    subclass,
    name = "RandomForestClassifier",
    module = "pyspark.ml.connect.classification"
)]
pub struct PyRandomForestClassifier {
    inner: RandomForestClassifier,
    unsendable: Vec<&'static str>,
}

const RANDOM_FOREST_PARAMS: &[(&str, Kind)] = &[
    ("featuresCol", Kind::Str),
    ("labelCol", Kind::Str),
    ("predictionCol", Kind::Str),
    ("probabilityCol", Kind::Str),
    ("rawPredictionCol", Kind::Str),
    ("maxDepth", Kind::Int),
    ("maxBins", Kind::Int),
    ("minInstancesPerNode", Kind::Int),
    ("minInfoGain", Kind::Float),
    ("pruneTree", Kind::Bool),
    ("maxMemoryInMB", Kind::Int),
    ("cacheNodeIds", Kind::Bool),
    ("checkpointInterval", Kind::Int),
    ("impurity", Kind::Str),
    ("numTrees", Kind::Int),
    ("featureSubsetStrategy", Kind::Str),
    ("seed", Kind::Int),
    ("subsamplingRate", Kind::Float),
    ("leafCol", Kind::Str),
    ("minWeightFractionPerNode", Kind::Float),
    ("weightCol", Kind::Str),
    ("bootstrap", Kind::Bool),
];

native_pymethods! {
    PyRandomForestClassifier;
    jvm: "org.apache.spark.ml.classification.RandomForestClassifier", op: Estimator;
    params: [
        "bootstrap" => ("getBootstrap", get_bootstrap, param_bootstrap),
        "cacheNodeIds" => ("getCacheNodeIds", get_cache_node_ids, param_cache_node_ids),
        "checkpointInterval" => (
            "getCheckpointInterval",
            get_checkpoint_interval,
            param_checkpoint_interval
        ),
        "featureSubsetStrategy" => (
            "getFeatureSubsetStrategy",
            get_feature_subset_strategy,
            param_feature_subset_strategy
        ),
        "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
        "impurity" => ("getImpurity", get_impurity, param_impurity),
        "labelCol" => ("getLabelCol", get_label_col, param_label_col),
        "leafCol" => ("getLeafCol", get_leaf_col, param_leaf_col),
        "maxBins" => ("getMaxBins", get_max_bins, param_max_bins),
        "maxDepth" => ("getMaxDepth", get_max_depth, param_max_depth),
        "maxMemoryInMB" => ("getMaxMemoryInMB", get_max_memory_in_mb, param_max_memory_in_mb),
        "minInfoGain" => ("getMinInfoGain", get_min_info_gain, param_min_info_gain),
        "minInstancesPerNode" => (
            "getMinInstancesPerNode",
            get_min_instances_per_node,
            param_min_instances_per_node
        ),
        "minWeightFractionPerNode" => (
            "getMinWeightFractionPerNode",
            get_min_weight_fraction_per_node,
            param_min_weight_fraction_per_node
        ),
        "numTrees" => ("getNumTrees", get_num_trees, param_num_trees),
        "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
        "probabilityCol" => ("getProbabilityCol", get_probability_col, param_probability_col),
        "pruneTree" => ("getPruneTree", get_prune_tree, param_prune_tree),
        "rawPredictionCol" => (
            "getRawPredictionCol",
            get_raw_prediction_col,
            param_raw_prediction_col
        ),
        "seed" => ("getSeed", get_seed, param_seed),
        "subsamplingRate" => ("getSubsamplingRate", get_subsampling_rate, param_subsampling_rate),
        "thresholds" => ("getThresholds", get_thresholds, param_thresholds),
        "weightCol" => ("getWeightCol", get_weight_col, param_weight_col),
    ];
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, featuresCol='features', labelCol='label', predictionCol='prediction', probabilityCol='probability', rawPredictionCol='rawPrediction', maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, pruneTree=True, maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10, impurity='gini', numTrees=20, featureSubsetStrategy='auto', seed=None, subsamplingRate=1.0, leafCol='', minWeightFractionPerNode=0.0, weightCol=None, bootstrap=True)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: RandomForestClassifier::new(),
            unsendable: Vec::new(),
        };
        init_params(
            &mut this,
            "RandomForestClassifier",
            &[],
            RANDOM_FOREST_PARAMS,
            args,
            kwargs,
        )?;
        Ok(this)
    }
    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, featuresCol='features', labelCol='label', predictionCol='prediction', probabilityCol='probability', rawPredictionCol='rawPrediction', maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, pruneTree=True, maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10, impurity='gini', numTrees=20, featureSubsetStrategy='auto', seed=None, subsamplingRate=1.0, leafCol='', minWeightFractionPerNode=0.0, weightCol=None, bootstrap=True)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(
            slf,
            "RandomForestClassifier",
            RANDOM_FOREST_PARAMS,
            args,
            kwargs,
        )
    }

    #[pyo3(name = "setFeaturesCol")]
    fn set_feature_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "featuresCol", Kind::Str, value)
    }

    #[pyo3(name = "setLabelCol")]
    fn set_label_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "labelCol", Kind::Str, value)
    }

    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "predictionCol", Kind::Str, value)
    }

    #[pyo3(name = "setProbabilityCol")]
    fn set_probability_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "probabilityCol", Kind::Str, value)
    }

    #[pyo3(name = "setRawPredictionCol")]
    fn set_raw_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "rawPredictionCol", Kind::Str, value)
    }

    #[pyo3(name = "setMaxDepth")]
    fn set_max_depth<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxDepth", Kind::Int, value)
    }

    #[pyo3(name = "setMaxBins")]
    fn set_max_bins<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxBins", Kind::Int, value)
    }

    #[pyo3(name = "setMinInstancesPerNode")]
    fn set_min_instances_per_node<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "minInstancesPerNode", Kind::Int, value)
    }

    #[pyo3(name = "setMinInfoGain")]
    fn set_min_info_gain<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "minInfoGain", Kind::Float, value)
    }

    #[pyo3(name = "setPruneTree")]
    fn set_prune_tree<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "pruneTree", Kind::Bool, value)
    }

    #[pyo3(name = "setMaxMemoryInMB")]
    fn set_max_memory_in_mb<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxMemoryInMB", Kind::Int, value)
    }

    #[pyo3(name = "setCacheNodeIds")]
    fn set_cache_node_ids<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "cacheNodeIds", Kind::Bool, value)
    }

    #[pyo3(name = "setCheckpointInterval")]
    fn set_checkpoint_interval<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "checkpointInterval", Kind::Int, value)
    }

    #[pyo3(name = "setImpurity")]
    fn set_impurity<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "impurity", Kind::Str, value)
    }

    #[pyo3(name = "setNumTrees")]
    fn set_num_trees<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "numTrees", Kind::Int, value)
    }

    #[pyo3(name = "setFeatureSubsetStrategy")]
    fn set_feature_subset_strategy<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "featureSubsetStrategy", Kind::Str, value)
    }

    #[pyo3(name = "setSeed")]
    fn set_seed<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "seed", Kind::Int, value)
    }

    #[pyo3(name = "setSubsamplingRate")]
    fn set_subsampling_rate<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "subsamplingRate", Kind::Float, value)
    }

    #[pyo3(name = "setLeafCol")]
    fn set_leaf_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "leafCol", Kind::Str, value)
    }

    #[pyo3(name = "setMinWeightFractionPerNode")]
    fn set_min_weight_fraction_per_node<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "minWeightFractionPerNode", Kind::Float, value)
    }

    #[pyo3(name = "setWeightCol")]
    fn set_weight_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "weightCol", Kind::Str, value)
    }

    #[pyo3(name = "setBootstrap")]
    fn set_bootstrap<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "bootstrap", Kind::Bool, value)
    }

    #[pyo3(name = "setThresholds")]
    fn set_thresholds<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "thresholds", Kind::FloatList, value)
    }

    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<Py<classification_api::PyRandomForestClassificationModel>>> {
        fit_with(slf, dataset, params, || {
            fit_native_model(slf, dataset, |py, model, df| {
                classification_api::PyRandomForestClassificationModel::fitted(
                    py,
                    model,
                    df,
                    object::SummaryDataset::Train,
                )
            })
        })
    }

    /// The fits of a copy of the estimator, one per param map, as `Estimator.fitMultiple`, of
    /// PySpark's keywords.
    #[pyo3(name = "fitMultiple")]
    #[allow(non_snake_case)]
    fn fit_multiple<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        paramMaps: &Bound<'py, PyAny>,
    ) -> PyResult<Py<params_api::PyFitMultipleIterator>> {
        params_api::fit_multiple(slf.as_any(), dataset, paramMaps)
    }

    /// The supportedFeatureSubsetStrategies of the class, as PySpark's class attribute.
    #[classattr]
    #[allow(non_snake_case)]
    fn supportedFeatureSubsetStrategies() -> Vec<&'static str> {
        vec!["auto", "all", "onethird", "sqrt", "log2"]
    }

    /// The supportedImpurities of the class, as PySpark's class attribute.
    #[classattr]
    #[allow(non_snake_case)]
    fn supportedImpurities() -> Vec<&'static str> {
        vec!["entropy", "gini"]
    }
}

impl params_api::NativeInstance for PyRandomForestClassifier {
    fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        object::new_py(
            py,
            Self {
                inner: self.inner.clone(),
                unsendable: self.unsendable.clone(),
            },
        )
    }

    fn uid(&self) -> String {
        Estimator::operator(&self.inner).uid.clone()
    }

    fn set_uid(&mut self, uid: &str) {
        Estimator::operator_mut(&mut self.inner).uid = uid.to_string();
    }
}

/// Gradient-boosted regressor fitted and cached by Spark Connect.
#[pyclass(
    subclass,
    name = "GBTRegressor",
    module = "pyspark.ml.connect.regression"
)]
pub struct PyGBTRegressor {
    inner: GBTRegressor,
    unsendable: Vec<&'static str>,
}

const GBT_REGRESSOR_PARAMS: &[(&str, Kind)] = &[
    ("featuresCol", Kind::Str),
    ("labelCol", Kind::Str),
    ("predictionCol", Kind::Str),
    ("maxDepth", Kind::Int),
    ("maxBins", Kind::Int),
    ("minInstancesPerNode", Kind::Int),
    ("minInfoGain", Kind::Float),
    ("maxMemoryInMB", Kind::Int),
    ("cacheNodeIds", Kind::Bool),
    ("subsamplingRate", Kind::Float),
    ("checkpointInterval", Kind::Int),
    ("lossType", Kind::Str),
    ("maxIter", Kind::Int),
    ("stepSize", Kind::Float),
    ("seed", Kind::Int),
    ("impurity", Kind::Str),
    ("featureSubsetStrategy", Kind::Str),
    ("validationTol", Kind::Float),
    ("validationIndicatorCol", Kind::Str),
    ("leafCol", Kind::Str),
    ("minWeightFractionPerNode", Kind::Float),
    ("weightCol", Kind::Str),
];

native_pymethods! {
    PyGBTRegressor;
    jvm: "org.apache.spark.ml.regression.GBTRegressor", op: Estimator;
    params: [
    "cacheNodeIds" => ("getCacheNodeIds", get_cache_node_ids, param_cache_node_ids),
    "checkpointInterval" => (
        "getCheckpointInterval",
        get_checkpoint_interval,
        param_checkpoint_interval
    ),
    "featureSubsetStrategy" => (
        "getFeatureSubsetStrategy",
        get_feature_subset_strategy,
        param_feature_subset_strategy
    ),
    "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
    "impurity" => ("getImpurity", get_impurity, param_impurity),
    "labelCol" => ("getLabelCol", get_label_col, param_label_col),
    "leafCol" => ("getLeafCol", get_leaf_col, param_leaf_col),
    "lossType" => ("getLossType", get_loss_type, param_loss_type),
    "maxBins" => ("getMaxBins", get_max_bins, param_max_bins),
    "maxDepth" => ("getMaxDepth", get_max_depth, param_max_depth),
    "maxIter" => ("getMaxIter", get_max_iter, param_max_iter),
    "maxMemoryInMB" => ("getMaxMemoryInMB", get_max_memory_in_mb, param_max_memory_in_mb),
    "minInfoGain" => ("getMinInfoGain", get_min_info_gain, param_min_info_gain),
    "minInstancesPerNode" => (
        "getMinInstancesPerNode",
        get_min_instances_per_node,
        param_min_instances_per_node
    ),
    "minWeightFractionPerNode" => (
        "getMinWeightFractionPerNode",
        get_min_weight_fraction_per_node,
        param_min_weight_fraction_per_node
    ),
    "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
    "seed" => ("getSeed", get_seed, param_seed),
    "stepSize" => ("getStepSize", get_step_size, param_step_size),
    "subsamplingRate" => ("getSubsamplingRate", get_subsampling_rate, param_subsampling_rate),
    "validationIndicatorCol" => (
        "getValidationIndicatorCol",
        get_validation_indicator_col,
        param_validation_indicator_col
    ),
    "validationTol" => ("getValidationTol", get_validation_tol, param_validation_tol),
    "weightCol" => ("getWeightCol", get_weight_col, param_weight_col),
    ];
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, featuresCol='features', labelCol='label', predictionCol='prediction', maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256, cacheNodeIds=False, subsamplingRate=1.0, checkpointInterval=10, lossType='squared', maxIter=20, stepSize=0.1, seed=None, impurity='variance', featureSubsetStrategy='all', validationTol=0.1, validationIndicatorCol=None, leafCol='', minWeightFractionPerNode=0.0, weightCol=None)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: GBTRegressor::new(),
            unsendable: Vec::new(),
        };
        init_params(
            &mut this,
            "GBTRegressor",
            &[],
            GBT_REGRESSOR_PARAMS,
            args,
            kwargs,
        )?;
        Ok(this)
    }
    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, featuresCol='features', labelCol='label', predictionCol='prediction', maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256, cacheNodeIds=False, subsamplingRate=1.0, checkpointInterval=10, lossType='squared', maxIter=20, stepSize=0.1, seed=None, impurity='variance', featureSubsetStrategy='all', validationTol=0.1, validationIndicatorCol=None, leafCol='', minWeightFractionPerNode=0.0, weightCol=None)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(slf, "GBTRegressor", GBT_REGRESSOR_PARAMS, args, kwargs)
    }

    #[pyo3(name = "setFeaturesCol")]
    fn set_feature_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "featuresCol", Kind::Str, value)
    }

    #[pyo3(name = "setLabelCol")]
    fn set_label_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "labelCol", Kind::Str, value)
    }

    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "predictionCol", Kind::Str, value)
    }

    #[pyo3(name = "setMaxDepth")]
    fn set_max_depth<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxDepth", Kind::Int, value)
    }

    #[pyo3(name = "setMaxBins")]
    fn set_max_bins<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxBins", Kind::Int, value)
    }

    #[pyo3(name = "setMinInstancesPerNode")]
    fn set_min_instances_per_node<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "minInstancesPerNode", Kind::Int, value)
    }

    #[pyo3(name = "setMinInfoGain")]
    fn set_min_info_gain<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "minInfoGain", Kind::Float, value)
    }

    #[pyo3(name = "setMaxMemoryInMB")]
    fn set_max_memory_in_mb<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxMemoryInMB", Kind::Int, value)
    }

    #[pyo3(name = "setCacheNodeIds")]
    fn set_cache_node_ids<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "cacheNodeIds", Kind::Bool, value)
    }

    #[pyo3(name = "setCheckpointInterval")]
    fn set_checkpoint_interval<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "checkpointInterval", Kind::Int, value)
    }

    #[pyo3(name = "setImpurity")]
    fn set_impurity<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "impurity", Kind::Str, value)
    }

    #[pyo3(name = "setFeatureSubsetStrategy")]
    fn set_feature_subset_strategy<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "featureSubsetStrategy", Kind::Str, value)
    }

    #[pyo3(name = "setSeed")]
    fn set_seed<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "seed", Kind::Int, value)
    }

    #[pyo3(name = "setSubsamplingRate")]
    fn set_subsampling_rate<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "subsamplingRate", Kind::Float, value)
    }

    #[pyo3(name = "setLeafCol")]
    fn set_leaf_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "leafCol", Kind::Str, value)
    }

    #[pyo3(name = "setMinWeightFractionPerNode")]
    fn set_min_weight_fraction_per_node<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "minWeightFractionPerNode", Kind::Float, value)
    }

    #[pyo3(name = "setWeightCol")]
    fn set_weight_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "weightCol", Kind::Str, value)
    }

    #[pyo3(name = "setLossType")]
    fn set_loss_type<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "lossType", Kind::Str, value)
    }

    #[pyo3(name = "setMaxIter")]
    fn set_max_iter<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxIter", Kind::Int, value)
    }

    #[pyo3(name = "setStepSize")]
    fn set_step_size<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "stepSize", Kind::Float, value)
    }

    #[pyo3(name = "setValidationIndicatorCol")]
    fn set_validation_indicator_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "validationIndicatorCol", Kind::Str, value)
    }

    /// The supportedImpurities of the class, as PySpark's class attribute.
    #[classattr]
    #[allow(non_snake_case)]
    fn supportedImpurities() -> Vec<&'static str> {
        vec!["variance"]
    }

    /// The supportedFeatureSubsetStrategies of the class, as PySpark's class attribute.
    #[classattr]
    #[allow(non_snake_case)]
    fn supportedFeatureSubsetStrategies() -> Vec<&'static str> {
        vec!["auto", "all", "onethird", "sqrt", "log2"]
    }

    /// The supportedLossTypes of the class, as PySpark's class attribute.
    #[classattr]
    #[allow(non_snake_case)]
    fn supportedLossTypes() -> Vec<&'static str> {
        vec!["squared", "absolute"]
    }


    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<Py<regression_api::PyGBTRegressionModel>>> {
        fit_with(slf, dataset, params, || {
            fit_native_model(slf, dataset, |py, model, df| {
                regression_api::PyGBTRegressionModel::fitted(
                    py,
                    model,
                    df,
                    object::SummaryDataset::None,
                )
            })
        })
    }

    /// The fits of a copy of the estimator, one per param map, as `Estimator.fitMultiple`, of
    /// PySpark's keywords.
    #[pyo3(name = "fitMultiple")]
    #[allow(non_snake_case)]
    fn fit_multiple<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        paramMaps: &Bound<'py, PyAny>,
    ) -> PyResult<Py<params_api::PyFitMultipleIterator>> {
        params_api::fit_multiple(slf.as_any(), dataset, paramMaps)
    }
}

impl params_api::NativeInstance for PyGBTRegressor {
    fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        object::new_py(
            py,
            Self {
                inner: self.inner.clone(),
                unsendable: self.unsendable.clone(),
            },
        )
    }

    fn uid(&self) -> String {
        Estimator::operator(&self.inner).uid.clone()
    }

    fn set_uid(&mut self, uid: &str) {
        Estimator::operator_mut(&mut self.inner).uid = uid.to_string();
    }
}

/// Decision tree regressor fitted and cached by Spark Connect.
#[pyclass(
    subclass,
    name = "DecisionTreeRegressor",
    module = "pyspark.ml.connect.regression"
)]
pub struct PyDecisionTreeRegressor {
    inner: DecisionTreeRegressor,
    unsendable: Vec<&'static str>,
}

const DECISION_TREE_REGRESSOR_PARAMS: &[(&str, Kind)] = &[
    ("featuresCol", Kind::Str),
    ("labelCol", Kind::Str),
    ("predictionCol", Kind::Str),
    ("maxDepth", Kind::Int),
    ("maxBins", Kind::Int),
    ("minInstancesPerNode", Kind::Int),
    ("minInfoGain", Kind::Float),
    ("maxMemoryInMB", Kind::Int),
    ("cacheNodeIds", Kind::Bool),
    ("checkpointInterval", Kind::Int),
    ("impurity", Kind::Str),
    ("seed", Kind::Int),
    ("varianceCol", Kind::Str),
    ("leafCol", Kind::Str),
    ("minWeightFractionPerNode", Kind::Float),
    ("weightCol", Kind::Str),
];

native_pymethods! {
    PyDecisionTreeRegressor;
    jvm: "org.apache.spark.ml.regression.DecisionTreeRegressor", op: Estimator;
    params: [
    "cacheNodeIds" => ("getCacheNodeIds", get_cache_node_ids, param_cache_node_ids),
    "checkpointInterval" => (
        "getCheckpointInterval",
        get_checkpoint_interval,
        param_checkpoint_interval
    ),
    "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
    "impurity" => ("getImpurity", get_impurity, param_impurity),
    "labelCol" => ("getLabelCol", get_label_col, param_label_col),
    "leafCol" => ("getLeafCol", get_leaf_col, param_leaf_col),
    "maxBins" => ("getMaxBins", get_max_bins, param_max_bins),
    "maxDepth" => ("getMaxDepth", get_max_depth, param_max_depth),
    "maxMemoryInMB" => ("getMaxMemoryInMB", get_max_memory_in_mb, param_max_memory_in_mb),
    "minInfoGain" => ("getMinInfoGain", get_min_info_gain, param_min_info_gain),
    "minInstancesPerNode" => (
        "getMinInstancesPerNode",
        get_min_instances_per_node,
        param_min_instances_per_node
    ),
    "minWeightFractionPerNode" => (
        "getMinWeightFractionPerNode",
        get_min_weight_fraction_per_node,
        param_min_weight_fraction_per_node
    ),
    "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
    "seed" => ("getSeed", get_seed, param_seed),
    "varianceCol" => ("getVarianceCol", get_variance_col, param_variance_col),
    "weightCol" => ("getWeightCol", get_weight_col, param_weight_col),
    ];
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, featuresCol='features', labelCol='label', predictionCol='prediction', maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10, impurity='variance', seed=None, varianceCol=None, weightCol=None, leafCol='', minWeightFractionPerNode=0.0)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: DecisionTreeRegressor::new(),
            unsendable: Vec::new(),
        };
        init_params(
            &mut this,
            "DecisionTreeRegressor",
            &[],
            DECISION_TREE_REGRESSOR_PARAMS,
            args,
            kwargs,
        )?;
        Ok(this)
    }
    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, featuresCol='features', labelCol='label', predictionCol='prediction', maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10, impurity='variance', seed=None, varianceCol=None, weightCol=None, leafCol='', minWeightFractionPerNode=0.0)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(
            slf,
            "DecisionTreeRegressor",
            DECISION_TREE_REGRESSOR_PARAMS,
            args,
            kwargs,
        )
    }

    #[pyo3(name = "setFeaturesCol")]
    fn set_feature_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "featuresCol", Kind::Str, value)
    }

    #[pyo3(name = "setLabelCol")]
    fn set_label_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "labelCol", Kind::Str, value)
    }

    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "predictionCol", Kind::Str, value)
    }

    #[pyo3(name = "setMaxDepth")]
    fn set_max_depth<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxDepth", Kind::Int, value)
    }

    #[pyo3(name = "setMaxBins")]
    fn set_max_bins<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxBins", Kind::Int, value)
    }

    #[pyo3(name = "setMinInstancesPerNode")]
    fn set_min_instances_per_node<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "minInstancesPerNode", Kind::Int, value)
    }

    #[pyo3(name = "setMinInfoGain")]
    fn set_min_info_gain<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "minInfoGain", Kind::Float, value)
    }

    #[pyo3(name = "setMaxMemoryInMB")]
    fn set_max_memory_in_mb<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxMemoryInMB", Kind::Int, value)
    }

    #[pyo3(name = "setCacheNodeIds")]
    fn set_cache_node_ids<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "cacheNodeIds", Kind::Bool, value)
    }

    #[pyo3(name = "setCheckpointInterval")]
    fn set_checkpoint_interval<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "checkpointInterval", Kind::Int, value)
    }

    #[pyo3(name = "setImpurity")]
    fn set_impurity<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "impurity", Kind::Str, value)
    }

    #[pyo3(name = "setSeed")]
    fn set_seed<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "seed", Kind::Int, value)
    }

    #[pyo3(name = "setLeafCol")]
    fn set_leaf_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "leafCol", Kind::Str, value)
    }

    #[pyo3(name = "setMinWeightFractionPerNode")]
    fn set_min_weight_fraction_per_node<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "minWeightFractionPerNode", Kind::Float, value)
    }

    #[pyo3(name = "setWeightCol")]
    fn set_weight_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "weightCol", Kind::Str, value)
    }

    #[pyo3(name = "setVarianceCol")]
    fn set_variance_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "varianceCol", Kind::Str, value)
    }

    /// The supportedImpurities of the class, as PySpark's class attribute.
    #[classattr]
    #[allow(non_snake_case)]
    fn supportedImpurities() -> Vec<&'static str> {
        vec!["variance"]
    }


    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<Py<regression_api::PyDecisionTreeRegressionModel>>> {
        fit_with(slf, dataset, params, || {
            fit_native_model(slf, dataset, |py, model, df| {
                regression_api::PyDecisionTreeRegressionModel::fitted(
                    py,
                    model,
                    df,
                    object::SummaryDataset::None,
                )
            })
        })
    }

    /// The fits of a copy of the estimator, one per param map, as `Estimator.fitMultiple`, of
    /// PySpark's keywords.
    #[pyo3(name = "fitMultiple")]
    #[allow(non_snake_case)]
    fn fit_multiple<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        paramMaps: &Bound<'py, PyAny>,
    ) -> PyResult<Py<params_api::PyFitMultipleIterator>> {
        params_api::fit_multiple(slf.as_any(), dataset, paramMaps)
    }
}

impl params_api::NativeInstance for PyDecisionTreeRegressor {
    fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        object::new_py(
            py,
            Self {
                inner: self.inner.clone(),
                unsendable: self.unsendable.clone(),
            },
        )
    }

    fn uid(&self) -> String {
        Estimator::operator(&self.inner).uid.clone()
    }

    fn set_uid(&mut self, uid: &str) {
        Estimator::operator_mut(&mut self.inner).uid = uid.to_string();
    }
}
/// Random forest regressor fitted and cached by Spark Connect.
#[pyclass(
    subclass,
    name = "RandomForestRegressor",
    module = "pyspark.ml.connect.regression"
)]
pub struct PyRandomForestRegressor {
    inner: RandomForestRegressor,
    unsendable: Vec<&'static str>,
}

const RANDOM_FOREST_REGRESSOR_PARAMS: &[(&str, Kind)] = &[
    ("featuresCol", Kind::Str),
    ("labelCol", Kind::Str),
    ("predictionCol", Kind::Str),
    ("maxDepth", Kind::Int),
    ("maxBins", Kind::Int),
    ("minInstancesPerNode", Kind::Int),
    ("minInfoGain", Kind::Float),
    ("maxMemoryInMB", Kind::Int),
    ("cacheNodeIds", Kind::Bool),
    ("checkpointInterval", Kind::Int),
    ("impurity", Kind::Str),
    ("subsamplingRate", Kind::Float),
    ("seed", Kind::Int),
    ("numTrees", Kind::Int),
    ("featureSubsetStrategy", Kind::Str),
    ("leafCol", Kind::Str),
    ("minWeightFractionPerNode", Kind::Float),
    ("weightCol", Kind::Str),
    ("bootstrap", Kind::Bool),
];

native_pymethods! {
    PyRandomForestRegressor;
    jvm: "org.apache.spark.ml.regression.RandomForestRegressor", op: Estimator;
    params: [
    "bootstrap" => ("getBootstrap", get_bootstrap, param_bootstrap),
    "cacheNodeIds" => ("getCacheNodeIds", get_cache_node_ids, param_cache_node_ids),
    "checkpointInterval" => (
        "getCheckpointInterval",
        get_checkpoint_interval,
        param_checkpoint_interval
    ),
    "featureSubsetStrategy" => (
        "getFeatureSubsetStrategy",
        get_feature_subset_strategy,
        param_feature_subset_strategy
    ),
    "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
    "impurity" => ("getImpurity", get_impurity, param_impurity),
    "labelCol" => ("getLabelCol", get_label_col, param_label_col),
    "leafCol" => ("getLeafCol", get_leaf_col, param_leaf_col),
    "maxBins" => ("getMaxBins", get_max_bins, param_max_bins),
    "maxDepth" => ("getMaxDepth", get_max_depth, param_max_depth),
    "maxMemoryInMB" => ("getMaxMemoryInMB", get_max_memory_in_mb, param_max_memory_in_mb),
    "minInfoGain" => ("getMinInfoGain", get_min_info_gain, param_min_info_gain),
    "minInstancesPerNode" => (
        "getMinInstancesPerNode",
        get_min_instances_per_node,
        param_min_instances_per_node
    ),
    "minWeightFractionPerNode" => (
        "getMinWeightFractionPerNode",
        get_min_weight_fraction_per_node,
        param_min_weight_fraction_per_node
    ),
    "numTrees" => ("getNumTrees", get_num_trees, param_num_trees),
    "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
    "seed" => ("getSeed", get_seed, param_seed),
    "subsamplingRate" => ("getSubsamplingRate", get_subsampling_rate, param_subsampling_rate),
    "weightCol" => ("getWeightCol", get_weight_col, param_weight_col),
    ];
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, featuresCol='features', labelCol='label', predictionCol='prediction', maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10, impurity='variance', subsamplingRate=1.0, seed=None, numTrees=20, featureSubsetStrategy='auto', leafCol='', minWeightFractionPerNode=0.0, weightCol=None, bootstrap=True)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: RandomForestRegressor::new(),
            unsendable: Vec::new(),
        };
        init_params(
            &mut this,
            "RandomForestRegressor",
            &[],
            RANDOM_FOREST_REGRESSOR_PARAMS,
            args,
            kwargs,
        )?;
        Ok(this)
    }
    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, featuresCol='features', labelCol='label', predictionCol='prediction', maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10, impurity='variance', subsamplingRate=1.0, seed=None, numTrees=20, featureSubsetStrategy='auto', leafCol='', minWeightFractionPerNode=0.0, weightCol=None, bootstrap=True)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(
            slf,
            "RandomForestRegressor",
            RANDOM_FOREST_REGRESSOR_PARAMS,
            args,
            kwargs,
        )
    }

    #[pyo3(name = "setFeaturesCol")]
    fn set_feature_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "featuresCol", Kind::Str, value)
    }

    #[pyo3(name = "setLabelCol")]
    fn set_label_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "labelCol", Kind::Str, value)
    }

    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "predictionCol", Kind::Str, value)
    }

    #[pyo3(name = "setMaxDepth")]
    fn set_max_depth<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxDepth", Kind::Int, value)
    }

    #[pyo3(name = "setMaxBins")]
    fn set_max_bins<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxBins", Kind::Int, value)
    }

    #[pyo3(name = "setMinInstancesPerNode")]
    fn set_min_instances_per_node<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "minInstancesPerNode", Kind::Int, value)
    }

    #[pyo3(name = "setMinInfoGain")]
    fn set_min_info_gain<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "minInfoGain", Kind::Float, value)
    }

    #[pyo3(name = "setMaxMemoryInMB")]
    fn set_max_memory_in_mb<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxMemoryInMB", Kind::Int, value)
    }

    #[pyo3(name = "setCacheNodeIds")]
    fn set_cache_node_ids<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "cacheNodeIds", Kind::Bool, value)
    }

    #[pyo3(name = "setCheckpointInterval")]
    fn set_checkpoint_interval<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "checkpointInterval", Kind::Int, value)
    }

    #[pyo3(name = "setImpurity")]
    fn set_impurity<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "impurity", Kind::Str, value)
    }

    #[pyo3(name = "setNumTrees")]
    fn set_num_trees<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "numTrees", Kind::Int, value)
    }

    #[pyo3(name = "setFeatureSubsetStrategy")]
    fn set_feature_subset_strategy<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "featureSubsetStrategy", Kind::Str, value)
    }

    #[pyo3(name = "setSeed")]
    fn set_seed<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "seed", Kind::Int, value)
    }

    #[pyo3(name = "setSubsamplingRate")]
    fn set_subsampling_rate<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "subsamplingRate", Kind::Float, value)
    }

    #[pyo3(name = "setLeafCol")]
    fn set_leaf_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "leafCol", Kind::Str, value)
    }

    #[pyo3(name = "setMinWeightFractionPerNode")]
    fn set_min_weight_fraction_per_node<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "minWeightFractionPerNode", Kind::Float, value)
    }

    #[pyo3(name = "setWeightCol")]
    fn set_weight_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "weightCol", Kind::Str, value)
    }

    #[pyo3(name = "setBootstrap")]
    fn set_bootstrap<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "bootstrap", Kind::Bool, value)
    }

    /// The supportedImpurities of the class, as PySpark's class attribute.
    #[classattr]
    #[allow(non_snake_case)]
    fn supportedImpurities() -> Vec<&'static str> {
        vec!["variance"]
    }

    /// The supportedFeatureSubsetStrategies of the class, as PySpark's class attribute.
    #[classattr]
    #[allow(non_snake_case)]
    fn supportedFeatureSubsetStrategies() -> Vec<&'static str> {
        vec!["auto", "all", "onethird", "sqrt", "log2"]
    }


    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<Py<regression_api::PyRandomForestRegressionModel>>> {
        fit_with(slf, dataset, params, || {
            fit_native_model(slf, dataset, |py, model, df| {
                regression_api::PyRandomForestRegressionModel::fitted(
                    py,
                    model,
                    df,
                    object::SummaryDataset::None,
                )
            })
        })
    }

    /// The fits of a copy of the estimator, one per param map, as `Estimator.fitMultiple`, of
    /// PySpark's keywords.
    #[pyo3(name = "fitMultiple")]
    #[allow(non_snake_case)]
    fn fit_multiple<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        paramMaps: &Bound<'py, PyAny>,
    ) -> PyResult<Py<params_api::PyFitMultipleIterator>> {
        params_api::fit_multiple(slf.as_any(), dataset, paramMaps)
    }
}

impl params_api::NativeInstance for PyRandomForestRegressor {
    fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        object::new_py(
            py,
            Self {
                inner: self.inner.clone(),
                unsendable: self.unsendable.clone(),
            },
        )
    }

    fn uid(&self) -> String {
        Estimator::operator(&self.inner).uid.clone()
    }

    fn set_uid(&mut self, uid: &str) {
        Estimator::operator_mut(&mut self.inner).uid = uid.to_string();
    }
}

/// K-means clustering backed by Spark Connect's JVM ML API.
#[pyclass(subclass, name = "KMeans", module = "pyspark.ml.connect.clustering")]
pub struct PyKMeans {
    inner: KMeans,
    unsendable: Vec<&'static str>,
}

const KMEANS_PARAMS: &[(&str, Kind)] = &[
    ("featuresCol", Kind::Str),
    ("predictionCol", Kind::Str),
    ("k", Kind::Int),
    ("initMode", Kind::Str),
    ("initSteps", Kind::Int),
    ("tol", Kind::Float),
    ("maxIter", Kind::Int),
    ("seed", Kind::Int),
    ("distanceMeasure", Kind::Str),
    ("weightCol", Kind::Str),
    ("solver", Kind::Str),
    ("maxBlockSizeInMB", Kind::Float),
];

native_pymethods! {
    PyKMeans;
    jvm: "org.apache.spark.ml.clustering.KMeans", op: Estimator;
    params: [
        "distanceMeasure" => ("getDistanceMeasure", get_distance_measure, param_distance_measure),
        "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
        "initMode" => ("getInitMode", get_init_mode, param_init_mode),
        "initSteps" => ("getInitSteps", get_init_steps, param_init_steps),
        "intermediateStorageLevel" => (
            "getIntermediateStorageLevel",
            get_intermediate_storage_level,
            param_intermediate_storage_level
        ),
        "k" => ("getK", get_k, param_k),
        "maxBlockSizeInMB" => (
            "getMaxBlockSizeInMB",
            get_max_block_size_in_mb,
            param_max_block_size_in_mb
        ),
        "maxIter" => ("getMaxIter", get_max_iter, param_max_iter),
        "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
        "seed" => ("getSeed", get_seed, param_seed),
        "solver" => ("getSolver", get_solver, param_solver),
        "tol" => ("getTol", get_tol, param_tol),
        "weightCol" => ("getWeightCol", get_weight_col, param_weight_col),
    ];
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, featuresCol='features', predictionCol='prediction', k=2, initMode='k-means||', initSteps=2, tol=0.0001, maxIter=20, seed=None, distanceMeasure='euclidean', weightCol=None, solver='auto', maxBlockSizeInMB=0.0)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: KMeans::new(),
            unsendable: Vec::new(),
        };
        init_params(&mut this, "KMeans", &[], KMEANS_PARAMS, args, kwargs)?;
        Ok(this)
    }
    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, featuresCol='features', predictionCol='prediction', k=2, initMode='k-means||', initSteps=2, tol=0.0001, maxIter=20, seed=None, distanceMeasure='euclidean', weightCol=None, solver='auto', maxBlockSizeInMB=0.0)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(slf, "KMeans", KMEANS_PARAMS, args, kwargs)
    }

    #[pyo3(name = "setFeaturesCol")]
    fn set_feature_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "featuresCol", Kind::Str, value)
    }

    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "predictionCol", Kind::Str, value)
    }

    #[pyo3(name = "setK")]
    fn set_k<'py>(slf: Bound<'py, Self>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "k", Kind::Int, value)
    }

    #[pyo3(name = "setInitMode")]
    fn set_init_mode<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "initMode", Kind::Str, value)
    }

    #[pyo3(name = "setInitSteps")]
    fn set_init_steps<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "initSteps", Kind::Int, value)
    }

    #[pyo3(name = "setTol")]
    fn set_tol<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "tol", Kind::Float, value)
    }

    #[pyo3(name = "setMaxIter")]
    fn set_max_iter<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxIter", Kind::Int, value)
    }

    #[pyo3(name = "setSeed")]
    fn set_seed<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "seed", Kind::Int, value)
    }

    #[pyo3(name = "setDistanceMeasure")]
    fn set_distance_measure<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "distanceMeasure", Kind::Str, value)
    }

    #[pyo3(name = "setWeightCol")]
    fn set_weight_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "weightCol", Kind::Str, value)
    }

    #[pyo3(name = "setSolver")]
    fn set_solver<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "solver", Kind::Str, value)
    }

    #[pyo3(name = "setMaxBlockSizeInMB")]
    fn set_max_block_size_in_mb<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxBlockSizeInMB", Kind::Float, value)
    }

    #[pyo3(name = "setIntermediateStorageLevel")]
    fn set_intermediate_storage_level<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "intermediateStorageLevel", Kind::Str, value)
    }

    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<Py<clustering_api::PyKMeansModel>>> {
        fit_with(slf, dataset, params, || {
            fit_native_model(slf, dataset, |py, model, df| {
                clustering_api::PyKMeansModel::fitted(
                    py,
                    model,
                    df,
                    object::SummaryDataset::Transformed,
                )
            })
        })
    }

    /// The fits of a copy of the estimator, one per param map, as `Estimator.fitMultiple`, of
    /// PySpark's keywords.
    #[pyo3(name = "fitMultiple")]
    #[allow(non_snake_case)]
    fn fit_multiple<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        paramMaps: &Bound<'py, PyAny>,
    ) -> PyResult<Py<params_api::PyFitMultipleIterator>> {
        params_api::fit_multiple(slf.as_any(), dataset, paramMaps)
    }
}

impl params_api::NativeInstance for PyKMeans {
    fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        object::new_py(
            py,
            Self {
                inner: self.inner.clone(),
                unsendable: self.unsendable.clone(),
            },
        )
    }

    fn uid(&self) -> String {
        Estimator::operator(&self.inner).uid.clone()
    }

    fn set_uid(&mut self, uid: &str) {
        Estimator::operator_mut(&mut self.inner).uid = uid.to_string();
    }
}

/// Native LDA backed by Spark Connect's JVM ML API.
#[pyclass(subclass, name = "LDA", module = "pyspark.ml.connect.clustering")]
pub struct PyLDA {
    inner: LDA,
    unsendable: Vec<&'static str>,
}

const LDA_PARAMS: &[(&str, Kind)] = &[
    ("featuresCol", Kind::Str),
    ("maxIter", Kind::Int),
    ("seed", Kind::Int),
    ("checkpointInterval", Kind::Int),
    ("k", Kind::Int),
    ("optimizer", Kind::Str),
    ("learningOffset", Kind::Float),
    ("learningDecay", Kind::Float),
    ("subsamplingRate", Kind::Float),
    ("optimizeDocConcentration", Kind::Bool),
    ("docConcentration", Kind::FloatList),
    ("topicConcentration", Kind::Float),
    ("topicDistributionCol", Kind::Str),
    ("keepLastCheckpoint", Kind::Bool),
];

native_pymethods! {
    PyLDA;
    jvm: "org.apache.spark.ml.clustering.LDA", op: Estimator;
    params: [
        "checkpointInterval" => (
            "getCheckpointInterval",
            get_checkpoint_interval,
            param_checkpoint_interval
        ),
        "docConcentration" => (
            "getDocConcentration",
            get_doc_concentration,
            param_doc_concentration
        ),
        "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
        "k" => ("getK", get_k, param_k),
        "keepLastCheckpoint" => (
            "getKeepLastCheckpoint",
            get_keep_last_checkpoint,
            param_keep_last_checkpoint
        ),
        "learningDecay" => ("getLearningDecay", get_learning_decay, param_learning_decay),
        "learningOffset" => ("getLearningOffset", get_learning_offset, param_learning_offset),
        "maxIter" => ("getMaxIter", get_max_iter, param_max_iter),
        "optimizeDocConcentration" => (
            "getOptimizeDocConcentration",
            get_optimize_doc_concentration,
            param_optimize_doc_concentration
        ),
        "optimizer" => ("getOptimizer", get_optimizer, param_optimizer),
        "seed" => ("getSeed", get_seed, param_seed),
        "subsamplingRate" => ("getSubsamplingRate", get_subsampling_rate, param_subsampling_rate),
        "topicConcentration" => (
            "getTopicConcentration",
            get_topic_concentration,
            param_topic_concentration
        ),
        "topicDistributionCol" => (
            "getTopicDistributionCol",
            get_topic_distribution_col,
            param_topic_distribution_col
        ),
    ];
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, featuresCol='features', maxIter=20, seed=None, checkpointInterval=10, k=10, optimizer='online', learningOffset=1024.0, learningDecay=0.51, subsamplingRate=0.05, optimizeDocConcentration=True, docConcentration=None, topicConcentration=None, topicDistributionCol='topicDistribution', keepLastCheckpoint=True)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: LDA::new(),
            unsendable: Vec::new(),
        };
        init_params(&mut this, "LDA", &[], LDA_PARAMS, args, kwargs)?;
        Ok(this)
    }

    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, featuresCol='features', maxIter=20, seed=None, checkpointInterval=10, k=10, optimizer='online', learningOffset=1024.0, learningDecay=0.51, subsamplingRate=0.05, optimizeDocConcentration=True, docConcentration=None, topicConcentration=None, topicDistributionCol='topicDistribution', keepLastCheckpoint=True)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(slf, "LDA", LDA_PARAMS, args, kwargs)
    }

    #[pyo3(name = "setFeaturesCol")]
    fn set_feature_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "featuresCol", Kind::Str, value)
    }

    #[pyo3(name = "setMaxIter")]
    fn set_max_iter<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxIter", Kind::Int, value)
    }

    #[pyo3(name = "setSeed")]
    fn set_seed<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "seed", Kind::Int, value)
    }

    #[pyo3(name = "setCheckpointInterval")]
    fn set_checkpoint_interval<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "checkpointInterval", Kind::Int, value)
    }

    #[pyo3(name = "setK")]
    fn set_k<'py>(slf: Bound<'py, Self>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "k", Kind::Int, value)
    }

    #[pyo3(name = "setOptimizer")]
    fn set_optimizer<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "optimizer", Kind::Str, value)
    }

    #[pyo3(name = "setLearningOffset")]
    fn set_learning_offset<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "learningOffset", Kind::Float, value)
    }

    #[pyo3(name = "setLearningDecay")]
    fn set_learning_decay<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "learningDecay", Kind::Float, value)
    }

    #[pyo3(name = "setSubsamplingRate")]
    fn set_subsampling_rate<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "subsamplingRate", Kind::Float, value)
    }

    #[pyo3(name = "setOptimizeDocConcentration")]
    fn set_optimize_doc_concentration<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "optimizeDocConcentration", Kind::Bool, value)
    }

    #[pyo3(name = "setDocConcentration")]
    fn set_doc_concentration<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "docConcentration", Kind::FloatList, value)
    }

    #[pyo3(name = "setTopicConcentration")]
    fn set_topic_concentration<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "topicConcentration", Kind::Float, value)
    }

    #[pyo3(name = "setTopicDistributionCol")]
    fn set_topic_distribution_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "topicDistributionCol", Kind::Str, value)
    }

    #[pyo3(name = "setKeepLastCheckpoint")]
    fn set_keep_last_checkpoint<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "keepLastCheckpoint", Kind::Bool, value)
    }

    /// The model of a fit, as PySpark's `LDA._create_model`: a DistributedLDAModel of the optimizer
    /// "em", else a LocalLDAModel. Neither has a training summary.
    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<Py<clustering_api::PyLDAModel>>> {
        // The model of the base class, of the `make` of the subclass, of which `Fitted` makes the
        // object of the subclass.
        fit_with(slf, dataset, params, || {
            if slf.try_borrow()?.inner.optimizer() == "em" {
                return fit_native_model(slf, dataset, |py, model, df| {
                    clustering_api::PyDistributedLDAModel::fitted(
                        py,
                        model,
                        df,
                        object::SummaryDataset::None,
                    )
                });
            }
            fit_native_model(slf, dataset, |py, model, df| {
                clustering_api::PyLocalLDAModel::fitted(py, model, df, object::SummaryDataset::None)
            })
        })
    }

    /// The fits of a copy of the estimator, one per param map, as `Estimator.fitMultiple`, of
    /// PySpark's keywords.
    #[pyo3(name = "fitMultiple")]
    #[allow(non_snake_case)]
    fn fit_multiple<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        paramMaps: &Bound<'py, PyAny>,
    ) -> PyResult<Py<params_api::PyFitMultipleIterator>> {
        params_api::fit_multiple(slf.as_any(), dataset, paramMaps)
    }
}

impl params_api::NativeInstance for PyLDA {
    fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        object::new_py(
            py,
            Self {
                inner: self.inner.clone(),
                unsendable: self.unsendable.clone(),
            },
        )
    }

    fn uid(&self) -> String {
        Estimator::operator(&self.inner).uid.clone()
    }

    fn set_uid(&mut self, uid: &str) {
        Estimator::operator_mut(&mut self.inner).uid = uid.to_string();
    }
}

/// Native PowerIterationClustering backed by Spark Connect's JVM ML API. It is not an estimator:
/// `assignClusters` runs the algorithm, and it saves as PySpark's
/// `PowerIterationClusteringWrapper` transformer, of its uid.
#[pyclass(
    subclass,
    name = "PowerIterationClustering",
    module = "pyspark.ml.connect.clustering"
)]
pub struct PyPowerIterationClustering {
    inner: PowerIterationClustering,
    unsendable: Vec<&'static str>,
    uid: String,
}

const POWER_ITERATION_CLUSTERING_PARAMS: &[(&str, Kind)] = &[
    ("k", Kind::Int),
    ("maxIter", Kind::Int),
    ("initMode", Kind::Str),
    ("srcCol", Kind::Str),
    ("dstCol", Kind::Str),
    ("weightCol", Kind::Str),
];

native_pymethods! {
    PyPowerIterationClustering;
    jvm: "org.apache.spark.ml.clustering.PowerIterationClusteringWrapper", op: Transformer;
    params: [
        "dstCol" => ("getDstCol", get_dst_col, param_dst_col),
        "initMode" => ("getInitMode", get_init_mode, param_init_mode),
        "k" => ("getK", get_k, param_k),
        "maxIter" => ("getMaxIter", get_max_iter, param_max_iter),
        "srcCol" => ("getSrcCol", get_src_col, param_src_col),
        "weightCol" => ("getWeightCol", get_weight_col, param_weight_col),
    ];
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, k=2, maxIter=20, initMode='random', srcCol='src', dstCol='dst', weightCol=None)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: PowerIterationClustering::new(),
            unsendable: Vec::new(),
            uid: spark_connect::ml::MlOperator::new(
                "org.apache.spark.ml.clustering.PowerIterationClustering",
                spark_connect::ml::OperatorType::Transformer,
            )
            .uid,
        };
        init_params(
            &mut this,
            "PowerIterationClustering",
            &[],
            POWER_ITERATION_CLUSTERING_PARAMS,
            args,
            kwargs,
        )?;
        Ok(this)
    }

    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, k=2, maxIter=20, initMode='random', srcCol='src', dstCol='dst', weightCol=None)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(
            slf,
            "PowerIterationClustering",
            POWER_ITERATION_CLUSTERING_PARAMS,
            args,
            kwargs,
        )
    }

    #[pyo3(name = "setK")]
    fn set_k<'py>(slf: Bound<'py, Self>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "k", Kind::Int, value)
    }

    #[pyo3(name = "setMaxIter")]
    fn set_max_iter<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxIter", Kind::Int, value)
    }

    #[pyo3(name = "setInitMode")]
    fn set_init_mode<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "initMode", Kind::Str, value)
    }

    #[pyo3(name = "setSrcCol")]
    fn set_src_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "srcCol", Kind::Str, value)
    }

    #[pyo3(name = "setDstCol")]
    fn set_dst_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "dstCol", Kind::Str, value)
    }

    #[pyo3(name = "setWeightCol")]
    fn set_weight_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "weightCol", Kind::Str, value)
    }

    /// As PySpark's `assignClusters`, a helper call (`invoke_helper_relation`) of the dataset and
    /// the params: of the active session of the thread, whose absence is the AssertionError of its
    /// `assert`, and of the dataset as `serialize` sends it, a DataFrame as its relation and
    /// another value as a literal, which the server rejects when the result is analyzed or
    /// collected (a value that cannot be one raises PySpark's error of its conversion).
    #[pyo3(name = "assignClusters")]
    fn assign_clusters(slf: &Bound<'_, Self>, dataset: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        use spark_connect::ml::{attribute_relation, FetchArg, MlTarget};
        let py = slf.py();
        let algorithm = snapshot(slf)?;
        let session = crate::session::active_session(py)
            .ok_or_else(|| pyo3::exceptions::PyAssertionError::new_err(()))?
            .borrow(py)
            .session
            .clone();
        let mut args = vec![object::fetch_arg_of(dataset)?];
        args.extend(
            algorithm
                .assign_clusters_arguments()
                .into_iter()
                .map(FetchArg::Literal),
        );
        Ok(PyDataFrame::new(
            attribute_relation(
                &MlTarget::helper(&session),
                "powerIterationClusteringAssignClusters",
                args,
            )
            .to_pyerr()?,
        ))
    }
}

impl params_api::NativeInstance for PyPowerIterationClustering {
    fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        object::new_py(
            py,
            Self {
                inner: self.inner.clone(),
                unsendable: self.unsendable.clone(),
                uid: self.uid.clone(),
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

/// Gaussian mixture estimator fitted and cached by Spark Connect.
#[pyclass(
    subclass,
    name = "GaussianMixture",
    module = "pyspark.ml.connect.clustering"
)]
pub struct PyGaussianMixture {
    inner: GaussianMixture,
    unsendable: Vec<&'static str>,
}

const GAUSSIAN_MIXTURE_PARAMS: &[(&str, Kind)] = &[
    ("featuresCol", Kind::Str),
    ("predictionCol", Kind::Str),
    ("k", Kind::Int),
    ("probabilityCol", Kind::Str),
    ("tol", Kind::Float),
    ("maxIter", Kind::Int),
    ("seed", Kind::Int),
    ("aggregationDepth", Kind::Int),
    ("weightCol", Kind::Str),
];

native_pymethods! {
    PyGaussianMixture;
    jvm: "org.apache.spark.ml.clustering.GaussianMixture", op: Estimator;
    params: [
        "aggregationDepth" => (
            "getAggregationDepth",
            get_aggregation_depth,
            param_aggregation_depth
        ),
        "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
        "k" => ("getK", get_k, param_k),
        "maxIter" => ("getMaxIter", get_max_iter, param_max_iter),
        "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
        "probabilityCol" => ("getProbabilityCol", get_probability_col, param_probability_col),
        "seed" => ("getSeed", get_seed, param_seed),
        "tol" => ("getTol", get_tol, param_tol),
        "weightCol" => ("getWeightCol", get_weight_col, param_weight_col),
    ];
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, featuresCol='features', predictionCol='prediction', k=2, probabilityCol='probability', tol=0.01, maxIter=100, seed=None, aggregationDepth=2, weightCol=None)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: GaussianMixture::new(),
            unsendable: Vec::new(),
        };
        init_params(
            &mut this,
            "GaussianMixture",
            &[],
            GAUSSIAN_MIXTURE_PARAMS,
            args,
            kwargs,
        )?;
        Ok(this)
    }
    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, featuresCol='features', predictionCol='prediction', k=2, probabilityCol='probability', tol=0.01, maxIter=100, seed=None, aggregationDepth=2, weightCol=None)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(
            slf,
            "GaussianMixture",
            GAUSSIAN_MIXTURE_PARAMS,
            args,
            kwargs,
        )
    }

    #[pyo3(name = "setFeaturesCol")]
    fn set_feature_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "featuresCol", Kind::Str, value)
    }

    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "predictionCol", Kind::Str, value)
    }

    #[pyo3(name = "setK")]
    fn set_k<'py>(slf: Bound<'py, Self>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "k", Kind::Int, value)
    }

    #[pyo3(name = "setProbabilityCol")]
    fn set_probability_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "probabilityCol", Kind::Str, value)
    }

    #[pyo3(name = "setTol")]
    fn set_tol<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "tol", Kind::Float, value)
    }

    #[pyo3(name = "setMaxIter")]
    fn set_max_iter<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxIter", Kind::Int, value)
    }

    #[pyo3(name = "setSeed")]
    fn set_seed<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "seed", Kind::Int, value)
    }

    #[pyo3(name = "setAggregationDepth")]
    fn set_aggregation_depth<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "aggregationDepth", Kind::Int, value)
    }

    #[pyo3(name = "setWeightCol")]
    fn set_weight_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "weightCol", Kind::Str, value)
    }

    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<Py<clustering_api::PyGaussianMixtureModel>>> {
        fit_with(slf, dataset, params, || {
            fit_native_model(slf, dataset, |py, model, df| {
                clustering_api::PyGaussianMixtureModel::fitted(
                    py,
                    model,
                    df,
                    object::SummaryDataset::Transformed,
                )
            })
        })
    }

    /// The fits of a copy of the estimator, one per param map, as `Estimator.fitMultiple`, of
    /// PySpark's keywords.
    #[pyo3(name = "fitMultiple")]
    #[allow(non_snake_case)]
    fn fit_multiple<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        paramMaps: &Bound<'py, PyAny>,
    ) -> PyResult<Py<params_api::PyFitMultipleIterator>> {
        params_api::fit_multiple(slf.as_any(), dataset, paramMaps)
    }
}

impl params_api::NativeInstance for PyGaussianMixture {
    fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        object::new_py(
            py,
            Self {
                inner: self.inner.clone(),
                unsendable: self.unsendable.clone(),
            },
        )
    }

    fn uid(&self) -> String {
        Estimator::operator(&self.inner).uid.clone()
    }

    fn set_uid(&mut self, uid: &str) {
        Estimator::operator_mut(&mut self.inner).uid = uid.to_string();
    }
}

/// Bisecting K-means fitted and cached by Spark Connect.
#[pyclass(
    subclass,
    name = "BisectingKMeans",
    module = "pyspark.ml.connect.clustering"
)]
pub struct PyBisectingKMeans {
    inner: BisectingKMeans,
    unsendable: Vec<&'static str>,
}

const BISECTING_KMEANS_PARAMS: &[(&str, Kind)] = &[
    ("featuresCol", Kind::Str),
    ("predictionCol", Kind::Str),
    ("maxIter", Kind::Int),
    ("seed", Kind::Int),
    ("k", Kind::Int),
    ("minDivisibleClusterSize", Kind::Float),
    ("distanceMeasure", Kind::Str),
    ("weightCol", Kind::Str),
];

native_pymethods! {
    PyBisectingKMeans;
    jvm: "org.apache.spark.ml.clustering.BisectingKMeans", op: Estimator;
    params: [
        "distanceMeasure" => ("getDistanceMeasure", get_distance_measure, param_distance_measure),
        "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
        "k" => ("getK", get_k, param_k),
        "maxIter" => ("getMaxIter", get_max_iter, param_max_iter),
        "minDivisibleClusterSize" => (
            "getMinDivisibleClusterSize",
            get_min_divisible_cluster_size,
            param_min_divisible_cluster_size
        ),
        "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
        "seed" => ("getSeed", get_seed, param_seed),
        "weightCol" => ("getWeightCol", get_weight_col, param_weight_col),
    ];
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, featuresCol='features', predictionCol='prediction', maxIter=20, seed=None, k=4, minDivisibleClusterSize=1.0, distanceMeasure='euclidean', weightCol=None)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: BisectingKMeans::new(),
            unsendable: Vec::new(),
        };
        init_params(
            &mut this,
            "BisectingKMeans",
            &[],
            BISECTING_KMEANS_PARAMS,
            args,
            kwargs,
        )?;
        Ok(this)
    }
    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, featuresCol='features', predictionCol='prediction', maxIter=20, seed=None, k=4, minDivisibleClusterSize=1.0, distanceMeasure='euclidean', weightCol=None)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(
            slf,
            "BisectingKMeans",
            BISECTING_KMEANS_PARAMS,
            args,
            kwargs,
        )
    }

    #[pyo3(name = "setFeaturesCol")]
    fn set_feature_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "featuresCol", Kind::Str, value)
    }

    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "predictionCol", Kind::Str, value)
    }

    #[pyo3(name = "setK")]
    fn set_k<'py>(slf: Bound<'py, Self>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "k", Kind::Int, value)
    }

    #[pyo3(name = "setMaxIter")]
    fn set_max_iter<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxIter", Kind::Int, value)
    }

    #[pyo3(name = "setSeed")]
    fn set_seed<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "seed", Kind::Int, value)
    }

    #[pyo3(name = "setDistanceMeasure")]
    fn set_distance_measure<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "distanceMeasure", Kind::Str, value)
    }

    #[pyo3(name = "setWeightCol")]
    fn set_weight_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "weightCol", Kind::Str, value)
    }

    #[pyo3(name = "setMinDivisibleClusterSize")]
    fn set_min_divisible_cluster_size<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "minDivisibleClusterSize", Kind::Float, value)
    }

    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<Py<clustering_api::PyBisectingKMeansModel>>> {
        fit_with(slf, dataset, params, || {
            fit_native_model(slf, dataset, |py, model, df| {
                clustering_api::PyBisectingKMeansModel::fitted(
                    py,
                    model,
                    df,
                    object::SummaryDataset::Transformed,
                )
            })
        })
    }

    /// The fits of a copy of the estimator, one per param map, as `Estimator.fitMultiple`, of
    /// PySpark's keywords.
    #[pyo3(name = "fitMultiple")]
    #[allow(non_snake_case)]
    fn fit_multiple<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        paramMaps: &Bound<'py, PyAny>,
    ) -> PyResult<Py<params_api::PyFitMultipleIterator>> {
        params_api::fit_multiple(slf.as_any(), dataset, paramMaps)
    }
}

impl params_api::NativeInstance for PyBisectingKMeans {
    fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        object::new_py(
            py,
            Self {
                inner: self.inner.clone(),
                unsendable: self.unsendable.clone(),
            },
        )
    }

    fn uid(&self) -> String {
        Estimator::operator(&self.inner).uid.clone()
    }

    fn set_uid(&mut self, uid: &str) {
        Estimator::operator_mut(&mut self.inner).uid = uid.to_string();
    }
}

native!(
    PyLogisticRegression => LogisticRegression,
    PyLinearSVC => LinearSVC,
    PyFMClassifier => FMClassifier,
    PyMultilayerPerceptronClassifier => MultilayerPerceptronClassifier,
    PyLinearRegression => LinearRegression,
    PyAFTSurvivalRegression => AFTSurvivalRegression,
    PyIsotonicRegression => IsotonicRegression,
    PyFMRegressor => FMRegressor,
    PyGeneralizedLinearRegression => GeneralizedLinearRegression,
    PyDecisionTreeClassifier => DecisionTreeClassifier,
    PyNaiveBayes => NaiveBayes,
    PyGBTClassifier => GBTClassifier,
    PyGBTRegressor => GBTRegressor,
    PyRandomForestClassifier => RandomForestClassifier,
    PyDecisionTreeRegressor => DecisionTreeRegressor,
    PyRandomForestRegressor => RandomForestRegressor,
    PyLDA => LDA,
    PyPowerIterationClustering => PowerIterationClustering,
    PyGaussianMixture => GaussianMixture,
    PyKMeans => KMeans,
    PyBisectingKMeans => BisectingKMeans,
);
