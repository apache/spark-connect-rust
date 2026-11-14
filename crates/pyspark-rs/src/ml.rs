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
    AFTSurvivalRegression, BisectingKMeans, Classifier, CrossValidator, DecisionTreeClassifier,
    DecisionTreeRegressor, Estimator, Evaluator, FMClassifier, FMRegressor, GBTClassifier,
    GBTRegressor, GaussianMixture, GeneralizedLinearRegression, IsotonicRegression, KMeans,
    LinearRegression, LinearSVC, LogisticRegression, MlMatrix, MlVector, Model,
    MultilayerPerceptronClassifier, NaiveBayes, OneVsRest, Params, Pipeline,
    PowerIterationClustering, RandomForestClassifier, RandomForestRegressor, Transformer, LDA,
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
pub(crate) mod regression_api;

pub(crate) mod evaluation;
pub use evaluation::{
    PyBinaryClassificationEvaluator, PyClusteringEvaluator, PyMulticlassClassificationEvaluator,
    PyMultilabelClassificationEvaluator, PyRankingEvaluator, PyRegressionEvaluator,
};

pub(crate) mod feature;
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

/// Whether `dataset` is a pandas DataFrame, of an imported pandas.
fn is_pandas(dataset: &Bound<'_, PyAny>) -> PyResult<bool> {
    let pandas = dataset
        .py()
        .import("sys")?
        .getattr("modules")?
        .get_item("pandas")
        .ok();
    match pandas {
        Some(pandas) => dataset.is_instance(&pandas.getattr("DataFrame")?),
        None => Ok(false),
    }
}

/// Native ML runs on Spark DataFrames. Pandas input must be converted explicitly by the caller.
fn reject_pandas(dataset: &Bound<'_, PyAny>, method: &str) -> PyResult<()> {
    if is_pandas(dataset)? {
        return Err(PyTypeError::new_err(format!(
            "{method}() requires a Spark DataFrame; pandas DataFrames are not supported. \
             Convert the input with spark.createDataFrame(...) first."
        )));
    }
    Ok(())
}

/// The dataset of `method`, a Spark DataFrame. A pandas DataFrame raises the TypeError of
/// [`reject_pandas`]. Another object raises the `AttributeError` of `sparkSession`, the attribute
/// that `pyspark.ml` reads of the dataset first (`try_remote_fit`, `try_remote_transform_relation`
/// and `try_remote_evaluate` of `pyspark.ml.util`), and an object that has it raises a TypeError.
fn dataset_of(dataset: &Bound<'_, PyAny>, method: &str) -> PyResult<DataFrame> {
    if let Ok(df) = dataset.extract::<PyRef<'_, PyDataFrame>>() {
        return Ok(df.dataframe.clone());
    }
    reject_pandas(dataset, method)?;
    dataset.getattr("sparkSession")?;
    Err(PyTypeError::new_err(format!(
        "{method}() takes a Spark DataFrame, not '{}'",
        dataset.get_type().name()?
    )))
}

/// The dataset of `method`, a call of `pyspark.ml` that reads nothing of it, but sends it as an
/// argument of a helper or attribute call (`invoke_helper_relation`, `try_remote_attribute_relation`),
/// as `serialize` of `pyspark.ml.connect.serialize` sends an argument that is not a DataFrame: a
/// `(value, DataType)` pair as a literal of that type, an ML vector or matrix as itself, and
/// another value as the literal of `LiteralExpression._from_value`, which raises PySpark's error of
/// a value that cannot be one (`UNSUPPORTED_DATA_TYPE`, also of a pandas DataFrame, where a fit,
/// transform or evaluate raises the TypeError of [`reject_pandas`]). The server rejects a value that
/// PySpark sends, which this client does not send but refuses with a TypeError.
pub(crate) fn helper_dataset_of(dataset: &Bound<'_, PyAny>, method: &str) -> PyResult<DataFrame> {
    if let Ok(df) = dataset.extract::<PyRef<'_, PyDataFrame>>() {
        return Ok(df.dataframe.clone());
    }
    if !serialized_as_is(dataset)? {
        crate::functions::to_value_column(dataset)?;
    }
    Err(PyTypeError::new_err(format!(
        "{method}() takes a Spark DataFrame, not '{}'",
        dataset.get_type().name()?
    )))
}

/// Whether `serialize` of `pyspark.ml.connect.serialize` sends `value` without
/// `LiteralExpression._from_value`: a `(value, DataType)` pair, and an ML vector or matrix of the
/// classes that `serialize_param` takes.
fn serialized_as_is(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = value.py();
    if let Ok(pair) = value.cast::<PyTuple>() {
        if pair.len() == 2 {
            let data_type = py.import("pyspark.sql.types")?.getattr("DataType")?;
            if pair.get_item(1)?.is_instance(&data_type)? {
                return Ok(true);
            }
        }
    }
    let linalg = py.import("pyspark.ml.linalg")?;
    for class in ["SparseVector", "DenseVector", "SparseMatrix", "DenseMatrix"] {
        if value.is_instance(&linalg.getattr(class)?)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Fit a native estimator on `dataset`, a Spark DataFrame, as `try_remote_fit` of `pyspark.ml`
/// fits it: the fit sends the params as they are given, and the JVM estimator reads its columns
/// by those names exactly, and raises its own errors of their types (a featuresCol of a type that
/// it does not take) and values (a label that is not a class index). `select` names the columns of
/// the dataset that the fit sends; with None, the whole dataset is sent, as `pyspark.ml` sends it.
fn fit_native<T>(
    slf: &Bound<'_, T>,
    dataset: &Bound<'_, PyAny>,
    select: Option<&[&str]>,
) -> PyResult<PyMLModel>
where
    T: Native + PyClass,
    T::Core: Estimator,
{
    let df = dataset_of(dataset, "fit")?;
    let estimator = snapshot(slf)?;
    let df = match select {
        Some(columns) => df.select(columns.iter().map(|name| exact_column(name))),
        None => df,
    };
    fit_core(slf.py(), estimator, &df)
}

/// Fit a native estimator on `dataset`, as [`fit_native`], and make its typed model of the core
/// model and the training DataFrame with `model` (`PyLinearSVCModel::fitted`), which has the
/// defaults of the estimator, as `_copyValues` copies them to the model of a fit.
fn fit_native_model<T, M>(
    slf: &Bound<'_, T>,
    dataset: &Bound<'_, PyAny>,
    model: impl FnOnce(Python<'_>, Box<dyn Model>, &DataFrame) -> PyResult<M>,
) -> PyResult<M>
where
    T: Native + PyClass<Frozen = False>,
    T::Core: Estimator,
    M: params_api::ParamsHolder + pyo3::PyTypeInfo,
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
    Ok(model)
}

/// The fit of a native estimator whose core cannot fit it, as Pipeline and CrossValidator.
/// The core raises NOT_IMPLEMENTED of `feature` before any request, whatever its params are, so
/// the params are not serialized first, and PySpark's serialization errors of them are not
/// raised. Pandas inputs are rejected before reaching this unsupported operation.
fn fit_unimplemented<T>(
    slf: &Bound<'_, T>,
    dataset: &Bound<'_, PyAny>,
    feature: &str,
) -> PyResult<PyMLModel>
where
    T: Native + PyClass,
    T::Core: Estimator,
{
    let py = slf.py();
    reject_pandas(dataset, "fit")?;
    match dataset.extract::<PyRef<'_, PyDataFrame>>() {
        Ok(df) => fit_core(py, slf.try_borrow()?.inner().clone(), &df.dataframe),
        Err(_) => Err(crate::column::raise_pyspark(
            py,
            "PySparkNotImplementedError",
            "NOT_IMPLEMENTED",
            &[("feature", feature)],
        )),
    }
}

fn fit_core<C: Estimator>(py: Python<'_>, mut estimator: C, df: &DataFrame) -> PyResult<PyMLModel> {
    let session = df.spark_session();
    let model = interruptible(py, &session, || estimator.fit(df).map_err(Box::new))?;
    Ok(PyMLModel {
        model,
        session_id: session.session_id().to_string(),
    })
}

/// The column of the name `name` exactly, a name with a dot too.
fn exact_column(name: &str) -> spark_connect::Column {
    spark_connect::column::col(&format!("`{}`", name.replace('`', "``")))
}

/// A fitted ML model (result of `Estimator.fit`), wrapping a boxed core `Model`.
#[pyclass(name = "MLModel", module = "pyspark.ml.connect")]
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

/// The value of the param `name` of `params` that is set, as PySpark's `getOrDefault` reads a set
/// param: None when it is not set, `Some(None)` when it is set to None (a null).
fn set_literal<'a>(
    params: &'a Params,
    name: &str,
) -> Option<Option<&'a spark_connect_proto::expression::literal::LiteralType>> {
    use spark_connect_proto::expression::{literal::LiteralType, ExprType};

    match params.get_param(name)?.expr_type.as_ref() {
        Some(ExprType::Literal(literal)) => match literal.literal_type.as_ref() {
            Some(LiteralType::Null(_)) | None => Some(None),
            value => Some(value),
        },
        _ => Some(None),
    }
}

/// `_checkThresholdConsistency` of PySpark's LogisticRegression, which its constructor and its
/// `setParams` make once they have set the params that they are given, of the params of `params`:
/// when both `threshold` and `thresholds` are set, `thresholds` must have two values, whose
/// equivalent threshold, `1 / (1 + thresholds[0] / thresholds[1])`, is `threshold` within 1e-5,
/// else PySpark's ValueError, which leaves them set, as PySpark's `_set` does. A None of either,
/// and a division by zero, raise the TypeError and the ZeroDivisionError of PySpark's arithmetic.
/// (Its setters make no check, as PySpark's, whose `clear` of the other is a no-op over Connect.)
fn check_threshold_consistency(py: Python<'_>, params: &Params) -> PyResult<()> {
    use pyo3::exceptions::PyZeroDivisionError;
    use spark_connect_proto::expression::literal::LiteralType;

    let (Some(threshold), Some(thresholds)) = (
        set_literal(params, "threshold"),
        set_literal(params, "thresholds"),
    ) else {
        return Ok(());
    };
    let Some(LiteralType::Array(thresholds)) = thresholds else {
        return Err(PyTypeError::new_err(
            "object of type 'NoneType' has no len()",
        ));
    };
    let ts: Vec<f64> = thresholds
        .elements
        .iter()
        .filter_map(|element| match element.literal_type {
            Some(LiteralType::Double(value)) => Some(value),
            _ => None,
        })
        .collect();
    if ts.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "Logistic Regression getThreshold only applies to binary classification, but \
             thresholds has length != 2. thresholds: {}",
            PyList::new(py, &ts)?.str()?
        )));
    }
    let division = || PyZeroDivisionError::new_err("float division by zero");
    if ts[1] == 0.0 {
        return Err(division());
    }
    let denominator = 1.0 + ts[0] / ts[1];
    if denominator == 0.0 {
        return Err(division());
    }
    let t = 1.0 / denominator;
    let Some(&LiteralType::Double(t2)) = threshold else {
        return Err(PyTypeError::new_err(
            "unsupported operand type(s) for -: 'NoneType' and 'float'",
        ));
    };
    if (t2 - t).abs() >= 1e-5 {
        // Of PySpark's `%g`s.
        let message = pyo3::types::PyString::new(
            py,
            "Logistic Regression getThreshold found inconsistent values for threshold (%g) and \
             thresholds (equivalent to %g)",
        )
        .call_method1("__mod__", ((t2, t),))?
        .extract::<String>()?;
        return Err(PyValueError::new_err(message));
    }
    Ok(())
}

#[pymethods]
impl PyLogisticRegression {
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
        check_threshold_consistency(args.py(), this.inner.params())?;
        Ok(this)
    }
    /// `setParams` of PySpark's LogisticRegression: as its constructor, it sets the params that it
    /// is given (see [`set_params_of`]), then checks `threshold` and `thresholds`, whichever call
    /// set them (see [`check_threshold_consistency`]).
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
        check_threshold_consistency(slf.py(), slf.try_borrow()?.inner.params())?;
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
    /// Fit `dataset`, a Spark DataFrame, as PySpark's `pyspark.ml` fits it (see [`fit_native`]):
    /// its model outputs the JVM model's `rawPrediction` and `probability` vectors and `double`
    /// prediction.
    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<PyMLModel>> {
        fit_with(slf, dataset, params, || fit_native(slf, dataset, None))
    }
}

/// Linear support vector classifier backed by Spark Connect's JVM ML API.
#[pyclass(name = "LinearSVC", module = "pyspark.ml.connect.classification")]
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
        "aggregationDepth" => ("getAggregationDepth", get_aggregation_depth, param_aggregation_depth),
        "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
        "fitIntercept" => ("getFitIntercept", get_fit_intercept, param_fit_intercept),
        "labelCol" => ("getLabelCol", get_label_col, param_label_col),
        "maxBlockSizeInMB" => ("getMaxBlockSizeInMB", get_max_block_size_in_mb, param_max_block_size_in_mb),
        "maxIter" => ("getMaxIter", get_max_iter, param_max_iter),
        "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
        "rawPredictionCol" => ("getRawPredictionCol", get_raw_prediction_col, param_raw_prediction_col),
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
    ) -> PyResult<Fitted<classification_api::PyLinearSVCModel>> {
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
    ) -> PyResult<params_api::PyFitMultipleIterator> {
        params_api::fit_multiple(slf.as_any(), dataset, paramMaps)
    }
}

impl params_api::NativeInstance for PyLinearSVC {
    fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        Py::new(
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
#[pyclass(name = "FMClassifier", module = "pyspark.ml.connect.classification")]
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

#[pymethods]
impl PyFMClassifier {
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, featuresCol='features', labelCol='label', predictionCol='prediction', probabilityCol='probability', rawPredictionCol='rawPrediction', factorSize=8, fitIntercept=True, fitLinear=True, regParam=0.0, miniBatchFraction=1.0, initStd=0.01, maxIter=100, stepSize=1.0, tol=1e-06, solver='adamW', thresholds=None, seed=None)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: FMClassifier::new(),
            unsendable: Vec::new(),
        };
        init_params(
            &mut this,
            "FMClassifier",
            &[],
            FMCLASSIFIER_PARAMS,
            args,
            kwargs,
        )?;
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

    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<PyMLModel>> {
        fit_with(slf, dataset, params, || fit_native(slf, dataset, None))
    }
}

/// Native MultilayerPerceptronClassifier.
#[pyclass(
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

#[pymethods]
impl PyMultilayerPerceptronClassifier {
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

    #[pyo3(name = "setTol")]
    fn set_tol<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "tol", Kind::Float, value)
    }

    #[pyo3(name = "setSeed")]
    fn set_seed<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "seed", Kind::Int, value)
    }

    #[pyo3(name = "setLayers")]
    fn set_layers<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "layers", Kind::IntList, value)
    }

    #[pyo3(name = "setBlockSize")]
    fn set_block_size<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "blockSize", Kind::Int, value)
    }

    #[pyo3(name = "setStepSize")]
    fn set_step_size<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "stepSize", Kind::Float, value)
    }

    #[pyo3(name = "setSolver")]
    fn set_solver<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "solver", Kind::Str, value)
    }

    #[pyo3(name = "setInitialWeights")]
    fn set_initial_weights<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "initialWeights", Kind::Vector, value)
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
    ) -> PyResult<Fitted<PyMLModel>> {
        fit_with(slf, dataset, params, || fit_native(slf, dataset, None))
    }
}

/// Native OneVsRest.
#[pyclass(name = "OneVsRest", module = "pyspark.ml.connect.classification")]
pub struct PyOneVsRest {
    inner: OneVsRest,
    unsendable: Vec<&'static str>,
    classifier: Option<Py<PyAny>>,
}

const ONE_VS_REST_PARAMS: &[(&str, Kind)] = &[
    ("featuresCol", Kind::Str),
    ("labelCol", Kind::Str),
    ("predictionCol", Kind::Str),
    ("rawPredictionCol", Kind::Str),
    ("weightCol", Kind::Str),
    ("parallelism", Kind::ClampedInt),
];

#[pymethods]
impl PyOneVsRest {
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, featuresCol='features', labelCol='label', predictionCol='prediction', rawPredictionCol='rawPrediction', classifier=None, weightCol=None, parallelism=1)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: OneVsRest::new(),
            unsendable: Vec::new(),
            classifier: None,
        };
        let remaining = kwargs.map(|kw| kw.copy()).transpose()?;
        if let Some(kw) = &remaining {
            if let Some(classifier) = kw.get_item("classifier")? {
                if !classifier.is_none() {
                    native_classifier(&classifier, false)?;
                    this.classifier = Some(classifier.unbind());
                }
                kw.del_item("classifier")?;
            }
        }
        init_params(
            &mut this,
            "OneVsRest",
            &[],
            ONE_VS_REST_PARAMS,
            args,
            remaining.as_ref(),
        )?;
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
        // The classifier is no param, and as in the constructor, it is set by setClassifier.
        let remaining = kwargs.map(|kw| kw.copy()).transpose()?;
        if let Some(kw) = &remaining {
            if let Some(classifier) = kw.get_item("classifier")? {
                Self::set_classifier(slf.clone(), &classifier)?;
                kw.del_item("classifier")?;
            }
        }
        set_params_of(
            slf,
            "OneVsRest",
            ONE_VS_REST_PARAMS,
            args,
            remaining.as_ref(),
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

    #[pyo3(name = "setRawPredictionCol")]
    fn set_raw_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "rawPredictionCol", Kind::Str, value)
    }

    #[pyo3(name = "setWeightCol")]
    fn set_weight_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "weightCol", Kind::Str, value)
    }

    #[pyo3(name = "setParallelism")]
    fn set_parallelism<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "parallelism", Kind::ClampedInt, value)
    }

    #[pyo3(name = "setClassifier")]
    fn set_classifier<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        if value.is_none() {
            slf.try_borrow_mut()?.classifier = None;
        } else {
            native_classifier(value, false)?;
            slf.try_borrow_mut()?.classifier = Some(value.clone().unbind());
        }
        Ok(slf)
    }

    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<PyMLModel>> {
        // A list of param maps fits once per map (`fit_with`), each in PySpark's order below.
        fit_with(slf, dataset, params, || -> PyResult<PyMLModel> {
            let py = slf.py();
            reject_pandas(dataset, "fit")?;
            let classifier = slf
                .try_borrow()?
                .classifier
                .as_ref()
                .map(|x| x.clone_ref(py))
                .ok_or_else(|| PyValueError::new_err("OneVsRest requires a native classifier"))?;
            // PySpark's `_fit` reads the max label first, by `dataset.agg`: another object raises the
            // AttributeError of `agg`.
            let df = match dataset.extract::<PyRef<'_, PyDataFrame>>() {
                Ok(df) => df.dataframe.clone(),
                Err(_) => {
                    dataset.getattr("agg")?;
                    return Err(PyTypeError::new_err(format!(
                        "fit() takes a Spark DataFrame, not '{}'",
                        dataset.get_type().name()?
                    )));
                }
            };
            let mut core = slf.try_borrow()?.inner.clone();
            core.check_required_params().to_pyerr()?;
            let session = df.spark_session();
            let max = interruptible(py, &session, || core.max_label(&df).map_err(Box::new))?;
            // `numClasses = int(max) + 1`, by Python's own `int` and `min` below, so that a null, NaN
            // or infinite max, and a null parallelism, raise their errors of this Python.
            let builtins = py.import("builtins")?;
            let num_classes = builtins.getattr("int")?.call1((max,))?.add(1)?;
            if !native_classifier(classifier.bind(py), false)?.supports_weight()
                && core.weight_col().is_some_and(|s| !s.is_empty())
            {
                py.import("warnings")?.call_method1(
                    "warn",
                    ("weightCol is ignored, as it is not supported by this classifier",),
                )?;
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
            let model = interruptible(py, &session, || {
                core.fit_classes(&df, num_classes, threads as usize)
                    .map_err(Box::new)
            })?;
            Ok(PyMLModel {
                model,
                session_id: session.session_id().to_string(),
            })
        })
    }
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
#[pyclass(name = "LinearRegression", module = "pyspark.ml.connect.regression")]
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

#[pymethods]
impl PyLinearRegression {
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
    ) -> PyResult<Fitted<PyMLModel>> {
        fit_with(slf, dataset, params, || fit_native(slf, dataset, None))
    }
}

/// Generalized linear regression fitted and cached by Spark Connect.
#[pyclass(
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

#[pymethods]
impl PyGeneralizedLinearRegression {
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
    ) -> PyResult<Fitted<PyMLModel>> {
        fit_with(slf, dataset, params, || fit_native(slf, dataset, None))
    }
}

/// Weibull accelerated failure time regression using Spark's JVM implementation.
#[pyclass(
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

#[pymethods]
impl PyAFTSurvivalRegression {
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
    ) -> PyResult<Fitted<PyMLModel>> {
        fit_with(slf, dataset, params, || fit_native(slf, dataset, None))
    }
}

/// Univariate isotonic or antitonic regression using Spark's JVM implementation.
#[pyclass(name = "IsotonicRegression", module = "pyspark.ml.connect.regression")]
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

#[pymethods]
impl PyIsotonicRegression {
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
    ) -> PyResult<Fitted<PyMLModel>> {
        fit_with(slf, dataset, params, || fit_native(slf, dataset, None))
    }
}

/// Factorization machines regression using Spark's JVM implementation.
#[pyclass(name = "FMRegressor", module = "pyspark.ml.connect.regression")]
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

#[pymethods]
impl PyFMRegressor {
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
    ) -> PyResult<Fitted<PyMLModel>> {
        fit_with(slf, dataset, params, || fit_native(slf, dataset, None))
    }
}

/// Decision tree classifier fitted and cached by Spark Connect.
#[pyclass(
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

#[pymethods]
impl PyDecisionTreeClassifier {
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
    ) -> PyResult<Fitted<PyMLModel>> {
        fit_with(slf, dataset, params, || fit_native(slf, dataset, None))
    }
}

/// Naive Bayes classifier fitted and cached by Spark Connect.
#[pyclass(name = "NaiveBayes", module = "pyspark.ml.connect.classification")]
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

#[pymethods]
impl PyNaiveBayes {
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
    ) -> PyResult<Fitted<PyMLModel>> {
        fit_with(slf, dataset, params, || fit_native(slf, dataset, None))
    }
}

/// Gradient-boosted binary classifier fitted and cached by Spark Connect.
#[pyclass(name = "GBTClassifier", module = "pyspark.ml.connect.classification")]
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

#[pymethods]
impl PyGBTClassifier {
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
    ) -> PyResult<Fitted<PyMLModel>> {
        fit_with(slf, dataset, params, || fit_native(slf, dataset, None))
    }
}

/// Random forest classifier fitted and cached by Spark Connect.
#[pyclass(
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

#[pymethods]
impl PyRandomForestClassifier {
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
    ) -> PyResult<Fitted<PyMLModel>> {
        fit_with(slf, dataset, params, || fit_native(slf, dataset, None))
    }
}

/// Gradient-boosted regressor fitted and cached by Spark Connect.
#[pyclass(name = "GBTRegressor", module = "pyspark.ml.connect.regression")]
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

#[pymethods]
impl PyGBTRegressor {
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

    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<PyMLModel>> {
        fit_with(slf, dataset, params, || fit_native(slf, dataset, None))
    }
}

/// Decision tree regressor fitted and cached by Spark Connect.
#[pyclass(
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

#[pymethods]
impl PyDecisionTreeRegressor {
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

    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<PyMLModel>> {
        fit_with(slf, dataset, params, || fit_native(slf, dataset, None))
    }
}
/// Random forest regressor fitted and cached by Spark Connect.
#[pyclass(
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

#[pymethods]
impl PyRandomForestRegressor {
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

    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<PyMLModel>> {
        fit_with(slf, dataset, params, || fit_native(slf, dataset, None))
    }
}

/// K-means clustering backed by Spark Connect's JVM ML API.
#[pyclass(name = "KMeans", module = "pyspark.ml.connect.clustering")]
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
        "intermediateStorageLevel" => ("getIntermediateStorageLevel", get_intermediate_storage_level, param_intermediate_storage_level),
        "k" => ("getK", get_k, param_k),
        "maxBlockSizeInMB" => ("getMaxBlockSizeInMB", get_max_block_size_in_mb, param_max_block_size_in_mb),
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
    ) -> PyResult<Fitted<clustering_api::PyKMeansModel>> {
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
    ) -> PyResult<params_api::PyFitMultipleIterator> {
        params_api::fit_multiple(slf.as_any(), dataset, paramMaps)
    }
}

impl params_api::NativeInstance for PyKMeans {
    fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        Py::new(
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
#[pyclass(name = "LDA", module = "pyspark.ml.connect.clustering")]
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

#[pymethods]
impl PyLDA {
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

    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<PyMLModel>> {
        fit_with(slf, dataset, params, || fit_native(slf, dataset, None))
    }
}

/// Native PowerIterationClustering backed by Spark Connect's JVM ML API.
#[pyclass(
    name = "PowerIterationClustering",
    module = "pyspark.ml.connect.clustering"
)]
pub struct PyPowerIterationClustering {
    inner: PowerIterationClustering,
    unsendable: Vec<&'static str>,
}

const POWER_ITERATION_CLUSTERING_PARAMS: &[(&str, Kind)] = &[
    ("k", Kind::Int),
    ("maxIter", Kind::Int),
    ("initMode", Kind::Str),
    ("srcCol", Kind::Str),
    ("dstCol", Kind::Str),
    ("weightCol", Kind::Str),
];

#[pymethods]
impl PyPowerIterationClustering {
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, k=2, maxIter=20, initMode='random', srcCol='src', dstCol='dst', weightCol=None)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: PowerIterationClustering::new(),
            unsendable: Vec::new(),
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

    #[pyo3(name = "assignClusters")]
    fn assign_clusters(slf: &Bound<'_, Self>, dataset: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let df = helper_dataset_of(dataset, "assignClusters")?;
        let algorithm = snapshot(slf)?;
        Ok(PyDataFrame::new(algorithm.assign_clusters(&df).to_pyerr()?))
    }
}

/// Gaussian mixture estimator fitted and cached by Spark Connect.
#[pyclass(name = "GaussianMixture", module = "pyspark.ml.connect.clustering")]
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

#[pymethods]
impl PyGaussianMixture {
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
    ) -> PyResult<Fitted<PyMLModel>> {
        fit_with(slf, dataset, params, || fit_native(slf, dataset, None))
    }
}

/// Bisecting K-means fitted and cached by Spark Connect.
#[pyclass(name = "BisectingKMeans", module = "pyspark.ml.connect.clustering")]
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

#[pymethods]
impl PyBisectingKMeans {
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
    ) -> PyResult<Fitted<PyMLModel>> {
        fit_with(slf, dataset, params, || fit_native(slf, dataset, None))
    }
}

/// Pipeline estimator (chains stages by name).
const PIPELINE_PARAMS: &[(&str, Kind)] = &[("stages", Kind::StrList)];

#[pyclass(name = "Pipeline", module = "pyspark.ml.connect.pipeline")]
pub struct PyPipeline {
    inner: Pipeline,
    unsendable: Vec<&'static str>,
}

#[pymethods]
impl PyPipeline {
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, stages=None)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: Pipeline::new(),
            unsendable: Vec::new(),
        };
        init_params(&mut this, "Pipeline", &[], PIPELINE_PARAMS, args, kwargs)?;
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
        set_params_of(slf, "Pipeline", PIPELINE_PARAMS, args, kwargs)
    }
    #[pyo3(name = "setStages")]
    fn set_stages<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "stages", Kind::StrList, value)
    }
    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<PyMLModel>> {
        fit_with(slf, dataset, params, || {
            fit_unimplemented(slf, dataset, "Pipeline.fit")
        })
    }
}

/// CrossValidator (k-fold tuning).
const CROSS_VALIDATOR_PARAMS: &[(&str, Kind)] = &[
    ("numFolds", Kind::Int),
    ("parallelism", Kind::Int),
    ("seed", Kind::Int),
];

#[pyclass(name = "CrossValidator", module = "pyspark.ml.connect.tuning")]
pub struct PyCrossValidator {
    inner: CrossValidator,
    unsendable: Vec<&'static str>,
}

#[pymethods]
impl PyCrossValidator {
    #[new]
    #[pyo3(
        signature = (*args, **kwargs),
        text_signature = "(*, numFolds=None, parallelism=None, seed=None)"
    )]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: CrossValidator::new(),
            unsendable: Vec::new(),
        };
        init_params(
            &mut this,
            "CrossValidator",
            &[],
            CROSS_VALIDATOR_PARAMS,
            args,
            kwargs,
        )?;
        Ok(this)
    }
    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, numFolds=None, parallelism=None, seed=None)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(slf, "CrossValidator", CROSS_VALIDATOR_PARAMS, args, kwargs)
    }
    #[pyo3(name = "setNumFolds")]
    fn set_num_folds<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "numFolds", Kind::Int, value)
    }
    #[pyo3(name = "setParallelism")]
    fn set_parallelism<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "parallelism", Kind::Int, value)
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
    ) -> PyResult<Fitted<PyMLModel>> {
        fit_with(slf, dataset, params, || {
            fit_unimplemented(slf, dataset, "CrossValidator.fit")
        })
    }
}

native!(
    PyLogisticRegression => LogisticRegression,
    PyLinearSVC => LinearSVC,
    PyFMClassifier => FMClassifier,
    PyMultilayerPerceptronClassifier => MultilayerPerceptronClassifier,
    PyOneVsRest => OneVsRest,
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
    PyPipeline => Pipeline,
    PyCrossValidator => CrossValidator,
);
