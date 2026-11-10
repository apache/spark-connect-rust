//! PyO3 wrappers for `spark_connect::ml` (mirrors `pyspark.ml.connect`).
//!
//! Each wrapper holds a core estimator, transformer or evaluator, whose params are those that
//! Python set, as PySpark's `_paramMap`: a fit, transform or evaluate sends only them, and the
//! server uses its defaults for the others, as with PySpark. The constructors are keyword-only,
//! as PySpark's `@keyword_only` ones, and they and the setters convert each value with the
//! `TypeConverters` of the param in PySpark, and send it as PySpark sends a Python value. A
//! setter sets the param of the wrapper itself and returns the wrapper, as PySpark's
//! `Params._set` does. `fit`, `transform` and `evaluate` clone the core object under a short
//! borrow, so that another thread may call a setter while they wait for the server.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::pyclass::boolean_struct::False;
use pyo3::types::{PyDict, PyTuple, PyType};
use pyo3::PyClass;
use spark_connect::ml::{
    BinaryClassificationEvaluator, CrossValidator, Estimator, Evaluator, KMeans, LinearRegression,
    LinearSVC, LogisticRegression, MaxAbsScaler, Model, MulticlassClassificationEvaluator, Params,
    Pipeline, RandomForestClassifier, RegressionEvaluator, StandardScaler, StringIndexer,
    Transformer, VectorAssembler,
};
use spark_connect::DataFrame;

use crate::dataframe::PyDataFrame;
use crate::errors::ResultExt;
use crate::interrupt::interruptible;

/// The kind of a param: the method of PySpark's `TypeConverters` that converts its value, and
/// the literal that PySpark sends of the converted value.
#[derive(Clone, Copy)]
enum Kind {
    /// `toInt`: an `Integer` literal when the int fits 32 bits, else a `Long`.
    Int,
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
}

impl Kind {
    fn converter(self) -> &'static str {
        match self {
            Kind::Int => "toInt",
            Kind::Float => "toFloat",
            Kind::Str => "toString",
            Kind::Bool => "toBoolean",
            Kind::StrList => "toListString",
            Kind::FloatList => "toListFloat",
        }
    }
}

/// `pyspark.ml.param.TypeConverters`, which converts the values of the params of PySpark.
static TYPE_CONVERTERS: pyo3::sync::PyOnceLock<Py<PyType>> = pyo3::sync::PyOnceLock::new();

/// The value of a param, as PySpark's `Params._set` converts it.
enum Converted {
    /// None, which leaves the param unset, so that the server uses its default. PySpark sends a
    /// null, which the server rejects (a NullPointerException for a number).
    Clear,
    Int(i64),
    /// An int beyond a long, which PySpark's `LiteralExpression` cannot send.
    Unsendable,
    Float(f64),
    Str(String),
    Bool(bool),
    StrList(Vec<String>),
    FloatList(Vec<f64>),
}

/// Convert `value` of the param `name` with its `TypeConverters` method, raising PySpark's
/// TypeError of `Params._set` for a value that it cannot convert.
fn convert(name: &str, kind: Kind, value: &Bound<'_, PyAny>) -> PyResult<Converted> {
    if value.is_none() {
        return Ok(Converted::Clear);
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
        Kind::Int => converted
            .extract::<i64>()
            .map_or(Converted::Unsendable, Converted::Int),
        Kind::Float => Converted::Float(converted.extract()?),
        Kind::Str => Converted::Str(converted.extract()?),
        Kind::Bool => Converted::Bool(converted.extract()?),
        Kind::StrList => Converted::StrList(converted.extract()?),
        Kind::FloatList => Converted::FloatList(converted.extract()?),
    })
}

/// A wrapper of a core ML object, whose params Python sets.
trait Native {
    type Core: Clone;
    fn inner(&self) -> &Self::Core;
    fn params_mut(&mut self) -> &mut Params;
    /// The params whose value PySpark cannot send: a fit, transform or evaluate raises PySpark's
    /// error of its serialization of the params, before any request.
    fn unsendable(&self) -> &[&'static str];
    fn unsendable_mut(&mut self) -> &mut Vec<&'static str>;
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
    let params = std::mem::take(target.params_mut());
    let params = match value {
        Converted::Clear => params.clear_param(name),
        Converted::Unsendable => {
            target.unsendable_mut().push(name);
            params.clear_param(name)
        }
        Converted::Int(value) => params.set_param_int_narrow(name, value),
        Converted::Float(value) => params.set_param_double(name, value),
        Converted::Str(value) => params.set_param_string(name, &value),
        Converted::Bool(value) => params.set_param_bool(name, value),
        Converted::StrList(values) => params.set_param_string_array(name, values),
        Converted::FloatList(values) => params.set_param_double_array(name, values),
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

/// Set the params that a constructor of `class` is given, as PySpark's `__init__` binds and
/// `_set`s them. `positional` names the params taken by position: none for a keyword-only
/// constructor, which raises the TypeError of PySpark's `@keyword_only` for a positional
/// argument. `params` are the params that it takes.
fn init_params<T: Native>(
    target: &mut T,
    class: &str,
    positional: &[&'static str],
    params: &[(&'static str, Kind)],
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    if positional.is_empty() && !args.is_empty() {
        return Err(PyTypeError::new_err(
            "Method __init__ forces keyword arguments.",
        ));
    }
    if args.len() > positional.len() {
        return Err(PyTypeError::new_err(format!(
            "{class}.__init__() takes from 1 to {} positional arguments but {} were given",
            positional.len() + 1,
            args.len() + 1
        )));
    }
    let mut given: Vec<(&'static str, Bound<'_, PyAny>)> =
        positional.iter().copied().zip(args.iter()).collect();
    if let Some(kwargs) = kwargs {
        for (key, value) in kwargs.iter() {
            let key = key.extract::<String>()?;
            let Some(&(name, _)) = params.iter().find(|(name, _)| *name == key) else {
                return Err(PyTypeError::new_err(format!(
                    "{class}.__init__() got an unexpected keyword argument '{key}'"
                )));
            };
            if given.iter().any(|(set, _)| *set == name) {
                return Err(PyTypeError::new_err(format!(
                    "{class}.__init__() got multiple values for argument '{name}'"
                )));
            }
            given.push((name, value));
        }
    }
    for (name, value) in given {
        let kind = params
            .iter()
            .find(|(param, _)| *param == name)
            .map(|(_, kind)| *kind)
            .expect("a positional param is a param");
        let converted = convert(name, kind, &value)?;
        apply(target, name, converted);
    }
    Ok(())
}

/// The core object of `slf`, cloned under a short borrow, so that a setter that another
/// thread calls while a fit, transform or evaluate waits for the server does not fail. As
/// PySpark's serialization of the params, it first raises `VALUE_NOT_BETWEEN` of an int
/// that PySpark cannot send.
fn snapshot<T: Native + PyClass>(slf: &Bound<'_, T>) -> PyResult<T::Core> {
    let this = slf.try_borrow()?;
    if !this.unsendable().is_empty() {
        return Err(crate::functions::value_not_between(slf.py()));
    }
    Ok(this.inner().clone())
}

fn fit_native<T>(slf: &Bound<'_, T>, df: &PyDataFrame) -> PyResult<PyMLModel>
where
    T: Native + PyClass,
    T::Core: Estimator,
{
    let mut estimator = snapshot(slf)?;
    let session = df.dataframe.spark_session();
    let model = interruptible(slf.py(), &session, || {
        estimator.fit(&df.dataframe).map_err(Box::new)
    })?;
    Ok(PyMLModel {
        model,
        session_id: session.session_id().to_string(),
    })
}

fn evaluate_native<T>(slf: &Bound<'_, T>, df: &PyDataFrame) -> PyResult<f64>
where
    T: Native + PyClass,
    T::Core: Evaluator,
{
    let evaluator = snapshot(slf)?;
    interruptible(slf.py(), &df.dataframe.spark_session(), || {
        evaluator.evaluate(&df.dataframe).map_err(Box::new)
    })
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
    /// Apply the model to a DataFrame.
    fn transform(&self, py: Python<'_>, df: &PyDataFrame) -> PyResult<PyDataFrame> {
        let mut model = self.model.clone_box();
        let session = df.dataframe.spark_session();
        if session.session_id() != self.session_id {
            // As PySpark, the transform of a DataFrame of another session refers to the model
            // on the server of that session, which raises CONNECT_ML.CACHE_INVALID when the
            // DataFrame is analyzed or run, as it caches no such model. (The Rust API rejects
            // the transform itself.)
            let relation = spark_connect_proto::Relation {
                rel_type: Some(spark_connect_proto::relation::RelType::MlRelation(
                    Box::new(model.build_ml_relation(&df.dataframe).to_pyerr()?),
                )),
                ..Default::default()
            };
            return Ok(PyDataFrame::new(
                DataFrame::from_relation(session, relation, None).to_pyerr()?,
            ));
        }
        let out = py
            .detach(|| model.transform(&df.dataframe).map_err(Box::new))
            .to_pyerr()?;
        Ok(PyDataFrame::new(out))
    }
}

/// StandardScaler estimator.
#[pyclass(name = "StandardScaler", module = "pyspark.ml.connect.feature")]
pub struct PyStandardScaler {
    inner: StandardScaler,
    unsendable: Vec<&'static str>,
}

const COLUMN_PARAMS: &[(&str, Kind)] = &[("inputCol", Kind::Str), ("outputCol", Kind::Str)];

#[pymethods]
impl PyStandardScaler {
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, inputCol=None, outputCol=None)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: StandardScaler::new(),
            unsendable: Vec::new(),
        };
        init_params(
            &mut this,
            "StandardScaler",
            &[],
            COLUMN_PARAMS,
            args,
            kwargs,
        )?;
        Ok(this)
    }
    #[pyo3(name = "setInputCol")]
    fn set_input_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "inputCol", Kind::Str, value)
    }
    #[pyo3(name = "setOutputCol")]
    fn set_output_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "outputCol", Kind::Str, value)
    }
    fn fit(slf: &Bound<'_, Self>, df: &PyDataFrame) -> PyResult<PyMLModel> {
        fit_native(slf, df)
    }
}

/// MaxAbsScaler estimator.
#[pyclass(name = "MaxAbsScaler", module = "pyspark.ml.connect.feature")]
pub struct PyMaxAbsScaler {
    inner: MaxAbsScaler,
    unsendable: Vec<&'static str>,
}

#[pymethods]
impl PyMaxAbsScaler {
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, inputCol=None, outputCol=None)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: MaxAbsScaler::new(),
            unsendable: Vec::new(),
        };
        init_params(&mut this, "MaxAbsScaler", &[], COLUMN_PARAMS, args, kwargs)?;
        Ok(this)
    }
    #[pyo3(name = "setInputCol")]
    fn set_input_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "inputCol", Kind::Str, value)
    }
    #[pyo3(name = "setOutputCol")]
    fn set_output_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "outputCol", Kind::Str, value)
    }
    fn fit(slf: &Bound<'_, Self>, df: &PyDataFrame) -> PyResult<PyMLModel> {
        fit_native(slf, df)
    }
}

/// StringIndexer estimator.
#[pyclass(name = "StringIndexer", module = "pyspark.ml.connect.feature")]
pub struct PyStringIndexer {
    inner: StringIndexer,
    unsendable: Vec<&'static str>,
}

#[pymethods]
impl PyStringIndexer {
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, inputCol=None, outputCol=None)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: StringIndexer::new(),
            unsendable: Vec::new(),
        };
        init_params(&mut this, "StringIndexer", &[], COLUMN_PARAMS, args, kwargs)?;
        Ok(this)
    }
    #[pyo3(name = "setInputCol")]
    fn set_input_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "inputCol", Kind::Str, value)
    }
    #[pyo3(name = "setOutputCol")]
    fn set_output_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "outputCol", Kind::Str, value)
    }
    fn fit(slf: &Bound<'_, Self>, df: &PyDataFrame) -> PyResult<PyMLModel> {
        fit_native(slf, df)
    }
}

/// VectorAssembler transformer.
#[pyclass(name = "VectorAssembler", module = "pyspark.ml.connect.feature")]
pub struct PyVectorAssembler {
    inner: VectorAssembler,
    unsendable: Vec<&'static str>,
}

#[pymethods]
impl PyVectorAssembler {
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, inputCols=None, outputCol=None)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: VectorAssembler::new(),
            unsendable: Vec::new(),
        };
        let params = &[("inputCols", Kind::StrList), ("outputCol", Kind::Str)];
        init_params(&mut this, "VectorAssembler", &[], params, args, kwargs)?;
        Ok(this)
    }
    #[pyo3(name = "setInputCols")]
    fn set_input_cols<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "inputCols", Kind::StrList, value)
    }
    #[pyo3(name = "setOutputCol")]
    fn set_output_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "outputCol", Kind::Str, value)
    }
    fn transform(slf: &Bound<'_, Self>, df: &PyDataFrame) -> PyResult<PyDataFrame> {
        let mut transformer = snapshot(slf)?;
        let out = slf
            .py()
            .detach(|| transformer.transform(&df.dataframe).map_err(Box::new))
            .to_pyerr()?;
        Ok(PyDataFrame::new(out))
    }
}

/// LogisticRegression estimator.
#[pyclass(
    name = "LogisticRegression",
    module = "pyspark.ml.connect.classification"
)]
pub struct PyLogisticRegression {
    inner: LogisticRegression,
    unsendable: Vec<&'static str>,
}

#[pymethods]
impl PyLogisticRegression {
    #[new]
    #[pyo3(
        signature = (*args, **kwargs),
        text_signature = "(*, featuresCol=None, labelCol=None, predictionCol=None, maxIter=None)"
    )]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: LogisticRegression::new(),
            unsendable: Vec::new(),
        };
        let params = &[
            ("featuresCol", Kind::Str),
            ("labelCol", Kind::Str),
            ("predictionCol", Kind::Str),
            ("maxIter", Kind::Int),
        ];
        init_params(&mut this, "LogisticRegression", &[], params, args, kwargs)?;
        Ok(this)
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
    #[pyo3(name = "setMaxIter")]
    fn set_max_iter<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxIter", Kind::Int, value)
    }
    fn fit(slf: &Bound<'_, Self>, df: &PyDataFrame) -> PyResult<PyMLModel> {
        fit_native(slf, df)
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

#[pymethods]
impl PyLinearSVC {
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

    fn fit(slf: &Bound<'_, Self>, df: &PyDataFrame) -> PyResult<PyMLModel> {
        fit_native(slf, df)
    }
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

    fn fit(slf: &Bound<'_, Self>, df: &PyDataFrame) -> PyResult<PyMLModel> {
        fit_native(slf, df)
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

    fn fit(slf: &Bound<'_, Self>, df: &PyDataFrame) -> PyResult<PyMLModel> {
        fit_native(slf, df)
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

#[pymethods]
impl PyKMeans {
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

    fn fit(slf: &Bound<'_, Self>, df: &PyDataFrame) -> PyResult<PyMLModel> {
        fit_native(slf, df)
    }
}

/// RegressionEvaluator.
#[pyclass(name = "RegressionEvaluator", module = "pyspark.ml.connect.evaluation")]
pub struct PyRegressionEvaluator {
    inner: RegressionEvaluator,
    unsendable: Vec<&'static str>,
}

#[pymethods]
impl PyRegressionEvaluator {
    #[new]
    #[pyo3(
        signature = (*args, **kwargs),
        text_signature = "(*, labelCol=None, predictionCol=None, metricName=None)"
    )]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: RegressionEvaluator::new(),
            unsendable: Vec::new(),
        };
        let params = &[
            ("labelCol", Kind::Str),
            ("predictionCol", Kind::Str),
            ("metricName", Kind::Str),
        ];
        init_params(&mut this, "RegressionEvaluator", &[], params, args, kwargs)?;
        Ok(this)
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
    #[pyo3(name = "setMetricName")]
    fn set_metric_name<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "metricName", Kind::Str, value)
    }
    fn evaluate(slf: &Bound<'_, Self>, df: &PyDataFrame) -> PyResult<f64> {
        evaluate_native(slf, df)
    }
}

/// BinaryClassificationEvaluator.
#[pyclass(
    name = "BinaryClassificationEvaluator",
    module = "pyspark.ml.connect.evaluation"
)]
pub struct PyBinaryClassificationEvaluator {
    inner: BinaryClassificationEvaluator,
    unsendable: Vec<&'static str>,
}

#[pymethods]
impl PyBinaryClassificationEvaluator {
    #[new]
    #[pyo3(
        signature = (*args, **kwargs),
        text_signature = "(*, labelCol=None, rawPredictionCol=None, metricName=None)"
    )]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: BinaryClassificationEvaluator::new(),
            unsendable: Vec::new(),
        };
        let params = &[
            ("labelCol", Kind::Str),
            ("rawPredictionCol", Kind::Str),
            ("metricName", Kind::Str),
        ];
        init_params(
            &mut this,
            "BinaryClassificationEvaluator",
            &[],
            params,
            args,
            kwargs,
        )?;
        Ok(this)
    }
    #[pyo3(name = "setLabelCol")]
    fn set_label_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "labelCol", Kind::Str, value)
    }
    #[pyo3(name = "setRawPredictionCol")]
    fn set_raw_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "rawPredictionCol", Kind::Str, value)
    }
    #[pyo3(name = "setMetricName")]
    fn set_metric_name<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "metricName", Kind::Str, value)
    }
    fn evaluate(slf: &Bound<'_, Self>, df: &PyDataFrame) -> PyResult<f64> {
        evaluate_native(slf, df)
    }
}

/// Pipeline estimator (chains stages by name).
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
        let params = &[("stages", Kind::StrList)];
        init_params(&mut this, "Pipeline", &[], params, args, kwargs)?;
        Ok(this)
    }
    #[pyo3(name = "setStages")]
    fn set_stages<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "stages", Kind::StrList, value)
    }
    fn fit(slf: &Bound<'_, Self>, df: &PyDataFrame) -> PyResult<PyMLModel> {
        fit_native(slf, df)
    }
}

/// MulticlassClassificationEvaluator.
#[pyclass(
    name = "MulticlassClassificationEvaluator",
    module = "pyspark.ml.connect.evaluation"
)]
pub struct PyMulticlassClassificationEvaluator {
    inner: MulticlassClassificationEvaluator,
    unsendable: Vec<&'static str>,
}

/// The params of MulticlassClassificationEvaluator, in the order of PySpark's constructor,
/// which takes them by position too (it is not `@keyword_only`).
const MULTICLASS_PARAMS: &[(&str, Kind)] = &[
    ("metricName", Kind::Str),
    ("labelCol", Kind::Str),
    ("predictionCol", Kind::Str),
];

#[pymethods]
impl PyMulticlassClassificationEvaluator {
    #[new]
    #[pyo3(
        signature = (*args, **kwargs),
        text_signature = "(metricName=None, labelCol=None, predictionCol=None)"
    )]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: MulticlassClassificationEvaluator::new(),
            unsendable: Vec::new(),
        };
        init_params(
            &mut this,
            "MulticlassClassificationEvaluator",
            &["metricName", "labelCol", "predictionCol"],
            MULTICLASS_PARAMS,
            args,
            kwargs,
        )?;
        Ok(this)
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
    #[pyo3(name = "setMetricName")]
    fn set_metric_name<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "metricName", Kind::Str, value)
    }
    fn evaluate(slf: &Bound<'_, Self>, df: &PyDataFrame) -> PyResult<f64> {
        evaluate_native(slf, df)
    }
}

/// CrossValidator (k-fold tuning).
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
        let params = &[
            ("numFolds", Kind::Int),
            ("parallelism", Kind::Int),
            ("seed", Kind::Int),
        ];
        init_params(&mut this, "CrossValidator", &[], params, args, kwargs)?;
        Ok(this)
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
    fn fit(slf: &Bound<'_, Self>, df: &PyDataFrame) -> PyResult<PyMLModel> {
        fit_native(slf, df)
    }
}

native!(
    PyStandardScaler => StandardScaler,
    PyMaxAbsScaler => MaxAbsScaler,
    PyStringIndexer => StringIndexer,
    PyVectorAssembler => VectorAssembler,
    PyLogisticRegression => LogisticRegression,
    PyLinearSVC => LinearSVC,
    PyLinearRegression => LinearRegression,
    PyRandomForestClassifier => RandomForestClassifier,
    PyKMeans => KMeans,
    PyRegressionEvaluator => RegressionEvaluator,
    PyBinaryClassificationEvaluator => BinaryClassificationEvaluator,
    PyMulticlassClassificationEvaluator => MulticlassClassificationEvaluator,
    PyPipeline => Pipeline,
    PyCrossValidator => CrossValidator,
);
