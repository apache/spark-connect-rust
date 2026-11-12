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
    BinaryClassificationEvaluator, CrossValidator, DecisionTreeClassifier, DecisionTreeRegressor,
    Estimator, Evaluator, GBTClassifier, KMeans, LinearRegression, LinearSVC, LogisticRegression,
    MaxAbsScaler, Model, MulticlassClassificationEvaluator, Params, Pipeline,
    RandomForestClassifier, RandomForestRegressor, RegressionEvaluator, StandardScaler,
    StringIndexer, Transformer, VectorAssembler,
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
    /// None, which PySpark sends as a null literal: the server rejects a null number (a
    /// NullPointerException), and reads a null bootstrap as false.
    Null,
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
    fn params(&self) -> &Params;
    fn unsendable(&self) -> &[&'static str];
    fn unsendable_mut(&mut self) -> &mut Vec<&'static str>;
    /// The name of the param `name` in the JVM class, where it is not the name of PySpark's.
    fn jvm_name(name: &'static str) -> &'static str {
        name
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
    let jvm = T::jvm_name(name);
    let params = std::mem::take(target.params_mut());
    let params = match value {
        Converted::Null => params.set_param_null(jvm),
        Converted::Unsendable => {
            target.unsendable_mut().push(name);
            params.clear_param(jvm)
        }
        Converted::Int(value) => params.set_param_int_narrow(jvm, value),
        Converted::Float(value) => params.set_param_double(jvm, value),
        Converted::Str(value) => params.set_param_string(jvm, &value),
        Converted::Bool(value) => params.set_param_bool(jvm, value),
        Converted::StrList(values) => params.set_param_string_array(jvm, values),
        Converted::FloatList(values) => params.set_param_double_array(jvm, values),
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
    Ok(this.inner().clone())
}

fn fit_native<T>(slf: &Bound<'_, T>, df: &PyDataFrame) -> PyResult<PyMLModel>
where
    T: Native + PyClass,
    T::Core: Estimator,
{
    let estimator = snapshot(slf)?;
    fit_core(slf.py(), estimator, df)
}

/// The fit of a native estimator whose core cannot fit it, as Pipeline and CrossValidator.
/// The core raises NOT_IMPLEMENTED before any request, whatever its params are, so the params
/// are not serialized first, and PySpark's serialization errors of them are not raised.
fn fit_unimplemented<T>(slf: &Bound<'_, T>, df: &PyDataFrame) -> PyResult<PyMLModel>
where
    T: Native + PyClass,
    T::Core: Estimator,
{
    let estimator = slf.try_borrow()?.inner().clone();
    fit_core(slf.py(), estimator, df)
}

fn fit_core<C: Estimator>(
    py: Python<'_>,
    mut estimator: C,
    df: &PyDataFrame,
) -> PyResult<PyMLModel> {
    let session = df.dataframe.spark_session();
    let model = interruptible(py, &session, || {
        estimator.fit(&df.dataframe).map_err(Box::new)
    })?;
    Ok(PyMLModel {
        model,
        session_id: session.session_id().to_string(),
    })
}

/// The input columns of an evaluator, as the JVM evaluator reads them: `doubles`, a prediction
/// that it reads as a double only, and `score`, a probability that it reads as a double or a
/// vector, where PySpark's evaluators read any numeric column, and an array, as a tensor.
struct EvaluatorInputs {
    doubles: Vec<String>,
    score: Option<String>,
}

/// Evaluate `dataset` with the JVM evaluator of `slf`, whose input columns `inputs` gives, as
/// PySpark's evaluators evaluate it (see [`evaluation_dataset`] and [`evaluator_input`]).
fn evaluate_native<T>(
    slf: &Bound<'_, T>,
    dataset: &Bound<'_, PyAny>,
    inputs: impl FnOnce(&T::Core) -> EvaluatorInputs,
) -> PyResult<f64>
where
    T: Native + PyClass,
    T::Core: Evaluator,
{
    let py = slf.py();
    let evaluator = snapshot(slf)?;
    let df = evaluation_dataset(dataset)?;
    let df = evaluator_input(py, df, inputs(&evaluator))?;
    interruptible(py, &df.spark_session(), || {
        evaluator.evaluate(&df).map_err(Box::new)
    })
}

/// The dataset of `evaluate`, as PySpark's evaluators take it: a DataFrame, or a pandas DataFrame,
/// which PySpark evaluates locally, and which is sent here with `createDataFrame` of the active
/// session, as the evaluators of this client run on the server. Another object raises the
/// `AttributeError` of the `select` that PySpark's `aggregate_dataframe` calls on it.
fn evaluation_dataset(dataset: &Bound<'_, PyAny>) -> PyResult<DataFrame> {
    let py = dataset.py();
    if let Ok(df) = dataset.extract::<PyRef<'_, PyDataFrame>>() {
        return Ok(df.dataframe.clone());
    }
    // A pandas DataFrame is of an imported pandas.
    let pandas = py
        .import("sys")?
        .getattr("modules")?
        .get_item("pandas")
        .ok();
    if let Some(pandas) = pandas {
        if dataset.is_instance(&pandas.getattr("DataFrame")?)? {
            let df = py
                .import("pyspark.sql")?
                .getattr("SparkSession")?
                .call_method0("active")?
                .call_method1("createDataFrame", (dataset,))?;
            return Ok(df.extract::<PyRef<'_, PyDataFrame>>()?.dataframe.clone());
        }
    }
    dataset.getattr("select")?;
    Err(PyTypeError::new_err(format!(
        "evaluate() takes a DataFrame or a pandas DataFrame, not '{}'",
        dataset.get_type().name()?
    )))
}

/// `df` as the JVM evaluator reads it, with its columns of `inputs` converted: a prediction
/// (`inputs.doubles`) of another numeric type is cast to a double, and a score (`inputs.score`) of
/// another numeric type too, while an array score is sent as a vector by `array_to_vector`, which
/// the JVM evaluator reads at its element 1, as PySpark's reads the column 1 of an array. The
/// columns are found in the schema of `df` regardless of case, as Spark resolves them; another
/// type is sent as it is, which the JVM evaluator checks.
fn evaluator_input(py: Python<'_>, df: DataFrame, inputs: EvaluatorInputs) -> PyResult<DataFrame> {
    use spark_connect::types::DataType;

    let schema = interruptible(py, &df.spark_session(), || df.schema().map_err(Box::new))?;
    let DataType::Struct { fields } = schema else {
        return Ok(df);
    };
    let mut df = df;
    for field in fields {
        let named = |name: &String| name.eq_ignore_ascii_case(&field.name);
        let score = inputs.score.as_ref().is_some_and(named);
        if !score && !inputs.doubles.iter().any(named) {
            continue;
        }
        // The column of this name exactly, a name with a dot too.
        let column = spark_connect::column::col(&format!("`{}`", field.name.replace('`', "``")));
        let converted = match &field.data_type {
            DataType::Byte
            | DataType::Short
            | DataType::Integer
            | DataType::Long
            | DataType::Float
            | DataType::Decimal { .. } => column.cast(DataType::Double),
            DataType::Array { .. } if score => {
                spark_connect::Column::new(spark_connect::expression::Expression::function(
                    "array_to_vector",
                    vec![column.expression().clone()],
                ))
            }
            _ => continue,
        };
        df = df.with_column(&field.name, converted);
    }
    Ok(df)
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
            // the transform itself.) Its input is the placeholder of the DataFrame, as in the
            // plans of `pyspark.ml`: the core resolves it, counting its levels for the depth
            // bound, where an input put in whole would count as one level.
            let input =
                DataFrame::from_sent_relation(session.clone(), df.dataframe.placeholder_relation());
            let relation = spark_connect_proto::Relation {
                rel_type: Some(spark_connect_proto::relation::RelType::MlRelation(
                    Box::new(model.build_ml_relation(&input).to_pyerr()?),
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

    fn fit(slf: &Bound<'_, Self>, df: &PyDataFrame) -> PyResult<PyMLModel> {
        fit_native(slf, df)
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

    fn fit(slf: &Bound<'_, Self>, df: &PyDataFrame) -> PyResult<PyMLModel> {
        fit_native(slf, df)
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
        text_signature = "(*, metricName='rmse', labelCol='label', predictionCol='prediction')"
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
    /// Evaluate `dataset`, a DataFrame or a pandas DataFrame, as PySpark's evaluator, which reads a
    /// prediction of any numeric type.
    fn evaluate(slf: &Bound<'_, Self>, dataset: &Bound<'_, PyAny>) -> PyResult<f64> {
        evaluate_native(slf, dataset, |evaluator| EvaluatorInputs {
            doubles: vec![evaluator.prediction_col().to_string()],
            score: None,
        })
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

/// The JVM evaluator scores its rawPredictionCol, so the probability column of PySpark's
/// evaluator is sent as that param: a probability vector is read at its element 1, as the
/// positive class of PySpark's evaluator.
impl Native for PyBinaryClassificationEvaluator {
    type Core = BinaryClassificationEvaluator;
    fn inner(&self) -> &BinaryClassificationEvaluator {
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
    fn jvm_name(name: &'static str) -> &'static str {
        if name == "probabilityCol" {
            "rawPredictionCol"
        } else {
            name
        }
    }
}

#[pymethods]
impl PyBinaryClassificationEvaluator {
    #[new]
    #[pyo3(
        signature = (*args, **kwargs),
        text_signature = "(*, metricName='areaUnderROC', labelCol='label', probabilityCol='probability')"
    )]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: BinaryClassificationEvaluator::new(),
            unsendable: Vec::new(),
        };
        // As PySpark's, whose constructor sets the probability column "probability" by default,
        // where the JVM evaluator's default is "rawPrediction".
        apply(
            &mut this,
            "probabilityCol",
            Converted::Str("probability".to_string()),
        );
        let params = &[
            ("labelCol", Kind::Str),
            ("probabilityCol", Kind::Str),
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
    #[pyo3(name = "setProbabilityCol")]
    fn set_probability_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "probabilityCol", Kind::Str, value)
    }
    #[pyo3(name = "setMetricName")]
    fn set_metric_name<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "metricName", Kind::Str, value)
    }
    /// Evaluate `dataset`, a DataFrame or a pandas DataFrame, as PySpark's evaluator, which reads a
    /// probability of any numeric type, or an array of the probabilities of the classes.
    fn evaluate(slf: &Bound<'_, Self>, dataset: &Bound<'_, PyAny>) -> PyResult<f64> {
        evaluate_native(slf, dataset, |evaluator| EvaluatorInputs {
            doubles: Vec::new(),
            score: Some(evaluator.raw_prediction_col().to_string()),
        })
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
        fit_unimplemented(slf, df)
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
        text_signature = "(metricName='accuracy', labelCol='label', predictionCol='prediction')"
    )]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: MulticlassClassificationEvaluator::new(),
            unsendable: Vec::new(),
        };
        // As PySpark's, whose constructor sets the metric "accuracy" by default, where the JVM
        // evaluator's default (that of the Rust API) is "f1".
        apply(
            &mut this,
            "metricName",
            Converted::Str("accuracy".to_string()),
        );
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
    /// Evaluate `dataset`, a DataFrame or a pandas DataFrame, as PySpark's evaluator, which reads a
    /// prediction of any numeric type.
    fn evaluate(slf: &Bound<'_, Self>, dataset: &Bound<'_, PyAny>) -> PyResult<f64> {
        evaluate_native(slf, dataset, |evaluator| EvaluatorInputs {
            doubles: vec![evaluator.prediction_col().to_string()],
            score: None,
        })
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
        fit_unimplemented(slf, df)
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
    PyDecisionTreeClassifier => DecisionTreeClassifier,
    PyGBTClassifier => GBTClassifier,
    PyRandomForestClassifier => RandomForestClassifier,
    PyDecisionTreeRegressor => DecisionTreeRegressor,
    PyRandomForestRegressor => RandomForestRegressor,
    PyKMeans => KMeans,
    PyRegressionEvaluator => RegressionEvaluator,
    PyMulticlassClassificationEvaluator => MulticlassClassificationEvaluator,
    PyPipeline => Pipeline,
    PyCrossValidator => CrossValidator,
);
