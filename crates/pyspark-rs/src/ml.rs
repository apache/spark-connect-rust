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
//! a DataFrame or a pandas DataFrame, as PySpark's do (see [`Dataset`]).

use pyo3::exceptions::{PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pyclass::boolean_struct::False;
use pyo3::types::{PyDict, PyList, PyTuple, PyType};
use pyo3::PyClass;
use spark_connect::ml::{
    AFTSurvivalRegression, BinaryClassificationEvaluator, BisectingKMeans, Classifier,
    CrossValidator, DecisionTreeClassifier, DecisionTreeRegressor, Estimator, Evaluator,
    FMClassifier, FMRegressor, GBTClassifier, GBTRegressor, GaussianMixture,
    GeneralizedLinearRegression, IsotonicRegression, KMeans, LinearRegression, LinearSVC,
    LogisticRegression, MlOperator, MlVector, Model, MulticlassClassificationEvaluator,
    MultilayerPerceptronClassifier, NaiveBayes, OneVsRest, Params, Pipeline,
    RandomForestClassifier, RandomForestRegressor, RegressionEvaluator, StringIndexer, Transformer,
    VectorAssembler,
};
use spark_connect::types::StructField;
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
    IntList,
    Vector,
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
            Kind::IntList => "toListInt",
            Kind::Vector => "toVector",
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
    IntList(Vec<i64>),
    Vector(MlVector),
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
        Kind::IntList => converted
            .extract::<Vec<i64>>()
            .map_or(Converted::Unsendable, Converted::IntList),
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
        Converted::IntList(values) => params.set_param_int_array(jvm, values),
        Converted::Vector(value) => params.set_param_vector(jvm, value),
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
/// keyed by its Param objects, and the native classes have none; a list or tuple of maps is fitted
/// as one model per map, which they do not do either.
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

/// Check the `params` of a fit, as the `Estimator.fit` of `pyspark.ml` checks it, before the fit.
/// It also takes a list or tuple of param maps, which it fits one model per map of: the native
/// classes cannot, so it raises [`NO_PARAM_MAP`]. Any other value raises PySpark's TypeError.
fn check_fit_params(params: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    let Some(params) = params else {
        return Ok(());
    };
    if overrides_nothing(params)? {
        return Ok(());
    }
    if params.is_instance_of::<PyList>() || params.is_instance_of::<PyTuple>() {
        return Err(PyTypeError::new_err(NO_PARAM_MAP));
    }
    Err(PyTypeError::new_err(format!(
        "Params must be either a param map or a list/tuple of param maps, but got {}.",
        params.get_type()
    )))
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
    Ok(this.inner().clone())
}

/// A dataset of a fit, a transform or an evaluate, as the classes of PySpark's
/// `pyspark.ml.connect` take it: a DataFrame, or a pandas DataFrame, which PySpark's compute on
/// locally, and which is sent here with `createDataFrame` of the active session, as the native
/// classes run on the server.
enum Dataset<'py> {
    Spark(DataFrame),
    Pandas(Bound<'py, PyAny>),
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

/// The dataset of `method`, `dataset`. Another object raises the `AttributeError` of `attribute`,
/// the attribute that PySpark reads of it first: `select` in a fit (by `select` of the columns
/// that it reads) and an evaluate (by `aggregate_dataframe`), `columns` in a transform (by
/// `Transformer._transform`). An object that has it raises a TypeError.
fn dataset_of<'py>(
    dataset: &Bound<'py, PyAny>,
    method: &str,
    attribute: &str,
) -> PyResult<Dataset<'py>> {
    if let Ok(df) = dataset.extract::<PyRef<'_, PyDataFrame>>() {
        return Ok(Dataset::Spark(df.dataframe.clone()));
    }
    if is_pandas(dataset)? {
        return Ok(Dataset::Pandas(dataset.clone()));
    }
    dataset.getattr(attribute)?;
    Err(PyTypeError::new_err(format!(
        "{method}() takes a DataFrame or a pandas DataFrame, not '{}'",
        dataset.get_type().name()?
    )))
}

/// The DataFrame of the pandas DataFrame `pdf`, of `createDataFrame` of the active session. Its
/// columns are sent as Arrow arrays, built without pandas' conversion of NaN to null (which the
/// non-nullable element type of an array rejects), so that a NaN element or value stays NaN, as
/// PySpark computes on it locally. A column that Arrow takes only with that conversion, such as the
/// strings with a NaN, is taken as pandas takes it. Its column names are the `str` of the labels
/// of `pdf`, as createDataFrame names them (a label that is not a string, such as 0, is a name).
fn active_dataframe(pdf: &Bound<'_, PyAny>) -> PyResult<DataFrame> {
    let py = pdf.py();
    let pa = py.import("pyarrow")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("from_pandas", false)?;
    let arrays = PyList::empty(py);
    let names = PyList::empty(py);
    for item in pdf.call_method0("items")?.try_iter()? {
        let (name, column): (Bound<'_, PyAny>, Bound<'_, PyAny>) = item?.extract()?;
        let array = match pa.call_method("array", (&column,), Some(&kwargs)) {
            Ok(array) => array,
            Err(_) => pa.call_method1("array", (column,))?,
        };
        arrays.append(array)?;
        names.append(name.str()?)?;
    }
    let table = pa
        .getattr("Table")?
        .call_method1("from_arrays", (arrays, names))?;
    let df = py
        .import("pyspark.sql")?
        .getattr("SparkSession")?
        .call_method0("active")?
        .call_method1("createDataFrame", (table,))?;
    Ok(df.extract::<PyRef<'_, PyDataFrame>>()?.dataframe.clone())
}

/// The columns that the fit of a class of PySpark's `pyspark.ml.connect` reads, as the params
/// that name them, of their defaults: LogisticRegression's selects its features and its labels.
type Reads = &'static [(&'static str, Option<&'static str>)];

const LOGISTIC_REGRESSION_READS: Reads = &[
    ("featuresCol", Some("features")),
    ("labelCol", Some("label")),
];

/// The string param `name` of `params` if it is set, else `default`. None when it is set to
/// another value (a null), or is not set and has no default, which the server rejects.
fn string_or(params: &Params, name: &str, default: Option<&str>) -> Option<String> {
    match params.get_param(name) {
        Some(_) => string_param(params, name),
        None => default.map(str::to_string),
    }
}

/// The names of the columns `reads` of `params` (see [`string_or`]), or None when one of them has
/// none.
fn read_columns(params: &Params, reads: Reads) -> Option<Vec<String>> {
    reads
        .iter()
        .map(|&(name, default)| string_or(params, name, default))
        .collect()
}

/// The fit of a native estimator, on `dataset` as its JVM estimator reads it: an array column that
/// the JVM estimator reads as an ML vector is sent as that vector (see [`Vectors`]), and a pandas
/// DataFrame with `createDataFrame` of the active session (see [`Dataset`]). With `reads`, of an
/// estimator of a class of PySpark's `pyspark.ml.connect`, only the columns that PySpark's fit
/// reads are sent: their `select`, or `dataset[columns]` of a pandas DataFrame, which raises the
/// KeyError of pandas for a missing column, as PySpark's `aggregate_dataframe`. So the JVM
/// estimator checks no other column, as PySpark's fit reads no other, and the dataset may have
/// an output column (a `prediction`, say). The other estimators, of the classes that PySpark's
/// `pyspark.ml.connect` does not have, fit the whole dataset, as their JVM estimators read it, and
/// take vectors, arrays of numbers and pandas DataFrames, where classic `pyspark.ml` estimators
/// take only a `VectorUDT` column.
fn fit_native<T>(
    slf: &Bound<'_, T>,
    dataset: &Bound<'_, PyAny>,
    reads: Option<Reads>,
) -> PyResult<PyMLModel>
where
    T: Native + PyClass,
    T::Core: Estimator,
{
    fit_native_checked(slf, dataset, reads, |_, _, _| Ok(()))
}

/// [`fit_native`], which first runs `check` on the DataFrame that the fit sends, of the estimator
/// `estimator`: the checks that PySpark's `_fit` makes of its dataset before it fits.
fn fit_native_checked<T>(
    slf: &Bound<'_, T>,
    dataset: &Bound<'_, PyAny>,
    reads: Option<Reads>,
    check: impl FnOnce(Python<'_>, &T::Core, &DataFrame) -> PyResult<()>,
) -> PyResult<PyMLModel>
where
    T: Native + PyClass,
    T::Core: Estimator,
{
    let py = slf.py();
    let dataset = dataset_of(dataset, "fit", "select")?;
    let mut estimator = snapshot(slf)?;
    let columns = reads.and_then(|reads| read_columns(estimator.params(), reads));
    let (df, select) = match dataset {
        Dataset::Spark(df) => (df, columns),
        Dataset::Pandas(pdf) => {
            let pdf = match &columns {
                Some(columns) => pdf.get_item(PyList::new(py, columns)?)?,
                None => pdf,
            };
            (active_dataframe(&pdf)?, None)
        }
    };
    let vectors = Vectors::of(estimator.operator(), estimator.params());
    // Found in the schema of the whole DataFrame, which is read once for a fit and a transform
    // of it: the selection keeps the columns as they are. The JVM estimator reads its columns by
    // the spelling of the schema (see [`spell_param`]).
    let labelled = estimator.params().get_param("labelCol").is_some();
    let fields = if vectors.input.is_some() || labelled {
        schema_fields(py, &df)?
    } else {
        Vec::new()
    };
    let array = vectors.input_field(&fields).filter(is_number_array);
    let params = vectors.spell_features(std::mem::take(estimator.params_mut()), &fields);
    *estimator.params_mut() = spell_param(params, &fields, "labelCol", None);
    let df = match select {
        Some(columns) => df.select(
            columns
                .iter()
                .map(|name| exact_column(spelling(&fields, name))),
        ),
        None => df,
    };
    check(py, &estimator, &df)?;
    let df = match array {
        Some(field) => df.with_column(&field.name, vector_of(&field)),
        None => df,
    };
    fit_core(py, estimator, &df)
}

/// The checks of the labels of a LogisticRegression fit, as PySpark's `_fit` makes them of the
/// labels that its query collects: `collect_set` of the label column, which has the distinct
/// labels and no null. There must be at least two, each an integer in `[0, numClasses)`, with
/// PySpark's ValueErrors. PySpark's query also counts the rows and reads the size of the features,
/// which these checks do not read, and array_size raises on the vector features that the native
/// LogisticRegression takes, so that query is not sent.
fn check_labels(py: Python<'_>, df: &DataFrame, label: &str) -> PyResult<()> {
    use spark_connect::functions::{col, collect_set};
    use spark_connect::row::Value;

    let rows = interruptible(py, &df.spark_session(), || {
        df.agg(vec![collect_set(col(label)).expression().clone()])
            .collect()
            .map_err(Box::new)
    })?;
    let classes = match rows.first().and_then(|row| row.get(0)) {
        Some(Value::List(classes)) => classes.as_slice(),
        _ => &[],
    };
    if classes.len() < 2 {
        return Err(PyValueError::new_err(
            "Training dataset distinct labels must >= 2.",
        ));
    }
    // A label is in `range(0, numClasses)` as PySpark's `in` finds it: an integer, or a number
    // that equals one (a bool is 0 or 1).
    let integer = |value: i64| value >= 0 && value.unsigned_abs() < classes.len() as u64;
    let number = |value: f64| value.fract() == 0.0 && value >= 0.0 && value < classes.len() as f64;
    let in_range = classes.iter().all(|class| match class {
        Value::Byte(v) => integer(i64::from(*v)),
        Value::Short(v) => integer(i64::from(*v)),
        Value::Integer(v) => integer(i64::from(*v)),
        Value::Long(v) => integer(*v),
        Value::Float(v) => number(f64::from(*v)),
        Value::Double(v) => number(*v),
        Value::Decimal { value, .. } => value.parse::<f64>().is_ok_and(number),
        Value::Bool(v) => integer(i64::from(*v)),
        _ => false,
    });
    if !in_range {
        return Err(PyValueError::new_err(
            "Training labels must be integers in [0, numClasses).",
        ));
    }
    Ok(())
}

/// The fit of a native estimator whose core cannot fit it, as Pipeline and CrossValidator.
/// The core raises NOT_IMPLEMENTED of `feature` before any request, whatever its params are, so
/// the params are not serialized first, and PySpark's serialization errors of them are not
/// raised. So does the fit of a pandas DataFrame, or of another object, which is not sent.
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

/// The columns of a native estimator, and of the model of its fit, that the JVM reads as ML
/// vectors, as their fit and transform send them, so that they take the columns of PySpark's
/// `pyspark.ml.connect` classes, which compute with numpy and torch on arrays: an array of numbers
/// of the `input` column is sent as a vector by `array_to_vector`, and a vector as it is.
#[derive(Default)]
struct Vectors {
    input: Option<String>,
}

impl Vectors {
    /// The vector column of the JVM estimator or model `operator`, of the params `params`, as
    /// Spark's ML classes read it: those of the classification, regression and clustering packages
    /// read the vector of their `featuresCol` (`features` unless set). Of the feature package, the
    /// StringIndexer and the VectorAssembler read no vector.
    fn of(operator: &spark_connect::ml::MlOperator, params: &Params) -> Self {
        let class = operator
            .name
            .strip_suffix("Model")
            .unwrap_or(&operator.name);
        match class.strip_prefix("org.apache.spark.ml.feature.") {
            Some(_) => Self::default(),
            None => Self {
                input: Some(
                    string_param(params, "featuresCol").unwrap_or_else(|| "features".to_string()),
                ),
            },
        }
    }

    /// The field of the `input` column in `fields`, the schema of a DataFrame, found regardless of
    /// case as Spark resolves it, of any type. None when there is no `input`, or no such column.
    fn input_field(&self, fields: &[StructField]) -> Option<StructField> {
        let input = self.input.as_ref()?;
        fields
            .iter()
            .find(|field| field.name.eq_ignore_ascii_case(input))
            .cloned()
    }

    /// `params`, with the featuresCol spelled as the input column of `fields` is (see
    /// [`spell_param`]), where the JVM estimator or model reads a vector of its featuresCol.
    fn spell_features(&self, params: Params, fields: &[StructField]) -> Params {
        match self.input {
            Some(_) => spell_param(params, fields, "featuresCol", Some("features")),
            None => params,
        }
    }
}

/// Whether `field` is an array of numbers, which an input column sends as its ML vector.
fn is_number_array(field: &StructField) -> bool {
    use spark_connect::types::DataType;

    match &field.data_type {
        DataType::Array { element_type, .. } => matches!(
            **element_type,
            DataType::Byte
                | DataType::Short
                | DataType::Integer
                | DataType::Long
                | DataType::Float
                | DataType::Double
                | DataType::Decimal { .. }
        ),
        _ => false,
    }
}

/// `params`, whose string param `name` is spelled as the field of `fields` that it names. Spark
/// resolves a column name regardless of case, but the JVM reads the param as it is given, so the
/// param takes the spelling of the schema where the two differ. An unset param names `default`,
/// and is set only where that differs too. Without a `default`, an unset param is left as it is.
fn spell_param(
    params: Params,
    fields: &[StructField],
    name: &str,
    default: Option<&str>,
) -> Params {
    let Some(given) = string_or(&params, name, default) else {
        return params;
    };
    let spelled = spelling(fields, &given);
    if spelled == given {
        params
    } else {
        params.set_param_string(name, spelled)
    }
}

/// The spelling in `fields`, the schema of a DataFrame, of the column `name`, which Spark resolves
/// regardless of case; `name` itself when `fields` has no such column.
fn spelling<'a>(fields: &'a [StructField], name: &'a str) -> &'a str {
    fields
        .iter()
        .find(|field| field.name.eq_ignore_ascii_case(name))
        .map_or(name, |field| field.name.as_str())
}

/// The fields of the schema of `df`, read lazily, as a transform is, which is lazy as the
/// transforms of PySpark's JVM models are: a schema that the server does not give (of a DataFrame
/// too deep for it, say) is empty, and leaves the columns as they are, as the server raises its
/// error when the transformed DataFrame is analyzed or run, as without the read. A signal raises.
fn lazy_schema_fields(py: Python<'_>, df: &DataFrame) -> PyResult<Vec<StructField>> {
    use spark_connect::types::DataType;

    let session = df.spark_session();
    match crate::interrupt::interruptible_outcome(py, &session, || df.schema().map_err(Box::new))? {
        Ok(DataType::Struct { fields }) => Ok(fields),
        _ => Ok(Vec::new()),
    }
}

/// The ML vector of `field`, an array column of numbers: `array_to_vector` of it, which takes
/// the arrays of the numbers that Spark casts up to doubles. An array of decimals, which it does
/// not take, is cast to an `array<double>` first, as PySpark's MaxAbsScaler computes on decimals.
fn vector_of(field: &spark_connect::types::StructField) -> spark_connect::Column {
    use spark_connect::types::DataType;

    let column = exact_column(&field.name);
    array_to_vector(match &field.data_type {
        DataType::Array { element_type, .. }
            if matches!(**element_type, DataType::Decimal { .. }) =>
        {
            column.cast(DataType::Array {
                element_type: Box::new(DataType::Double),
                contains_null: true,
            })
        }
        _ => column,
    })
}

/// The string param `name` of `params`, if it is set to a string, as the core reads it (a null is
/// not a string).
fn string_param(params: &Params, name: &str) -> Option<String> {
    use spark_connect_proto::expression::{literal::LiteralType, ExprType};

    match params.get_param(name)?.expr_type.as_ref()? {
        ExprType::Literal(literal) => match literal.literal_type.as_ref()? {
            LiteralType::String(value) => Some(value.clone()),
            _ => None,
        },
        _ => None,
    }
}

/// The fields of the schema of `df`, which is read from the server once (and then cached).
fn schema_fields(
    py: Python<'_>,
    df: &DataFrame,
) -> PyResult<Vec<spark_connect::types::StructField>> {
    use spark_connect::types::DataType;

    let schema = interruptible(py, &df.spark_session(), || df.schema().map_err(Box::new))?;
    Ok(match schema {
        DataType::Struct { fields } => fields,
        _ => Vec::new(),
    })
}

/// The column of the name `name` exactly, a name with a dot too.
fn exact_column(name: &str) -> spark_connect::Column {
    spark_connect::column::col(&format!("`{}`", name.replace('`', "``")))
}

/// `array_to_vector` of `column`, the ML vector of an array of numbers.
fn array_to_vector(column: spark_connect::Column) -> spark_connect::Column {
    spark_connect::Column::new(spark_connect::expression::Expression::function(
        "array_to_vector",
        vec![column.expression().clone()],
    ))
}

/// `vector_to_array` of the column `name`, of PySpark's default dtype `float64`: the
/// `array<double>` of an ML vector.
fn vector_to_array(name: &str) -> spark_connect::Column {
    spark_connect::Column::new(spark_connect::expression::Expression::function(
        "vector_to_array",
        vec![
            exact_column(name).expression().clone(),
            spark_connect::column::lit_string("float64")
                .expression()
                .clone(),
        ],
    ))
}

/// The `array<double>` of the ML vector column `name`, of the values of `vector_to_array`, as a
/// `pandas_udf` of PySpark's `transform_dataframe_column` outputs it: a nullable column of
/// nullable elements (`containsNull`), where `vector_to_array` is of neither. Its elements are
/// cast to nullable doubles, and it is the array where the vector is not null, else null.
fn vector_array(name: &str) -> spark_connect::Column {
    use spark_connect::types::DataType;

    spark_connect::column::when(
        exact_column(name).is_not_null(),
        vector_to_array(name).cast(DataType::Array {
            element_type: Box::new(DataType::Double),
            contains_null: true,
        }),
    )
}

/// The input columns of an evaluator, as the JVM evaluator reads them: `doubles`, a prediction
/// that it reads as a double only, and `score`, a probability that it reads as a double or a
/// vector, where PySpark's evaluators read any numeric column, and an array, as a tensor. `label`
/// is the label column, which the JVM evaluator reads as its labelCol.
struct EvaluatorInputs {
    doubles: Vec<String>,
    score: Option<String>,
    label: String,
}

impl EvaluatorInputs {
    /// The columns that PySpark's evaluator reads of its dataset, as its `_get_input_cols` does:
    /// the prediction or the probability, then the label.
    fn reads(&self) -> Vec<String> {
        self.doubles
            .iter()
            .chain(&self.score)
            .cloned()
            .chain([self.label.clone()])
            .collect()
    }
}

/// Evaluate `dataset` with the JVM evaluator of `slf`, whose input columns `inputs` gives, as
/// PySpark's evaluators evaluate it (see [`evaluation_dataset`] and [`evaluator_input`]). The
/// JVM evaluator reads its columns by the spelling of the schema of the dataset.
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
    let mut evaluator = snapshot(slf)?;
    let inputs = inputs(&evaluator);
    let df = evaluation_dataset(dataset, &inputs)?;
    let fields = schema_fields(py, &df)?;
    // An evaluator that exposes no params is sent as it is (see `Evaluator::params_mut`).
    if let Some(evaluator_params) = evaluator.params_mut() {
        let mut params = std::mem::take(evaluator_params);
        params = spell_param(params, &fields, "labelCol", Some("label"));
        if !inputs.doubles.is_empty() {
            params = spell_param(params, &fields, "predictionCol", Some("prediction"));
        }
        if inputs.score.is_some() {
            params = spell_param(params, &fields, "rawPredictionCol", Some("rawPrediction"));
        }
        *evaluator_params = params;
    }
    let df = evaluator_input(df, &fields, inputs);
    interruptible(py, &df.spark_session(), || {
        evaluator.evaluate(&df).map_err(Box::new)
    })
}

/// The dataset of `evaluate`, as PySpark's evaluators take it (see [`Dataset`]): a pandas DataFrame,
/// which PySpark evaluates locally, is sent with `createDataFrame` of the active session, as the
/// evaluators of this client run on the server. Only the columns that PySpark's evaluator reads
/// are sent, as `pdf[columns]` selects them, which raises the KeyError of pandas for a missing one.
fn evaluation_dataset(dataset: &Bound<'_, PyAny>, inputs: &EvaluatorInputs) -> PyResult<DataFrame> {
    match dataset_of(dataset, "evaluate", "select")? {
        Dataset::Spark(df) => Ok(df),
        Dataset::Pandas(pdf) => {
            let columns = PyList::new(pdf.py(), inputs.reads())?;
            active_dataframe(&pdf.get_item(columns)?)
        }
    }
}

/// `df` as the JVM evaluator reads it, whose schema has the fields `fields`, with its columns of
/// `inputs` converted: a prediction (`inputs.doubles`) of another numeric type is cast to a
/// double, and a score (`inputs.score`) of another numeric type too, while an array score is sent
/// as a vector by `array_to_vector`, which the JVM evaluator reads at its element 1, as PySpark's
/// reads the column 1 of an array. The columns are found in `fields` regardless of case, as Spark
/// resolves them; another type is sent as it is, which the JVM evaluator checks.
fn evaluator_input(
    mut df: DataFrame,
    fields: &[StructField],
    inputs: EvaluatorInputs,
) -> DataFrame {
    use spark_connect::types::DataType;

    for field in fields {
        let named = |name: &String| name.eq_ignore_ascii_case(&field.name);
        let score = inputs.score.as_ref().is_some_and(named);
        if !score && !inputs.doubles.iter().any(named) {
            continue;
        }
        let converted = match &field.data_type {
            DataType::Byte
            | DataType::Short
            | DataType::Integer
            | DataType::Long
            | DataType::Float
            | DataType::Decimal { .. } => exact_column(&field.name).cast(DataType::Double),
            DataType::Array { .. } if score => array_to_vector(exact_column(&field.name)),
            _ => continue,
        };
        df = df.with_column(&field.name, converted);
    }
    df
}

/// The model of a class of PySpark's `pyspark.ml.connect`, LogisticRegression, whose transform
/// outputs what PySpark's `Transformer._transform` outputs: the columns of its `_output_columns`,
/// of their types and in their order, appended to the dataset, which may not have any of them
/// already. The other native models, of the classes that PySpark's `pyspark.ml.connect` does not
/// have, output what their JVM models output, which check it.
enum PySparkModel {
    /// LogisticRegressionModel: of its `featuresCol`, and its `bigint` prediction, and its
    /// `array<double>` probability unless its `probabilityCol` is empty (PySpark's `if prob_col`),
    /// of which the JVM model, whose rawPredictionCol is empty (see [`PyLogisticRegression`]),
    /// outputs no raw prediction. (A null column param, which the server rejects, fits no model.)
    LogisticRegression {
        features: Option<String>,
        prediction: Option<String>,
        probability: Option<String>,
    },
}

impl PySparkModel {
    fn of(operator: &MlOperator, params: &Params) -> Option<Self> {
        match operator.name.as_str() {
            "org.apache.spark.ml.classification.LogisticRegressionModel" => {
                // An empty output column is not output, as by PySpark's `if prob_col`.
                let output = |name, default| {
                    string_or(params, name, Some(default)).filter(|name| !name.is_empty())
                };
                Some(Self::LogisticRegression {
                    features: string_or(params, "featuresCol", Some("features")),
                    prediction: output("predictionCol", "prediction"),
                    probability: output("probabilityCol", "probability"),
                })
            }
            _ => None,
        }
    }

    /// The column that PySpark's transform reads (`_input_columns`).
    fn input(&self) -> Option<&str> {
        match self {
            Self::LogisticRegression { features, .. } => features.as_deref(),
        }
    }

    /// The columns that PySpark's transform appends, in their order (`_output_columns`).
    fn outputs(&self) -> Vec<String> {
        match self {
            Self::LogisticRegression {
                prediction,
                probability,
                ..
            } => prediction.iter().chain(probability).cloned().collect(),
        }
    }

    /// The transform of the empty pandas DataFrame `copy` (a shallow copy of it), as PySpark's
    /// transform computes it locally: a LogisticRegression model raises the ValueError of
    /// `np.stack` in PySpark's transform function, which has no array to stack.
    fn transform_empty<'py>(&self, _copy: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Self::LogisticRegression { .. } => {
                Err(PyValueError::new_err("need at least one array to stack"))
            }
        }
    }

    /// PySpark's ValueError of `Transformer._transform` of an output column that `dataset` has
    /// already: one of `list(dataset.columns)`, of their exact names, before any transform.
    fn check_outputs(&self, dataset: &Bound<'_, PyAny>) -> PyResult<()> {
        let existing = dataset
            .py()
            .get_type::<PyList>()
            .call1((dataset.getattr("columns")?,))?;
        for output in self.outputs() {
            if existing.contains(output)? {
                return Err(PyValueError::new_err(
                    "Transformers does not allow output column having the same name with \
                     existing columns.",
                ));
            }
        }
        Ok(())
    }
}

/// The transform of the pandas DataFrame `pdf`, as PySpark's `transform_dataframe_column`
/// returns it: a shallow copy of `pdf` (`copy(deep=False)`) with the output columns inserted
/// after its columns. PySpark's computes them locally, where `transform` computes them on the
/// server, of `pdf` sent with `createDataFrame` of the active session, and they are taken back by
/// `toPandas`, of the rows in their order, with the index of `pdf`. A model of a class of
/// PySpark's (see [`PySparkModel`]) is sent only the column that PySpark's reads, `pdf[input]`,
/// which raises the KeyError of pandas for a missing column, as PySpark's; another transform is
/// sent the whole of `pdf`, and its outputs are the columns that the server appends.
fn transform_pandas<'py>(
    pdf: &Bound<'py, PyAny>,
    pyspark: Option<&PySparkModel>,
    transform: impl FnOnce(&DataFrame) -> PyResult<DataFrame>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = pdf.py();
    let kwargs = PyDict::new(py);
    kwargs.set_item("deep", false)?;
    let copy = pdf.call_method("copy", (), Some(&kwargs))?;
    let sent = match pyspark.and_then(PySparkModel::input) {
        Some(input) => copy.get_item(input)?.call_method0("to_frame")?,
        None => copy.clone(),
    };
    // The server cannot infer the schema of an empty pandas DataFrame, so PySpark's own result of
    // no rows is computed here, as PySpark computes it.
    if let (Some(model), 0) = (pyspark, pdf.len()?) {
        return model.transform_empty(&copy);
    }
    let df = active_dataframe(&sent)?;
    let out = transform(&df)?;
    let outputs = match pyspark {
        Some(model) => model.outputs(),
        None => {
            let sent: Vec<String> = schema_fields(py, &df)?
                .into_iter()
                .map(|field| field.name)
                .collect();
            schema_fields(py, &out)?
                .into_iter()
                .map(|field| field.name)
                .filter(|name| !sent.contains(name))
                .collect()
        }
    };
    let selected = out.select(outputs.iter().map(|name| exact_column(name)));
    let result = PyDataFrame::new(selected)
        .to_pandas(py, None)?
        .into_bound(py);
    result.setattr("index", copy.getattr("index")?)?;
    for output in &outputs {
        let at = copy.getattr("columns")?.len()?;
        copy.call_method1("insert", (at, output, result.get_item(output)?))?;
    }
    Ok(copy)
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
    /// Apply the model to `dataset`, a DataFrame or a pandas DataFrame, as PySpark's
    /// `Transformer.transform` takes it, whose transform of a pandas DataFrame is a pandas
    /// DataFrame (see [`transform_pandas`]). The model of a class of PySpark's raises PySpark's
    /// ValueError of an output column that the dataset has already (see [`PySparkModel`]).
    #[pyo3(signature = (dataset, params=None))]
    fn transform(
        &self,
        py: Python<'_>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        check_params(params)?;
        let pyspark = PySparkModel::of(self.model.operator(), self.model.params());
        let input = dataset_of(dataset, "transform", "columns")?;
        if let Some(model) = &pyspark {
            model.check_outputs(dataset)?;
        }
        let transform = |df: &DataFrame| self.transform_dataframe(py, df, pyspark.as_ref());
        Ok(match input {
            Dataset::Spark(df) => Py::new(py, PyDataFrame::new(transform(&df)?))?.into_any(),
            Dataset::Pandas(pdf) => transform_pandas(&pdf, pyspark.as_ref(), transform)?.unbind(),
        })
    }
}

impl PyMLModel {
    /// The transform of the DataFrame `df`. As PySpark's models of `pyspark.ml.connect`, it takes
    /// an array of numbers where the JVM model reads a vector, and keeps that column as it is in
    /// its output (see [`Vectors`]). A LogisticRegression model outputs PySpark's `bigint`
    /// prediction, and then its `array<double>` probability, of the vector that the JVM model
    /// outputs in a column of its own, before its prediction.
    fn transform_dataframe(
        &self,
        py: Python<'_>,
        df: &DataFrame,
        pyspark: Option<&PySparkModel>,
    ) -> PyResult<DataFrame> {
        use spark_connect::types::DataType;

        let mut model = self.model.clone_box();
        let session = df.spark_session();
        let vectors = Vectors::of(model.operator(), model.params());
        // The JVM model reads its featuresCol by the spelling of the schema of the input.
        let fields = if vectors.input.is_some() {
            lazy_schema_fields(py, df)?
        } else {
            Vec::new()
        };
        let params = std::mem::take(model.params_mut());
        *model.params_mut() = vectors.spell_features(params, &fields);
        // The JVM model appends its probability vector before its prediction: it is output in a
        // column of its own, whose array is appended after the prediction, as PySpark's is.
        let probability = match pyspark {
            Some(PySparkModel::LogisticRegression {
                probability: Some(probability),
                ..
            }) => {
                let vector = format!("{}__probability", model.operator().uid);
                let params = std::mem::take(model.params_mut());
                *model.params_mut() = params.set_param_string("probabilityCol", &vector);
                Some((probability.clone(), vector))
            }
            _ => None,
        };
        // An array input is sent as its vector, which the JVM model reads, and kept as it is in a
        // column of its own, which takes its place again in the output.
        let array = vectors
            .input_field(&fields)
            .filter(is_number_array)
            .map(|field| (format!("{}__input", model.operator().uid), field));
        let input = match &array {
            Some((kept, field)) => df.with_columns(vec![
                (kept.clone(), exact_column(&field.name)),
                (field.name.clone(), vector_of(field)),
            ]),
            None => df.clone(),
        };
        let out = if session.session_id() != self.session_id {
            // As PySpark, the transform of a DataFrame of another session refers to the model
            // on the server of that session, which raises CONNECT_ML.CACHE_INVALID when the
            // DataFrame is analyzed or run, as it caches no such model. (The Rust API rejects
            // the transform itself.) Its input is the placeholder of the DataFrame, as in the
            // plans of `pyspark.ml`: the core resolves it, counting its levels for the depth
            // bound, where an input put in whole would count as one level.
            let placeholder =
                DataFrame::from_sent_relation(session.clone(), input.placeholder_relation());
            let relation = spark_connect_proto::Relation {
                rel_type: Some(spark_connect_proto::relation::RelType::MlRelation(
                    Box::new(model.build_ml_relation(&placeholder).to_pyerr()?),
                )),
                ..Default::default()
            };
            DataFrame::from_relation(session, relation, None).to_pyerr()?
        } else {
            py.detach(|| model.transform(&input).map_err(Box::new))
                .to_pyerr()?
        };
        // The JVM model appends its output columns, and keeps the columns of its input, in place.
        // The columns that take the place of others keep it, and the others are appended in
        // their order, as by PySpark's `withColumns` of its outputs.
        let mut columns = Vec::new();
        let mut dropped = Vec::new();
        if let Some((kept, field)) = &array {
            columns.push((field.name.clone(), exact_column(kept)));
            dropped.push(kept.as_str());
        }
        if let Some(PySparkModel::LogisticRegression {
            prediction: Some(prediction),
            ..
        }) = pyspark
        {
            columns.push((
                prediction.clone(),
                exact_column(prediction).cast(DataType::Long),
            ));
        }
        if let Some((probability, vector)) = &probability {
            columns.push((probability.clone(), vector_array(vector)));
            dropped.push(vector.as_str());
        }
        let out = if columns.is_empty() {
            out
        } else {
            out.with_columns(columns)
        };
        Ok(if dropped.is_empty() {
            out
        } else {
            out.drop(dropped)
        })
    }
}

const COLUMN_PARAMS: &[(&str, Kind)] = &[("inputCol", Kind::Str), ("outputCol", Kind::Str)];

const VECTOR_ASSEMBLER_PARAMS: &[(&str, Kind)] =
    &[("inputCols", Kind::StrList), ("outputCol", Kind::Str)];

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
    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, inputCol=None, outputCol=None)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(slf, "StringIndexer", COLUMN_PARAMS, args, kwargs)
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
    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyMLModel> {
        check_fit_params(params)?;
        fit_native(slf, dataset, None)
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
        init_params(
            &mut this,
            "VectorAssembler",
            &[],
            VECTOR_ASSEMBLER_PARAMS,
            args,
            kwargs,
        )?;
        Ok(this)
    }
    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, inputCols=None, outputCol=None)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(
            slf,
            "VectorAssembler",
            VECTOR_ASSEMBLER_PARAMS,
            args,
            kwargs,
        )
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
    /// Transform `dataset`, a DataFrame or a pandas DataFrame, as PySpark's transformers take it
    /// (see [`transform_pandas`]).
    #[pyo3(signature = (dataset, params=None))]
    fn transform(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        check_params(params)?;
        let py = slf.py();
        let input = dataset_of(dataset, "transform", "columns")?;
        let mut transformer = snapshot(slf)?;
        let mut transform =
            |df: &DataFrame| py.detach(|| transformer.transform(df).map_err(Box::new));
        Ok(match input {
            Dataset::Spark(df) => {
                Py::new(py, PyDataFrame::new(transform(&df).to_pyerr()?))?.into_any()
            }
            Dataset::Pandas(pdf) => {
                transform_pandas(&pdf, None, |df| transform(df).to_pyerr())?.unbind()
            }
        })
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

/// The params of LogisticRegression, in the order of PySpark's constructor, which also takes the
/// params of its PyTorch training, `numTrainWorkers`, `batchSize`, `learningRate`, `momentum` and
/// `seed`. The L-BFGS of the JVM estimator has none of them, so the constructor raises the TypeError
/// of an unexpected keyword argument for them, rather than take a value that no fit would use.
const LOGISTIC_REGRESSION_PARAMS: &[(&str, Kind)] = &[
    ("featuresCol", Kind::Str),
    ("labelCol", Kind::Str),
    ("predictionCol", Kind::Str),
    ("probabilityCol", Kind::Str),
    ("maxIter", Kind::Int),
    ("tol", Kind::Float),
];

#[pymethods]
impl PyLogisticRegression {
    #[new]
    #[pyo3(
        signature = (*args, **kwargs),
        text_signature = "(*, featuresCol='features', labelCol='label', predictionCol='prediction', probabilityCol='probability', maxIter=100, tol=1e-06)"
    )]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: LogisticRegression::new(),
            unsendable: Vec::new(),
        };
        // As PySpark's, whose model outputs no raw prediction: the JVM model outputs none of an
        // empty rawPredictionCol, a param that PySpark's LogisticRegression does not have, so that
        // a dataset with a `rawPrediction` column is fitted and transformed, as with PySpark.
        apply(&mut this, "rawPredictionCol", Converted::Str(String::new()));
        init_params(
            &mut this,
            "LogisticRegression",
            &[],
            LOGISTIC_REGRESSION_PARAMS,
            args,
            kwargs,
        )?;
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
    #[pyo3(name = "setTol")]
    fn set_tol<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "tol", Kind::Float, value)
    }
    /// Fit `dataset`, of which the fit reads the features and the labels, as PySpark's (see
    /// [`fit_native`]). As PySpark's, it fits no pandas DataFrame yet.
    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyMLModel> {
        check_fit_params(params)?;
        if is_pandas(dataset)? {
            return Err(PyNotImplementedError::new_err(
                "Fitting pandas dataframe is not supported yet.",
            ));
        }
        fit_native_checked(
            slf,
            dataset,
            Some(LOGISTIC_REGRESSION_READS),
            |py, estimator, df| check_labels(py, df, estimator.label_col()),
        )
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
    ) -> PyResult<PyMLModel> {
        check_fit_params(params)?;
        fit_native(slf, dataset, None)
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
    ) -> PyResult<PyMLModel> {
        check_fit_params(params)?;
        fit_native(slf, dataset, None)
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
    ) -> PyResult<PyMLModel> {
        check_fit_params(params)?;
        fit_native(slf, dataset, None)
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
    ("parallelism", Kind::Int),
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
        set_param(slf, "parallelism", Kind::Int, value)
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
    ) -> PyResult<PyMLModel> {
        check_fit_params(params)?;
        let classifier = slf
            .try_borrow()?
            .classifier
            .as_ref()
            .map(|x| x.clone_ref(slf.py()))
            .ok_or_else(|| PyValueError::new_err("OneVsRest requires a native classifier"))?;
        let classifier = native_classifier(classifier.bind(slf.py()), true)?;
        {
            let mut this = slf.try_borrow_mut()?;
            if !classifier.supports_weight()
                && this.inner.weight_col().is_some_and(|s| !s.is_empty())
            {
                slf.py().import("warnings")?.call_method1(
                    "warn",
                    ("weightCol is ignored, as it is not supported by this classifier",),
                )?;
            }
            // PySpark's fit trains the binary classifiers in a ThreadPool of `parallelism`
            // processes, which raises this ValueError when there are fewer than one.
            if this.inner.parallelism() < 1 {
                return Err(PyValueError::new_err(
                    "Number of processes must be at least 1",
                ));
            }
            this.inner = this.inner.clone().set_classifier(classifier);
        }
        fit_native(slf, dataset, None)
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
    ) -> PyResult<PyMLModel> {
        check_fit_params(params)?;
        fit_native(slf, dataset, None)
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
    ) -> PyResult<PyMLModel> {
        check_fit_params(params)?;
        fit_native(slf, dataset, None)
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
    ) -> PyResult<PyMLModel> {
        check_fit_params(params)?;
        fit_native(slf, dataset, None)
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
    ) -> PyResult<PyMLModel> {
        check_fit_params(params)?;
        fit_native(slf, dataset, None)
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
    ) -> PyResult<PyMLModel> {
        check_fit_params(params)?;
        fit_native(slf, dataset, None)
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
    ) -> PyResult<PyMLModel> {
        check_fit_params(params)?;
        fit_native(slf, dataset, None)
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
    ) -> PyResult<PyMLModel> {
        check_fit_params(params)?;
        fit_native(slf, dataset, None)
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
    ) -> PyResult<PyMLModel> {
        check_fit_params(params)?;
        fit_native(slf, dataset, None)
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
    ) -> PyResult<PyMLModel> {
        check_fit_params(params)?;
        fit_native(slf, dataset, None)
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
    ) -> PyResult<PyMLModel> {
        check_fit_params(params)?;
        fit_native(slf, dataset, None)
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
    ) -> PyResult<PyMLModel> {
        check_fit_params(params)?;
        fit_native(slf, dataset, None)
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
    ) -> PyResult<PyMLModel> {
        check_fit_params(params)?;
        fit_native(slf, dataset, None)
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

    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyMLModel> {
        check_fit_params(params)?;
        fit_native(slf, dataset, None)
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
    #[pyo3(signature = (dataset, params=None))]
    fn evaluate(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<f64> {
        check_params(params)?;
        evaluate_native(slf, dataset, |evaluator| EvaluatorInputs {
            doubles: vec![evaluator.prediction_col().to_string()],
            score: None,
            label: evaluator.label_col().to_string(),
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
    #[pyo3(signature = (dataset, params=None))]
    fn evaluate(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<f64> {
        check_params(params)?;
        evaluate_native(slf, dataset, |evaluator| EvaluatorInputs {
            doubles: Vec::new(),
            score: Some(evaluator.raw_prediction_col().to_string()),
            label: evaluator.label_col().to_string(),
        })
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
    ) -> PyResult<PyMLModel> {
        check_fit_params(params)?;
        fit_native(slf, dataset, None)
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
    ) -> PyResult<PyMLModel> {
        check_fit_params(params)?;
        fit_native(slf, dataset, None)
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
    ) -> PyResult<PyMLModel> {
        check_fit_params(params)?;
        fit_unimplemented(slf, dataset, "Pipeline.fit")
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
    #[pyo3(signature = (dataset, params=None))]
    fn evaluate(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<f64> {
        check_params(params)?;
        evaluate_native(slf, dataset, |evaluator| EvaluatorInputs {
            doubles: vec![evaluator.prediction_col().to_string()],
            score: None,
            label: evaluator.label_col().to_string(),
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
    ) -> PyResult<PyMLModel> {
        check_fit_params(params)?;
        fit_unimplemented(slf, dataset, "CrossValidator.fit")
    }
}

native!(
    PyStringIndexer => StringIndexer,
    PyVectorAssembler => VectorAssembler,
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
    PyGaussianMixture => GaussianMixture,
    PyKMeans => KMeans,
    PyBisectingKMeans => BisectingKMeans,
    PyRegressionEvaluator => RegressionEvaluator,
    PyMulticlassClassificationEvaluator => MulticlassClassificationEvaluator,
    PyPipeline => Pipeline,
    PyCrossValidator => CrossValidator,
);
