// SPDX-License-Identifier: Apache-2.0
//! PySpark's validators of `pyspark.ml.tuning`, natively, as PySpark 4.3.0 has them over Spark
//! Connect: `CrossValidator`, `TrainValidationSplit` and their models, of any estimator and
//! evaluator (a native class, PySpark's own or the user's).
//! - **Fit:** `_fit`, on the client: the splits of the core ([`k_fold`], [`train_validation_split`]),
//!   cached as `_cache_spark_dataset` caches them; the fits and metrics of PySpark's
//!   `_parallelFitTasks` (the estimator's `fitMultiple`) in a pool of `parallelism` threads; the
//!   sub-models of `collectSubModels`; PySpark's numpy metrics; and the fit of the best param map
//!   on the whole dataset, whose model gets the params of the validator (`_copyValues`).
//! - **Copy:** PySpark's `copy` of each.
//! - **Persistence:** the writers and readers of `_ValidatorSharedReadWrite`, in PySpark's Python
//!   format, so that each of PySpark and this client loads what the other saves.
//!
//! Their params are PySpark's Python values, in the orders of PySpark's `_paramMap` and
//! `_defaultParamMap` ([`ParamValues`]), as their metadata writes them.
use super::pipeline_api::{
    extra_or_empty, load_metadata_dict, load_params_instance, metadata_str, writable,
};
use super::*;
use spark_connect::ml::{join_path, k_fold, save_metadata, train_validation_split, Metadata};
use spark_connect::SparkSession;
use spark_connect_core::error::{SparkError, SparkErrorKind};

use crate::interrupt::interruptible_outcome;

/// The classes of the metadata of the validators, PySpark's, of which PySpark loads what this
/// client saves.
const CROSS_VALIDATOR_CLASS: &str = "pyspark.ml.tuning.CrossValidator";
const CROSS_VALIDATOR_MODEL_CLASS: &str = "pyspark.ml.tuning.CrossValidatorModel";
const TRAIN_VALIDATION_SPLIT_CLASS: &str = "pyspark.ml.tuning.TrainValidationSplit";
const TRAIN_VALIDATION_SPLIT_MODEL_CLASS: &str = "pyspark.ml.tuning.TrainValidationSplitModel";

/// The params of the constructor and `setParams` of a CrossValidator, in PySpark's order.
const CROSS_VALIDATOR_PARAMS: &[(&str, Kind)] = &[
    ("estimator", Kind::Identity),
    ("estimatorParamMaps", Kind::Identity),
    ("evaluator", Kind::Identity),
    ("numFolds", Kind::Int),
    ("seed", Kind::Int),
    ("parallelism", Kind::Int),
    ("collectSubModels", Kind::Bool),
    ("foldCol", Kind::Str),
];

/// The params of the constructor and `setParams` of a TrainValidationSplit, in PySpark's order.
const TRAIN_VALIDATION_SPLIT_PARAMS: &[(&str, Kind)] = &[
    ("estimator", Kind::Identity),
    ("estimatorParamMaps", Kind::Identity),
    ("evaluator", Kind::Identity),
    ("trainRatio", Kind::Float),
    ("parallelism", Kind::Int),
    ("collectSubModels", Kind::Bool),
    ("seed", Kind::Int),
];

/// The params that a validator writes apart from the others of its metadata.
const SKIP_PARAMS: &[&str] = &["estimator", "evaluator", "estimatorParamMaps"];

/// PySpark's error of a save of the sub-models of a model of none.
const NO_SUB_MODELS: &str =
    "When persisting tuning models, you can only set persistSubModels to true \
                             if the tuning was done with collectSubModels set to true. To save the \
                             sub-models, try rerunning fitting with collectSubModels set to true.";

/// The params of a validator, as PySpark's `_paramMap` and `_defaultParamMap` of a Python class
/// hold them: the set values, as their converters converted them, in the order of their first
/// setting; the defaults that the object holds itself (of `_copyValues` or of a load); and the
/// order of its defaults, which is the order in which the constructor of the class sets them, or
/// the order of their names (`_resetUid` and the `_copyValues` of `Params.copy` rebuild the maps
/// of `params`, which is of the names).
struct ParamValues {
    set: Vec<(&'static str, Py<PyAny>)>,
    own_defaults: params_api::OwnDefaults,
    sorted_defaults: bool,
}

impl ParamValues {
    fn new() -> Self {
        Self {
            set: Vec::new(),
            own_defaults: Vec::new(),
            sorted_defaults: false,
        }
    }

    fn get(&self, py: Python<'_>, name: &str) -> Option<Py<PyAny>> {
        self.set
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, value)| value.clone_ref(py))
    }

    /// Set `name` to `value`, as a dict sets a key: in its place, if it has it.
    fn put(&mut self, name: &'static str, value: Py<PyAny>) {
        match self.set.iter_mut().find(|(n, _)| *n == name) {
            Some((_, old)) => *old = value,
            None => self.set.push((name, value)),
        }
    }

    fn unset(&mut self, name: &str) {
        self.set.retain(|(n, _)| *n != name);
    }

    /// Put the maps in the order of the names of the params, as `_resetUid` and `Params.copy` do.
    fn sort(&mut self) {
        self.set.sort_by_key(|(name, _)| *name);
        self.sorted_defaults = true;
    }

    fn clone_ref(&self, py: Python<'_>) -> Self {
        Self {
            set: self
                .set
                .iter()
                .map(|(name, value)| (*name, value.clone_ref(py)))
                .collect(),
            own_defaults: params_api::clone_own_defaults(py, &self.own_defaults),
            sorted_defaults: self.sorted_defaults,
        }
    }

    /// The names of the defaults of a class whose constructor sets them in the order `defaults`,
    /// in the order of `_defaultParamMap`.
    fn default_names(&self, defaults: &[&'static str]) -> Vec<&'static str> {
        let mut names = defaults.to_vec();
        if self.sorted_defaults {
            names.sort_unstable();
        }
        names
    }
}

/// The Python value of a converted value, as the converter returned it, which PySpark's
/// `_paramMap` holds.
fn converted_to_py(py: Python<'_>, value: Converted) -> PyResult<Py<PyAny>> {
    Ok(match value {
        Converted::Null => py.None(),
        Converted::Int(value) => value.into_py_any(py)?,
        // An int beyond a long, as the `repr` of the converted value.
        Converted::Unsendable(text) => py
            .import("ast")?
            .getattr("literal_eval")?
            .call1((text,))?
            .unbind(),
        Converted::Float(value) => value.into_py_any(py)?,
        Converted::Str(value) => value.into_py_any(py)?,
        Converted::Bool(value) => value.into_py_any(py)?,
        Converted::StrList(value) => value.into_py_any(py)?,
        Converted::FloatList(value) => value.into_py_any(py)?,
        Converted::IntList(value) => value.into_py_any(py)?,
        Converted::FloatListList(value) => value.into_py_any(py)?,
        Converted::Raw(value) => value,
        Converted::Vector(_) | Converted::Matrix(_) => {
            return Err(PyTypeError::new_err(
                "a validator has no param of a vector or a matrix",
            ))
        }
    })
}

/// What the code of the four classes shares: their params, and the order in which the
/// constructor of each sets its defaults.
trait Validator: params_api::ParamsHolder + params_api::NativeInstance {
    /// The names of the defaults, in the order in which the constructor sets them (the MRO of
    /// PySpark's class: `HasCollectSubModels`, `HasParallelism`, `HasSeed`, then the class's).
    const DEFAULTS: &'static [&'static str];
    /// The name of the class of the metadata of a save.
    const METADATA_CLASS: &'static str;
    fn values(&self) -> &ParamValues;
    fn values_mut(&mut self) -> &mut ParamValues;
}

/// The `ParamsHolder`, `NativeInstance` and `Validator` of a validator class `$ty` of the field
/// `values`, the uid `uid`, and the fields `$field` that a duplicate shares.
macro_rules! validator {
    ($ty:ty, $class:expr, [$($default:literal),*], [$($field:ident),*]) => {
        impl params_api::ParamsHolder for $ty {
            fn set_value(&self, py: Python<'_>, name: &str) -> PyResult<Option<Py<PyAny>>> {
                Ok(self.values.get(py, name))
            }

            fn put(&mut self, py: Python<'_>, name: &'static str, value: Converted) -> PyResult<()> {
                let value = converted_to_py(py, value)?;
                self.values.put(name, value);
                Ok(())
            }

            fn own_default(&self, py: Python<'_>, name: &str) -> Option<Py<PyAny>> {
                params_api::own_default_of(py, &self.values.own_defaults, name)
            }

            fn put_own_default(&mut self, name: &'static str, value: Py<PyAny>) -> bool {
                params_api::put_own_default_in(&mut self.values.own_defaults, name, value);
                true
            }

            fn unset(&mut self, name: &'static str) -> PyResult<()> {
                self.values.unset(name);
                Ok(())
            }
        }

        impl params_api::NativeInstance for $ty {
            fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
                Py::new(
                    py,
                    Self {
                        uid: self.uid.clone(),
                        values: self.values.clone_ref(py),
                        $( $field: self.$field.clone_ref(py), )*
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

        impl Validator for $ty {
            const DEFAULTS: &'static [&'static str] = &[$($default),*];
            const METADATA_CLASS: &'static str = $class;

            fn values(&self) -> &ParamValues {
                &self.values
            }

            fn values_mut(&mut self) -> &mut ParamValues {
                &mut self.values
            }
        }
    };
}

/// PySpark's `CrossValidator`: k-fold cross validation of an estimator over its param maps.
#[pyclass(name = "CrossValidator", module = "pyspark.ml.connect.tuning")]
pub struct PyCrossValidator {
    uid: String,
    values: ParamValues,
}

validator!(
    PyCrossValidator,
    CROSS_VALIDATOR_CLASS,
    [
        "collectSubModels",
        "parallelism",
        "seed",
        "numFolds",
        "foldCol"
    ],
    []
);

/// PySpark's `CrossValidatorModel`: the best model of a cross validation, the average and the
/// standard deviation of the metrics of each param map over the folds, and the sub-models of each
/// fold and param map, as PySpark's attributes of the instance.
#[pyclass(name = "CrossValidatorModel", module = "pyspark.ml.connect.tuning")]
pub struct PyCrossValidatorModel {
    uid: String,
    values: ParamValues,
    #[pyo3(get, set, name = "bestModel")]
    best_model: Py<PyAny>,
    #[pyo3(get, set, name = "avgMetrics")]
    avg_metrics: Py<PyAny>,
    #[pyo3(get, set, name = "subModels")]
    sub_models: Py<PyAny>,
    #[pyo3(get, set, name = "stdMetrics")]
    std_metrics: Py<PyAny>,
}

validator!(
    PyCrossValidatorModel,
    CROSS_VALIDATOR_MODEL_CLASS,
    ["seed", "numFolds", "foldCol"],
    [best_model, avg_metrics, sub_models, std_metrics]
);

/// PySpark's `TrainValidationSplit`: the validation of an estimator over its param maps on one
/// random split of the dataset.
#[pyclass(name = "TrainValidationSplit", module = "pyspark.ml.connect.tuning")]
pub struct PyTrainValidationSplit {
    uid: String,
    values: ParamValues,
}

validator!(
    PyTrainValidationSplit,
    TRAIN_VALIDATION_SPLIT_CLASS,
    ["collectSubModels", "parallelism", "seed", "trainRatio"],
    []
);

/// PySpark's `TrainValidationSplitModel`: the best model of a train validation split, the metric
/// of each param map, and the sub-models of each param map, as PySpark's attributes of the
/// instance.
#[pyclass(
    name = "TrainValidationSplitModel",
    module = "pyspark.ml.connect.tuning"
)]
pub struct PyTrainValidationSplitModel {
    uid: String,
    values: ParamValues,
    #[pyo3(get, set, name = "bestModel")]
    best_model: Py<PyAny>,
    #[pyo3(get, set, name = "validationMetrics")]
    validation_metrics: Py<PyAny>,
    #[pyo3(get, set, name = "subModels")]
    sub_models: Py<PyAny>,
}

validator!(
    PyTrainValidationSplitModel,
    TRAIN_VALIDATION_SPLIT_MODEL_CLASS,
    ["seed", "trainRatio"],
    [best_model, validation_metrics, sub_models]
);

/// `value or []`, of which PySpark's models keep their metrics.
fn or_empty_list(py: Python<'_>, value: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    match value {
        Some(value) if value.is_truthy()? => Ok(value.clone().unbind()),
        _ => Ok(PyList::empty(py).into_any().unbind()),
    }
}

impl PyCrossValidatorModel {
    /// `CrossValidatorModel(bestModel, avgMetrics, subModels, stdMetrics)`: of a new uid.
    fn of(
        py: Python<'_>,
        best_model: Py<PyAny>,
        avg_metrics: Option<&Bound<'_, PyAny>>,
        sub_models: Py<PyAny>,
        std_metrics: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            uid: params_api::random_uid(py, "CrossValidatorModel")?,
            values: ParamValues::new(),
            best_model,
            avg_metrics: or_empty_list(py, avg_metrics)?,
            sub_models,
            std_metrics: or_empty_list(py, std_metrics)?,
        })
    }
}

impl PyTrainValidationSplitModel {
    /// `TrainValidationSplitModel(bestModel, validationMetrics, subModels)`: of a new uid.
    fn of(
        py: Python<'_>,
        best_model: Py<PyAny>,
        validation_metrics: Option<&Bound<'_, PyAny>>,
        sub_models: Py<PyAny>,
    ) -> PyResult<Self> {
        Ok(Self {
            uid: params_api::random_uid(py, "TrainValidationSplitModel")?,
            values: ParamValues::new(),
            best_model,
            validation_metrics: or_empty_list(py, validation_metrics)?,
            sub_models,
        })
    }
}

/// A validator of the uid of the class `class` and of the params `given` of its constructor
/// (`_set(**kwargs)`), as PySpark's `__init__`.
fn of_init_params<'py>(
    py: Python<'py>,
    class: &str,
    params: &[(&'static str, Kind)],
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<(String, ParamValues)> {
    let uid = params_api::random_uid(py, class)?;
    let mut values = ParamValues::new();
    for (name, kind, value) in bind_params("__init__", class, &[], params, args, kwargs)? {
        values.put(name, converted_to_py(py, convert(name, kind, &value)?)?);
    }
    Ok((uid, values))
}

/// Set the params `given` to `setParams` of `slf`, as `_set(**kwargs)`.
fn set_params_of<'py, T: Validator>(
    slf: Bound<'py, T>,
    class: &str,
    params: &[(&'static str, Kind)],
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, T>> {
    for (name, _, value) in bind_params("setParams", class, &[], params, args, kwargs)? {
        params_api::set_by_name(&slf, name, &value)?;
    }
    Ok(slf)
}

/// `self._set(name=value)` of a setter.
fn set_one<'py, T: Validator>(
    slf: Bound<'py, T>,
    name: &str,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, T>> {
    params_api::set_by_name(&slf, name, value)?;
    Ok(slf)
}

/// A copy of the estimator `slf`, as `CrossValidator.copy` and `TrainValidationSplit.copy`:
/// `Params.copy` (of the same uid, of the params of `slf` and of `extra`), with a copy of its
/// estimator and of its evaluator, if set, of `extra` (its param maps stay the same).
fn copy_validator<'py, T: Validator>(
    slf: &Bound<'py, T>,
    extra: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, T>> {
    let py = slf.py();
    let extra = extra_or_empty(py, extra);
    let that = params_api::copy(slf, Some(&extra))?;
    that.try_borrow_mut()?.values_mut().sort();
    for (name, setter) in [("estimator", "setEstimator"), ("evaluator", "setEvaluator")] {
        let value = slf.try_borrow()?.values().get(py, name);
        if let Some(value) = value {
            let copy = value.bind(py).call_method1("copy", (&extra,))?;
            that.as_any().call_method1(setter, (copy,))?;
        }
    }
    Ok(that)
}

/// `from._copyValues(to, extra)`, as PySpark's `Params._copyValues` of a validator `from` and a
/// new object `to` (its model, or a copy of its model): each default of `from` of a param that
/// `to` has is the default of `to`, and each set value of `from` (or the value of `extra`, a param
/// map of `from`) of a param that `to` has is set in `to`, in the order of the names of the params
/// of `from`.
fn copy_values<F: Validator, T: Validator>(
    from: &Bound<'_, F>,
    to: &Bound<'_, T>,
    extra: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let py = from.py();
    let extras = match extra {
        Some(extra) if !extra.is_none() => params_api::extra_values(from.as_any(), extra)?,
        _ => Vec::new(),
    };
    let mut names: Vec<&'static str> = F::DEFAULTS
        .iter()
        .copied()
        .chain(["estimator", "estimatorParamMaps", "evaluator"])
        .collect();
    names.sort_unstable();
    names.dedup();
    for name in names {
        if !to.as_any().call_method1("hasParam", (name,))?.is_truthy()? {
            continue;
        }
        if let Some(default) = params_api::default_of(from, name)? {
            to.try_borrow_mut()?.put_own_default(name, default);
        }
        let value = match extras.iter().find(|(n, _)| *n == name) {
            Some((_, value)) => Some(value.clone().unbind()),
            None => from.try_borrow()?.values().get(py, name),
        };
        if let Some(value) = value {
            params_api::set_by_name(to, name, value.bind(py))?;
        }
    }
    Ok(())
}

/// The names of the params of the validators, of which their own defaults are held.
const VALIDATOR_PARAM_NAMES: &[&str] = &[
    "collectSubModels",
    "estimator",
    "estimatorParamMaps",
    "evaluator",
    "foldCol",
    "numFolds",
    "parallelism",
    "seed",
    "trainRatio",
];

/// The `&'static str` of the name of a param of a validator.
fn static_name(name: &str) -> PyResult<&'static str> {
    VALIDATOR_PARAM_NAMES
        .iter()
        .copied()
        .find(|n| *n == name)
        .ok_or_else(|| PyValueError::new_err(format!("Cannot find param with name {name}.")))
}

/// The DataFrame of `dataset`, of which a validator's split reads `select`: an object without it
/// raises its AttributeError, as the split of PySpark does (of a pandas DataFrame).
fn split_dataframe(dataset: &Bound<'_, PyAny>) -> PyResult<DataFrame> {
    if let Ok(df) = dataset.extract::<PyRef<'_, PyDataFrame>>() {
        return Ok(df.dataframe.clone());
    }
    dataset.getattr("select")?;
    Err(PyTypeError::new_err(format!(
        "a validator splits a Spark DataFrame, not '{}'",
        dataset.get_type().name()?
    )))
}

/// The seed of a split, `F.rand(seed)`: the int of the param, or a random seed of None.
fn split_seed<T: Validator>(slf: &Bound<'_, T>) -> PyResult<Option<i64>> {
    let seed = slf.as_any().call_method1("getOrDefault", ("seed",))?;
    if seed.is_none() {
        return Ok(None);
    }
    Ok(Some(seed.extract()?))
}

/// The Python DataFrame of `df`.
fn py_dataframe<'py>(py: Python<'py>, df: DataFrame) -> PyResult<Bound<'py, PyAny>> {
    Ok(Bound::new(py, PyDataFrame::new(df))?.into_any())
}

/// The result of a split of the core, whose errors of values (an empty fold) are PySpark's
/// ValueErrors.
fn split_result<T>(result: Result<T, Box<SparkError>>) -> PyResult<T> {
    result.or_else(|error| {
        if error.kind == SparkErrorKind::ValueError && error.error_class.is_empty() {
            Err(PyValueError::new_err(error.message.clone()))
        } else {
            Err(error).to_pyerr()
        }
    })
}

/// The folds of `dataset` of the cross validator `slf`, as `CrossValidator._kFold`: of the core's
/// [`k_fold`], as Python DataFrames.
fn cross_validator_folds<'py>(
    slf: &Bound<'py, PyCrossValidator>,
    dataset: &Bound<'py, PyAny>,
) -> PyResult<Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
    let py = slf.py();
    let n_folds = slf.call_method1("getOrDefault", ("numFolds",))?;
    let fold_col = slf.call_method1("getOrDefault", ("foldCol",))?;
    let random = !fold_col.is_truthy()?;
    let seed = if random { split_seed(slf)? } else { None };
    let n_folds: i64 = n_folds.extract()?;
    if random && n_folds == 0 {
        // PySpark's `1.0 / nFolds`.
        return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
            "float division by zero",
        ));
    }
    let fold_col: String = if random {
        String::new()
    } else {
        fold_col.extract()?
    };
    let df = split_dataframe(dataset)?;
    let uid = slf.try_borrow()?.uid.clone();
    let session = df.spark_session();
    let folds = split_result(interruptible_outcome(py, &session, || {
        k_fold(&df, n_folds, &fold_col, seed, &uid).map_err(Box::new)
    })?)?;
    folds
        .into_iter()
        .map(|(train, validation)| Ok((py_dataframe(py, train)?, py_dataframe(py, validation)?)))
        .collect()
}

/// The end of a `with` of the context manager `cm` of the result of its block: `__exit__` of the
/// error of the block, if any, else of none. The context managers here (`_cache_spark_dataset`)
/// do not suppress an error.
fn exit_context<T>(cm: &Bound<'_, PyAny>, result: PyResult<T>) -> PyResult<T> {
    let py = cm.py();
    match result {
        Ok(value) => {
            cm.call_method1("__exit__", (py.None(), py.None(), py.None()))?;
            Ok(value)
        }
        Err(error) => {
            cm.call_method1(
                "__exit__",
                (error.get_type(py), error.value(py), error.traceback(py)),
            )?;
            Err(error)
        }
    }
}

/// Run `body` of `train` and `validation` cached, as `with _cache_spark_dataset(train) as train,
/// _cache_spark_dataset(validation) as validation:`.
fn with_cached<'py, R>(
    train: &Bound<'py, PyAny>,
    validation: &Bound<'py, PyAny>,
    body: impl FnOnce(&Bound<'py, PyAny>, &Bound<'py, PyAny>) -> PyResult<R>,
) -> PyResult<R> {
    let cache = train
        .py()
        .import("pyspark.ml.util")?
        .getattr("_cache_spark_dataset")?;
    let train_cm = cache.call1((train,))?;
    let train = train_cm.call_method0("__enter__")?;
    let result = (|| {
        let validation_cm = cache.call1((validation,))?;
        let validation = validation_cm.call_method0("__enter__")?;
        let result = body(&train, &validation);
        exit_context(&validation_cm, result)
    })();
    exit_context(&train_cm, result)
}

/// PySpark's pool of the fits of a validator, `ThreadPool(processes=min(parallelism,
/// numModels))`. Closed after the fit (PySpark leaves it open), its threads end when their tasks
/// do.
fn thread_pool<'py>(slf: &Bound<'py, PyAny>, num_models: usize) -> PyResult<Bound<'py, PyAny>> {
    let py = slf.py();
    let parallelism = slf.call_method0("getParallelism")?;
    let processes = py
        .import("builtins")?
        .getattr("min")?
        .call1((parallelism, num_models))?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("processes", processes)?;
    py.import("multiprocessing.pool")?
        .getattr("ThreadPool")?
        .call((), Some(&kwargs))
}

/// The fits of `est` of each param map of `epm` on `train`, each evaluated by `eva` on
/// `validation`, in the threads of `pool`, as `_parallelFitTasks` and the `imap_unordered` of the
/// validators: `on_result(index, metric, sub_model)` of each, in the order in which they end.
#[allow(clippy::too_many_arguments)]
fn parallel_fits<'py>(
    pool: &Bound<'py, PyAny>,
    dataset: &Bound<'py, PyAny>,
    est: &Bound<'py, PyAny>,
    train: &Bound<'py, PyAny>,
    eva: &Bound<'py, PyAny>,
    validation: &Bound<'py, PyAny>,
    epm: &Bound<'py, PyAny>,
    collect: &Bound<'py, PyAny>,
    mut on_result: impl FnMut(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>) -> PyResult<()>,
) -> PyResult<()> {
    let py = pool.py();
    let tasks = py
        .import("pyspark.ml.tuning")?
        .getattr("_parallelFitTasks")?
        .call1((est, train, eva, validation, epm, collect))?;
    let target = py
        .import("pyspark.util")?
        .getattr("inheritable_thread_target")?
        .call1((dataset.getattr("sparkSession")?,))?;
    let tasks = py
        .import("builtins")?
        .getattr("map")?
        .call1((target, tasks))?;
    // `lambda f: f()`.
    let call = py
        .import("operator")?
        .getattr("methodcaller")?
        .call1(("__call__",))?;
    for item in pool
        .call_method1("imap_unordered", (call, tasks))?
        .try_iter()?
    {
        let item = item?;
        on_result(item.get_item(0)?, item.get_item(1)?, item.get_item(2)?)?;
    }
    Ok(())
}

/// The index of the best metric of `metrics`, as `np.argmax(metrics)` of an evaluator whose larger
/// metrics are better, else `np.argmin(metrics)`.
fn best_index<'py>(
    eva: &Bound<'py, PyAny>,
    metrics: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = eva.py().import("numpy")?;
    if eva.call_method0("isLargerBetter")?.is_truthy()? {
        np.call_method1("argmax", (metrics,))
    } else {
        np.call_method1("argmin", (metrics,))
    }
}

/// `len(range(n))`, of PySpark's `range(nFolds)`.
fn range_len(n: &Bound<'_, PyAny>) -> PyResult<usize> {
    n.py()
        .import("builtins")?
        .getattr("range")?
        .call1((n,))?
        .len()
}

/// The fit of the cross validator `slf` of `dataset`, as `CrossValidator._fit`.
fn cross_validator_fit(
    slf: &Bound<'_, PyCrossValidator>,
    dataset: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let py = slf.py();
    let est = slf.call_method0("getEstimator")?;
    let epm = slf.call_method0("getEstimatorParamMaps")?;
    let num_models = epm.len()?;
    let eva = slf.call_method0("getEvaluator")?;
    let n_folds = range_len(&slf.call_method1("getOrDefault", ("numFolds",))?)?;
    let metrics_all = PyList::empty(py);
    for _ in 0..n_folds {
        metrics_all.append(PyList::new(py, vec![0.0f64; num_models])?)?;
    }
    let pool = thread_pool(slf.as_any(), num_models)?;
    let result = (|| -> PyResult<Option<Bound<'_, PyList>>> {
        let collect = slf.call_method0("getCollectSubModels")?;
        let sub_models = if collect.is_truthy()? {
            let folds = PyList::empty(py);
            for _ in 0..n_folds {
                folds.append(PyList::new(py, (0..num_models).map(|_| py.None()))?)?;
            }
            Some(folds)
        } else {
            None
        };
        let datasets = cross_validator_folds(slf, dataset)?;
        for i in 0..n_folds {
            let (train, validation) = datasets.get(i).ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err("list index out of range")
            })?;
            with_cached(train, validation, |train, validation| {
                parallel_fits(
                    &pool,
                    dataset,
                    &est,
                    train,
                    &eva,
                    validation,
                    &epm,
                    &collect,
                    |j, metric, sub_model| {
                        metrics_all.get_item(i)?.set_item(&j, metric)?;
                        if let Some(sub_models) = &sub_models {
                            sub_models.get_item(i)?.set_item(&j, sub_model)?;
                        }
                        Ok(())
                    },
                )
            })?;
        }
        Ok(sub_models)
    })();
    pool.call_method0("close")?;
    let sub_models = result?;
    let (metrics, std_metrics): (Bound<'_, PyAny>, Bound<'_, PyAny>) = py
        .import("pyspark.ml.tuning")?
        .getattr("CrossValidator")?
        .call_method1("_gen_avg_and_std_metrics", (&metrics_all,))?
        .extract()?;
    let index = best_index(&eva, &metrics)?;
    let best_model = est.call_method1("fit", (dataset, epm.get_item(index)?))?;
    let sub_models = match sub_models {
        Some(sub_models) => sub_models.into_any().unbind(),
        None => py.None(),
    };
    let model = PyCrossValidatorModel::of(
        py,
        best_model.unbind(),
        Some(&metrics),
        sub_models,
        Some(&std_metrics),
    )?;
    let model = Bound::new(py, model)?;
    copy_values(slf, &model, None)?;
    Ok(model.into_any().unbind())
}

/// The fit of the train validation split `slf` of `dataset`, as `TrainValidationSplit._fit`.
fn train_validation_split_fit(
    slf: &Bound<'_, PyTrainValidationSplit>,
    dataset: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let py = slf.py();
    let est = slf.call_method0("getEstimator")?;
    let epm = slf.call_method0("getEstimatorParamMaps")?;
    let num_models = epm.len()?;
    let eva = slf.call_method0("getEvaluator")?;
    let train_ratio: f64 = slf
        .call_method1("getOrDefault", ("trainRatio",))?
        .extract()?;
    let seed = split_seed(slf)?;
    let df = split_dataframe(dataset)?;
    let uid = slf.try_borrow()?.uid.clone();
    let (train, validation) = train_validation_split(&df, train_ratio, seed, &uid);
    let (train, validation) = (py_dataframe(py, train)?, py_dataframe(py, validation)?);
    let (metrics, sub_models) = with_cached(&train, &validation, |train, validation| {
        let collect = slf.call_method0("getCollectSubModels")?;
        let sub_models = if collect.is_truthy()? {
            Some(PyList::new(py, (0..num_models).map(|_| py.None()))?)
        } else {
            None
        };
        let pool = thread_pool(slf.as_any(), num_models)?;
        let metrics = PyList::new(py, (0..num_models).map(|_| py.None()))?;
        let result = parallel_fits(
            &pool,
            dataset,
            &est,
            train,
            &eva,
            validation,
            &epm,
            &collect,
            |j, metric, sub_model| {
                metrics.as_any().set_item(&j, metric)?;
                if let Some(sub_models) = &sub_models {
                    sub_models.as_any().set_item(&j, sub_model)?;
                }
                Ok(())
            },
        );
        pool.call_method0("close")?;
        result.map(|()| (metrics, sub_models))
    })?;
    let index = best_index(&eva, metrics.as_any())?;
    let best_model = est.call_method1("fit", (dataset, epm.get_item(index)?))?;
    let sub_models = match sub_models {
        Some(sub_models) => sub_models.into_any().unbind(),
        None => py.None(),
    };
    let model = PyTrainValidationSplitModel::of(
        py,
        best_model.unbind(),
        Some(metrics.as_any()),
        sub_models,
    )?;
    let model = Bound::new(py, model)?;
    copy_values(slf, &model, None)?;
    Ok(model.into_any().unbind())
}

/// The transform of `dataset` by the best model of a validator's model `slf`, as
/// `Transformer.transform` (a param map transforms of a copy, which has sub-models) and the
/// `_transform` of the models.
fn transform_best<'py>(
    slf: &Bound<'py, PyAny>,
    best_model: Py<PyAny>,
    dataset: &Bound<'py, PyAny>,
    params: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Some(map) = params.filter(|p| p.cast::<PyDict>().is_ok_and(|d| !d.is_empty())) {
        let copy = slf.call_method1("copy", (map,))?;
        return copy.call_method1("transform", (dataset,));
    }
    check_params(params)?;
    best_model
        .bind(slf.py())
        .call_method1("transform", (dataset,))
}

/// `[model.copy() for model in models]`.
fn copied_models<'py>(models: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let copies = PyList::empty(models.py());
    for model in models.try_iter()? {
        copies.append(model?.call_method0("copy")?)?;
    }
    Ok(copies)
}

/// `list(value)`.
fn list_of<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    value
        .py()
        .import("builtins")?
        .getattr("list")?
        .call1((value,))
}

native_pymethods! {
    PyCrossValidator;
    persistence: python;
    params: [
        "collectSubModels" => (
            "getCollectSubModels",
            get_collect_sub_models,
            param_collect_sub_models
        ),
        "estimator" => ("getEstimator", get_estimator, param_estimator),
        "estimatorParamMaps" => (
            "getEstimatorParamMaps",
            get_estimator_param_maps,
            param_estimator_param_maps
        ),
        "evaluator" => ("getEvaluator", get_evaluator, param_evaluator),
        "foldCol" => ("getFoldCol", get_fold_col, param_fold_col),
        "numFolds" => ("getNumFolds", get_num_folds, param_num_folds),
        "parallelism" => ("getParallelism", get_parallelism, param_parallelism),
        "seed" => ("getSeed", get_seed, param_seed),
    ];

    #[new]
    #[pyo3(
        signature = (*args, **kwargs),
        text_signature = "(*, estimator=None, estimatorParamMaps=None, evaluator=None, \
                          numFolds=3, seed=None, parallelism=1, collectSubModels=False, \
                          foldCol='')"
    )]
    fn new(
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let (uid, values) =
            of_init_params(py, "CrossValidator", CROSS_VALIDATOR_PARAMS, args, kwargs)?;
        Ok(Self { uid, values })
    }

    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, estimator=None, estimatorParamMaps=None, evaluator=None, \
                          numFolds=3, seed=None, parallelism=1, collectSubModels=False, \
                          foldCol='')"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(slf, "CrossValidator", CROSS_VALIDATOR_PARAMS, args, kwargs)
    }

    #[pyo3(name = "setEstimator")]
    fn set_estimator<'py>(slf: Bound<'py, Self>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        set_one(slf, "estimator", value)
    }

    #[pyo3(name = "setEstimatorParamMaps")]
    fn set_estimator_param_maps<'py>(slf: Bound<'py, Self>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        set_one(slf, "estimatorParamMaps", value)
    }

    #[pyo3(name = "setEvaluator")]
    fn set_evaluator<'py>(slf: Bound<'py, Self>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        set_one(slf, "evaluator", value)
    }

    #[pyo3(name = "setNumFolds")]
    fn set_num_folds<'py>(slf: Bound<'py, Self>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        set_one(slf, "numFolds", value)
    }

    #[pyo3(name = "setFoldCol")]
    fn set_fold_col<'py>(slf: Bound<'py, Self>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        set_one(slf, "foldCol", value)
    }

    #[pyo3(name = "setSeed")]
    fn set_seed<'py>(slf: Bound<'py, Self>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        set_one(slf, "seed", value)
    }

    #[pyo3(name = "setParallelism")]
    fn set_parallelism<'py>(slf: Bound<'py, Self>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        set_one(slf, "parallelism", value)
    }

    #[pyo3(name = "setCollectSubModels")]
    fn set_collect_sub_models<'py>(slf: Bound<'py, Self>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        set_one(slf, "collectSubModels", value)
    }

    /// The cross validation of `dataset`, as `Estimator.fit` (a param map fits a copy, a list or
    /// tuple of them fits once each) and `CrossValidator._fit`.
    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<Py<PyAny>>> {
        fit_with(slf, dataset, params, || cross_validator_fit(slf, dataset))
    }

    /// The fits of a copy of the validator, one per param map, as `Estimator.fitMultiple`.
    #[pyo3(name = "fitMultiple")]
    #[allow(non_snake_case)]
    fn fit_multiple<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        paramMaps: &Bound<'py, PyAny>,
    ) -> PyResult<params_api::PyFitMultipleIterator> {
        params_api::fit_multiple(slf.as_any(), dataset, paramMaps)
    }

    /// A copy, as `CrossValidator.copy` (see [`copy_validator`]).
    #[pyo3(name = "copy", signature = (extra=None))]
    fn copy<'py>(slf: &Bound<'py, Self>, extra: Option<&Bound<'py, PyAny>>) -> PyResult<Bound<'py, Self>> {
        copy_validator(slf, extra)
    }

    /// The writer, as `CrossValidator.write` over Spark Connect (`RemoteMLWriter`, which saves
    /// as `CrossValidatorWriter`).
    fn write(slf: &Bound<'_, Self>) -> object::PyMLWriter {
        object::PyMLWriter::new(object::WriteTarget::Python(
            slf.clone().into_any().unbind(),
            save_cross_validator,
        ))
    }

    /// Save to `path`, as `MLWritable.save`: `self.write().save(path)`.
    fn save(slf: &Bound<'_, Self>, path: &str) -> PyResult<()> {
        slf.call_method0("write")?.call_method1("save", (path,))?;
        Ok(())
    }

    /// The reader, as `CrossValidator.read` over Spark Connect (`CrossValidatorReader`).
    #[classmethod]
    fn read(cls: &Bound<'_, pyo3::types::PyType>) -> object::PyMLReader {
        let _ = cls;
        object::PyMLReader::python(load_cross_validator)
    }

    /// Load from `path`, as `MLReadable.load`: `cls.read().load(path)`.
    #[classmethod]
    fn load(cls: &Bound<'_, pyo3::types::PyType>, path: &str) -> PyResult<Py<PyAny>> {
        Ok(cls.call_method0("read")?.call_method1("load", (path,))?.unbind())
    }
}

native_pymethods! {
    PyCrossValidatorModel;
    persistence: python;
    params: [
        "estimator" => ("getEstimator", get_estimator, param_estimator),
        "estimatorParamMaps" => (
            "getEstimatorParamMaps",
            get_estimator_param_maps,
            param_estimator_param_maps
        ),
        "evaluator" => ("getEvaluator", get_evaluator, param_evaluator),
        "foldCol" => ("getFoldCol", get_fold_col, param_fold_col),
        "numFolds" => ("getNumFolds", get_num_folds, param_num_folds),
        "seed" => ("getSeed", get_seed, param_seed),
    ];

    #[new]
    #[pyo3(signature = (bestModel, avgMetrics=None, subModels=None, stdMetrics=None))]
    #[allow(non_snake_case)]
    fn new(
        py: Python<'_>,
        bestModel: Py<PyAny>,
        avgMetrics: Option<&Bound<'_, PyAny>>,
        subModels: Option<Py<PyAny>>,
        stdMetrics: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Self::of(py, bestModel, avgMetrics, subModels.unwrap_or_else(|| py.None()), stdMetrics)
    }

    /// The transform of `dataset` by the best model, as `CrossValidatorModel._transform`.
    #[pyo3(signature = (dataset, params=None))]
    fn transform<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        params: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let best_model = slf.try_borrow()?.best_model.clone_ref(slf.py());
        transform_best(slf.as_any(), best_model, dataset, params)
    }

    /// A copy, as `CrossValidatorModel.copy`: of a new uid, of a copy of the best model of
    /// `extra`, of copies of the metrics and the sub-models (which a model of no sub-models fails,
    /// of PySpark's `assert`), and of the params of the model and of `extra`.
    #[pyo3(name = "copy", signature = (extra=None))]
    fn copy(slf: &Bound<'_, Self>, extra: Option<&Bound<'_, PyAny>>) -> PyResult<Py<Self>> {
        let py = slf.py();
        let extra = extra_or_empty(py, extra);
        let (best_model, avg_metrics, sub_models, std_metrics) = {
            let this = slf.try_borrow()?;
            (
                this.best_model.clone_ref(py),
                this.avg_metrics.clone_ref(py),
                this.sub_models.clone_ref(py),
                this.std_metrics.clone_ref(py),
            )
        };
        let best_model = best_model.bind(py).call_method1("copy", (&extra,))?;
        let avg_metrics = list_of(avg_metrics.bind(py))?;
        if sub_models.is_none(py) {
            return Err(pyo3::exceptions::PyAssertionError::new_err(()));
        }
        let folds = PyList::empty(py);
        for fold in sub_models.bind(py).try_iter()? {
            folds.append(copied_models(&fold?)?)?;
        }
        let std_metrics = list_of(std_metrics.bind(py))?;
        let model = Self::of(
            py,
            best_model.unbind(),
            Some(&avg_metrics),
            folds.into_any().unbind(),
            Some(&std_metrics),
        )?;
        let model = Bound::new(py, model)?;
        copy_values(slf, &model, Some(&extra))?;
        Ok(model.unbind())
    }

    /// The writer, as `CrossValidatorModel.write` over Spark Connect (`RemoteMLWriter`, which saves
    /// as `CrossValidatorModelWriter`, of its option `persistSubModels`).
    fn write(slf: &Bound<'_, Self>) -> object::PyMLWriter {
        object::PyMLWriter::new(object::WriteTarget::Python(
            slf.clone().into_any().unbind(),
            save_cross_validator_model,
        ))
    }

    fn save(slf: &Bound<'_, Self>, path: &str) -> PyResult<()> {
        slf.call_method0("write")?.call_method1("save", (path,))?;
        Ok(())
    }

    #[classmethod]
    fn read(cls: &Bound<'_, pyo3::types::PyType>) -> object::PyMLReader {
        let _ = cls;
        object::PyMLReader::python(load_cross_validator_model)
    }

    #[classmethod]
    fn load(cls: &Bound<'_, pyo3::types::PyType>, path: &str) -> PyResult<Py<PyAny>> {
        Ok(cls.call_method0("read")?.call_method1("load", (path,))?.unbind())
    }
}

native_pymethods! {
    PyTrainValidationSplit;
    persistence: python;
    params: [
        "collectSubModels" => (
            "getCollectSubModels",
            get_collect_sub_models,
            param_collect_sub_models
        ),
        "estimator" => ("getEstimator", get_estimator, param_estimator),
        "estimatorParamMaps" => (
            "getEstimatorParamMaps",
            get_estimator_param_maps,
            param_estimator_param_maps
        ),
        "evaluator" => ("getEvaluator", get_evaluator, param_evaluator),
        "parallelism" => ("getParallelism", get_parallelism, param_parallelism),
        "seed" => ("getSeed", get_seed, param_seed),
        "trainRatio" => ("getTrainRatio", get_train_ratio, param_train_ratio),
    ];

    #[new]
    #[pyo3(
        signature = (*args, **kwargs),
        text_signature = "(*, estimator=None, estimatorParamMaps=None, evaluator=None, \
                          trainRatio=0.75, parallelism=1, collectSubModels=False, seed=None)"
    )]
    fn new(
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let (uid, values) = of_init_params(
            py,
            "TrainValidationSplit",
            TRAIN_VALIDATION_SPLIT_PARAMS,
            args,
            kwargs,
        )?;
        Ok(Self { uid, values })
    }

    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, estimator=None, estimatorParamMaps=None, evaluator=None, \
                          trainRatio=0.75, parallelism=1, collectSubModels=False, seed=None)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(
            slf,
            "TrainValidationSplit",
            TRAIN_VALIDATION_SPLIT_PARAMS,
            args,
            kwargs,
        )
    }

    #[pyo3(name = "setEstimator")]
    fn set_estimator<'py>(slf: Bound<'py, Self>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        set_one(slf, "estimator", value)
    }

    #[pyo3(name = "setEstimatorParamMaps")]
    fn set_estimator_param_maps<'py>(slf: Bound<'py, Self>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        set_one(slf, "estimatorParamMaps", value)
    }

    #[pyo3(name = "setEvaluator")]
    fn set_evaluator<'py>(slf: Bound<'py, Self>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        set_one(slf, "evaluator", value)
    }

    #[pyo3(name = "setTrainRatio")]
    fn set_train_ratio<'py>(slf: Bound<'py, Self>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        set_one(slf, "trainRatio", value)
    }

    #[pyo3(name = "setSeed")]
    fn set_seed<'py>(slf: Bound<'py, Self>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        set_one(slf, "seed", value)
    }

    #[pyo3(name = "setParallelism")]
    fn set_parallelism<'py>(slf: Bound<'py, Self>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        set_one(slf, "parallelism", value)
    }

    #[pyo3(name = "setCollectSubModels")]
    fn set_collect_sub_models<'py>(slf: Bound<'py, Self>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        set_one(slf, "collectSubModels", value)
    }

    /// The validation of `dataset`, as `Estimator.fit` and `TrainValidationSplit._fit`.
    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<Py<PyAny>>> {
        fit_with(slf, dataset, params, || train_validation_split_fit(slf, dataset))
    }

    #[pyo3(name = "fitMultiple")]
    #[allow(non_snake_case)]
    fn fit_multiple<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        paramMaps: &Bound<'py, PyAny>,
    ) -> PyResult<params_api::PyFitMultipleIterator> {
        params_api::fit_multiple(slf.as_any(), dataset, paramMaps)
    }

    /// A copy, as `TrainValidationSplit.copy` (see [`copy_validator`]).
    #[pyo3(name = "copy", signature = (extra=None))]
    fn copy<'py>(slf: &Bound<'py, Self>, extra: Option<&Bound<'py, PyAny>>) -> PyResult<Bound<'py, Self>> {
        copy_validator(slf, extra)
    }

    /// The writer, as `TrainValidationSplit.write` over Spark Connect (`RemoteMLWriter`, which
    /// saves as `TrainValidationSplitWriter`).
    fn write(slf: &Bound<'_, Self>) -> object::PyMLWriter {
        object::PyMLWriter::new(object::WriteTarget::Python(
            slf.clone().into_any().unbind(),
            save_train_validation_split,
        ))
    }

    fn save(slf: &Bound<'_, Self>, path: &str) -> PyResult<()> {
        slf.call_method0("write")?.call_method1("save", (path,))?;
        Ok(())
    }

    #[classmethod]
    fn read(cls: &Bound<'_, pyo3::types::PyType>) -> object::PyMLReader {
        let _ = cls;
        object::PyMLReader::python(load_train_validation_split)
    }

    #[classmethod]
    fn load(cls: &Bound<'_, pyo3::types::PyType>, path: &str) -> PyResult<Py<PyAny>> {
        Ok(cls.call_method0("read")?.call_method1("load", (path,))?.unbind())
    }
}

native_pymethods! {
    PyTrainValidationSplitModel;
    persistence: python;
    params: [
        "estimator" => ("getEstimator", get_estimator, param_estimator),
        "estimatorParamMaps" => (
            "getEstimatorParamMaps",
            get_estimator_param_maps,
            param_estimator_param_maps
        ),
        "evaluator" => ("getEvaluator", get_evaluator, param_evaluator),
        "seed" => ("getSeed", get_seed, param_seed),
        "trainRatio" => ("getTrainRatio", get_train_ratio, param_train_ratio),
    ];

    #[new]
    #[pyo3(signature = (bestModel, validationMetrics=None, subModels=None))]
    #[allow(non_snake_case)]
    fn new(
        py: Python<'_>,
        bestModel: Py<PyAny>,
        validationMetrics: Option<&Bound<'_, PyAny>>,
        subModels: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        Self::of(py, bestModel, validationMetrics, subModels.unwrap_or_else(|| py.None()))
    }

    /// The transform of `dataset` by the best model, as `TrainValidationSplitModel._transform`.
    #[pyo3(signature = (dataset, params=None))]
    fn transform<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        params: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let best_model = slf.try_borrow()?.best_model.clone_ref(slf.py());
        transform_best(slf.as_any(), best_model, dataset, params)
    }

    /// A copy, as `TrainValidationSplitModel.copy`: of a new uid, of a copy of the best model of
    /// `extra`, of copies of the metrics and the sub-models (which a model of no sub-models fails,
    /// of PySpark's `assert`), and of the params of the model and of `extra`.
    #[pyo3(name = "copy", signature = (extra=None))]
    fn copy(slf: &Bound<'_, Self>, extra: Option<&Bound<'_, PyAny>>) -> PyResult<Py<Self>> {
        let py = slf.py();
        let extra = extra_or_empty(py, extra);
        let (best_model, validation_metrics, sub_models) = {
            let this = slf.try_borrow()?;
            (
                this.best_model.clone_ref(py),
                this.validation_metrics.clone_ref(py),
                this.sub_models.clone_ref(py),
            )
        };
        let best_model = best_model.bind(py).call_method1("copy", (&extra,))?;
        let validation_metrics = list_of(validation_metrics.bind(py))?;
        if sub_models.is_none(py) {
            return Err(pyo3::exceptions::PyAssertionError::new_err(()));
        }
        let sub_models = copied_models(sub_models.bind(py))?;
        let model = Self::of(
            py,
            best_model.unbind(),
            Some(&validation_metrics),
            sub_models.into_any().unbind(),
        )?;
        let model = Bound::new(py, model)?;
        copy_values(slf, &model, Some(&extra))?;
        Ok(model.unbind())
    }

    /// The writer, as `TrainValidationSplitModel.write` over Spark Connect (`RemoteMLWriter`,
    /// which saves as `TrainValidationSplitModelWriter`, of its option `persistSubModels`).
    fn write(slf: &Bound<'_, Self>) -> object::PyMLWriter {
        object::PyMLWriter::new(object::WriteTarget::Python(
            slf.clone().into_any().unbind(),
            save_train_validation_split_model,
        ))
    }

    fn save(slf: &Bound<'_, Self>, path: &str) -> PyResult<()> {
        slf.call_method0("write")?.call_method1("save", (path,))?;
        Ok(())
    }

    #[classmethod]
    fn read(cls: &Bound<'_, pyo3::types::PyType>) -> object::PyMLReader {
        let _ = cls;
        object::PyMLReader::python(load_train_validation_split_model)
    }

    #[classmethod]
    fn load(cls: &Bound<'_, pyo3::types::PyType>, path: &str) -> PyResult<Py<PyAny>> {
        Ok(cls.call_method0("read")?.call_method1("load", (path,))?.unbind())
    }
}

// ---------------------------------------------------------------------------------------------
// Persistence, of `_ValidatorSharedReadWrite` and the writers and readers of `pyspark.ml.tuning`.
// ---------------------------------------------------------------------------------------------

/// The map of the uids of the nested stages of `estimator` to them, as
/// `MetaAlgorithmReadWrite.getUidMap`.
fn uid_map<'py>(estimator: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    estimator
        .py()
        .import("pyspark.ml.util")?
        .getattr("MetaAlgorithmReadWrite")?
        .call_method1("getUidMap", (estimator,))
}

/// Check that the validator `instance` saves, as `_ValidatorSharedReadWrite.validateParams`: its
/// evaluator, its estimator and the estimator's nested stages write (see [`writable`]), and each
/// param of its param maps is a param of one of them.
fn validate_params(instance: &Bound<'_, PyAny>) -> PyResult<()> {
    let estimator = instance.call_method0("getEstimator")?;
    let evaluator = instance.call_method0("getEvaluator")?;
    let uids = uid_map(&estimator)?;
    let mut elems = vec![evaluator];
    for stage in uids.call_method0("values")?.try_iter()? {
        elems.push(stage?);
    }
    for elem in elems {
        if !writable(&elem)? {
            return Err(PyValueError::new_err(format!(
                "Validator write will fail because it contains {} which is not writable.",
                elem.getattr("uid")?.str()?
            )));
        }
    }
    for param_map in instance.call_method0("getEstimatorParamMaps")?.try_iter()? {
        for param in param_map?.try_iter()? {
            let param = param?;
            if !uids.contains(param.getattr("parent")?)? {
                return Err(PyValueError::new_err(format!(
                    "Validator save requires all Params in estimatorParamMaps to apply to its \
                     Estimator, An extraneous Param was found: {}",
                    param.repr()?
                )));
            }
        }
    }
    Ok(())
}

/// `json.dumps(value, separators=(",", ":"))`, as `DefaultParamsWriter` writes the metadata.
fn compact_json(value: &Bound<'_, PyAny>) -> PyResult<String> {
    let py = value.py();
    let kwargs = PyDict::new(py);
    kwargs.set_item("separators", (",", ":"))?;
    py.import("json")?
        .getattr("dumps")?
        .call((value,), Some(&kwargs))?
        .extract()
}

/// The params of `slf` that its metadata writes, as `DefaultParamsWriter.extractJsonParams`
/// (`extractParamMap()` of the params not in `skip`): of each default in the order of
/// `_defaultParamMap`, its set value or the default, and then each set param without a default, in
/// the order of `_paramMap`.
fn json_params<'py, T: Validator>(
    slf: &Bound<'py, T>,
    skip: &[&str],
) -> PyResult<Bound<'py, PyDict>> {
    let py = slf.py();
    let this = slf.try_borrow()?;
    let values = this.values();
    let defaults = values.default_names(T::DEFAULTS);
    drop(this);
    let params = PyDict::new(py);
    for name in &defaults {
        let value = match slf.try_borrow()?.values().get(py, name) {
            Some(value) => Some(value),
            None => params_api::default_of(slf, name)?,
        };
        if let Some(value) = value {
            params.set_item(*name, value)?;
        }
    }
    let set = slf
        .try_borrow()?
        .values()
        .set
        .iter()
        .map(|(name, value)| (*name, value.clone_ref(py)))
        .collect::<Vec<_>>();
    for (name, value) in set {
        if !defaults.contains(&name) {
            params.set_item(name, value)?;
        }
    }
    for name in skip {
        if params.contains(*name)? {
            params.del_item(*name)?;
        }
    }
    Ok(params)
}

/// The defaults of `slf`, as `_defaultParamMap`, in its order.
fn default_params<'py, T: Validator>(slf: &Bound<'py, T>) -> PyResult<Bound<'py, PyDict>> {
    let py = slf.py();
    let names = slf.try_borrow()?.values().default_names(T::DEFAULTS);
    let defaults = PyDict::new(py);
    for name in names {
        if let Some(value) = params_api::default_of(slf, name)? {
            defaults.set_item(name, value)?;
        }
    }
    Ok(defaults)
}

/// Save the validator `instance` to `path`, as `_ValidatorSharedReadWrite.saveImpl`: each param
/// value of its param maps that is a stage (an estimator that is not a meta-estimator, a
/// transformer or an evaluator) saved by its own writer at `epm_<name><n>`, else as its JSON; the
/// metadata of its params and of `extra`; then its evaluator and its estimator.
fn save_impl<T: Validator>(
    instance: &Bound<'_, T>,
    session: &SparkSession,
    path: &str,
    extra: Vec<(&str, Bound<'_, PyAny>)>,
) -> PyResult<()> {
    let py = instance.py();
    let util = py.import("pyspark.ml.util")?;
    let meta_algorithm = util.getattr("MetaAlgorithmReadWrite")?;
    let ml_writable = util.getattr("MLWritable")?;
    let base = py.import("pyspark.ml.base")?;
    let estimator_class = base.getattr("Estimator")?;
    let transformer_class = base.getattr("Transformer")?;
    let evaluator_class = py.import("pyspark.ml.evaluation")?.getattr("Evaluator")?;
    let dumps = py.import("json")?.getattr("dumps")?;
    let mut num_params_not_json = 0;
    let json_param_maps = PyList::empty(py);
    for param_map in instance
        .as_any()
        .call_method0("getEstimatorParamMaps")?
        .try_iter()?
    {
        let json_param_map = PyList::empty(py);
        for item in param_map?.call_method0("items")?.try_iter()? {
            let item = item?;
            let (param, value) = (item.get_item(0)?, item.get_item(1)?);
            let json_param = PyDict::new(py);
            json_param.set_item("parent", param.getattr("parent")?)?;
            json_param.set_item("name", param.getattr("name")?)?;
            let stage = (value.is_instance(&estimator_class)?
                && !meta_algorithm
                    .call_method1("isMetaEstimator", (&value,))?
                    .is_truthy()?)
                || value.is_instance(&transformer_class)?
                || value.is_instance(&evaluator_class)?;
            if stage {
                let relative_path =
                    format!("epm_{}{num_params_not_json}", param.getattr("name")?.str()?);
                let param_path = join_path(path, &relative_path);
                num_params_not_json += 1;
                value.call_method1("save", (param_path,))?;
                json_param.set_item("value", relative_path)?;
                json_param.set_item("isJson", false)?;
            } else if value.is_instance(&ml_writable)? {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "ValidatorSharedReadWrite.saveImpl does not handle parameters of type: \
                     MLWritable that are not Estimator/Evaluator/Transformer, and if parameter is \
                     estimator, it cannot be meta estimator such as Validator or OneVsRest",
                ));
            } else {
                json_param.set_item("value", dumps.call1((&value,))?)?;
                json_param.set_item("isJson", true)?;
            }
            json_param_map.append(json_param)?;
        }
        json_param_maps.append(json_param_map)?;
    }
    let params = json_params(instance, SKIP_PARAMS)?;
    params.set_item("estimatorParamMaps", json_param_maps)?;
    let param_map = compact_json(params.as_any())?;
    let default_param_map = compact_json(default_params(instance)?.as_any())?;
    let extra = extra
        .iter()
        .map(|(key, value)| Ok((key.to_string(), compact_json(value)?)))
        .collect::<PyResult<Vec<_>>>()?;
    let uid: String = instance.as_any().getattr("uid")?.extract()?;
    interruptible(py, session, || {
        let version = session.version()?;
        let mut metadata = Metadata::new(
            T::METADATA_CLASS,
            &uid,
            &version,
            param_map,
            default_param_map,
        );
        metadata.extra = extra;
        save_metadata(session, path, &metadata.to_json()).map_err(Box::new)
    })?;
    instance
        .as_any()
        .call_method0("getEvaluator")?
        .call_method1("save", (join_path(path, "evaluator"),))?;
    instance
        .as_any()
        .call_method0("getEstimator")?
        .call_method1("save", (join_path(path, "estimator"),))?;
    Ok(())
}

/// Whether a model's writer saves the sub-models, as
/// `getValidatorModelWriterPersistSubModelsParam`: of its option `persistSubModels` ("true" or
/// "false" of any case), else of whether the model has sub-models.
fn persist_sub_models(
    instance: &Bound<'_, PyAny>,
    options: &std::collections::HashMap<String, String>,
) -> PyResult<bool> {
    match options.get("persistsubmodels") {
        Some(value) => match value.to_lowercase().as_str() {
            "true" => Ok(true),
            "false" => Ok(false),
            other => Err(PyValueError::new_err(format!(
                "persistSubModels option value {other} is invalid, the possible values are \
                 True, 'True' or False, 'False'"
            ))),
        },
        None => Ok(!instance.getattr("subModels")?.is_none()),
    }
}

/// `model.write().session(spark).save(path)`.
fn save_model_at(model: &Bound<'_, PyAny>, spark: &Bound<'_, PyAny>, path: String) -> PyResult<()> {
    model
        .call_method0("write")?
        .call_method1("session", (spark,))?
        .call_method1("save", (path,))?;
    Ok(())
}

/// Save the cross validator `instance`, as `CrossValidatorWriter.saveImpl`.
fn save_cross_validator(
    instance: &Bound<'_, PyAny>,
    _spark: &Bound<'_, PyAny>,
    session: &SparkSession,
    path: &str,
    _options: &std::collections::HashMap<String, String>,
) -> PyResult<()> {
    validate_params(instance)?;
    save_impl(
        instance.cast::<PyCrossValidator>()?,
        session,
        path,
        Vec::new(),
    )
}

/// Save the train validation split `instance`, as `TrainValidationSplitWriter.saveImpl`.
fn save_train_validation_split(
    instance: &Bound<'_, PyAny>,
    _spark: &Bound<'_, PyAny>,
    session: &SparkSession,
    path: &str,
    _options: &std::collections::HashMap<String, String>,
) -> PyResult<()> {
    validate_params(instance)?;
    save_impl(
        instance.cast::<PyTrainValidationSplit>()?,
        session,
        path,
        Vec::new(),
    )
}

/// Save the model `instance`, as `CrossValidatorModelWriter.saveImpl`: the validator's metadata of
/// `avgMetrics`, `persistSubModels` and the `stdMetrics` of a model that has them; the best model at
/// `bestModel`; and, of `persistSubModels`, each sub-model at `subModels/fold<i>/<j>`.
fn save_cross_validator_model(
    instance: &Bound<'_, PyAny>,
    spark: &Bound<'_, PyAny>,
    session: &SparkSession,
    path: &str,
    options: &std::collections::HashMap<String, String>,
) -> PyResult<()> {
    let py = instance.py();
    validate_params(instance)?;
    let persist = persist_sub_models(instance, options)?;
    let mut extra = vec![
        ("avgMetrics", instance.getattr("avgMetrics")?),
        (
            "persistSubModels",
            persist.into_pyobject(py)?.to_owned().into_any(),
        ),
    ];
    let std_metrics = instance.getattr("stdMetrics")?;
    if std_metrics.is_truthy()? {
        extra.push(("stdMetrics", std_metrics));
    }
    let model = instance.cast::<PyCrossValidatorModel>()?;
    save_impl(model, session, path, extra)?;
    save_model_at(
        &instance.getattr("bestModel")?,
        spark,
        join_path(path, "bestModel"),
    )?;
    if persist {
        let sub_models = instance.getattr("subModels")?;
        if sub_models.is_none() {
            return Err(PyValueError::new_err(NO_SUB_MODELS));
        }
        let sub_models_path = join_path(path, "subModels");
        let n_folds = range_len(&instance.call_method0("getNumFolds")?)?;
        let n_maps = instance.call_method0("getEstimatorParamMaps")?.len()?;
        for split_index in 0..n_folds {
            let split_path = join_path(&sub_models_path, &format!("fold{split_index}"));
            for param_index in 0..n_maps {
                let model = sub_models.get_item(split_index)?.get_item(param_index)?;
                save_model_at(
                    &model,
                    spark,
                    join_path(&split_path, &param_index.to_string()),
                )?;
            }
        }
    }
    Ok(())
}

/// Save the model `instance`, as `TrainValidationSplitModelWriter.saveImpl`: the validator's
/// metadata of `validationMetrics` and `persistSubModels`; the best model at `bestModel`; and, of
/// `persistSubModels`, each sub-model at `subModels/<j>`.
fn save_train_validation_split_model(
    instance: &Bound<'_, PyAny>,
    spark: &Bound<'_, PyAny>,
    session: &SparkSession,
    path: &str,
    options: &std::collections::HashMap<String, String>,
) -> PyResult<()> {
    let py = instance.py();
    validate_params(instance)?;
    let persist = persist_sub_models(instance, options)?;
    let extra = vec![
        ("validationMetrics", instance.getattr("validationMetrics")?),
        (
            "persistSubModels",
            persist.into_pyobject(py)?.to_owned().into_any(),
        ),
    ];
    let model = instance.cast::<PyTrainValidationSplitModel>()?;
    save_impl(model, session, path, extra)?;
    save_model_at(
        &instance.getattr("bestModel")?,
        spark,
        join_path(path, "bestModel"),
    )?;
    if persist {
        let sub_models = instance.getattr("subModels")?;
        if sub_models.is_none() {
            return Err(PyValueError::new_err(NO_SUB_MODELS));
        }
        let sub_models_path = join_path(path, "subModels");
        let n_maps = instance.call_method0("getEstimatorParamMaps")?.len()?;
        for param_index in 0..n_maps {
            let model = sub_models.get_item(param_index)?;
            save_model_at(
                &model,
                spark,
                join_path(&sub_models_path, &param_index.to_string()),
            )?;
        }
    }
    Ok(())
}

/// The estimator, the evaluator and the param maps of the validator saved at `path` of the
/// metadata `metadata`, as `_ValidatorSharedReadWrite.load`: each param of a param map is the
/// param of the name of its stage (of the uid of `parent`), of the value of its JSON, or of the
/// stage saved at its path. A metadata of PySpark's JVM writer has the JSON of each field.
fn load_shared<'py>(
    py: Python<'py>,
    spark: &Bound<'py, PyAny>,
    session: &SparkSession,
    path: &str,
    metadata: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyList>)> {
    let evaluator = load_params_instance(py, spark, session, &join_path(path, "evaluator"))?;
    let estimator = load_params_instance(py, spark, session, &join_path(path, "estimator"))?;
    let uids = uid_map(&estimator)?;
    uids.set_item(evaluator.getattr("uid")?, &evaluator)?;
    let json_param_maps = metadata
        .get_item("paramMap")?
        .get_item("estimatorParamMaps")?;
    let python_writer = metadata
        .get_item("class")?
        .call_method1("startswith", ("pyspark.ml.",))?
        .is_truthy()?;
    let loads = py.import("json")?.getattr("loads")?;
    let param_maps = PyList::empty(py);
    for json_param_map in json_param_maps.try_iter()? {
        let param_map = PyDict::new(py);
        for json_param in json_param_map?.try_iter()? {
            let json_param = json_param?;
            let stage = uids.get_item(json_param.get_item("parent")?)?;
            let param = stage.getattr(
                json_param
                    .get_item("name")?
                    .cast_into::<pyo3::types::PyString>()?,
            )?;
            let field = |key: &str| -> PyResult<Bound<'py, PyAny>> {
                let value = json_param.get_item(key)?;
                if python_writer {
                    Ok(value)
                } else {
                    loads.call1((value,))
                }
            };
            let is_json = !json_param.contains("isJson")? || field("isJson")?.is_truthy()?;
            let value = if is_json {
                loads.call1((json_param.get_item("value")?,))?
            } else {
                let relative_path = metadata_str(&field("value")?)?;
                load_params_instance(py, spark, session, &join_path(path, &relative_path))?
            };
            param_map.set_item(param, value)?;
        }
        param_maps.append(param_map)?;
    }
    Ok((estimator, evaluator, param_maps))
}

/// `VersionUtils.majorMinorVersion` of the version of a metadata.
fn major_minor_version(version: &Bound<'_, PyAny>) -> PyResult<(i64, i64)> {
    version
        .py()
        .import("pyspark.util")?
        .getattr("VersionUtils")?
        .call_method1("majorMinorVersion", (version,))?
        .extract()
}

/// Set the params of the metadata `metadata` in `instance`, as `DefaultParamsReader.getAndSetParams`
/// (of the params not in `skip`), and its defaults, of the converters of the params, as
/// `_setDefault` of a metadata of Spark 2.4 or later, which has them.
fn get_and_set_params<T: Validator>(
    instance: &Bound<'_, T>,
    metadata: &Bound<'_, PyAny>,
    skip: &[&str],
) -> PyResult<()> {
    let py = instance.py();
    let param_map = metadata.get_item("paramMap")?;
    for name in param_map.try_iter()? {
        let name = name?;
        let param = instance.as_any().call_method1("getParam", (&name,))?;
        let key: String = name.extract()?;
        if !skip.contains(&key.as_str()) {
            instance
                .as_any()
                .call_method1("set", (&param, param_map.get_item(&name)?))?;
        }
    }
    let (major, minor) = major_minor_version(&metadata.get_item("sparkVersion")?)?;
    if major > 2 || (major == 2 && minor >= 4) {
        if !metadata.contains("defaultParamMap")? {
            return Err(pyo3::exceptions::PyAssertionError::new_err(
                "Error loading metadata: Expected `defaultParamMap` section not found",
            ));
        }
        let defaults = metadata.get_item("defaultParamMap")?;
        for name in defaults.try_iter()? {
            let name = name?;
            let param = instance.as_any().call_method1("getParam", (&name,))?;
            let mut value = defaults.get_item(&name)?;
            if !value.is_none() {
                value = param
                    .getattr("typeConverter")?
                    .call1((&value,))
                    .map_err(|error| {
                        if error.is_instance_of::<PyTypeError>(py) {
                            PyTypeError::new_err(format!(
                                "Invalid default param value given for param \"{}\". {}",
                                name,
                                error.value(py)
                            ))
                        } else {
                            error
                        }
                    })?;
            }
            let key = static_name(&name.extract::<String>()?)?;
            instance
                .try_borrow_mut()?
                .put_own_default(key, value.unbind());
        }
    }
    Ok(())
}

/// `instance._resetUid(uid)` of a loaded validator: of `str(uid)`, with its maps in the order of
/// the names of the params.
fn reset_uid<T: Validator>(instance: &Bound<'_, T>, uid: &Bound<'_, PyAny>) -> PyResult<()> {
    let uid = uid.str()?.to_string();
    let mut this = instance.try_borrow_mut()?;
    this.set_uid(&uid);
    this.values_mut().sort();
    Ok(())
}

/// The estimator, evaluator and param maps of a loaded validator, set in `instance` as its
/// constructor of them sets them (`estimator=..., estimatorParamMaps=..., evaluator=...`).
fn set_shared<T: Validator>(
    instance: &Bound<'_, T>,
    estimator: &Bound<'_, PyAny>,
    param_maps: &Bound<'_, PyList>,
    evaluator: &Bound<'_, PyAny>,
) -> PyResult<()> {
    params_api::set_by_name(instance, "estimator", estimator)?;
    params_api::set_by_name(instance, "estimatorParamMaps", param_maps.as_any())?;
    params_api::set_by_name(instance, "evaluator", evaluator)?;
    Ok(())
}

/// Load the cross validator at `path`, as `CrossValidatorReader.load`.
fn load_cross_validator(
    py: Python<'_>,
    spark: &Bound<'_, PyAny>,
    session: &SparkSession,
    path: &str,
) -> PyResult<Py<PyAny>> {
    let metadata = load_metadata_dict(py, session, path)?;
    let (estimator, evaluator, param_maps) = load_shared(py, spark, session, path, &metadata)?;
    let (uid, values) = (
        params_api::random_uid(py, "CrossValidator")?,
        ParamValues::new(),
    );
    let cv = Bound::new(py, PyCrossValidator { uid, values })?;
    set_shared(&cv, &estimator, &param_maps, &evaluator)?;
    reset_uid(&cv, &metadata.get_item("uid")?)?;
    get_and_set_params(&cv, &metadata, &["estimatorParamMaps"])?;
    Ok(cv.into_any().unbind())
}

/// Load the train validation split at `path`, as `TrainValidationSplitReader.load`.
fn load_train_validation_split(
    py: Python<'_>,
    spark: &Bound<'_, PyAny>,
    session: &SparkSession,
    path: &str,
) -> PyResult<Py<PyAny>> {
    let metadata = load_metadata_dict(py, session, path)?;
    let (estimator, evaluator, param_maps) = load_shared(py, spark, session, path, &metadata)?;
    let (uid, values) = (
        params_api::random_uid(py, "TrainValidationSplit")?,
        ParamValues::new(),
    );
    let tvs = Bound::new(py, PyTrainValidationSplit { uid, values })?;
    set_shared(&tvs, &estimator, &param_maps, &evaluator)?;
    reset_uid(&tvs, &metadata.get_item("uid")?)?;
    get_and_set_params(&tvs, &metadata, &["estimatorParamMaps"])?;
    Ok(tvs.into_any().unbind())
}

/// Load the model at `path`, as `CrossValidatorModelReader.load`: of its best model and metrics,
/// and of its sub-models where they were saved, of PySpark's list of the folds, which repeats one
/// list of the models (`[[None] * n] * numFolds`), so that each fold has the models of the last.
fn load_cross_validator_model(
    py: Python<'_>,
    spark: &Bound<'_, PyAny>,
    session: &SparkSession,
    path: &str,
) -> PyResult<Py<PyAny>> {
    let metadata = load_metadata_dict(py, session, path)?;
    let (estimator, evaluator, param_maps) = load_shared(py, spark, session, path, &metadata)?;
    let num_folds = metadata.get_item("paramMap")?.get_item("numFolds")?;
    let best_model = load_params_instance(py, spark, session, &join_path(path, "bestModel"))?;
    let avg_metrics = metadata.get_item("avgMetrics")?;
    let std_metrics = if metadata.contains("stdMetrics")? {
        Some(metadata.get_item("stdMetrics")?)
    } else {
        None
    };
    let persist = metadata.contains("persistSubModels")?
        && metadata.get_item("persistSubModels")?.is_truthy()?;
    let sub_models = if persist {
        let n_folds = range_len(&num_folds)?;
        let models = PyList::new(py, (0..param_maps.len()).map(|_| py.None()))?;
        let folds = PyList::new(py, vec![models.clone(); n_folds])?;
        for split_index in 0..n_folds {
            for param_index in 0..param_maps.len() {
                let model_path = join_path(
                    &join_path(&join_path(path, "subModels"), &format!("fold{split_index}")),
                    &param_index.to_string(),
                );
                let model = load_params_instance(py, spark, session, &model_path)?;
                folds.get_item(split_index)?.set_item(param_index, model)?;
            }
        }
        folds.into_any().unbind()
    } else {
        py.None()
    };
    let model = PyCrossValidatorModel::of(
        py,
        best_model.unbind(),
        Some(&avg_metrics),
        sub_models,
        std_metrics.as_ref(),
    )?;
    let model = Bound::new(py, model)?;
    reset_uid(&model, &metadata.get_item("uid")?)?;
    set_shared(&model, &estimator, &param_maps, &evaluator)?;
    get_and_set_params(&model, &metadata, &["estimatorParamMaps"])?;
    Ok(model.into_any().unbind())
}

/// Load the model at `path`, as `TrainValidationSplitModelReader.load`: of its best model and
/// metrics, and of its sub-models where they were saved.
fn load_train_validation_split_model(
    py: Python<'_>,
    spark: &Bound<'_, PyAny>,
    session: &SparkSession,
    path: &str,
) -> PyResult<Py<PyAny>> {
    let metadata = load_metadata_dict(py, session, path)?;
    let (estimator, evaluator, param_maps) = load_shared(py, spark, session, path, &metadata)?;
    let best_model = load_params_instance(py, spark, session, &join_path(path, "bestModel"))?;
    let validation_metrics = metadata.get_item("validationMetrics")?;
    let persist = metadata.contains("persistSubModels")?
        && metadata.get_item("persistSubModels")?.is_truthy()?;
    let sub_models = if persist {
        let models = PyList::new(py, (0..param_maps.len()).map(|_| py.None()))?;
        for param_index in 0..param_maps.len() {
            let model_path = join_path(&join_path(path, "subModels"), &param_index.to_string());
            models.set_item(
                param_index,
                load_params_instance(py, spark, session, &model_path)?,
            )?;
        }
        models.into_any().unbind()
    } else {
        py.None()
    };
    let model = PyTrainValidationSplitModel::of(
        py,
        best_model.unbind(),
        Some(&validation_metrics),
        sub_models,
    )?;
    let model = Bound::new(py, model)?;
    reset_uid(&model, &metadata.get_item("uid")?)?;
    set_shared(&model, &estimator, &param_maps, &evaluator)?;
    get_and_set_params(&model, &metadata, &["estimatorParamMaps"])?;
    Ok(model.into_any().unbind())
}

/// Register the validators and their models.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCrossValidator>()?;
    m.add_class::<PyCrossValidatorModel>()?;
    m.add_class::<PyTrainValidationSplit>()?;
    m.add_class::<PyTrainValidationSplitModel>()?;
    Ok(())
}
