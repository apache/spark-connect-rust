// SPDX-License-Identifier: Apache-2.0
//! PySpark's validators of `pyspark.ml.tuning` in the Rust API, as PySpark 4.3.0 runs them over
//! Spark Connect, on the client: [`CrossValidator`] and [`TrainValidationSplit`] and their models,
//! of any estimator and evaluator, and their persistence of PySpark's Python format
//! (`_ValidatorSharedReadWrite`), which PySpark loads, and which they load of PySpark.
use super::meta::{
    check_not_model, converted, json_to_ml_value, load_evaluator, load_python_metadata, load_stage,
    metadata_item, metadata_str, ml_value_json, param_doc, python_class_of, python_float,
    python_float_json, python_str_repr, CROSS_VALIDATOR_CLASS, CROSS_VALIDATOR_MODEL_CLASS,
    TRAIN_VALIDATION_SPLIT_CLASS, TRAIN_VALIDATION_SPLIT_MODEL_CLASS,
};
use super::*;
use spark_connect_core::error::Result;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

impl Clone for Box<dyn EstimatorStage> {
    fn clone(&self) -> Self {
        self.clone_stage()
    }
}

impl Clone for Box<dyn EvaluatorStage> {
    fn clone(&self) -> Self {
        self.clone_stage()
    }
}

/// A random seed, drawn for the default seed of a class of validators.
fn random_seed() -> i64 {
    let hex = uuid::Uuid::new_v4().simple().to_string();
    i64::from_str_radix(&hex[..15], 16).unwrap_or_default()
}

// The default seed of each class of validators and of its model, drawn once per process: PySpark's
// default seed is `hash(type(self).__name__)`, which is the same for every instance of a class
// within a process (and differs between processes, as Python randomizes the hash of a string).
static CROSS_VALIDATOR_SEED: OnceLock<i64> = OnceLock::new();
static CROSS_VALIDATOR_MODEL_SEED: OnceLock<i64> = OnceLock::new();
static TRAIN_VALIDATION_SPLIT_SEED: OnceLock<i64> = OnceLock::new();
static TRAIN_VALIDATION_SPLIT_MODEL_SEED: OnceLock<i64> = OnceLock::new();

/// The default seed of the class whose cell is `cell`, the same in each call of the process.
fn process_seed(cell: &'static OnceLock<i64>) -> i64 {
    *cell.get_or_init(random_seed)
}

/// The value of the param `name` of the validator of the JVM class `class` and the uid `uid` in
/// `map`, converted as its param's type converter converts it, if the map sets it.
fn own_value(class: &str, uid: &str, map: &ParamMap, name: &str) -> Result<Option<MlValue>> {
    map.get(uid, name)
        .map(|value| converted(class, name, value))
        .transpose()
}

/// The value of an own param of [`own_value`], as `read` makes of it, or PySpark's `TypeError`.
fn own_as<T>(
    class: &str,
    uid: &str,
    map: &ParamMap,
    name: &str,
    read: impl Fn(&MlValue) -> Option<T>,
) -> Result<Option<T>> {
    match own_value(class, uid, map, name)? {
        None => Ok(None),
        Some(value) => read(&value).map(Some).ok_or_else(|| {
            SparkError::value_msg(format!(
                "the param {name} of {uid} has no value of its type"
            ))
        }),
    }
}

/// The params of a validator, of PySpark's `_ValidatorParams` and `HasParallelism` and
/// `HasCollectSubModels`: its estimator, param maps and evaluator, its seed (the default of its
/// class's seed, see [`process_seed`]), its parallelism and whether it keeps the sub-models.
#[derive(Clone)]
struct Common {
    estimator: Option<Box<dyn EstimatorStage>>,
    param_maps: Vec<ParamMap>,
    evaluator: Option<Box<dyn EvaluatorStage>>,
    seed: Option<i64>,
    default_seed: i64,
    parallelism: i64,
    collect_sub_models: bool,
}

impl Common {
    /// The common params of the model of a validator: the validator's explicit seed (PySpark's
    /// `_copyValues` copies a set param), and the default seed of the model's class.
    fn model_common(&self, default_seed: i64) -> Self {
        let mut model = self.clone();
        model.default_seed = default_seed;
        model
    }

    fn new(default_seed: i64) -> Self {
        Self {
            estimator: None,
            param_maps: Vec::new(),
            evaluator: None,
            seed: None,
            default_seed,
            parallelism: 1,
            collect_sub_models: false,
        }
    }

    /// Set the values of `map` of the validator's own params (its seed, parallelism and
    /// `collectSubModels`), of the validator of the class `class` and the uid `uid`, as PySpark's
    /// `Params.copy(extra)` of a validator sets them.
    fn apply_own(&mut self, class: &str, uid: &str, map: &ParamMap) -> Result<()> {
        if let Some(seed) = own_as(class, uid, map, "seed", MlValue::as_i64)? {
            self.seed = Some(seed);
        }
        if let Some(parallelism) = own_as(class, uid, map, "parallelism", MlValue::as_i64)? {
            self.parallelism = parallelism;
        }
        if let Some(collect) = own_as(class, uid, map, "collectSubModels", MlValue::as_bool)? {
            self.collect_sub_models = collect;
        }
        Ok(())
    }

    /// Set the values of `map` of the params of the validator's estimator and evaluator, as the
    /// validators' `copy(extra)` copies them (after the validator's own params).
    fn apply_stages(&mut self, map: &ParamMap) -> Result<()> {
        if let Some(estimator) = &mut self.estimator {
            estimator.apply_param_map(map)?;
        }
        if let Some(evaluator) = &mut self.evaluator {
            evaluator.apply_param_map(map)?;
        }
        Ok(())
    }

    fn estimator(&self) -> Result<&dyn EstimatorStage> {
        self.estimator
            .as_deref()
            .ok_or_else(|| SparkError::value_msg("the validator has no estimator: set_estimator"))
    }

    fn evaluator(&self) -> Result<&dyn EvaluatorStage> {
        self.evaluator
            .as_deref()
            .ok_or_else(|| SparkError::value_msg("the validator has no evaluator: set_evaluator"))
    }

    fn seed(&self) -> i64 {
        self.seed.unwrap_or(self.default_seed)
    }
}

impl std::fmt::Debug for Common {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValidatorParams")
            .field(
                "estimator",
                &self.estimator.as_ref().map(|e| e.operator().uid.clone()),
            )
            .field("param_maps", &self.param_maps.len())
            .field(
                "evaluator",
                &self.evaluator.as_ref().map(|e| e.operator().uid.clone()),
            )
            .field("seed", &self.seed())
            .field("parallelism", &self.parallelism)
            .field("collect_sub_models", &self.collect_sub_models)
            .finish()
    }
}

/// The fits of a copy of `estimator` of each map of `maps` on `train`, each evaluated by
/// `evaluator` on the transform of `validation` by the model of the map, in `parallelism`
/// threads (at most one per map), as PySpark's `_parallelFitTasks` in its `ThreadPool`: the metric
/// and the model of each map, in the order of the maps.
fn parallel_fits(
    estimator: &dyn EstimatorStage,
    maps: &[ParamMap],
    evaluator: &dyn EvaluatorStage,
    train: &DataFrame,
    validation: &DataFrame,
    parallelism: i64,
) -> Result<Vec<(f64, Box<dyn Model>)>> {
    let threads = parallelism.min(maps.len() as i64).max(1) as usize;
    let next = std::sync::atomic::AtomicUsize::new(0);
    type Slot = std::sync::Mutex<Option<Result<(f64, Box<dyn Model>)>>>;
    let results: Vec<Slot> = maps.iter().map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if index >= maps.len() {
                    break;
                }
                let result = (|| {
                    let mut copy = estimator.clone_stage();
                    copy.apply_param_map(&maps[index])?;
                    let model = copy.fit(train)?;
                    // As `model.transform(validation, epm[index])`: of a copy of the map, as
                    // `Transformer.transform` copies the model of a map that is not empty.
                    let mut evaluated = model.clone_box();
                    if !maps[index].is_empty() {
                        evaluated.apply_param_map(&maps[index])?;
                    }
                    let metric = evaluator.evaluate(&evaluated.transform(validation)?)?;
                    Ok((metric, model))
                })();
                *results[index].lock().unwrap_or_else(|e| e.into_inner()) = Some(result);
            });
        }
    });
    results
        .into_iter()
        .map(|slot| {
            slot.into_inner()
                .unwrap_or_else(|e| e.into_inner())
                .unwrap_or_else(|| Err(SparkError::value_msg("a fit of a param map did not run")))
        })
        .collect()
}

/// PySpark's check of its pool of `min(parallelism, number of param maps)` threads.
fn check_pool(parallelism: i64, maps: usize) -> Result<()> {
    if parallelism.min(maps as i64) < 1 {
        return Err(SparkError::value_msg(
            "Number of processes must be at least 1",
        ));
    }
    Ok(())
}

/// Run `body` of `dataset` cached, as `_cache_spark_dataset` does by default: persisted
/// (`MEMORY_AND_DISK_DESER`), and unpersisted after, or of a copy saved under
/// `SPARKML_TEMP_DFS_PATH` when that is set.
fn with_cached<R>(dataset: &DataFrame, body: impl FnOnce(&DataFrame) -> Result<R>) -> Result<R> {
    let level = proto::StorageLevel {
        use_disk: true,
        use_memory: true,
        use_off_heap: false,
        deserialized: true,
        replication: 1,
    };
    with_cached_at(dataset, level, true, body)
}

/// Run `body` of `dataset` cached at `level`, as `_cache_spark_dataset(dataset,
/// storageLevel=level, enable=enable)`: of `dataset` itself, uncached, if not `enable`; else of a
/// copy saved under `SPARKML_TEMP_DFS_PATH`, if set, which the ML helper deletes after; else
/// persisted at `level`, and unpersisted after `body` returns.
pub(super) fn with_cached_at<R>(
    dataset: &DataFrame,
    level: proto::StorageLevel,
    enable: bool,
    body: impl FnOnce(&DataFrame) -> Result<R>,
) -> Result<R> {
    if !enable {
        return body(dataset);
    }
    match std::env::var("SPARKML_TEMP_DFS_PATH") {
        Ok(dir) if !dir.is_empty() => {
            let path = join_path(&dir, &uuid::Uuid::new_v4().simple().to_string());
            dataset.write().save(Some(&path))?;
            let loaded = dataset.session.read().load(Some(&path));
            let result = body(&loaded);
            let removed = handle_overwrite(&dataset.session, &path);
            let value = result?;
            removed?;
            Ok(value)
        }
        _ => {
            dataset.persist(level)?;
            let result = body(dataset);
            let unpersisted = dataset.unpersist(false);
            let value = result?;
            unpersisted?;
            Ok(value)
        }
    }
}

/// The index of the best of `metrics`, as numpy's `argmax` for an evaluator whose larger metrics
/// are better, else `argmin`: the first of the best, and the first NaN of any.
fn best_index(metrics: &[f64], larger_is_better: bool) -> usize {
    if let Some(nan) = metrics.iter().position(|m| m.is_nan()) {
        return nan;
    }
    let mut best = 0;
    for (i, &metric) in metrics.iter().enumerate().skip(1) {
        let better = if larger_is_better {
            metric > metrics[best]
        } else {
            metric < metrics[best]
        };
        if better {
            best = i;
        }
    }
    best
}

/// The mean and the standard deviation of each metric over the folds, as numpy's `mean` and `std`
/// of axis 0 (the deviation of the population).
fn mean_and_std(metrics: &[Vec<f64>], num_models: usize) -> (Vec<f64>, Vec<f64>) {
    let n = metrics.len() as f64;
    let mut means = vec![0.0; num_models];
    for fold in metrics {
        for (j, m) in fold.iter().enumerate() {
            means[j] += m;
        }
    }
    for mean in &mut means {
        *mean /= n;
    }
    let mut stds = vec![0.0; num_models];
    for fold in metrics {
        for (j, m) in fold.iter().enumerate() {
            stds[j] += (m - means[j]).powi(2);
        }
    }
    for std in &mut stds {
        *std = (*std / n).sqrt();
    }
    (means, stds)
}

/// The fit of the best map of `maps` on the whole dataset `df`, by a copy of `estimator`.
fn fit_best(
    estimator: &dyn EstimatorStage,
    maps: &[ParamMap],
    best: usize,
    df: &DataFrame,
) -> Result<Box<dyn Model>> {
    let mut copy = estimator.clone_stage();
    copy.apply_param_map(&maps[best])?;
    copy.fit(df)
}

// ---------------------------------------------------------------------------------------------
// CrossValidator and CrossValidatorModel.
// ---------------------------------------------------------------------------------------------

/// PySpark's `CrossValidator`: the selection of the best param map of an estimator by k-fold cross
/// validation, of an evaluator's metric averaged over the folds.
#[derive(Debug, Clone)]
pub struct CrossValidator {
    operator: MlOperator,
    params: Params,
    common: Common,
    num_folds: i64,
    fold_col: String,
}

impl CrossValidator {
    /// The names of PySpark's params of the class.
    pub const PARAM_NAMES: &'static [&'static str] = &[
        "collectSubModels",
        "estimator",
        "estimatorParamMaps",
        "evaluator",
        "foldCol",
        "numFolds",
        "parallelism",
        "seed",
    ];

    /// A validator of no estimator, param maps or evaluator, of 3 folds and no fold column, of a
    /// new uid (`CrossValidator_<12 hex digits>`).
    pub fn new() -> Self {
        Self {
            operator: MlOperator::new(
                "org.apache.spark.ml.tuning.CrossValidator",
                OperatorType::Estimator,
            ),
            params: Params::new(),
            common: Common::new(process_seed(&CROSS_VALIDATOR_SEED)),
            num_folds: 3,
            fold_col: String::new(),
        }
    }

    pub fn set_estimator(mut self, estimator: impl EstimatorStage + 'static) -> Self {
        self.common.estimator = Some(Box::new(estimator));
        self
    }

    pub fn set_estimator_param_maps(mut self, maps: Vec<ParamMap>) -> Self {
        self.common.param_maps = maps;
        self
    }

    pub fn set_evaluator(mut self, evaluator: impl EvaluatorStage + 'static) -> Self {
        self.common.evaluator = Some(Box::new(evaluator));
        self
    }

    /// The number of folds (default: 3).
    pub fn set_num_folds(mut self, value: i64) -> Self {
        self.num_folds = value;
        self
    }

    /// The column of the user's fold numbers, in `[0, numFolds)`, of no random split (default:
    /// none, "").
    pub fn set_fold_col(mut self, value: &str) -> Self {
        self.fold_col = value.to_string();
        self
    }

    /// The seed of the random split (default: a random seed of the validator).
    pub fn set_seed(mut self, value: i64) -> Self {
        self.common.seed = Some(value);
        self
    }

    /// The number of threads of the fits of the param maps (default: 1).
    pub fn set_parallelism(mut self, value: i64) -> Self {
        self.common.parallelism = value;
        self
    }

    /// Whether the model keeps the model of each fold and param map (default: false).
    pub fn set_collect_sub_models(mut self, value: bool) -> Self {
        self.common.collect_sub_models = value;
        self
    }

    pub fn estimator(&self) -> Option<&dyn EstimatorStage> {
        self.common.estimator.as_deref()
    }

    pub fn estimator_param_maps(&self) -> &[ParamMap] {
        &self.common.param_maps
    }

    pub fn evaluator(&self) -> Option<&dyn EvaluatorStage> {
        self.common.evaluator.as_deref()
    }

    pub fn num_folds(&self) -> i64 {
        self.num_folds
    }

    pub fn fold_col(&self) -> &str {
        &self.fold_col
    }

    pub fn seed(&self) -> i64 {
        self.common.seed()
    }

    pub fn parallelism(&self) -> i64 {
        self.common.parallelism
    }

    pub fn collect_sub_models(&self) -> bool {
        self.common.collect_sub_models
    }

    pub fn uid(&self) -> &str {
        &self.operator.uid
    }

    /// A copy of the validator, as `CrossValidator.copy(extra)`: of the same uid, of copies of its
    /// estimator and evaluator of the values of `extra` of their params (its param maps stay the
    /// same).
    pub fn copy(&self, extra: &ParamMap) -> Result<Self> {
        let mut that = self.clone();
        that.apply_param_map(extra)?;
        Ok(that)
    }

    /// The cross validation of `df`, as `CrossValidator._fit`, of the typed model.
    pub fn fit_validator(&mut self, df: &DataFrame) -> Result<CrossValidatorModel> {
        let estimator = self.common.estimator()?;
        let maps = &self.common.param_maps;
        let evaluator = self.common.evaluator()?;
        check_pool(self.common.parallelism, maps.len())?;
        let folds = k_fold(
            df,
            self.num_folds,
            &self.fold_col,
            Some(self.common.seed()),
            &self.operator.uid,
        )?;
        if folds.is_empty() {
            // As PySpark's mean and deviation of no fold, a scalar of which `list` fails.
            return Err(SparkError::value_msg(
                "'numpy.float64' object is not iterable",
            ));
        }
        let mut metrics_all = Vec::with_capacity(folds.len());
        let mut sub_models = self.common.collect_sub_models.then(Vec::new);
        for (train, validation) in &folds {
            let results = with_cached(train, |train| {
                with_cached(validation, |validation| {
                    parallel_fits(
                        estimator,
                        maps,
                        evaluator,
                        train,
                        validation,
                        self.common.parallelism,
                    )
                })
            })?;
            let (metrics, models): (Vec<f64>, Vec<Box<dyn Model>>) = results.into_iter().unzip();
            metrics_all.push(metrics);
            if let Some(sub_models) = &mut sub_models {
                sub_models.push(models);
            }
        }
        let (avg_metrics, std_metrics) = mean_and_std(&metrics_all, maps.len());
        let best = best_index(&avg_metrics, evaluator.is_larger_better());
        let best_model = fit_best(estimator, maps, best, df)?;
        Ok(CrossValidatorModel {
            operator: MlOperator::new(
                "org.apache.spark.ml.tuning.CrossValidatorModel",
                OperatorType::Model,
            ),
            params: Params::new(),
            best_model,
            avg_metrics,
            std_metrics,
            sub_models,
            common: self
                .common
                .model_common(process_seed(&CROSS_VALIDATOR_MODEL_SEED)),
            num_folds: self.num_folds,
            fold_col: self.fold_col.clone(),
        })
    }

    /// Load the validator saved at `path` of PySpark's Python format, as `CrossValidatorReader`.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let loaded = load_common(session, path, process_seed(&CROSS_VALIDATOR_SEED))?;
        let mut cv = Self::new();
        cv.operator = MlOperator::with_uid(
            "org.apache.spark.ml.tuning.CrossValidator",
            &loaded.uid,
            OperatorType::Estimator,
        );
        cv.common = loaded.common;
        if let Some(value) = loaded.ints.get("numFolds") {
            cv.num_folds = *value;
        }
        if let Some(value) = loaded.strings.get("foldCol") {
            cv.fold_col = value.clone();
        }
        Ok(cv)
    }
}

impl Default for CrossValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for CrossValidator {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }

    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }

    fn params(&self) -> &Params {
        &self.params
    }

    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn fit_impl(&mut self, df: &DataFrame) -> Result<Box<dyn Model>> {
        Ok(Box::new(self.fit_validator(df)?))
    }

    fn apply_param_map(&mut self, map: &ParamMap) -> Result<()> {
        let class = self.operator.name.clone();
        let uid = self.operator.uid.clone();
        if let Some(num_folds) = own_as(&class, &uid, map, "numFolds", MlValue::as_i64)? {
            self.num_folds = num_folds;
        }
        if let Some(fold_col) = own_as(&class, &uid, map, "foldCol", |v| {
            v.as_str().map(String::from)
        })? {
            self.fold_col = fold_col;
        }
        self.common.apply_own(&class, &uid, map)?;
        self.common.apply_stages(map)
    }

    fn write(&self) -> MlWriter {
        MlWriter::meta(Arc::new(ValidatorSave {
            class: CROSS_VALIDATOR_CLASS,
            uid: self.operator.uid.clone(),
            common: self.common.clone(),
            own: vec![
                ("numFolds", self.num_folds.to_string(), "3".to_string()),
                ("foldCol", json_string(&self.fold_col), "\"\"".to_string()),
            ],
            model: None,
        }))
    }
}

/// PySpark's `CrossValidatorModel`: the model of the best param map of a cross validation, the
/// mean and the standard deviation of the metric of each param map over the folds, and the model
/// of each fold and param map of `collectSubModels`.
#[derive(Debug, Clone)]
pub struct CrossValidatorModel {
    operator: MlOperator,
    params: Params,
    best_model: Box<dyn Model>,
    avg_metrics: Vec<f64>,
    std_metrics: Vec<f64>,
    sub_models: Option<Vec<Vec<Box<dyn Model>>>>,
    common: Common,
    num_folds: i64,
    fold_col: String,
}

impl CrossValidatorModel {
    /// The names of PySpark's params of the class.
    pub const PARAM_NAMES: &'static [&'static str] = &[
        "estimator",
        "estimatorParamMaps",
        "evaluator",
        "foldCol",
        "numFolds",
        "seed",
    ];

    /// The model of the best param map, as `bestModel`.
    pub fn best_model(&self) -> &dyn Model {
        self.best_model.as_ref()
    }

    /// The mean of the metric of each param map over the folds, as `avgMetrics`.
    pub fn avg_metrics(&self) -> &[f64] {
        &self.avg_metrics
    }

    /// The standard deviation of the metric of each param map over the folds, as `stdMetrics`.
    pub fn std_metrics(&self) -> &[f64] {
        &self.std_metrics
    }

    /// The model of each fold and param map, of `collectSubModels`, as `subModels`.
    pub fn sub_models(&self) -> Option<&[Vec<Box<dyn Model>>]> {
        self.sub_models.as_deref()
    }

    pub fn estimator_param_maps(&self) -> &[ParamMap] {
        &self.common.param_maps
    }

    pub fn num_folds(&self) -> i64 {
        self.num_folds
    }

    pub fn fold_col(&self) -> &str {
        &self.fold_col
    }

    pub fn seed(&self) -> i64 {
        self.common.seed()
    }

    pub fn uid(&self) -> &str {
        &self.operator.uid
    }

    /// Load the model saved at `path` of PySpark's Python format, as `CrossValidatorModelReader`:
    /// of the sub-models, where they were saved, of one list repeated for each fold, as PySpark's
    /// reader makes them.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let loaded = load_common(session, path, process_seed(&CROSS_VALIDATOR_MODEL_SEED))?;
        let num_folds = loaded.ints.get("numFolds").copied().unwrap_or(3);
        let best_model = load_model(session, &join_path(path, "bestModel"))?;
        let avg_metrics = floats(metadata_item(&loaded.metadata, "avgMetrics")?)?;
        let std_metrics = match loaded.metadata.get("stdMetrics") {
            Some(value) => floats(value)?,
            None => Vec::new(),
        };
        let sub_models = if persisted_sub_models(&loaded.metadata) {
            let mut models = Vec::with_capacity(loaded.common.param_maps.len());
            for index in 0..loaded.common.param_maps.len() {
                // PySpark's `[[None] * n] * numFolds`: each fold holds the models of the last.
                let path = join_path(
                    &join_path(
                        &join_path(path, "subModels"),
                        &format!("fold{}", num_folds - 1),
                    ),
                    &index.to_string(),
                );
                models.push(load_model(session, &path)?);
            }
            Some((0..num_folds).map(|_| models.clone()).collect())
        } else {
            None
        };
        Ok(Self {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.tuning.CrossValidatorModel",
                &loaded.uid,
                OperatorType::Model,
            ),
            params: Params::new(),
            best_model,
            avg_metrics,
            std_metrics,
            sub_models,
            common: loaded.common,
            num_folds,
            fold_col: loaded.strings.get("foldCol").cloned().unwrap_or_default(),
        })
    }
}

impl Transformer for CrossValidatorModel {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }

    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }

    fn params(&self) -> &Params {
        &self.params
    }

    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn transform_impl(&mut self, df: &DataFrame) -> Result<DataFrame> {
        self.best_model.transform(df)
    }

    /// Set the values of `map`, as `CrossValidatorModel.copy(extra)` of PySpark: a copy of the
    /// best model of `map`, then PySpark's `assert self.subModels is not None`, which fails of a
    /// model of no sub-models (see `no_sub_models`), copies of the sub-models (of no map: PySpark
    /// copies no extra into them), and the model's own params of `map` (`numFolds`, `foldCol` and
    /// `seed`), as `_copyValues` sets them.
    fn apply_param_map(&mut self, map: &ParamMap) -> Result<()> {
        let mut best_model = self.best_model.clone_box();
        best_model.apply_param_map(map)?;
        let Some(folds) = &self.sub_models else {
            return Err(no_sub_models("CrossValidatorModel"));
        };
        let sub_models = folds
            .iter()
            .map(|models| copied_models(models))
            .collect::<Result<Vec<_>>>()?;
        let class = self.operator.name.clone();
        let uid = self.operator.uid.clone();
        let num_folds = own_as(&class, &uid, map, "numFolds", MlValue::as_i64)?;
        let fold_col = own_as(&class, &uid, map, "foldCol", |v| {
            v.as_str().map(String::from)
        })?;
        let seed = own_as(&class, &uid, map, "seed", MlValue::as_i64)?;
        self.best_model = best_model;
        self.sub_models = Some(sub_models);
        if let Some(num_folds) = num_folds {
            self.num_folds = num_folds;
        }
        if let Some(fold_col) = fold_col {
            self.fold_col = fold_col;
        }
        if seed.is_some() {
            self.common.seed = seed;
        }
        Ok(())
    }
}

/// PySpark's `assert self.subModels is not None` of the `copy` of a model of the validator class
/// `class`: a bare `AssertionError` of the model of a fit of no `collectSubModels`, which fails an
/// outer validator's evaluation of a param map too, as its `model.transform(validation,
/// epm[index])` copies the model of the map.
fn no_sub_models(class: &str) -> SparkError {
    assertion_error(format!(
        "{class}.copy asserts that the model has sub-models (`assert self.subModels is not \
         None`), and this model has none: fit it with collect_sub_models to copy it of a param map"
    ))
}

/// Copies of `models`, as PySpark's `[sub_model.copy() for sub_model in models]` copies the
/// sub-models of a validator's model: of no param map (whose copy of a validator's model of no
/// sub-models fails as its own `copy` does).
fn copied_models(models: &[Box<dyn Model>]) -> Result<Vec<Box<dyn Model>>> {
    models
        .iter()
        .map(|model| {
            let mut copy = model.clone_box();
            copy.apply_param_map(&ParamMap::new())?;
            Ok(copy)
        })
        .collect()
}

impl Model for CrossValidatorModel {
    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }

    fn write(&self) -> Result<MlWriter> {
        let mut extra = vec![("avgMetrics", floats_json(&self.avg_metrics))];
        if !self.std_metrics.is_empty() {
            extra.push(("stdMetrics", floats_json(&self.std_metrics)));
        }
        Ok(MlWriter::meta(Arc::new(ValidatorSave {
            class: CROSS_VALIDATOR_MODEL_CLASS,
            uid: self.operator.uid.clone(),
            common: self.common.clone(),
            own: vec![
                ("numFolds", self.num_folds.to_string(), "3".to_string()),
                ("foldCol", json_string(&self.fold_col), "\"\"".to_string()),
            ],
            model: Some(ModelPart {
                extra,
                best_model: self.best_model.clone_box(),
                sub_models: self
                    .sub_models
                    .as_ref()
                    .map(|folds| SubModels::Folds(folds.clone())),
            }),
        })))
    }
}

// ---------------------------------------------------------------------------------------------
// TrainValidationSplit and TrainValidationSplitModel.
// ---------------------------------------------------------------------------------------------

/// PySpark's `TrainValidationSplit`: the selection of the best param map of an estimator by one
/// random split of a dataset into a training and a validation set.
#[derive(Debug, Clone)]
pub struct TrainValidationSplit {
    operator: MlOperator,
    params: Params,
    common: Common,
    train_ratio: f64,
}

impl TrainValidationSplit {
    /// The names of PySpark's params of the class.
    pub const PARAM_NAMES: &'static [&'static str] = &[
        "collectSubModels",
        "estimator",
        "estimatorParamMaps",
        "evaluator",
        "parallelism",
        "seed",
        "trainRatio",
    ];

    /// A validator of no estimator, param maps or evaluator, of a train ratio of 0.75, of a new
    /// uid (`TrainValidationSplit_<12 hex digits>`).
    pub fn new() -> Self {
        Self {
            operator: MlOperator::new(
                "org.apache.spark.ml.tuning.TrainValidationSplit",
                OperatorType::Estimator,
            ),
            params: Params::new(),
            common: Common::new(process_seed(&TRAIN_VALIDATION_SPLIT_SEED)),
            train_ratio: 0.75,
        }
    }

    pub fn set_estimator(mut self, estimator: impl EstimatorStage + 'static) -> Self {
        self.common.estimator = Some(Box::new(estimator));
        self
    }

    pub fn set_estimator_param_maps(mut self, maps: Vec<ParamMap>) -> Self {
        self.common.param_maps = maps;
        self
    }

    pub fn set_evaluator(mut self, evaluator: impl EvaluatorStage + 'static) -> Self {
        self.common.evaluator = Some(Box::new(evaluator));
        self
    }

    /// The ratio of the training set of the split (default: 0.75).
    pub fn set_train_ratio(mut self, value: f64) -> Self {
        self.train_ratio = value;
        self
    }

    pub fn set_seed(mut self, value: i64) -> Self {
        self.common.seed = Some(value);
        self
    }

    pub fn set_parallelism(mut self, value: i64) -> Self {
        self.common.parallelism = value;
        self
    }

    pub fn set_collect_sub_models(mut self, value: bool) -> Self {
        self.common.collect_sub_models = value;
        self
    }

    pub fn estimator(&self) -> Option<&dyn EstimatorStage> {
        self.common.estimator.as_deref()
    }

    pub fn estimator_param_maps(&self) -> &[ParamMap] {
        &self.common.param_maps
    }

    pub fn evaluator(&self) -> Option<&dyn EvaluatorStage> {
        self.common.evaluator.as_deref()
    }

    pub fn train_ratio(&self) -> f64 {
        self.train_ratio
    }

    pub fn seed(&self) -> i64 {
        self.common.seed()
    }

    pub fn parallelism(&self) -> i64 {
        self.common.parallelism
    }

    pub fn collect_sub_models(&self) -> bool {
        self.common.collect_sub_models
    }

    pub fn uid(&self) -> &str {
        &self.operator.uid
    }

    /// A copy, as `TrainValidationSplit.copy(extra)` (see [`CrossValidator::copy`]).
    pub fn copy(&self, extra: &ParamMap) -> Result<Self> {
        let mut that = self.clone();
        that.apply_param_map(extra)?;
        Ok(that)
    }

    /// The validation of `df`, as `TrainValidationSplit._fit`, of the typed model.
    pub fn fit_validator(&mut self, df: &DataFrame) -> Result<TrainValidationSplitModel> {
        let estimator = self.common.estimator()?;
        let maps = &self.common.param_maps;
        let evaluator = self.common.evaluator()?;
        let (train, validation) = train_validation_split(
            df,
            self.train_ratio,
            Some(self.common.seed()),
            &self.operator.uid,
        );
        let results = with_cached(&train, |train| {
            with_cached(&validation, |validation| {
                check_pool(self.common.parallelism, maps.len())?;
                parallel_fits(
                    estimator,
                    maps,
                    evaluator,
                    train,
                    validation,
                    self.common.parallelism,
                )
            })
        })?;
        let (validation_metrics, models): (Vec<f64>, Vec<Box<dyn Model>>) =
            results.into_iter().unzip();
        let best = best_index(&validation_metrics, evaluator.is_larger_better());
        let best_model = fit_best(estimator, maps, best, df)?;
        Ok(TrainValidationSplitModel {
            operator: MlOperator::new(
                "org.apache.spark.ml.tuning.TrainValidationSplitModel",
                OperatorType::Model,
            ),
            params: Params::new(),
            best_model,
            validation_metrics,
            sub_models: self.common.collect_sub_models.then_some(models),
            common: self
                .common
                .model_common(process_seed(&TRAIN_VALIDATION_SPLIT_MODEL_SEED)),
            train_ratio: self.train_ratio,
        })
    }

    /// Load the validator saved at `path` of PySpark's Python format, as
    /// `TrainValidationSplitReader`.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let loaded = load_common(session, path, process_seed(&TRAIN_VALIDATION_SPLIT_SEED))?;
        let mut tvs = Self::new();
        tvs.operator = MlOperator::with_uid(
            "org.apache.spark.ml.tuning.TrainValidationSplit",
            &loaded.uid,
            OperatorType::Estimator,
        );
        tvs.common = loaded.common;
        if let Some(value) = loaded.floats.get("trainRatio") {
            tvs.train_ratio = *value;
        }
        Ok(tvs)
    }
}

impl Default for TrainValidationSplit {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for TrainValidationSplit {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }

    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }

    fn params(&self) -> &Params {
        &self.params
    }

    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn fit_impl(&mut self, df: &DataFrame) -> Result<Box<dyn Model>> {
        Ok(Box::new(self.fit_validator(df)?))
    }

    fn apply_param_map(&mut self, map: &ParamMap) -> Result<()> {
        let class = self.operator.name.clone();
        let uid = self.operator.uid.clone();
        if let Some(train_ratio) = own_as(&class, &uid, map, "trainRatio", MlValue::as_f64)? {
            self.train_ratio = train_ratio;
        }
        self.common.apply_own(&class, &uid, map)?;
        self.common.apply_stages(map)
    }

    fn write(&self) -> MlWriter {
        MlWriter::meta(Arc::new(ValidatorSave {
            class: TRAIN_VALIDATION_SPLIT_CLASS,
            uid: self.operator.uid.clone(),
            common: self.common.clone(),
            own: vec![(
                "trainRatio",
                python_float_json(self.train_ratio),
                "0.75".to_string(),
            )],
            model: None,
        }))
    }
}

/// PySpark's `TrainValidationSplitModel`: the model of the best param map of a train validation
/// split, the metric of each param map, and the model of each param map of `collectSubModels`.
#[derive(Debug, Clone)]
pub struct TrainValidationSplitModel {
    operator: MlOperator,
    params: Params,
    best_model: Box<dyn Model>,
    validation_metrics: Vec<f64>,
    sub_models: Option<Vec<Box<dyn Model>>>,
    common: Common,
    train_ratio: f64,
}

impl TrainValidationSplitModel {
    /// The names of PySpark's params of the class.
    pub const PARAM_NAMES: &'static [&'static str] = &[
        "estimator",
        "estimatorParamMaps",
        "evaluator",
        "seed",
        "trainRatio",
    ];

    /// The model of the best param map, as `bestModel`.
    pub fn best_model(&self) -> &dyn Model {
        self.best_model.as_ref()
    }

    /// The metric of each param map, as `validationMetrics`.
    pub fn validation_metrics(&self) -> &[f64] {
        &self.validation_metrics
    }

    /// The model of each param map, of `collectSubModels`, as `subModels`.
    pub fn sub_models(&self) -> Option<&[Box<dyn Model>]> {
        self.sub_models.as_deref()
    }

    pub fn estimator_param_maps(&self) -> &[ParamMap] {
        &self.common.param_maps
    }

    pub fn train_ratio(&self) -> f64 {
        self.train_ratio
    }

    pub fn seed(&self) -> i64 {
        self.common.seed()
    }

    pub fn uid(&self) -> &str {
        &self.operator.uid
    }

    /// Load the model saved at `path` of PySpark's Python format, as
    /// `TrainValidationSplitModelReader`.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let loaded = load_common(
            session,
            path,
            process_seed(&TRAIN_VALIDATION_SPLIT_MODEL_SEED),
        )?;
        let best_model = load_model(session, &join_path(path, "bestModel"))?;
        let validation_metrics = floats(metadata_item(&loaded.metadata, "validationMetrics")?)?;
        let sub_models = if persisted_sub_models(&loaded.metadata) {
            let mut models = Vec::with_capacity(loaded.common.param_maps.len());
            for index in 0..loaded.common.param_maps.len() {
                let path = join_path(&join_path(path, "subModels"), &index.to_string());
                models.push(load_model(session, &path)?);
            }
            Some(models)
        } else {
            None
        };
        Ok(Self {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.tuning.TrainValidationSplitModel",
                &loaded.uid,
                OperatorType::Model,
            ),
            params: Params::new(),
            best_model,
            validation_metrics,
            sub_models,
            common: loaded.common,
            train_ratio: loaded.floats.get("trainRatio").copied().unwrap_or(0.75),
        })
    }
}

impl Transformer for TrainValidationSplitModel {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }

    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }

    fn params(&self) -> &Params {
        &self.params
    }

    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn transform_impl(&mut self, df: &DataFrame) -> Result<DataFrame> {
        self.best_model.transform(df)
    }

    /// Set the values of `map`, as `TrainValidationSplitModel.copy(extra)` of PySpark (see
    /// [`CrossValidatorModel`]'s): a copy of the best model of `map`, the `assert` of sub-models,
    /// copies of the sub-models, and the model's own params of `map` (`trainRatio` and `seed`).
    fn apply_param_map(&mut self, map: &ParamMap) -> Result<()> {
        let mut best_model = self.best_model.clone_box();
        best_model.apply_param_map(map)?;
        let Some(models) = &self.sub_models else {
            return Err(no_sub_models("TrainValidationSplitModel"));
        };
        let sub_models = copied_models(models)?;
        let class = self.operator.name.clone();
        let uid = self.operator.uid.clone();
        let train_ratio = own_as(&class, &uid, map, "trainRatio", MlValue::as_f64)?;
        let seed = own_as(&class, &uid, map, "seed", MlValue::as_i64)?;
        self.best_model = best_model;
        self.sub_models = Some(sub_models);
        if let Some(train_ratio) = train_ratio {
            self.train_ratio = train_ratio;
        }
        if seed.is_some() {
            self.common.seed = seed;
        }
        Ok(())
    }
}

impl Model for TrainValidationSplitModel {
    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }

    fn write(&self) -> Result<MlWriter> {
        Ok(MlWriter::meta(Arc::new(ValidatorSave {
            class: TRAIN_VALIDATION_SPLIT_MODEL_CLASS,
            uid: self.operator.uid.clone(),
            common: self.common.clone(),
            own: vec![(
                "trainRatio",
                python_float_json(self.train_ratio),
                "0.75".to_string(),
            )],
            model: Some(ModelPart {
                extra: vec![("validationMetrics", floats_json(&self.validation_metrics))],
                best_model: self.best_model.clone_box(),
                sub_models: self.sub_models.clone().map(SubModels::Maps),
            }),
        })))
    }
}

// ---------------------------------------------------------------------------------------------
// Persistence.
// ---------------------------------------------------------------------------------------------

/// The JSON of a list of floats, as Python's `json.dumps` writes it (of `NaN`).
fn floats_json(values: &[f64]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|v| python_float_json(*v))
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// The floats of a list of a metadata.
fn floats(value: &serde_json::Value) -> Result<Vec<f64>> {
    value
        .as_array()
        .ok_or_else(|| SparkError::value_msg("the metrics of the metadata are no list"))?
        .iter()
        .map(|v| {
            python_float(v)
                .ok_or_else(|| SparkError::value_msg("a metric of the metadata is no float"))
        })
        .collect()
}

/// Whether the sub-models were saved, of the `persistSubModels` of a model's metadata.
fn persisted_sub_models(metadata: &serde_json::Value) -> bool {
    metadata
        .get("persistSubModels")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

/// The model saved at `path`: a model of the server's class, or a model of a meta-algorithm.
fn load_model(session: &SparkSession, path: &str) -> Result<Box<dyn Model>> {
    match load_stage(session, path)? {
        PipelineStage::Model(model) => Ok(model),
        other => Err(SparkError::value_msg(format!(
            "the stage at {path} is no model: {other:?}"
        ))),
    }
}

/// The sub-models of a model to save: of each fold and param map, or of each param map.
#[derive(Clone)]
enum SubModels {
    Folds(Vec<Vec<Box<dyn Model>>>),
    Maps(Vec<Box<dyn Model>>),
}

/// What a validator's model saves besides the validator: the fields of its metadata (`extra`), its
/// best model and its sub-models.
#[derive(Clone)]
struct ModelPart {
    extra: Vec<(&'static str, String)>,
    best_model: Box<dyn Model>,
    sub_models: Option<SubModels>,
}

/// The save of a validator or its model, as `_ValidatorSharedReadWrite.saveImpl` and the writers of
/// `pyspark.ml.tuning`: its params (`own`, of the name, the JSON of the value and the JSON of the
/// default of each), its param maps, and its estimator and evaluator, of each of its own writer.
struct ValidatorSave {
    class: &'static str,
    uid: String,
    common: Common,
    own: Vec<(&'static str, String, String)>,
    model: Option<ModelPart>,
}

impl MetaSave for ValidatorSave {
    fn name(&self) -> String {
        self.uid.clone()
    }

    fn save_meta(
        &self,
        session: &SparkSession,
        path: &str,
        options: &HashMap<String, String>,
    ) -> Result<()> {
        let estimator = self.common.estimator()?;
        let evaluator = self.common.evaluator()?;
        let stages = validate_params(estimator, &self.common.param_maps)?;
        let persist = match &self.model {
            Some(model) => Some(persist_sub_models(model, options)?),
            None => None,
        };
        // The params, of the defaults in PySpark's order: collectSubModels, parallelism and seed
        // of a validator (its model has none of the first two), its own, and then the param maps.
        let mut params: Vec<(&str, String)> = Vec::new();
        let mut defaults: Vec<(&str, String)> = Vec::new();
        if self.model.is_none() {
            params.push((
                "collectSubModels",
                self.common.collect_sub_models.to_string(),
            ));
            defaults.push(("collectSubModels", "false".to_string()));
            params.push(("parallelism", self.common.parallelism.to_string()));
            defaults.push(("parallelism", "1".to_string()));
        }
        params.push(("seed", self.common.seed().to_string()));
        defaults.push(("seed", self.common.default_seed.to_string()));
        for (name, value, default) in &self.own {
            params.push((name, value.clone()));
            defaults.push((name, default.clone()));
        }
        params.push((
            "estimatorParamMaps",
            param_maps_json(&self.common.param_maps, &stages)?,
        ));
        let mut metadata = Metadata::new(
            self.class,
            &self.uid,
            &session.version()?,
            json_object(&params),
            json_object(&defaults),
        );
        if let (Some(model), Some(persist)) = (&self.model, persist) {
            for (key, value) in &model.extra {
                metadata.extra.push((key.to_string(), value.clone()));
            }
            metadata
                .extra
                .push(("persistSubModels".to_string(), persist.to_string()));
            // `CrossValidatorModelWriter` writes `stdMetrics` after `persistSubModels`.
            if let Some(position) = metadata.extra.iter().position(|(k, _)| k == "stdMetrics") {
                let std = metadata.extra.remove(position);
                metadata.extra.push(std);
            }
        }
        save_metadata(session, path, &metadata.to_json())?;
        evaluator
            .write()
            .session(session)
            .save(&join_path(path, "evaluator"))?;
        estimator
            .write()
            .session(session)
            .save(&join_path(path, "estimator"))?;
        if let (Some(model), Some(persist)) = (&self.model, persist) {
            model
                .best_model
                .write()?
                .session(session)
                .save(&join_path(path, "bestModel"))?;
            if persist {
                let sub_models = model
                    .sub_models
                    .as_ref()
                    .ok_or_else(|| SparkError::value_msg(NO_SUB_MODELS))?;
                let dir = join_path(path, "subModels");
                match sub_models {
                    SubModels::Folds(folds) => {
                        for (i, fold) in folds.iter().enumerate() {
                            let fold_dir = join_path(&dir, &format!("fold{i}"));
                            for (j, m) in fold.iter().enumerate() {
                                m.write()?
                                    .session(session)
                                    .save(&join_path(&fold_dir, &j.to_string()))?;
                            }
                        }
                    }
                    SubModels::Maps(models) => {
                        for (j, m) in models.iter().enumerate() {
                            m.write()?
                                .session(session)
                                .save(&join_path(&dir, &j.to_string()))?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// PySpark's error of a save of the sub-models of a model of none.
const NO_SUB_MODELS: &str = "When persisting tuning models, you can only set persistSubModels to \
                             true if the tuning was done with collectSubModels set to true. To \
                             save the sub-models, try rerunning fitting with collectSubModels set \
                             to true.";

/// Whether a model's writer saves the sub-models, as
/// `getValidatorModelWriterPersistSubModelsParam`: of the option `persistsubmodels` ("true" or
/// "false" of any case), else of whether the model has sub-models.
fn persist_sub_models(model: &ModelPart, options: &HashMap<String, String>) -> Result<bool> {
    let option = options
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case("persistSubModels"))
        .map(|(_, value)| value.to_lowercase());
    match option.as_deref() {
        Some("true") => Ok(true),
        Some("false") => Ok(false),
        Some(other) => Err(SparkError::value_msg(format!(
            "persistSubModels option value {other} is invalid, the possible values are True, \
             'True' or False, 'False'"
        ))),
        None => Ok(model.sub_models.is_some()),
    }
}

/// The uid and the JVM class of a stage's operator.
fn stage_of(operator: &MlOperator) -> (String, String) {
    (operator.uid.clone(), operator.name.clone())
}

/// The stages of `stage` and of its nested stages, as `MetaAlgorithmReadWrite.getAllNestedStages`
/// of PySpark, each as its uid and its JVM class: a nested validator is an error, as PySpark raises
/// it.
fn nested_stages(stage: &PipelineStage, out: &mut Vec<(String, String)>) -> Result<()> {
    out.push(stage_of(stage.operator()));
    let any = match stage {
        PipelineStage::Estimator(e) => e.as_any_stage(),
        PipelineStage::Transformer(t) => {
            check_not_model(t.operator())?;
            None
        }
        PipelineStage::Model(m) => Some(m.as_ref().as_any()),
    };
    if let Some(any) = any {
        if any.is::<CrossValidator>()
            || any.is::<TrainValidationSplit>()
            || any.is::<CrossValidatorModel>()
            || any.is::<TrainValidationSplitModel>()
        {
            return Err(SparkError::value_msg(
                "PySpark does not support nested validator.",
            ));
        }
        let stages = any
            .downcast_ref::<Pipeline>()
            .map(Pipeline::stages)
            .or_else(|| {
                any.downcast_ref::<PipelineModel>()
                    .map(PipelineModel::stages)
            });
        for nested in stages.unwrap_or_default() {
            nested_stages(nested, out)?;
        }
        // The classifier of a OneVsRest, and of its model the models too, as
        // `getAllNestedStages` of PySpark: each is a stage of no stage of its own.
        if let Some(ovr) = any.downcast_ref::<OneVsRest>() {
            out.extend(ovr.classifier().map(|c| stage_of(c.estimator().operator())));
        }
        if let Some(ovr) = any.downcast_ref::<OneVsRestModel>() {
            out.extend(ovr.classifier().map(|c| stage_of(c.estimator().operator())));
            out.extend(ovr.stage_models().iter().map(|m| stage_of(m.operator())));
        }
    }
    Ok(())
}

/// The stages of `estimator` and of its nested stages, each as its uid and JVM class, as
/// `MetaAlgorithmReadWrite.getUidMap` of PySpark gives them: a uid that two of the stages have is
/// an error, which PySpark raises at the save and at the load alike, of the estimator's class.
fn uid_map(estimator: &dyn EstimatorStage) -> Result<Vec<(String, String)>> {
    let mut stages = Vec::new();
    nested_stages(
        &PipelineStage::Estimator(estimator.clone_stage()),
        &mut stages,
    )?;
    // The uids in the order of their first stage, as the keys of PySpark's dict of the stages.
    let mut uids: Vec<&str> = Vec::with_capacity(stages.len());
    for (uid, _) in &stages {
        if !uids.contains(&uid.as_str()) {
            uids.push(uid);
        }
    }
    if uids.len() != stages.len() {
        let list: Vec<String> = uids.iter().map(|uid| format!("'{uid}'")).collect();
        return Err(SparkError::value_msg(format!(
            "{}.load found a compound estimator with stages with duplicate UIDs. List of UIDs: \
             [{}].",
            python_class_of(&estimator.operator().name),
            list.join(", ")
        )));
    }
    Ok(stages)
}

/// Check that the validator saves, as `_ValidatorSharedReadWrite.validateParams`: its estimator's
/// uids are distinct, and each param of its param maps is of its estimator or of a stage that the
/// estimator nests. The stages of the estimator, each as its uid and JVM class, are returned.
fn validate_params(
    estimator: &dyn EstimatorStage,
    maps: &[ParamMap],
) -> Result<Vec<(String, String)>> {
    let stages = uid_map(estimator)?;
    for map in maps {
        for pair in map.pairs() {
            if !stages.iter().any(|(uid, _)| *uid == pair.parent) {
                return Err(SparkError::value_msg(format!(
                    "Validator save requires all Params in estimatorParamMaps to apply to its \
                     Estimator, An extraneous Param was found: {}",
                    param_repr(&pair.parent, &pair.name)
                )));
            }
        }
    }
    Ok(stages)
}

/// PySpark's `repr` of the param `name` of the stage of the uid `parent`, `Param(parent=..,
/// name=.., doc=..)`. A uid is its class's name and a suffix, as PySpark's `Identifiable` makes it,
/// so the doc is the one of that class; a param that the tables do not have has an empty doc.
fn param_repr(parent: &str, name: &str) -> String {
    let class = parent.rsplit_once('_').map_or(parent, |(class, _)| class);
    let doc = param_doc(class, name).unwrap_or("");
    format!(
        "Param(parent={}, name={}, doc={})",
        python_str_repr(parent),
        python_str_repr(name),
        python_str_repr(doc)
    )
}

/// The param maps of a validator, as `_ValidatorSharedReadWrite.saveImpl` writes them: each value
/// converted by the type converter of its param (of its stage, in `stages`), as PySpark converts
/// the values of its grid when it builds them, and written as the JSON of its `json.dumps`, a
/// string.
fn param_maps_json(maps: &[ParamMap], stages: &[(String, String)]) -> Result<String> {
    let mut out = Vec::with_capacity(maps.len());
    for map in maps {
        let mut pairs = Vec::with_capacity(map.len());
        for pair in map.pairs() {
            let class = stages
                .iter()
                .find(|(uid, _)| *uid == pair.parent)
                .map_or("", |(_, class)| class.as_str());
            let value = converted(class, &pair.name, &pair.value)?;
            pairs.push(json_object(&[
                ("parent", json_string(&pair.parent)),
                ("name", json_string(&pair.name)),
                ("value", json_string(&ml_value_json(&value)?)),
                ("isJson", "true".to_string()),
            ]));
        }
        out.push(format!("[{}]", pairs.join(",")));
    }
    Ok(format!("[{}]", out.join(",")))
}

/// What a load of a validator or its model reads of its metadata: the uid, the common params, the
/// ints, floats and strings of its other params, and the metadata itself.
struct Loaded {
    uid: String,
    common: Common,
    ints: HashMap<String, i64>,
    floats: HashMap<String, f64>,
    strings: HashMap<String, String>,
    metadata: serde_json::Value,
}

/// The validator or model saved at `path`, as `_ValidatorSharedReadWrite.load` and
/// `getAndSetParams`: its estimator, evaluator and param maps (of a metadata of PySpark's Python
/// writer, or of its JVM writer, which has the JSON of each field), and its params.
fn load_common(session: &SparkSession, path: &str, default_seed: i64) -> Result<Loaded> {
    let metadata = load_python_metadata(session, path)?;
    let class = metadata_str(&metadata, "class")?;
    let python_writer = class.starts_with("pyspark.ml.");
    let evaluator = load_evaluator(session, &join_path(path, "evaluator"))?;
    let estimator = match load_stage(session, &join_path(path, "estimator"))? {
        PipelineStage::Estimator(estimator) => estimator,
        other => {
            return Err(SparkError::value_msg(format!(
                "the estimator of the validator at {path} is no estimator: {other:?}"
            )))
        }
    };
    let mut stages = uid_map(estimator.as_ref())?;
    stages.push(stage_of(evaluator.operator()));
    let param_map = metadata_item(&metadata, "paramMap")?;
    let mut maps = Vec::new();
    let json_maps = metadata_item(param_map, "estimatorParamMaps")?
        .as_array()
        .ok_or_else(|| SparkError::value_msg("the estimatorParamMaps are no list"))?;
    for json_map in json_maps {
        let mut map = ParamMap::new();
        let pairs = json_map
            .as_array()
            .ok_or_else(|| SparkError::value_msg("a param map of the metadata is no list"))?;
        for pair in pairs {
            let parent = metadata_str(pair, "parent")?;
            if !stages.iter().any(|(uid, _)| uid == parent) {
                return Err(SparkError::value_msg(format!(
                    "the param map of the metadata is of the stage {parent}, which the validator \
                     has not"
                )));
            }
            let name = metadata_str(pair, "name")?;
            let field = |key: &str| -> Result<serde_json::Value> {
                let value = metadata_item(pair, key)?.clone();
                if python_writer {
                    Ok(value)
                } else {
                    let text = value.as_str().ok_or_else(|| {
                        SparkError::value_msg("a field of the param is no string")
                    })?;
                    super::meta::parse_python_json(text)
                }
            };
            let is_json = match pair.get("isJson") {
                None => true,
                Some(_) => field("isJson")?.as_bool().unwrap_or(false),
            };
            if !is_json {
                return Err(SparkError::value_msg(format!(
                    "the value of the param {name} of the stage {parent} is a saved stage, which \
                     a param map of this client does not hold"
                )));
            }
            let text = metadata_str(pair, "value")?;
            let value = super::meta::parse_python_json(text)?;
            map.insert(parent, name, json_to_ml_value(&value));
        }
        maps.push(map);
    }
    let mut common = Common::new(default_seed);
    common.estimator = Some(estimator);
    common.evaluator = Some(evaluator);
    common.param_maps = maps;
    let mut ints = HashMap::new();
    let mut floats_of = HashMap::new();
    let mut strings = HashMap::new();
    if let Some(params) = param_map.as_object() {
        for (name, value) in params {
            match (name.as_str(), value) {
                ("estimatorParamMaps", _) => {}
                ("seed", v) => common.seed = v.as_i64(),
                ("parallelism", v) => common.parallelism = v.as_i64().unwrap_or(1),
                ("collectSubModels", v) => common.collect_sub_models = v.as_bool().unwrap_or(false),
                (name, v) if v.is_i64() => {
                    ints.insert(name.to_string(), v.as_i64().unwrap_or_default());
                }
                (name, v) if v.is_string() && python_float(v).is_none() => {
                    strings.insert(name.to_string(), v.as_str().unwrap_or_default().to_string());
                }
                (name, v) => {
                    if let Some(f) = python_float(v) {
                        floats_of.insert(name.to_string(), f);
                    }
                }
            }
        }
    }
    if let Some(seed) = metadata
        .get("defaultParamMap")
        .and_then(|d| d.get("seed"))
        .and_then(serde_json::Value::as_i64)
    {
        common.default_seed = seed;
    }
    Ok(Loaded {
        uid: metadata_str(&metadata, "uid")?.to_string(),
        common,
        ints,
        floats: floats_of,
        strings,
        metadata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_best_metric_is_numpys_argmax_or_argmin_of_the_first_nan() {
        assert_eq!(best_index(&[0.1, 0.9, 0.9, 0.3], true), 1);
        assert_eq!(best_index(&[0.1, 0.9, 0.05, 0.05], false), 2);
        assert_eq!(best_index(&[0.1, f64::NAN, 0.9, f64::NAN], true), 1);
        assert_eq!(best_index(&[0.1, f64::NAN], false), 1);
        assert_eq!(best_index(&[0.4], true), 0);
    }

    #[test]
    fn the_metrics_of_the_folds_are_numpys_mean_and_deviation_of_the_population() {
        let (means, stds) = mean_and_std(&[vec![1.0, 2.0], vec![3.0, 4.0]], 2);
        assert_eq!(means, vec![2.0, 3.0]);
        assert_eq!(stds, vec![1.0, 1.0]);
        let (means, stds) = mean_and_std(&[vec![0.5], vec![0.5], vec![0.5]], 1);
        assert_eq!((means, stds), (vec![0.5], vec![0.0]));
    }

    #[test]
    fn the_option_of_persisted_sub_models_is_of_any_case() {
        let part = ModelPart {
            extra: Vec::new(),
            best_model: Box::new(PipelineModel::of_stages(Vec::new())),
            sub_models: None,
        };
        let options = |k: &str, v: &str| HashMap::from([(k.to_string(), v.to_string())]);
        assert!(persist_sub_models(&part, &options("persistsubmodels", "True")).unwrap());
        assert!(!persist_sub_models(&part, &options("persistSubModels", "FALSE")).unwrap());
        assert!(!persist_sub_models(&part, &HashMap::new()).unwrap());
        let error = persist_sub_models(&part, &options("persistsubmodels", "maybe")).unwrap_err();
        assert!(error
            .to_string()
            .contains("persistSubModels option value maybe is invalid"));
    }

    #[test]
    fn param_maps_are_written_as_pyspark_writes_them_and_nested_validators_refused() {
        let svc = LinearSVC::new();
        let uid = svc.operator().uid.clone();
        let maps = ParamGridBuilder::new()
            .add_grid(&uid, "regParam", [MlValue::Double(0.1)])
            .build();
        let class = svc.operator().name.clone();
        let stages = vec![(uid.clone(), class.clone())];
        assert_eq!(
            param_maps_json(&maps, &stages).unwrap(),
            format!(
                "[[{{\"parent\":\"{uid}\",\"name\":\"regParam\",\"value\":\"0.1\",\"isJson\":true}}]]"
            )
        );
        assert_eq!(validate_params(&svc, &maps).unwrap(), stages);
        // PySpark's `repr` of the param: its parent, its name and the doc of its class's param.
        let other = LinearSVC::new();
        let foreign =
            vec![ParamMap::new().put(&other.operator().uid, "regParam", MlValue::Double(0.1))];
        let error = validate_params(&svc, &foreign).unwrap_err().to_string();
        let other_uid = other.operator().uid.clone();
        let expected = format!(
            "An extraneous Param was found: Param(parent='{other_uid}', name='regParam', \
             doc='regularization parameter (>= 0).')"
        );
        assert!(error.ends_with(&expected), "{error}");
        let unknown = vec![ParamMap::new().put("Other_1", "maxIter", MlValue::Int(1))];
        assert!(validate_params(&svc, &unknown)
            .unwrap_err()
            .to_string()
            .ends_with(
                "An extraneous Param was found: Param(parent='Other_1', name='maxIter', doc='')"
            ));
        let nested = CrossValidator::new().set_estimator(svc.clone());
        assert!(validate_params(&nested, &[])
            .unwrap_err()
            .to_string()
            .contains("PySpark does not support nested validator."));
        // A pipeline's nested stages are of its uid map.
        let pipeline = Pipeline::new().set_stages(vec![PipelineStage::estimator(svc)]);
        assert_eq!(
            validate_params(&pipeline, &maps)
                .unwrap()
                .iter()
                .map(|(uid, _)| uid.clone())
                .collect::<Vec<_>>(),
            vec![pipeline.uid().to_string(), uid]
        );
    }

    #[test]
    fn a_compound_estimator_with_a_uid_twice_fails_as_pyspark_does() {
        // As `getUidMap` of PySpark: the uids of the estimator and its nested stages are distinct,
        // and the message names the estimator's class and the uids in the order of the first.
        let svc = LinearSVC::new();
        let svc_uid = svc.operator().uid.clone();
        let pipeline = Pipeline::new().set_stages(vec![
            PipelineStage::estimator(svc.clone()),
            PipelineStage::estimator(svc.clone()),
        ]);
        let pipeline_uid = pipeline.uid().to_string();
        let expected = format!(
            "pyspark.ml.pipeline.Pipeline.load found a compound estimator with stages with \
             duplicate UIDs. List of UIDs: ['{pipeline_uid}', '{svc_uid}']."
        );
        let error = validate_params(&pipeline, &[]).unwrap_err().to_string();
        assert!(error.ends_with(&expected), "{error}");
        let error = uid_map(&pipeline).unwrap_err().to_string();
        assert!(error.ends_with(&expected), "{error}");
        // A validator's save refuses it before it writes anything, as its `validateParams` does.
        let cv = CrossValidator::new()
            .set_estimator(pipeline.clone())
            .set_evaluator(BinaryClassificationEvaluator::new());
        let error = cv
            .write()
            .session(&offline_session())
            .save("unused/path")
            .unwrap_err()
            .to_string();
        assert!(error.ends_with(&expected), "{error}");
        // Distinct stages of one class are no duplicates.
        let distinct = Pipeline::new().set_stages(vec![
            PipelineStage::estimator(svc.clone()),
            PipelineStage::estimator(LinearSVC::new()),
        ]);
        assert_eq!(uid_map(&distinct).unwrap().len(), 3);
    }

    #[test]
    fn validators_get_their_params_and_copy_their_estimators_of_extra() {
        let svc = LinearSVC::new().set_max_iter(5);
        let uid = svc.operator().uid.clone();
        let tvs = TrainValidationSplit::new()
            .set_estimator(svc)
            .set_train_ratio(0.5)
            .set_seed(3)
            .set_parallelism(2)
            .set_collect_sub_models(true);
        assert_eq!(
            (
                tvs.train_ratio(),
                tvs.seed(),
                tvs.parallelism(),
                tvs.collect_sub_models()
            ),
            (0.5, 3, 2, true)
        );
        assert!(tvs.uid().starts_with("TrainValidationSplit_"));
        let copy = tvs
            .copy(&ParamMap::new().put(&uid, "maxIter", MlValue::Int(9)))
            .unwrap();
        assert_eq!(copy.uid(), tvs.uid());
        let literal = copy
            .estimator()
            .unwrap()
            .params()
            .get_param("maxIter")
            .cloned();
        assert!(format!("{literal:?}").contains("Integer(9)"), "{literal:?}");
        let original = tvs
            .estimator()
            .unwrap()
            .params()
            .get_param("maxIter")
            .cloned();
        assert!(
            format!("{original:?}").contains("Integer(5)"),
            "{original:?}"
        );
        assert_eq!(TrainValidationSplit::default().train_ratio(), 0.75);
    }

    #[test]
    fn the_stages_of_a_pipeline_model_stage_are_nested_stages() {
        // A model stage is a `Box<dyn Model>`, whose stages are found through `dyn Model`: the
        // stages of a PipelineModel stage are nested stages, as `getAllNestedStages` of PySpark
        // finds them (the `Any` of the box itself is not the model's).
        let inner = PipelineModel::of_stages(Vec::new());
        let inner_uid = inner.uid().to_string();
        let outer = PipelineModel::of_stages(vec![PipelineStage::Model(Box::new(inner))]);
        let outer_uid = outer.uid().to_string();
        let mut uids = Vec::new();
        nested_stages(&PipelineStage::Model(Box::new(outer)), &mut uids).unwrap();
        let uids: Vec<String> = uids.into_iter().map(|(uid, _)| uid).collect();
        assert_eq!(uids, vec![outer_uid, inner_uid]);
    }

    #[test]
    fn the_nested_stages_of_a_one_vs_rest_are_its_classifier_and_its_models() {
        // As `getAllNestedStages` of PySpark's OneVsRest: its classifier, and of a model its
        // classifier and each of its models.
        let classifier = LinearSVC::new();
        let classifier_uid = classifier.operator().uid.clone();
        let estimator = OneVsRest::new().set_classifier(classifier.clone());
        let ovr_uid = estimator.operator().uid.clone();
        let mut uids = Vec::new();
        nested_stages(&PipelineStage::estimator(estimator), &mut uids).unwrap();
        let uids: Vec<String> = uids.into_iter().map(|(uid, _)| uid).collect();
        assert_eq!(uids, vec![ovr_uid.clone(), classifier_uid.clone()]);
        // A model of one class, whose model is a stand-in with no stages of its own.
        let class_model = PipelineModel::of_stages(Vec::new());
        let class_uid = class_model.uid().to_string();
        let model = OneVsRestModel {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.classification.OneVsRestModel",
                "OneVsRestModel_0123456789ab",
                OperatorType::Model,
            ),
            params: Params::new(),
            models: vec![Box::new(class_model)],
            session_id: String::new(),
            classifier: Some(classifier.into()),
        };
        let mut uids = Vec::new();
        nested_stages(&PipelineStage::Model(Box::new(model)), &mut uids).unwrap();
        let uids: Vec<String> = uids.into_iter().map(|(uid, _)| uid).collect();
        assert_eq!(
            uids,
            vec![
                "OneVsRestModel_0123456789ab".to_string(),
                classifier_uid,
                class_uid
            ]
        );
    }

    fn offline_session() -> SparkSession {
        // gRPC connects lazily, so the splits and the checks below build plans only.
        SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .expect("session")
    }

    #[test]
    fn a_validator_in_a_grid_sets_its_own_params_of_a_param_map() {
        // As `copy(extra)` of PySpark's validators: their own params, then their estimator's.
        let inner = CrossValidator::new()
            .set_estimator(LinearSVC::new())
            .set_num_folds(3)
            .set_seed(7);
        let uid = inner.uid().to_string();
        let map = ParamMap::new().put(&uid, "numFolds", MlValue::Int(2)).put(
            &uid,
            "seed",
            MlValue::Int(9),
        );
        let mut alone = inner.clone();
        alone.apply_param_map(&map).unwrap();
        assert_eq!((alone.num_folds(), alone.seed()), (2, 9));
        // And nested in a pipeline, as a stage of the estimator of an outer validator.
        let mut pipeline = Pipeline::new().set_stages(vec![PipelineStage::estimator(inner)]);
        pipeline.apply_param_map(&map).unwrap();
        let PipelineStage::Estimator(stage) = &pipeline.stages()[0] else {
            panic!("an estimator stage");
        };
        let nested = stage
            .as_any_stage()
            .and_then(|any| any.downcast_ref::<CrossValidator>())
            .expect("a cross validator");
        assert_eq!((nested.num_folds(), nested.seed()), (2, 9));
    }

    #[test]
    fn a_train_validation_split_in_a_grid_sets_its_own_params_of_a_param_map() {
        let tvs = TrainValidationSplit::new()
            .set_estimator(LinearSVC::new())
            .set_train_ratio(0.75);
        let uid = tvs.uid().to_string();
        let map = ParamMap::new()
            .put(&uid, "trainRatio", MlValue::Double(0.3))
            .put(&uid, "parallelism", MlValue::Int(2))
            .put(&uid, "collectSubModels", MlValue::Bool(true));
        let mut copy = tvs.clone();
        copy.apply_param_map(&map).unwrap();
        assert_eq!(
            (
                copy.train_ratio(),
                copy.parallelism(),
                copy.collect_sub_models()
            ),
            (0.3, 2, true)
        );
    }

    #[test]
    fn the_models_of_the_validators_apply_a_param_map_as_their_copy_in_pyspark() {
        // As `CrossValidatorModel.copy(extra)` and `TrainValidationSplitModel.copy(extra)`: a copy
        // of the best model of `extra`, PySpark's `assert self.subModels is not None` (a bare
        // AssertionError of a model of no sub-models, also of no extra), copies of the sub-models of
        // no extra (which fail as their own `copy` does), and the model's own params of `extra`, as
        // `_copyValues` sets the params of the model's uid.
        let assembler = VectorAssembler::new().set_output_col("v");
        let assembler_uid = assembler.operator().uid.clone();
        let best = || -> Box<dyn Model> {
            Box::new(PipelineModel::of_stages(vec![PipelineStage::transformer(
                assembler.clone(),
            )]))
        };
        let output_col = |model: &dyn Model| {
            let pipeline = model.downcast_ref::<PipelineModel>().expect("a pipeline");
            let PipelineStage::Transformer(stage) = &pipeline.stages()[0] else {
                panic!("a transformer stage");
            };
            stage.params().string_opt("outputCol").map(str::to_string)
        };
        let cv_model = |sub_models| CrossValidatorModel {
            operator: MlOperator::new(
                "org.apache.spark.ml.tuning.CrossValidatorModel",
                OperatorType::Model,
            ),
            params: Params::new(),
            best_model: best(),
            avg_metrics: vec![0.5],
            std_metrics: vec![0.0],
            sub_models,
            common: Common::new(0),
            num_folds: 3,
            fold_col: String::new(),
        };
        let tvs_model = |sub_models| TrainValidationSplitModel {
            operator: MlOperator::new(
                "org.apache.spark.ml.tuning.TrainValidationSplitModel",
                OperatorType::Model,
            ),
            params: Params::new(),
            best_model: best(),
            validation_metrics: vec![0.5],
            sub_models,
            common: Common::new(0),
            train_ratio: 0.75,
        };
        let of_best = ParamMap::new().put(&assembler_uid, "outputCol", MlValue::String("w".into()));
        let assertion = |result: Result<()>| {
            let error = result.expect_err("the assert of the sub-models");
            assert_eq!(error.kind, SparkErrorKind::AssertionError);
            assert!(
                error
                    .to_string()
                    .contains("(`assert self.subModels is not None`)"),
                "{error}"
            );
        };
        for map in [of_best.clone(), ParamMap::new()] {
            assertion(cv_model(None).apply_param_map(&map));
            assertion(tvs_model(None).apply_param_map(&map));
        }

        let mut model = cv_model(Some(vec![vec![best()], vec![best()]]));
        let uid = model.uid().to_string();
        let map = of_best
            .clone()
            .put(&uid, "numFolds", MlValue::Int(5))
            .put(&uid, "foldCol", MlValue::String("fold".into()))
            .put(&uid, "seed", MlValue::Int(9));
        model.apply_param_map(&map).unwrap();
        assert_eq!(
            (model.num_folds(), model.fold_col(), model.seed()),
            (5, "fold", 9)
        );
        assert_eq!(output_col(model.best_model()).as_deref(), Some("w"));
        let folds = model.sub_models().expect("sub-models");
        assert_eq!(folds.len(), 2);
        assert_eq!(output_col(folds[0][0].as_ref()).as_deref(), Some("v"));

        let mut model = tvs_model(Some(vec![best()]));
        let uid = model.uid().to_string();
        let map = of_best
            .clone()
            .put(&uid, "trainRatio", MlValue::Double(0.5))
            .put(&uid, "seed", MlValue::Int(9));
        model.apply_param_map(&map).unwrap();
        assert_eq!((model.train_ratio(), model.seed()), (0.5, 9));
        assert_eq!(output_col(model.best_model()).as_deref(), Some("w"));
        let sub_models = model.sub_models().expect("sub-models");
        assert_eq!(output_col(sub_models[0].as_ref()).as_deref(), Some("v"));

        // A sub-model of no sub-models of its own fails its copy, of PySpark's
        // `[sub_model.copy() for ...]`.
        assertion(cv_model(Some(vec![vec![Box::new(cv_model(None))]])).apply_param_map(&of_best));
        assertion(tvs_model(Some(vec![Box::new(tvs_model(None))])).apply_param_map(&of_best));
    }

    #[test]
    fn a_validator_of_no_folds_fails_as_pyspark_does() {
        // PySpark's `np.mean` and `np.std` of no fold are a scalar, which `list` fails on.
        let df = offline_session().range(10).expect("a range");
        let svc = LinearSVC::new();
        let uid = svc.operator().uid.clone();
        let grid = ParamGridBuilder::new()
            .add_grid(&uid, "regParam", [MlValue::Double(0.1)])
            .build();
        for (num_folds, fold_col) in [(-1, ""), (0, "fold"), (-2, "fold")] {
            let mut cv = CrossValidator::new()
                .set_estimator(svc.clone())
                .set_evaluator(BinaryClassificationEvaluator::new())
                .set_estimator_param_maps(grid.clone())
                .set_num_folds(num_folds)
                .set_fold_col(fold_col);
            let error = cv.fit_validator(&df).expect_err("a validator of no folds");
            assert!(
                error
                    .to_string()
                    .contains("'numpy.float64' object is not iterable"),
                "{num_folds} {fold_col}: {error}"
            );
        }
    }

    #[test]
    fn grid_values_are_converted_to_their_params_types_when_they_are_applied_and_saved() {
        let svc = LinearSVC::new();
        let uid = svc.operator().uid.clone();
        let maps = ParamGridBuilder::new()
            .add_grid(&uid, "regParam", [MlValue::Int(0), MlValue::Int(1)])
            .build();
        let mut copy = svc.clone();
        copy.apply_param_map(&maps[0]).unwrap();
        let literal = format!("{:?}", copy.params().get_param("regParam"));
        assert!(literal.contains("Double(0.0)"), "{literal}");
        // The saved grid holds the converted value, as the JSON of `0.0`.
        let stages = vec![(uid.clone(), svc.operator().name.clone())];
        let json = param_maps_json(&maps, &stages).unwrap();
        assert!(json.contains(r#""value":"0.0""#), "{json}");
        // A Boolean param refuses an int, as PySpark's `toBoolean` does.
        let map = ParamMap::new().put(&uid, "fitIntercept", MlValue::Int(0));
        let error = svc.clone().apply_param_map(&map).expect_err("an int");
        assert!(
            error
                .to_string()
                .contains("Boolean Param requires value of type bool. Found <class 'int'>."),
            "{error}"
        );
    }

    #[test]
    fn the_default_seed_of_a_class_of_validators_is_constant_in_a_process() {
        // PySpark's default seed is `hash(type(self).__name__)`: the same for each instance.
        assert_eq!(CrossValidator::new().seed(), CrossValidator::new().seed());
        assert_eq!(
            TrainValidationSplit::new().seed(),
            TrainValidationSplit::new().seed()
        );
        assert_eq!(CrossValidator::new().set_seed(3).seed(), 3);
    }
}
