// SPDX-License-Identifier: Apache-2.0
//! The typed accessors, summaries and persistence of the decision tree, random forest and
//! gradient-boosted tree classification models, and of their estimators, as PySpark's
//! `pyspark.ml.classification` and `pyspark.ml.tree` have them.
//!
//! A random forest's and a gradient-boosted model's trees are server-side objects of their own
//! ([`TreeEnsembleModel::tree_of`]). A fit of a random forest sets its training summary to the
//! `<model>.summary` object, with the training dataset itself (as PySpark's `_summary_dataset`
//! returns it), from which the server recreates the summary when it has lost it.

#[cfg(test)]
mod golden_tests;

use spark_connect_core::error::{Result, SparkError};

use super::{
    BinaryClassificationSummary, ClassificationModel, ClassificationSummary, MlSummary,
    ProbabilisticClassificationModel, TrainingSummary,
};
use crate::dataframe::DataFrame;
use crate::ml::tree_api::{as_f64s, fetch_value};
use crate::ml::{
    fetch, fit_via_command, java_string_hash, read_model, read_operator,
    DecisionTreeClassificationModel, DecisionTreeClassifier, DecisionTreeModel,
    DecisionTreeRegressionModel, Estimator, FetchArg, GBTClassificationModel, GBTClassifier,
    MlOperator, MlRef, MlTarget, Model, OperatorType, Params, PredictionModel,
    RandomForestClassificationModel, RandomForestClassifier, TreeEnsembleModel,
};
use crate::SparkSession;

const DECISION_TREE_CLASSIFIER: &str = "org.apache.spark.ml.classification.DecisionTreeClassifier";
const DECISION_TREE_MODEL: &str =
    "org.apache.spark.ml.classification.DecisionTreeClassificationModel";
const GBT_CLASSIFIER: &str = "org.apache.spark.ml.classification.GBTClassifier";
const GBT_MODEL: &str = "org.apache.spark.ml.classification.GBTClassificationModel";
const RANDOM_FOREST_CLASSIFIER: &str = "org.apache.spark.ml.classification.RandomForestClassifier";
const RANDOM_FOREST_MODEL: &str =
    "org.apache.spark.ml.classification.RandomForestClassificationModel";

/// The impurities of a classification tree (`_TreeClassifierParams.supportedImpurities`).
const CLASSIFIER_IMPURITIES: &[&str] = &["entropy", "gini"];
/// The impurity of a gradient-boosted tree (`_HasVarianceImpurity.supportedImpurities`).
const VARIANCE_IMPURITY: &[&str] = &["variance"];
/// The strategies of the features of each tree of an ensemble
/// (`_TreeEnsembleParams.supportedFeatureSubsetStrategies`).
const FEATURE_SUBSET_STRATEGIES: &[&str] = &["auto", "all", "onethird", "sqrt", "log2"];
/// The loss types of a gradient-boosted classifier (`_GBTClassifierParams.supportedLossTypes`).
const LOSS_TYPES: &[&str] = &["logistic"];

// The estimators: the typed fit, the persistence and the client-side lists of PySpark.

ml_params! {
    DecisionTreeClassifier {
        set_cache_node_ids(value: bool) => "cacheNodeIds", set_param_bool;
        cache_node_ids -> bool = |this| this.params.bool_or("cacheNodeIds", false);
        set_checkpoint_interval(value: i32) => "checkpointInterval", set_param_int32;
        checkpoint_interval -> i32 = |this| this.params.int_or("checkpointInterval", 10);
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        set_impurity(value: &str) => "impurity", set_param_string;
        impurity -> &str = |this| this.params.string_or("impurity", "gini");
        set_label_col(value: &str) => "labelCol", set_param_string;
        label_col -> &str = |this| this.params.string_or("labelCol", "label");
        set_leaf_col(value: &str) => "leafCol", set_param_string;
        leaf_col -> &str = |this| this.params.string_or("leafCol", "");
        set_max_bins(value: i32) => "maxBins", set_param_int32;
        max_bins -> i32 = |this| this.params.int_or("maxBins", 32);
        set_max_depth(value: i32) => "maxDepth", set_param_int32;
        max_depth -> i32 = |this| this.params.int_or("maxDepth", 5);
        set_max_memory_in_mb(value: i32) => "maxMemoryInMB", set_param_int32;
        max_memory_in_mb -> i32 = |this| this.params.int_or("maxMemoryInMB", 256);
        set_min_info_gain(value: f64) => "minInfoGain", set_param_double;
        min_info_gain -> f64 = |this| this.params.double_or("minInfoGain", 0.0);
        set_min_instances_per_node(value: i32) => "minInstancesPerNode", set_param_int32;
        min_instances_per_node -> i32 = |this| this.params.int_or("minInstancesPerNode", 1);
        set_min_weight_fraction_per_node(value: f64) => "minWeightFractionPerNode",
            set_param_double;
        min_weight_fraction_per_node -> f64 = |this| this.params.double_or(
            "minWeightFractionPerNode", 0.0);
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        set_probability_col(value: &str) => "probabilityCol", set_param_string;
        probability_col -> &str = |this| this.params.string_or("probabilityCol", "probability");
        set_prune_tree(value: bool) => "pruneTree", set_param_bool;
        prune_tree -> bool = |this| this.params.bool_or("pruneTree", true);
        set_raw_prediction_col(value: &str) => "rawPredictionCol", set_param_string;
        raw_prediction_col -> &str = |this| this.params.string_or("rawPredictionCol",
            "rawPrediction");
        set_seed(value: i64) => "seed", set_param_int_narrow;
        seed -> i64 = |this| this.params.long_opt("seed").unwrap_or(java_string_hash(
            "org.apache.spark.ml.classification.DecisionTreeClassifier"));
        set_thresholds(value: impl IntoIterator<Item = f64>) => "thresholds",
            set_param_double_array;
        thresholds -> Option<Vec<f64>> = |this| this.params.double_array_opt("thresholds");
        set_weight_col(value: &str) => "weightCol", set_param_string;
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol");
    }
}

impl DecisionTreeClassifier {
    /// Fit the decision tree classification model of `df` (`fit`), typed.
    pub fn fit(&mut self, df: &DataFrame) -> Result<DecisionTreeClassificationModel> {
        self.ensure_operator();
        let info = fit_via_command(&self.operator, &self.params, df)?;
        self.model_from_info(info, &df.session)
    }

    /// The estimator that `path` was saved with (`DecisionTreeClassifier.load`).
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (operator, params) = read_operator(
            session,
            DECISION_TREE_CLASSIFIER,
            OperatorType::Estimator,
            path,
        )?;
        Ok(Self { operator, params })
    }

    /// The impurities that the tree supports (`supportedImpurities`).
    pub fn supported_impurities(&self) -> &'static [&'static str] {
        CLASSIFIER_IMPURITIES
    }
}

ml_params! {
    RandomForestClassifier {
        set_bootstrap(value: bool) => "bootstrap", set_param_bool;
        bootstrap -> bool = |this| this.params.bool_or("bootstrap", true);
        set_cache_node_ids(value: bool) => "cacheNodeIds", set_param_bool;
        cache_node_ids -> bool = |this| this.params.bool_or("cacheNodeIds", false);
        set_checkpoint_interval(value: i32) => "checkpointInterval", set_param_int32;
        checkpoint_interval -> i32 = |this| this.params.int_or("checkpointInterval", 10);
        set_feature_subset_strategy(value: &str) => "featureSubsetStrategy", set_param_string;
        feature_subset_strategy -> &str = |this| this.params.string_or("featureSubsetStrategy",
            "auto");
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        set_impurity(value: &str) => "impurity", set_param_string;
        impurity -> &str = |this| this.params.string_or("impurity", "gini");
        set_label_col(value: &str) => "labelCol", set_param_string;
        label_col -> &str = |this| this.params.string_or("labelCol", "label");
        set_leaf_col(value: &str) => "leafCol", set_param_string;
        leaf_col -> &str = |this| this.params.string_or("leafCol", "");
        set_max_bins(value: i32) => "maxBins", set_param_int32;
        max_bins -> i32 = |this| this.params.int_or("maxBins", 32);
        set_max_depth(value: i32) => "maxDepth", set_param_int32;
        max_depth -> i32 = |this| this.params.int_or("maxDepth", 5);
        set_max_memory_in_mb(value: i32) => "maxMemoryInMB", set_param_int32;
        max_memory_in_mb -> i32 = |this| this.params.int_or("maxMemoryInMB", 256);
        set_min_info_gain(value: f64) => "minInfoGain", set_param_double;
        min_info_gain -> f64 = |this| this.params.double_or("minInfoGain", 0.0);
        set_min_instances_per_node(value: i32) => "minInstancesPerNode", set_param_int32;
        min_instances_per_node -> i32 = |this| this.params.int_or("minInstancesPerNode", 1);
        set_min_weight_fraction_per_node(value: f64) => "minWeightFractionPerNode",
            set_param_double;
        min_weight_fraction_per_node -> f64 = |this| this.params.double_or(
            "minWeightFractionPerNode", 0.0);
        set_num_trees(value: i32) => "numTrees", set_param_int32;
        num_trees -> i32 = |this| this.params.int_or("numTrees", 20);
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        set_probability_col(value: &str) => "probabilityCol", set_param_string;
        probability_col -> &str = |this| this.params.string_or("probabilityCol", "probability");
        set_prune_tree(value: bool) => "pruneTree", set_param_bool;
        prune_tree -> bool = |this| this.params.bool_or("pruneTree", true);
        set_raw_prediction_col(value: &str) => "rawPredictionCol", set_param_string;
        raw_prediction_col -> &str = |this| this.params.string_or("rawPredictionCol",
            "rawPrediction");
        set_seed(value: i64) => "seed", set_param_int_narrow;
        seed -> i64 = |this| this.params.long_opt("seed").unwrap_or(java_string_hash(
            "org.apache.spark.ml.classification.RandomForestClassifier"));
        set_subsampling_rate(value: f64) => "subsamplingRate", set_param_double;
        subsampling_rate -> f64 = |this| this.params.double_or("subsamplingRate", 1.0);
        set_thresholds(value: impl IntoIterator<Item = f64>) => "thresholds",
            set_param_double_array;
        thresholds -> Option<Vec<f64>> = |this| this.params.double_array_opt("thresholds");
        set_weight_col(value: &str) => "weightCol", set_param_string;
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol");
    }
}

impl RandomForestClassifier {
    /// Fit the random forest classification model of `df` (`fit`), typed. Its training summary
    /// is the summary of its `<model>.summary` object, which the server recreates from `df`.
    pub fn fit(&mut self, df: &DataFrame) -> Result<RandomForestClassificationModel> {
        self.ensure_operator();
        let info = fit_via_command(&self.operator, &self.params, df)?;
        let mut model = self.model_from_info(info, &df.session)?;
        model.training_summary = Some(
            MlTarget::object(&model.cached)
                .child("summary")
                .with_summary_dataset(df),
        );
        Ok(model)
    }

    /// The estimator that `path` was saved with (`RandomForestClassifier.load`).
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (operator, params) = read_operator(
            session,
            RANDOM_FOREST_CLASSIFIER,
            OperatorType::Estimator,
            path,
        )?;
        Ok(Self { operator, params })
    }

    /// The strategies of the features of each tree (`supportedFeatureSubsetStrategies`).
    pub fn supported_feature_subset_strategies(&self) -> &'static [&'static str] {
        FEATURE_SUBSET_STRATEGIES
    }

    /// The impurities that the trees support (`supportedImpurities`).
    pub fn supported_impurities(&self) -> &'static [&'static str] {
        CLASSIFIER_IMPURITIES
    }
}

ml_params! {
    GBTClassifier {
        set_cache_node_ids(value: bool) => "cacheNodeIds", set_param_bool;
        cache_node_ids -> bool = |this| this.params.bool_or("cacheNodeIds", false);
        set_checkpoint_interval(value: i32) => "checkpointInterval", set_param_int32;
        checkpoint_interval -> i32 = |this| this.params.int_or("checkpointInterval", 10);
        set_feature_subset_strategy(value: &str) => "featureSubsetStrategy", set_param_string;
        feature_subset_strategy -> &str = |this| this.params.string_or("featureSubsetStrategy",
            "all");
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        set_impurity(value: &str) => "impurity", set_param_string;
        impurity -> &str = |this| this.params.string_or("impurity", "variance");
        set_label_col(value: &str) => "labelCol", set_param_string;
        label_col -> &str = |this| this.params.string_or("labelCol", "label");
        set_leaf_col(value: &str) => "leafCol", set_param_string;
        leaf_col -> &str = |this| this.params.string_or("leafCol", "");
        set_loss_type(value: &str) => "lossType", set_param_string;
        loss_type -> &str = |this| this.params.string_or("lossType", "logistic");
        set_max_bins(value: i32) => "maxBins", set_param_int32;
        max_bins -> i32 = |this| this.params.int_or("maxBins", 32);
        set_max_depth(value: i32) => "maxDepth", set_param_int32;
        max_depth -> i32 = |this| this.params.int_or("maxDepth", 5);
        set_max_iter(value: i32) => "maxIter", set_param_int32;
        max_iter -> i32 = |this| this.params.int_or("maxIter", 20);
        set_max_memory_in_mb(value: i32) => "maxMemoryInMB", set_param_int32;
        max_memory_in_mb -> i32 = |this| this.params.int_or("maxMemoryInMB", 256);
        set_min_info_gain(value: f64) => "minInfoGain", set_param_double;
        min_info_gain -> f64 = |this| this.params.double_or("minInfoGain", 0.0);
        set_min_instances_per_node(value: i32) => "minInstancesPerNode", set_param_int32;
        min_instances_per_node -> i32 = |this| this.params.int_or("minInstancesPerNode", 1);
        set_min_weight_fraction_per_node(value: f64) => "minWeightFractionPerNode",
            set_param_double;
        min_weight_fraction_per_node -> f64 = |this| this.params.double_or(
            "minWeightFractionPerNode", 0.0);
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        set_probability_col(value: &str) => "probabilityCol", set_param_string;
        probability_col -> &str = |this| this.params.string_or("probabilityCol", "probability");
        set_raw_prediction_col(value: &str) => "rawPredictionCol", set_param_string;
        raw_prediction_col -> &str = |this| this.params.string_or("rawPredictionCol",
            "rawPrediction");
        set_seed(value: i64) => "seed", set_param_int_narrow;
        seed -> i64 = |this| this.params.long_opt("seed").unwrap_or(java_string_hash(
            "org.apache.spark.ml.classification.GBTClassifier"));
        set_step_size(value: f64) => "stepSize", set_param_double;
        step_size -> f64 = |this| this.params.double_or("stepSize", 0.1);
        set_subsampling_rate(value: f64) => "subsamplingRate", set_param_double;
        subsampling_rate -> f64 = |this| this.params.double_or("subsamplingRate", 1.0);
        set_thresholds(value: impl IntoIterator<Item = f64>) => "thresholds",
            set_param_double_array;
        thresholds -> Option<Vec<f64>> = |this| this.params.double_array_opt("thresholds");
        set_validation_indicator_col(value: &str) => "validationIndicatorCol", set_param_string;
        validation_indicator_col -> Option<&str> = |this| this.params.string_opt(
            "validationIndicatorCol");
        validation_tol -> f64 = |this| this.params.double_or("validationTol", 0.01) =>
            "validationTol";
        set_weight_col(value: &str) => "weightCol", set_param_string;
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol");
    }
}

impl GBTClassifier {
    /// Fit the gradient-boosted classification model of `df` (`fit`), typed.
    pub fn fit(&mut self, df: &DataFrame) -> Result<GBTClassificationModel> {
        self.ensure_operator();
        let info = fit_via_command(&self.operator, &self.params, df)?;
        self.model_from_info(info, &df.session)
    }

    /// The estimator that `path` was saved with (`GBTClassifier.load`).
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (operator, params) =
            read_operator(session, GBT_CLASSIFIER, OperatorType::Estimator, path)?;
        Ok(Self { operator, params })
    }

    /// The strategies of the features of each tree (`supportedFeatureSubsetStrategies`).
    pub fn supported_feature_subset_strategies(&self) -> &'static [&'static str] {
        FEATURE_SUBSET_STRATEGIES
    }

    /// The impurity that the trees support (`supportedImpurities`).
    pub fn supported_impurities(&self) -> &'static [&'static str] {
        VARIANCE_IMPURITY
    }

    /// The loss types that the classifier supports (`supportedLossTypes`).
    pub fn supported_loss_types(&self) -> &'static [&'static str] {
        LOSS_TYPES
    }
}

// The decision tree classification model.

ml_params! {
    DecisionTreeClassificationModel {
        cache_node_ids -> bool = |this| this.params.bool_or("cacheNodeIds", false) =>
            "cacheNodeIds";
        checkpoint_interval -> i32 = |this| this.params.int_or("checkpointInterval", 10) =>
            "checkpointInterval";
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        impurity -> &str = |this| this.params.string_or("impurity", "gini") => "impurity";
        label_col -> &str = |this| this.params.string_or("labelCol", "label") => "labelCol";
        set_leaf_col(value: &str) => "leafCol", set_param_string;
        leaf_col -> &str = |this| this.params.string_or("leafCol", "");
        max_bins -> i32 = |this| this.params.int_or("maxBins", 32) => "maxBins";
        max_depth -> i32 = |this| this.params.int_or("maxDepth", 5) => "maxDepth";
        max_memory_in_mb -> i32 = |this| this.params.int_or("maxMemoryInMB", 256) =>
            "maxMemoryInMB";
        min_info_gain -> f64 = |this| this.params.double_or("minInfoGain", 0.0) => "minInfoGain";
        min_instances_per_node -> i32 = |this| this.params.int_or("minInstancesPerNode", 1) =>
            "minInstancesPerNode";
        min_weight_fraction_per_node -> f64 = |this| this.params.double_or(
            "minWeightFractionPerNode", 0.0) => "minWeightFractionPerNode";
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        set_probability_col(value: &str) => "probabilityCol", set_param_string;
        probability_col -> &str = |this| this.params.string_or("probabilityCol", "probability");
        prune_tree -> bool = |this| this.params.bool_or("pruneTree", true) => "pruneTree";
        set_raw_prediction_col(value: &str) => "rawPredictionCol", set_param_string;
        raw_prediction_col -> &str = |this| this.params.string_or("rawPredictionCol",
            "rawPrediction");
        seed -> i64 = |this| this.params.long_opt("seed").unwrap_or(java_string_hash(
            "org.apache.spark.ml.classification.DecisionTreeClassificationModel")) => "seed";
        set_thresholds(value: impl IntoIterator<Item = f64>) => "thresholds",
            set_param_double_array;
        thresholds -> Option<Vec<f64>> = |this| this.params.double_array_opt("thresholds");
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol") => "weightCol";
    }
}

impl DecisionTreeClassificationModel {
    /// The model of the server-side object `ml_ref`, with the params `params`, which the trees of
    /// a forest are built of. It has a new uid.
    pub(crate) fn from_ml_ref(ml_ref: MlRef, params: Params) -> Self {
        Self {
            operator: MlOperator::new(DECISION_TREE_MODEL, OperatorType::Model),
            params,
            cached: ml_ref,
        }
    }

    /// The model that `path` was saved with (`DecisionTreeClassificationModel.load`).
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (object, operator, params) = read_model(session, DECISION_TREE_MODEL, path)?;
        Ok(Self {
            operator,
            ..Self::from_ml_ref(object, params)
        })
    }

    /// The impurities that the tree supports (`supportedImpurities`).
    pub fn supported_impurities(&self) -> &'static [&'static str] {
        CLASSIFIER_IMPURITIES
    }
}

impl DecisionTreeModel for DecisionTreeClassificationModel {}

impl ClassificationModel for DecisionTreeClassificationModel {}

impl ProbabilisticClassificationModel for DecisionTreeClassificationModel {}

impl PredictionModel for DecisionTreeClassificationModel {}

// The random forest classification model and its summaries.

ml_params! {
    RandomForestClassificationModel {
        bootstrap -> bool = |this| this.params.bool_or("bootstrap", true) => "bootstrap";
        cache_node_ids -> bool = |this| this.params.bool_or("cacheNodeIds", false) =>
            "cacheNodeIds";
        checkpoint_interval -> i32 = |this| this.params.int_or("checkpointInterval", 10) =>
            "checkpointInterval";
        feature_subset_strategy -> &str = |this| this.params.string_or("featureSubsetStrategy",
            "auto") => "featureSubsetStrategy";
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        impurity -> &str = |this| this.params.string_or("impurity", "gini") => "impurity";
        label_col -> &str = |this| this.params.string_or("labelCol", "label") => "labelCol";
        set_leaf_col(value: &str) => "leafCol", set_param_string;
        leaf_col -> &str = |this| this.params.string_or("leafCol", "");
        max_bins -> i32 = |this| this.params.int_or("maxBins", 32) => "maxBins";
        max_depth -> i32 = |this| this.params.int_or("maxDepth", 5) => "maxDepth";
        max_memory_in_mb -> i32 = |this| this.params.int_or("maxMemoryInMB", 256) =>
            "maxMemoryInMB";
        min_info_gain -> f64 = |this| this.params.double_or("minInfoGain", 0.0) => "minInfoGain";
        min_instances_per_node -> i32 = |this| this.params.int_or("minInstancesPerNode", 1) =>
            "minInstancesPerNode";
        min_weight_fraction_per_node -> f64 = |this| this.params.double_or(
            "minWeightFractionPerNode", 0.0) => "minWeightFractionPerNode";
        num_trees -> i32 = |this| this.params.int_or("numTrees", 20) => "numTrees";
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        set_probability_col(value: &str) => "probabilityCol", set_param_string;
        probability_col -> &str = |this| this.params.string_or("probabilityCol", "probability");
        prune_tree -> bool = |this| this.params.bool_or("pruneTree", true) => "pruneTree";
        set_raw_prediction_col(value: &str) => "rawPredictionCol", set_param_string;
        raw_prediction_col -> &str = |this| this.params.string_or("rawPredictionCol",
            "rawPrediction");
        seed -> i64 = |this| this.params.long_opt("seed").unwrap_or(java_string_hash(
            "org.apache.spark.ml.classification.RandomForestClassificationModel")) => "seed";
        subsampling_rate -> f64 = |this| this.params.double_or("subsamplingRate", 1.0) =>
            "subsamplingRate";
        set_thresholds(value: impl IntoIterator<Item = f64>) => "thresholds",
            set_param_double_array;
        thresholds -> Option<Vec<f64>> = |this| this.params.double_array_opt("thresholds");
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol") => "weightCol";
    }
}

impl RandomForestClassificationModel {
    /// The model of the server-side object `ml_ref`, with the params `params`. It has a new uid
    /// and no training summary.
    pub(crate) fn from_ml_ref(ml_ref: MlRef, params: Params) -> Self {
        Self {
            operator: MlOperator::new(RANDOM_FOREST_MODEL, OperatorType::Model),
            params,
            cached: ml_ref,
            training_summary: None,
        }
    }

    /// The model that `path` was saved with (`RandomForestClassificationModel.load`). It has no
    /// training summary.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (object, operator, params) = read_model(session, RANDOM_FOREST_MODEL, path)?;
        Ok(Self {
            operator,
            ..Self::from_ml_ref(object, params)
        })
    }

    /// Whether the model has a training summary (`hasSummary`): only a model that a fit returned
    /// has one.
    pub fn has_summary(&self) -> bool {
        self.training_summary.is_some()
    }

    /// The training summary (`summary`): the binary one when the model has two classes or fewer,
    /// as PySpark chooses it by `numClasses`. Errors when the model has no training summary.
    pub fn summary(&self) -> Result<RandomForestTrainingSummary> {
        let target = self.training_summary.clone().ok_or_else(|| {
            SparkError::value_msg("the random forest model has no training summary")
        })?;
        if self.num_classes()? <= 2 {
            Ok(RandomForestTrainingSummary::Binary(
                BinaryRandomForestClassificationTrainingSummary::new(target),
            ))
        } else {
            Ok(RandomForestTrainingSummary::Multiclass(
                RandomForestClassificationTrainingSummary::new(target),
            ))
        }
    }

    /// The summary of the model on `dataset` (`evaluate`), of the class that `summary` chooses.
    pub fn evaluate(&self, dataset: &DataFrame) -> Result<RandomForestSummary> {
        let model = self.model_target()?;
        let id =
            fetch(&model, "evaluate", vec![FetchArg::Input(dataset.clone())])?.into_string()?;
        let target = MlTarget::unowned(model.session(), &id);
        if self.num_classes()? <= 2 {
            Ok(RandomForestSummary::Binary(
                BinaryRandomForestClassificationSummary::new(target),
            ))
        } else {
            Ok(RandomForestSummary::Multiclass(
                RandomForestClassificationSummary::new(target),
            ))
        }
    }

    /// The strategies of the features of each tree (`supportedFeatureSubsetStrategies`).
    pub fn supported_feature_subset_strategies(&self) -> &'static [&'static str] {
        FEATURE_SUBSET_STRATEGIES
    }

    /// The impurities that the trees support (`supportedImpurities`).
    pub fn supported_impurities(&self) -> &'static [&'static str] {
        CLASSIFIER_IMPURITIES
    }
}

impl TreeEnsembleModel for RandomForestClassificationModel {
    type Tree = DecisionTreeClassificationModel;

    fn tree_of(&self, tree: MlRef) -> DecisionTreeClassificationModel {
        DecisionTreeClassificationModel::from_ml_ref(tree, Params::new())
    }
}

impl ClassificationModel for RandomForestClassificationModel {}

impl ProbabilisticClassificationModel for RandomForestClassificationModel {}

impl PredictionModel for RandomForestClassificationModel {}

/// The training summary of a random forest classification model, of the class that PySpark
/// chooses by `numClasses`: binary for two classes or fewer.
#[derive(Debug, Clone)]
pub enum RandomForestTrainingSummary {
    /// The summary of a binary model (`BinaryRandomForestClassificationTrainingSummary`).
    Binary(BinaryRandomForestClassificationTrainingSummary),
    /// The summary of a model with more than two classes
    /// (`RandomForestClassificationTrainingSummary`).
    Multiclass(RandomForestClassificationTrainingSummary),
}

/// The summary of a random forest classification model on a dataset, of the class that PySpark
/// chooses by `numClasses`: binary for two classes or fewer.
#[derive(Debug, Clone)]
pub enum RandomForestSummary {
    /// The summary of a binary model (`BinaryRandomForestClassificationSummary`).
    Binary(BinaryRandomForestClassificationSummary),
    /// The summary of a model with more than two classes (`RandomForestClassificationSummary`).
    Multiclass(RandomForestClassificationSummary),
}

/// The summary of a random forest classification model on a dataset
/// (`RandomForestClassificationSummary`): the members of a classification summary.
#[derive(Debug, Clone)]
pub struct RandomForestClassificationSummary {
    target: MlTarget,
}

impl RandomForestClassificationSummary {
    /// The summary of `target`, the target of the summary's server-side object (as PySpark's
    /// summary constructors take the object of the summary).
    pub fn new(target: MlTarget) -> Self {
        Self { target }
    }
}

impl MlSummary for RandomForestClassificationSummary {
    fn summary_target(&self) -> &MlTarget {
        &self.target
    }
}

impl ClassificationSummary for RandomForestClassificationSummary {}

/// The training summary of a random forest classification model with more than two classes
/// (`RandomForestClassificationTrainingSummary`).
#[derive(Debug, Clone)]
pub struct RandomForestClassificationTrainingSummary {
    target: MlTarget,
}

impl RandomForestClassificationTrainingSummary {
    /// The summary of `target`, the target of the summary's server-side object (as PySpark's
    /// summary constructors take the object of the summary).
    pub fn new(target: MlTarget) -> Self {
        Self { target }
    }
}

impl MlSummary for RandomForestClassificationTrainingSummary {
    fn summary_target(&self) -> &MlTarget {
        &self.target
    }
}

impl ClassificationSummary for RandomForestClassificationTrainingSummary {}

impl TrainingSummary for RandomForestClassificationTrainingSummary {}

/// The summary of a binary random forest classification model on a dataset
/// (`BinaryRandomForestClassificationSummary`): the members of a binary classification summary.
#[derive(Debug, Clone)]
pub struct BinaryRandomForestClassificationSummary {
    target: MlTarget,
}

impl BinaryRandomForestClassificationSummary {
    /// The summary of `target`, the target of the summary's server-side object (as PySpark's
    /// summary constructors take the object of the summary).
    pub fn new(target: MlTarget) -> Self {
        Self { target }
    }
}

impl MlSummary for BinaryRandomForestClassificationSummary {
    fn summary_target(&self) -> &MlTarget {
        &self.target
    }
}

impl ClassificationSummary for BinaryRandomForestClassificationSummary {}

impl BinaryClassificationSummary for BinaryRandomForestClassificationSummary {}

/// The training summary of a binary random forest classification model
/// (`BinaryRandomForestClassificationTrainingSummary`).
#[derive(Debug, Clone)]
pub struct BinaryRandomForestClassificationTrainingSummary {
    target: MlTarget,
}

impl BinaryRandomForestClassificationTrainingSummary {
    /// The summary of `target`, the target of the summary's server-side object (as PySpark's
    /// summary constructors take the object of the summary).
    pub fn new(target: MlTarget) -> Self {
        Self { target }
    }
}

impl MlSummary for BinaryRandomForestClassificationTrainingSummary {
    fn summary_target(&self) -> &MlTarget {
        &self.target
    }
}

impl ClassificationSummary for BinaryRandomForestClassificationTrainingSummary {}

impl BinaryClassificationSummary for BinaryRandomForestClassificationTrainingSummary {}

impl TrainingSummary for BinaryRandomForestClassificationTrainingSummary {}

// The gradient-boosted classification model.

ml_params! {
    GBTClassificationModel {
        cache_node_ids -> bool = |this| this.params.bool_or("cacheNodeIds", false) =>
            "cacheNodeIds";
        checkpoint_interval -> i32 = |this| this.params.int_or("checkpointInterval", 10) =>
            "checkpointInterval";
        feature_subset_strategy -> &str = |this| this.params.string_or("featureSubsetStrategy",
            "all") => "featureSubsetStrategy";
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        impurity -> &str = |this| this.params.string_or("impurity", "variance") => "impurity";
        label_col -> &str = |this| this.params.string_or("labelCol", "label") => "labelCol";
        set_leaf_col(value: &str) => "leafCol", set_param_string;
        leaf_col -> &str = |this| this.params.string_or("leafCol", "");
        loss_type -> &str = |this| this.params.string_or("lossType", "logistic") => "lossType";
        max_bins -> i32 = |this| this.params.int_or("maxBins", 32) => "maxBins";
        max_depth -> i32 = |this| this.params.int_or("maxDepth", 5) => "maxDepth";
        max_iter -> i32 = |this| this.params.int_or("maxIter", 20) => "maxIter";
        max_memory_in_mb -> i32 = |this| this.params.int_or("maxMemoryInMB", 256) =>
            "maxMemoryInMB";
        min_info_gain -> f64 = |this| this.params.double_or("minInfoGain", 0.0) => "minInfoGain";
        min_instances_per_node -> i32 = |this| this.params.int_or("minInstancesPerNode", 1) =>
            "minInstancesPerNode";
        min_weight_fraction_per_node -> f64 = |this| this.params.double_or(
            "minWeightFractionPerNode", 0.0) => "minWeightFractionPerNode";
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        set_probability_col(value: &str) => "probabilityCol", set_param_string;
        probability_col -> &str = |this| this.params.string_or("probabilityCol", "probability");
        set_raw_prediction_col(value: &str) => "rawPredictionCol", set_param_string;
        raw_prediction_col -> &str = |this| this.params.string_or("rawPredictionCol",
            "rawPrediction");
        seed -> i64 = |this| this.params.long_opt("seed").unwrap_or(java_string_hash(
            "org.apache.spark.ml.classification.GBTClassificationModel")) => "seed";
        step_size -> f64 = |this| this.params.double_or("stepSize", 0.1) => "stepSize";
        subsampling_rate -> f64 = |this| this.params.double_or("subsamplingRate", 1.0) =>
            "subsamplingRate";
        set_thresholds(value: impl IntoIterator<Item = f64>) => "thresholds",
            set_param_double_array;
        thresholds -> Option<Vec<f64>> = |this| this.params.double_array_opt("thresholds");
        validation_indicator_col -> Option<&str> = |this| this.params.string_opt(
            "validationIndicatorCol") => "validationIndicatorCol";
        validation_tol -> f64 = |this| this.params.double_or("validationTol", 0.01) =>
            "validationTol";
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol") => "weightCol";
    }
}

impl GBTClassificationModel {
    /// The model of the server-side object `ml_ref`, with the params `params`. It has a new uid.
    pub(crate) fn from_ml_ref(ml_ref: MlRef, params: Params) -> Self {
        Self {
            operator: MlOperator::new(GBT_MODEL, OperatorType::Model),
            params,
            cached: ml_ref,
        }
    }

    /// The model that `path` was saved with (`GBTClassificationModel.load`).
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (object, operator, params) = read_model(session, GBT_MODEL, path)?;
        Ok(Self {
            operator,
            ..Self::from_ml_ref(object, params)
        })
    }

    /// The loss of the model after each boosting iteration, on `dataset`
    /// (`evaluateEachIteration`).
    pub fn evaluate_each_iteration(&self, dataset: &DataFrame) -> Result<Vec<f64>> {
        let value = fetch_value(
            &self.model_target()?,
            "evaluateEachIteration",
            vec![FetchArg::Input(dataset.clone())],
        )?;
        as_f64s("evaluateEachIteration", value)
    }

    /// The strategies of the features of each tree (`supportedFeatureSubsetStrategies`).
    pub fn supported_feature_subset_strategies(&self) -> &'static [&'static str] {
        FEATURE_SUBSET_STRATEGIES
    }

    /// The impurity that the trees support (`supportedImpurities`).
    pub fn supported_impurities(&self) -> &'static [&'static str] {
        VARIANCE_IMPURITY
    }

    /// The loss types that the model supports (`supportedLossTypes`).
    pub fn supported_loss_types(&self) -> &'static [&'static str] {
        LOSS_TYPES
    }
}

impl TreeEnsembleModel for GBTClassificationModel {
    type Tree = DecisionTreeRegressionModel;

    fn tree_of(&self, tree: MlRef) -> DecisionTreeRegressionModel {
        DecisionTreeRegressionModel::from_ml_ref(tree, Params::new())
    }
}

impl ClassificationModel for GBTClassificationModel {}

impl ProbabilisticClassificationModel for GBTClassificationModel {}

impl PredictionModel for GBTClassificationModel {}
