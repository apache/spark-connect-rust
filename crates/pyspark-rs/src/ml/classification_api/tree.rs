// SPDX-License-Identifier: Apache-2.0
//! The typed models and summaries of the decision tree, random forest and gradient-boosted tree
//! classifiers: their models and the random forest summaries.
use super::*;

/// The number of classes of the model of `target`, as its `numClasses` (a fetch).
fn fetch_num_classes(py: Python<'_>, target: &spark_connect::ml::MlTarget) -> PyResult<i64> {
    let out = object::fetch_py(py, target, "numClasses", &[])?;
    object::fetch_output_to_py(py, out)?.bind(py).extract()
}

/// The trees of a random forest, as PySpark's `trees` makes them: of the tree classifier's model
/// class, each of the constructor of the class and of its server-side object
/// (`DecisionTreeClassificationModel(RemoteModelRef(m))`).
fn random_forest_trees(py: Python<'_>, model: &object::ModelCore) -> PyResult<Py<PyAny>> {
    let target = model.target()?;
    let objects = object::fetch_py(py, &target, "trees", &[])?
        .into_objects()
        .to_pyerr()?;
    let trees = objects
        .into_iter()
        .map(|object| {
            PyDecisionTreeClassificationModel::of_constructor(py, |_| Ok(object)).map(Bound::unbind)
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new(py, trees)?.into_any().unbind())
}

/// The trees of a gradient-boosted tree classifier, as PySpark's `trees` makes them: of the
/// regression tree model class, each of the constructor of the class and of its server-side
/// object (`DecisionTreeRegressionModel(RemoteModelRef(m))`).
fn gbt_trees(py: Python<'_>, model: &object::ModelCore) -> PyResult<Py<PyAny>> {
    let target = model.target()?;
    let objects = object::fetch_py(py, &target, "trees", &[])?
        .into_objects()
        .to_pyerr()?;
    let trees = objects
        .into_iter()
        .map(|object| {
            crate::ml::regression_api::PyDecisionTreeRegressionModel::of_constructor(py, |_| {
                Ok(object)
            })
            .map(Bound::unbind)
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new(py, trees)?.into_any().unbind())
}

/// The training summary object of a random forest model, as its `_summaryCls` makes it: the
/// binary one of two classes or fewer, of the `numClasses` that the model fetches.
fn random_forest_training_summary(
    py: Python<'_>,
    model: &object::ModelCore,
    target: spark_connect::ml::MlTarget,
) -> PyResult<Py<PyAny>> {
    if fetch_num_classes(py, &model.target()?)? <= 2 {
        Ok(crate::ml::object::new_py(
            py,
            PyBinaryRandomForestClassificationTrainingSummary::from(target),
        )?
        .into_any())
    } else {
        Ok(crate::ml::object::new_py(
            py,
            PyRandomForestClassificationTrainingSummary::from(target),
        )?
        .into_any())
    }
}

native_model! {
    PyDecisionTreeClassificationModel = "DecisionTreeClassificationModel"
        in "pyspark.ml.connect.classification",
        jvm: "org.apache.spark.ml.classification.DecisionTreeClassificationModel";
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
    setters: [
        "setFeaturesCol" => (set_features_col, "featuresCol"),
        "setLeafCol" => (set_leaf_col, "leafCol"),
        "setPredictionCol" => (set_prediction_col, "predictionCol"),
        "setProbabilityCol" => (set_probability_col, "probabilityCol"),
        "setRawPredictionCol" => (set_raw_prediction_col, "rawPredictionCol"),
        "setThresholds" => (set_thresholds, "thresholds"),
    ];
    properties: [
        "depth" => (depth, "depth"),
        "featureImportances" => (feature_importances, "featureImportances"),
        "numClasses" => (num_classes, "numClasses"),
        "numFeatures" => (num_features, "numFeatures"),
        "numNodes" => (num_nodes, "numNodes"),
        "toDebugString" => (to_debug_string, "toDebugString"),
    ];
    methods: [
        "predict" => (predict, "predict", (value)),
        "predictLeaf" => (predict_leaf, "predictLeaf", (value)),
        "predictProbability" => (predict_probability, "predictProbability", (value)),
        "predictRaw" => (predict_raw, "predictRaw", (value)),
    ];
    relations: [];
    relation_methods: [];
    summary: None;
    cached: [];

    /// The supportedImpurities of the class, as PySpark's class attribute.
    #[classattr]
    #[allow(non_snake_case)]
    fn supportedImpurities() -> Vec<&'static str> {
        vec!["entropy", "gini"]
    }
}

native_model! {
    PyRandomForestClassificationModel = "RandomForestClassificationModel"
        in "pyspark.ml.connect.classification",
        jvm: "org.apache.spark.ml.classification.RandomForestClassificationModel";
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
    params_only: ["numTrees" => num_trees];
    setters: [
        "setFeaturesCol" => (set_features_col, "featuresCol"),
        "setLeafCol" => (set_leaf_col, "leafCol"),
        "setPredictionCol" => (set_prediction_col, "predictionCol"),
        "setProbabilityCol" => (set_probability_col, "probabilityCol"),
        "setRawPredictionCol" => (set_raw_prediction_col, "rawPredictionCol"),
        "setThresholds" => (set_thresholds, "thresholds"),
    ];
    properties: [
        "featureImportances" => (feature_importances, "featureImportances"),
        "getNumTrees" => (get_num_trees, "getNumTrees"),
        "numClasses" => (num_classes, "numClasses"),
        "numFeatures" => (num_features, "numFeatures"),
        "toDebugString" => (to_debug_string, "toDebugString"),
        "totalNumNodes" => (total_num_nodes, "totalNumNodes"),
        "treeWeights" => (tree_weights, "treeWeights"),
    ];
    methods: [
        "predict" => (predict, "predict", (value)),
        "predictLeaf" => (predict_leaf, "predictLeaf", (value)),
        "predictProbability" => (predict_probability, "predictProbability", (value)),
        "predictRaw" => (predict_raw, "predictRaw", (value)),
    ];
    relations: [];
    relation_methods: [];
    summary: Some(random_forest_training_summary);
    cached: [ "trees" => (trees, random_forest_trees) ];

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

    /// The training summary of the fit, as `HasTrainingSummary.summary` over Spark Connect: the
    /// same object on every call, of a model that this client fitted, else PySpark's RuntimeError.
    #[getter]
    fn summary(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let this = slf.try_borrow()?;
        this.core
            .summary_object(slf.as_any(), random_forest_training_summary)
    }

    /// Whether the model has a training summary, as `HasTrainingSummary.hasSummary` over Spark
    /// Connect: of a model that this client fitted, not of a loaded one.
    #[getter]
    #[pyo3(name = "hasSummary")]
    fn has_summary(&self) -> bool {
        self.core.summary.is_some()
    }

    /// The summary of the model on `dataset`, as `RandomForestClassificationModel.evaluate`: of
    /// the summary that the server returns of it, binary of two classes or fewer.
    fn evaluate(slf: &Bound<'_, Self>, dataset: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if dataset.extract::<PyRef<'_, PyDataFrame>>().is_err() {
            return Err(PyTypeError::new_err(format!(
                "dataset must be a DataFrame but got {}.",
                dataset.get_type()
            )));
        }
        let target = slf.try_borrow()?.core.target()?;
        let output = object::fetch_py(py, &target, "evaluate", std::slice::from_ref(dataset))?;
        let summary = spark_connect::ml::evaluated_summary(&target, output).to_pyerr()?;
        if fetch_num_classes(py, &target)? <= 2 {
            Ok(crate::ml::object::new_py(py, PyBinaryRandomForestClassificationSummary::from(summary))?.into_any())
        } else {
            Ok(crate::ml::object::new_py(py, PyRandomForestClassificationSummary::from(summary))?.into_any())
        }
    }
}

native_model! {
    PyGBTClassificationModel = "GBTClassificationModel" in "pyspark.ml.connect.classification",
        jvm: "org.apache.spark.ml.classification.GBTClassificationModel";
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
    setters: [
        "setFeaturesCol" => (set_features_col, "featuresCol"),
        "setLeafCol" => (set_leaf_col, "leafCol"),
        "setPredictionCol" => (set_prediction_col, "predictionCol"),
        "setProbabilityCol" => (set_probability_col, "probabilityCol"),
        "setRawPredictionCol" => (set_raw_prediction_col, "rawPredictionCol"),
        "setThresholds" => (set_thresholds, "thresholds"),
    ];
    properties: [
        "featureImportances" => (feature_importances, "featureImportances"),
        "getNumTrees" => (get_num_trees, "getNumTrees"),
        "numClasses" => (num_classes, "numClasses"),
        "numFeatures" => (num_features, "numFeatures"),
        "toDebugString" => (to_debug_string, "toDebugString"),
        "totalNumNodes" => (total_num_nodes, "totalNumNodes"),
        "treeWeights" => (tree_weights, "treeWeights"),
    ];
    methods: [
        "evaluateEachIteration" => (evaluate_each_iteration, "evaluateEachIteration", (dataset)),
        "predict" => (predict, "predict", (value)),
        "predictLeaf" => (predict_leaf, "predictLeaf", (value)),
        "predictProbability" => (predict_probability, "predictProbability", (value)),
        "predictRaw" => (predict_raw, "predictRaw", (value)),
    ];
    relations: [];
    relation_methods: [];
    summary: None;
    cached: [ "trees" => (trees, gbt_trees) ];

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

native_summary! {
    PyRandomForestClassificationSummary = "RandomForestClassificationSummary"
        in "pyspark.ml.connect.classification";
    properties: [
        "accuracy" => (accuracy, "accuracy"),
        "falsePositiveRateByLabel" => (false_positive_rate_by_label, "falsePositiveRateByLabel"),
        "labelCol" => (label_col, "labelCol"),
        "labels" => (labels, "labels"),
        "precisionByLabel" => (precision_by_label, "precisionByLabel"),
        "predictionCol" => (prediction_col, "predictionCol"),
        "recallByLabel" => (recall_by_label, "recallByLabel"),
        "truePositiveRateByLabel" => (true_positive_rate_by_label, "truePositiveRateByLabel"),
        "weightCol" => (weight_col, "weightCol"),
        "weightedFalsePositiveRate" => (weighted_false_positive_rate, "weightedFalsePositiveRate"),
        "weightedPrecision" => (weighted_precision, "weightedPrecision"),
        "weightedRecall" => (weighted_recall, "weightedRecall"),
        "weightedTruePositiveRate" => (weighted_true_positive_rate, "weightedTruePositiveRate"),
    ];
    methods: [
        "fMeasureByLabel" => (
            f_measure_by_label, "fMeasureByLabel", (beta = 1.0), "($self, beta=1.0)"
        ),
        "weightedFMeasure" => (
            weighted_f_measure, "weightedFMeasure", (beta = 1.0), "($self, beta=1.0)"
        ),
    ];
    relations: [
        "predictions" => (predictions, "predictions"),
    ];
    relation_methods: [];
}

native_summary! {
    PyRandomForestClassificationTrainingSummary =
        "RandomForestClassificationTrainingSummary" in "pyspark.ml.connect.classification"
        extends PyRandomForestClassificationSummary;
    properties: [
        "objectiveHistory" => (objective_history, "objectiveHistory"),
        "totalIterations" => (total_iterations, "totalIterations"),
    ];
    methods: [];
    relations: [];
    relation_methods: [];
}

native_summary! {
    PyBinaryRandomForestClassificationSummary =
        "BinaryRandomForestClassificationSummary" in "pyspark.ml.connect.classification"
        extends PyRandomForestClassificationSummary;
    properties: [
        "areaUnderROC" => (area_under_roc, "areaUnderROC"),
        "scoreCol" => (score_col, "scoreCol"),
    ];
    methods: [];
    relations: [
        "fMeasureByThreshold" => (f_measure_by_threshold, "fMeasureByThreshold"),
        "pr" => (pr, "pr"),
        "precisionByThreshold" => (precision_by_threshold, "precisionByThreshold"),
        "recallByThreshold" => (recall_by_threshold, "recallByThreshold"),
        "roc" => (roc, "roc"),
    ];
    relation_methods: [];
}

native_summary! {
    PyBinaryRandomForestClassificationTrainingSummary =
        "BinaryRandomForestClassificationTrainingSummary" in "pyspark.ml.connect.classification"
        extends PyBinaryRandomForestClassificationSummary;
    properties: [
        "objectiveHistory" => (objective_history, "objectiveHistory"),
        "totalIterations" => (total_iterations, "totalIterations"),
    ];
    methods: [];
    relations: [];
    relation_methods: [];
}

/// Register the typed classes of the family.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDecisionTreeClassificationModel>()?;
    m.add_class::<PyRandomForestClassificationModel>()?;
    m.add_class::<PyGBTClassificationModel>()?;
    m.add_class::<PyRandomForestClassificationSummary>()?;
    m.add_class::<PyRandomForestClassificationTrainingSummary>()?;
    m.add_class::<PyBinaryRandomForestClassificationSummary>()?;
    m.add_class::<PyBinaryRandomForestClassificationTrainingSummary>()?;
    Ok(())
}
