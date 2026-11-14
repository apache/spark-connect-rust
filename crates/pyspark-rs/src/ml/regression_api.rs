// SPDX-License-Identifier: Apache-2.0
//! The typed models and summaries of the native regression estimators, as PySpark's
//! `pyspark.ml.regression` has them over Spark Connect (see [`super::object`]).
use super::*;

/// The trees of an ensemble, as PySpark's `trees` (a cached property of the model): each tree
/// is a `DecisionTreeRegressionModel` of the object that the server returns of its id.
fn make_trees(py: Python<'_>, model: &object::ModelCore) -> PyResult<Py<PyAny>> {
    let target = model.target()?;
    let objects = object::fetch_py(py, &target, "trees", &[])?
        .into_objects()
        .to_pyerr()?;
    let trees = objects
        .into_iter()
        .map(|tree| PyDecisionTreeRegressionModel::of_object(py, tree).and_then(|m| Py::new(py, m)))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new(py, trees)?.into_any().unbind())
}

/// The training summary object of a LinearRegressionModel, as `_summaryCls` makes it.
fn linear_regression_summary(
    py: Python<'_>,
    _model: &object::ModelCore,
    target: spark_connect::ml::MlTarget,
) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, PyLinearRegressionTrainingSummary::from(target))?.into_any())
}

/// The training summary object of a GeneralizedLinearRegressionModel, as `_summaryCls` makes it.
fn glr_summary(
    py: Python<'_>,
    _model: &object::ModelCore,
    target: spark_connect::ml::MlTarget,
) -> PyResult<Py<PyAny>> {
    Ok(Py::new(
        py,
        PyGeneralizedLinearRegressionTrainingSummary::from(target),
    )?
    .into_any())
}

native_model! {
    PyDecisionTreeRegressionModel = "DecisionTreeRegressionModel"
        in "pyspark.ml.connect.regression",
        jvm: "org.apache.spark.ml.regression.DecisionTreeRegressionModel";
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
    setters: [
        "setFeaturesCol" => (set__features_col, "featuresCol"),
        "setLeafCol" => (set__leaf_col, "leafCol"),
        "setPredictionCol" => (set__prediction_col, "predictionCol"),
        "setVarianceCol" => (set__variance_col, "varianceCol"),
    ];
    properties: [
        "depth" => (depth, "depth"),
        "featureImportances" => (feature_importances, "featureImportances"),
        "numFeatures" => (num_features, "numFeatures"),
        "numNodes" => (num_nodes, "numNodes"),
        "toDebugString" => (to_debug_string, "toDebugString"),
    ];
    methods: [
        "predict" => (predict, "predict", (value)),
        "predictLeaf" => (predict_leaf, "predictLeaf", (value)),
    ];
    relations: [];
    relation_methods: [];
    summary: None;
    cached: [];
    /// The supportedImpurities of the class, as PySpark's class attribute.
    #[classattr]
    #[allow(non_snake_case)]
    fn supportedImpurities() -> Vec<&'static str> {
        vec!["variance"]
    }
}

native_model! {
    PyRandomForestRegressionModel = "RandomForestRegressionModel"
        in "pyspark.ml.connect.regression",
        jvm: "org.apache.spark.ml.regression.RandomForestRegressionModel";
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
    "seed" => ("getSeed", get_seed, param_seed),
    "subsamplingRate" => ("getSubsamplingRate", get_subsampling_rate, param_subsampling_rate),
    "weightCol" => ("getWeightCol", get_weight_col, param_weight_col),
    ];
    params_only: ["numTrees" => param_num_trees];
    setters: [
        "setFeaturesCol" => (set__features_col, "featuresCol"),
        "setLeafCol" => (set__leaf_col, "leafCol"),
        "setPredictionCol" => (set__prediction_col, "predictionCol"),
    ];
    properties: [
        "featureImportances" => (feature_importances, "featureImportances"),
        "getNumTrees" => (get_num_trees_of_model, "getNumTrees"),
        "numFeatures" => (num_features, "numFeatures"),
        "toDebugString" => (to_debug_string, "toDebugString"),
        "totalNumNodes" => (total_num_nodes, "totalNumNodes"),
        "treeWeights" => (tree_weights, "treeWeights"),
    ];
    methods: [
        "predict" => (predict, "predict", (value)),
        "predictLeaf" => (predict_leaf, "predictLeaf", (value)),
    ];
    relations: [];
    relation_methods: [];
    summary: None;
    cached: ["trees" => (trees, make_trees)];
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
}

native_model! {
    PyGBTRegressionModel = "GBTRegressionModel"
        in "pyspark.ml.connect.regression",
        jvm: "org.apache.spark.ml.regression.GBTRegressionModel";
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
    setters: [
        "setFeaturesCol" => (set__features_col, "featuresCol"),
        "setLeafCol" => (set__leaf_col, "leafCol"),
        "setPredictionCol" => (set__prediction_col, "predictionCol"),
    ];
    properties: [
        "featureImportances" => (feature_importances, "featureImportances"),
        "getNumTrees" => (get_num_trees_of_model, "getNumTrees"),
        "numFeatures" => (num_features, "numFeatures"),
        "toDebugString" => (to_debug_string, "toDebugString"),
        "totalNumNodes" => (total_num_nodes, "totalNumNodes"),
        "treeWeights" => (tree_weights, "treeWeights"),
    ];
    methods: [
        "evaluateEachIteration" => (
            evaluate_each_iteration,
            "evaluateEachIteration",
            (dataset, loss)
        ),
        "predict" => (predict, "predict", (value)),
        "predictLeaf" => (predict_leaf, "predictLeaf", (value)),
    ];
    relations: [];
    relation_methods: [];
    summary: None;
    cached: ["trees" => (trees, make_trees)];
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
}

native_model! {
    PyLinearRegressionModel = "LinearRegressionModel"
        in "pyspark.ml.connect.regression",
        jvm: "org.apache.spark.ml.regression.LinearRegressionModel";
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
    setters: [
        "setFeaturesCol" => (set__features_col, "featuresCol"),
        "setPredictionCol" => (set__prediction_col, "predictionCol"),
    ];
    properties: [
        "coefficients" => (coefficients, "coefficients"),
        "intercept" => (intercept, "intercept"),
        "numFeatures" => (num_features, "numFeatures"),
        "scale" => (scale, "scale"),
    ];
    methods: [
        "predict" => (predict, "predict", (value)),
    ];
    relations: [];
    relation_methods: [];
    summary: Some(linear_regression_summary);
    cached: [];

    /// The training summary of the fit, as `HasTrainingSummary.summary` over Spark Connect: the
    /// same object on every call, of a model that this client fitted, else PySpark's RuntimeError.
    #[getter]
    fn summary(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        slf.try_borrow()?.core.summary_object(slf.as_any(), linear_regression_summary)
    }

    /// Whether the model has a training summary, as `HasTrainingSummary.hasSummary` over Spark
    /// Connect: of a model that this client fitted, not of a loaded one.
    #[getter]
    #[pyo3(name = "hasSummary")]
    fn has_summary(&self) -> bool {
        self.core.summary.is_some()
    }

    /// The summary of the model on `dataset`, as the model's `evaluate`: of the summary that
    /// the server returns of it, which this client does not release, as PySpark does not.
    fn evaluate(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyLinearRegressionSummary>> {
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
        Py::new(py, PyLinearRegressionSummary::from(summary))
    }

}

native_model! {
    PyGeneralizedLinearRegressionModel = "GeneralizedLinearRegressionModel"
        in "pyspark.ml.connect.regression",
        jvm: "org.apache.spark.ml.regression.GeneralizedLinearRegressionModel";
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
    setters: [
        "setFeaturesCol" => (set__features_col, "featuresCol"),
        "setLinkPredictionCol" => (set__link_prediction_col, "linkPredictionCol"),
        "setPredictionCol" => (set__prediction_col, "predictionCol"),
    ];
    properties: [
        "coefficients" => (coefficients, "coefficients"),
        "intercept" => (intercept, "intercept"),
        "numFeatures" => (num_features, "numFeatures"),
    ];
    methods: [
        "predict" => (predict, "predict", (value)),
    ];
    relations: [];
    relation_methods: [];
    summary: Some(glr_summary);
    cached: [];

    /// The training summary of the fit, as `HasTrainingSummary.summary` over Spark Connect: the
    /// same object on every call, of a model that this client fitted, else PySpark's RuntimeError.
    #[getter]
    fn summary(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        slf.try_borrow()?.core.summary_object(slf.as_any(), glr_summary)
    }

    /// Whether the model has a training summary, as `HasTrainingSummary.hasSummary` over Spark
    /// Connect: of a model that this client fitted, not of a loaded one.
    #[getter]
    #[pyo3(name = "hasSummary")]
    fn has_summary(&self) -> bool {
        self.core.summary.is_some()
    }

    /// The summary of the model on `dataset`, as the model's `evaluate`: of the summary that
    /// the server returns of it, which this client does not release, as PySpark does not.
    fn evaluate(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyGeneralizedLinearRegressionSummary>> {
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
        Py::new(py, PyGeneralizedLinearRegressionSummary::from(summary))
    }

}

native_model! {
    PyAFTSurvivalRegressionModel = "AFTSurvivalRegressionModel"
        in "pyspark.ml.connect.regression",
        jvm: "org.apache.spark.ml.regression.AFTSurvivalRegressionModel";
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
    setters: [
        "setFeaturesCol" => (set__features_col, "featuresCol"),
        "setPredictionCol" => (set__prediction_col, "predictionCol"),
        "setQuantileProbabilities" => (set__quantile_probabilities, "quantileProbabilities"),
        "setQuantilesCol" => (set__quantiles_col, "quantilesCol"),
    ];
    properties: [
        "coefficients" => (coefficients, "coefficients"),
        "intercept" => (intercept, "intercept"),
        "numFeatures" => (num_features, "numFeatures"),
        "scale" => (scale, "scale"),
    ];
    methods: [
        "predict" => (predict, "predict", (value)),
        "predictQuantiles" => (predict_quantiles, "predictQuantiles", (features)),
    ];
    relations: [];
    relation_methods: [];
    summary: None;
    cached: [];
}

native_model! {
    PyIsotonicRegressionModel = "IsotonicRegressionModel"
        in "pyspark.ml.connect.regression",
        jvm: "org.apache.spark.ml.regression.IsotonicRegressionModel";
    params: [
    "featureIndex" => ("getFeatureIndex", get_feature_index, param_feature_index),
    "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
    "isotonic" => ("getIsotonic", get_isotonic, param_isotonic),
    "labelCol" => ("getLabelCol", get_label_col, param_label_col),
    "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
    "weightCol" => ("getWeightCol", get_weight_col, param_weight_col),
    ];
    setters: [
        "setFeatureIndex" => (set__feature_index, "featureIndex"),
        "setFeaturesCol" => (set__features_col, "featuresCol"),
        "setPredictionCol" => (set__prediction_col, "predictionCol"),
    ];
    properties: [
        "boundaries" => (boundaries, "boundaries"),
        "numFeatures" => (num_features, "numFeatures"),
        "predictions" => (predictions, "predictions"),
    ];
    methods: [
        "predict" => (predict, "predict", (value)),
    ];
    relations: [];
    relation_methods: [];
    summary: None;
    cached: [];
}

native_model! {
    PyFMRegressionModel = "FMRegressionModel"
        in "pyspark.ml.connect.regression",
        jvm: "org.apache.spark.ml.regression.FMRegressionModel";
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
    setters: [
        "setFeaturesCol" => (set__features_col, "featuresCol"),
        "setPredictionCol" => (set__prediction_col, "predictionCol"),
    ];
    properties: [
        "factors" => (factors, "factors"),
        "intercept" => (intercept, "intercept"),
        "linear" => (linear, "linear"),
        "numFeatures" => (num_features, "numFeatures"),
    ];
    methods: [
        "predict" => (predict, "predict", (value)),
    ];
    relations: [];
    relation_methods: [];
    summary: None;
    cached: [];
}

native_summary! {
    PyLinearRegressionSummary = "LinearRegressionSummary"
        in "pyspark.ml.connect.regression";
    properties: [
        "coefficientStandardErrors" => (coefficient_standard_errors, "coefficientStandardErrors"),
        "degreesOfFreedom" => (degrees_of_freedom, "degreesOfFreedom"),
        "devianceResiduals" => (deviance_residuals, "devianceResiduals"),
        "explainedVariance" => (explained_variance, "explainedVariance"),
        "featuresCol" => (features_col, "featuresCol"),
        "labelCol" => (label_col, "labelCol"),
        "meanAbsoluteError" => (mean_absolute_error, "meanAbsoluteError"),
        "meanSquaredError" => (mean_squared_error, "meanSquaredError"),
        "numInstances" => (num_instances, "numInstances"),
        "pValues" => (p_values, "pValues"),
        "predictionCol" => (prediction_col, "predictionCol"),
        "r2" => (r2, "r2"),
        "r2adj" => (r2adj, "r2adj"),
        "rootMeanSquaredError" => (root_mean_squared_error, "rootMeanSquaredError"),
        "tValues" => (t_values, "tValues"),
    ];
    methods: [];
    relations: [
        "predictions" => (predictions, "predictions"),
        "residuals" => (residuals, "residuals"),
    ];
    relation_methods: [];
}

native_summary! {
    PyLinearRegressionTrainingSummary = "LinearRegressionTrainingSummary"
        in "pyspark.ml.connect.regression";
    properties: [
        "coefficientStandardErrors" => (coefficient_standard_errors, "coefficientStandardErrors"),
        "degreesOfFreedom" => (degrees_of_freedom, "degreesOfFreedom"),
        "devianceResiduals" => (deviance_residuals, "devianceResiduals"),
        "explainedVariance" => (explained_variance, "explainedVariance"),
        "featuresCol" => (features_col, "featuresCol"),
        "labelCol" => (label_col, "labelCol"),
        "meanAbsoluteError" => (mean_absolute_error, "meanAbsoluteError"),
        "meanSquaredError" => (mean_squared_error, "meanSquaredError"),
        "numInstances" => (num_instances, "numInstances"),
        "pValues" => (p_values, "pValues"),
        "predictionCol" => (prediction_col, "predictionCol"),
        "r2" => (r2, "r2"),
        "r2adj" => (r2adj, "r2adj"),
        "rootMeanSquaredError" => (root_mean_squared_error, "rootMeanSquaredError"),
        "tValues" => (t_values, "tValues"),
        "objectiveHistory" => (objective_history, "objectiveHistory"),
        "totalIterations" => (total_iterations, "totalIterations"),
    ];
    methods: [];
    relations: [
        "predictions" => (predictions, "predictions"),
        "residuals" => (residuals, "residuals"),
    ];
    relation_methods: [];
}

native_summary! {
    PyGeneralizedLinearRegressionSummary = "GeneralizedLinearRegressionSummary"
        in "pyspark.ml.connect.regression";
    properties: [
        "aic" => (aic, "aic"),
        "degreesOfFreedom" => (degrees_of_freedom, "degreesOfFreedom"),
        "deviance" => (deviance, "deviance"),
        "dispersion" => (dispersion, "dispersion"),
        "nullDeviance" => (null_deviance, "nullDeviance"),
        "numInstances" => (num_instances, "numInstances"),
        "predictionCol" => (prediction_col, "predictionCol"),
        "rank" => (rank, "rank"),
        "residualDegreeOfFreedom" => (residual_degree_of_freedom, "residualDegreeOfFreedom"),
        "residualDegreeOfFreedomNull" => (
            residual_degree_of_freedom_null,
            "residualDegreeOfFreedomNull"
        ),
    ];
    methods: [];
    relations: [
        "predictions" => (predictions, "predictions"),
    ];
    relation_methods: ["residuals" => (
            residuals,
            "residuals",
            (residualsType = "deviance"),
            "($self, residualsType='deviance')"
        )];
}

native_summary! {
    PyGeneralizedLinearRegressionTrainingSummary = "GeneralizedLinearRegressionTrainingSummary"
        in "pyspark.ml.connect.regression";
    properties: [
        "aic" => (aic, "aic"),
        "degreesOfFreedom" => (degrees_of_freedom, "degreesOfFreedom"),
        "deviance" => (deviance, "deviance"),
        "dispersion" => (dispersion, "dispersion"),
        "nullDeviance" => (null_deviance, "nullDeviance"),
        "numInstances" => (num_instances, "numInstances"),
        "predictionCol" => (prediction_col, "predictionCol"),
        "rank" => (rank, "rank"),
        "residualDegreeOfFreedom" => (residual_degree_of_freedom, "residualDegreeOfFreedom"),
        "residualDegreeOfFreedomNull" => (
            residual_degree_of_freedom_null,
            "residualDegreeOfFreedomNull"
        ),
        "coefficientStandardErrors" => (coefficient_standard_errors, "coefficientStandardErrors"),
        "numIterations" => (num_iterations, "numIterations"),
        "pValues" => (p_values, "pValues"),
        "solver" => (solver, "solver"),
        "tValues" => (t_values, "tValues"),
    ];
    methods: [];
    relations: [
        "predictions" => (predictions, "predictions"),
    ];
    relation_methods: [
        "residuals" => (
            residuals,
            "residuals",
            (residualsType = "deviance"),
            "($self, residualsType='deviance')"
        ),
    ];
}

/// Register the typed regression models and summaries.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDecisionTreeRegressionModel>()?;
    m.add_class::<PyRandomForestRegressionModel>()?;
    m.add_class::<PyGBTRegressionModel>()?;
    m.add_class::<PyLinearRegressionModel>()?;
    m.add_class::<PyLinearRegressionSummary>()?;
    m.add_class::<PyLinearRegressionTrainingSummary>()?;
    m.add_class::<PyGeneralizedLinearRegressionModel>()?;
    m.add_class::<PyGeneralizedLinearRegressionSummary>()?;
    m.add_class::<PyGeneralizedLinearRegressionTrainingSummary>()?;
    m.add_class::<PyAFTSurvivalRegressionModel>()?;
    m.add_class::<PyIsotonicRegressionModel>()?;
    m.add_class::<PyFMRegressionModel>()?;
    Ok(())
}
