// SPDX-License-Identifier: Apache-2.0
//! The typed models and summaries of FMClassifier: FMClassificationModel and its summaries.
use super::*;

native_model! {
    PyFMClassificationModel = "FMClassificationModel" in "pyspark.ml.connect.classification",
        jvm: "org.apache.spark.ml.classification.FMClassificationModel";
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
    setters: [
        "setFeaturesCol" => (set_features_col, "featuresCol"),
        "setPredictionCol" => (set_prediction_col, "predictionCol"),
        "setProbabilityCol" => (set_probability_col, "probabilityCol"),
        "setRawPredictionCol" => (set_raw_prediction_col, "rawPredictionCol"),
        "setThresholds" => (set_thresholds, "thresholds"),
    ];
    properties: [
        "factors" => (factors, "factors"),
        "intercept" => (intercept, "intercept"),
        "linear" => (linear, "linear"),
        "numClasses" => (num_classes, "numClasses"),
        "numFeatures" => (num_features, "numFeatures"),
    ];
    methods: [
        "predict" => (predict, "predict", (value)),
        "predictProbability" => (predict_probability, "predictProbability", (value)),
        "predictRaw" => (predict_raw, "predictRaw", (value)),
    ];
    relations: [];
    relation_methods: [];
    summary: Some(fm_training_summary);
    cached: [];

    /// The training summary of the fit, as `HasTrainingSummary.summary` (a method in PySpark
    /// 4.3.0): the same object on every call, of a model that this client fitted, else PySpark's
    /// RuntimeError.
    fn summary(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let this = slf.try_borrow()?;
        this.core.summary_object(slf.as_any(), fm_training_summary)
    }

    /// Whether the model has a training summary, as `HasTrainingSummary.hasSummary` over Spark
    /// Connect: of a model that this client fitted, not of a loaded one.
    #[getter]
    #[pyo3(name = "hasSummary")]
    fn has_summary(&self) -> bool {
        self.core.summary.is_some()
    }

    /// The summary of the model on `dataset`, as `FMClassificationModel.evaluate`: of the summary
    /// that the server returns of it, which this client does not release, as PySpark does not.
    fn evaluate(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyFMClassificationSummary>> {
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
        Py::new(py, PyFMClassificationSummary::from(summary))
    }
}

native_summary! {
    PyFMClassificationSummary = "FMClassificationSummary" in "pyspark.ml.connect.classification";
    properties: [
        "accuracy" => (accuracy, "accuracy"),
        "areaUnderROC" => (area_under_roc, "areaUnderROC"),
        "falsePositiveRateByLabel" => (false_positive_rate_by_label, "falsePositiveRateByLabel"),
        "labelCol" => (label_col, "labelCol"),
        "labels" => (labels, "labels"),
        "precisionByLabel" => (precision_by_label, "precisionByLabel"),
        "predictionCol" => (prediction_col, "predictionCol"),
        "recallByLabel" => (recall_by_label, "recallByLabel"),
        "scoreCol" => (score_col, "scoreCol"),
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
        "fMeasureByThreshold" => (f_measure_by_threshold, "fMeasureByThreshold"),
        "pr" => (pr, "pr"),
        "precisionByThreshold" => (precision_by_threshold, "precisionByThreshold"),
        "predictions" => (predictions, "predictions"),
        "recallByThreshold" => (recall_by_threshold, "recallByThreshold"),
        "roc" => (roc, "roc"),
    ];
    relation_methods: [];
}

native_summary! {
    PyFMClassificationTrainingSummary = "FMClassificationTrainingSummary"
        in "pyspark.ml.connect.classification";
    properties: [
        "accuracy" => (accuracy, "accuracy"),
        "areaUnderROC" => (area_under_roc, "areaUnderROC"),
        "falsePositiveRateByLabel" => (false_positive_rate_by_label, "falsePositiveRateByLabel"),
        "labelCol" => (label_col, "labelCol"),
        "labels" => (labels, "labels"),
        "objectiveHistory" => (objective_history, "objectiveHistory"),
        "precisionByLabel" => (precision_by_label, "precisionByLabel"),
        "predictionCol" => (prediction_col, "predictionCol"),
        "recallByLabel" => (recall_by_label, "recallByLabel"),
        "scoreCol" => (score_col, "scoreCol"),
        "totalIterations" => (total_iterations, "totalIterations"),
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
        "fMeasureByThreshold" => (f_measure_by_threshold, "fMeasureByThreshold"),
        "pr" => (pr, "pr"),
        "precisionByThreshold" => (precision_by_threshold, "precisionByThreshold"),
        "predictions" => (predictions, "predictions"),
        "recallByThreshold" => (recall_by_threshold, "recallByThreshold"),
        "roc" => (roc, "roc"),
    ];
    relation_methods: [];
}

/// The training summary object of a FMClassificationModel, as `_summaryCls` makes it (of no fetch).
fn fm_training_summary(
    py: Python<'_>,
    _model: &object::ModelCore,
    target: spark_connect::ml::MlTarget,
) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, PyFMClassificationTrainingSummary::from(target))?.into_any())
}

/// Register the typed classes of the family.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFMClassificationModel>()?;
    m.add_class::<PyFMClassificationSummary>()?;
    m.add_class::<PyFMClassificationTrainingSummary>()?;
    Ok(())
}
