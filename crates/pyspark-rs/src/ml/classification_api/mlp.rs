// SPDX-License-Identifier: Apache-2.0
//! The typed models and summaries of MultilayerPerceptronClassifier:
//! MultilayerPerceptronClassificationModel and its summaries.
use super::*;

native_model! {
    PyMultilayerPerceptronClassificationModel = "MultilayerPerceptronClassificationModel"
        in "pyspark.ml.connect.classification",
        jvm: "org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel";
    params: [
        "blockSize" => ("getBlockSize", get_block_size, param_block_size),
        "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
        "initialWeights" => ("getInitialWeights", get_initial_weights, param_initial_weights),
        "labelCol" => ("getLabelCol", get_label_col, param_label_col),
        "layers" => ("getLayers", get_layers, param_layers),
        "maxIter" => ("getMaxIter", get_max_iter, param_max_iter),
        "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
        "probabilityCol" => ("getProbabilityCol", get_probability_col, param_probability_col),
        "rawPredictionCol" => (
            "getRawPredictionCol",
            get_raw_prediction_col,
            param_raw_prediction_col
        ),
        "seed" => ("getSeed", get_seed, param_seed),
        "solver" => ("getSolver", get_solver, param_solver),
        "stepSize" => ("getStepSize", get_step_size, param_step_size),
        "thresholds" => ("getThresholds", get_thresholds, param_thresholds),
        "tol" => ("getTol", get_tol, param_tol),
    ];
    setters: [
        "setFeaturesCol" => (set_features_col, "featuresCol"),
        "setPredictionCol" => (set_prediction_col, "predictionCol"),
        "setProbabilityCol" => (set_probability_col, "probabilityCol"),
        "setRawPredictionCol" => (set_raw_prediction_col, "rawPredictionCol"),
        "setThresholds" => (set_thresholds, "thresholds"),
    ];
    properties: [
        "numClasses" => (num_classes, "numClasses"),
        "numFeatures" => (num_features, "numFeatures"),
        "weights" => (weights, "weights"),
    ];
    methods: [
        "predict" => (predict, "predict", (value)),
        "predictProbability" => (predict_probability, "predictProbability", (value)),
        "predictRaw" => (predict_raw, "predictRaw", (value)),
    ];
    relations: [];
    relation_methods: [];
    summary: Some(mlp_training_summary);
    cached: [];

    /// The training summary of the fit, as `HasTrainingSummary.summary` (a method in PySpark
    /// 4.3.0): the same object on every call, of a model that this client fitted, else PySpark's
    /// RuntimeError.
    fn summary(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let this = slf.try_borrow()?;
        this.core.summary_object(slf.as_any(), mlp_training_summary)
    }

    /// Whether the model has a training summary, as `HasTrainingSummary.hasSummary` over Spark
    /// Connect: of a model that this client fitted, not of a loaded one.
    #[getter]
    #[pyo3(name = "hasSummary")]
    fn has_summary(&self) -> bool {
        self.core.summary.is_some()
    }

    /// The summary of the model on `dataset`, as `MultilayerPerceptronClassificationModel.evaluate`
    /// (of the summary that the server returns of it, which this client does not release).
    fn evaluate(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyMultilayerPerceptronClassificationSummary>> {
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
        Py::new(py, PyMultilayerPerceptronClassificationSummary::from(summary))
    }
}

native_summary! {
    PyMultilayerPerceptronClassificationSummary =
        "MultilayerPerceptronClassificationSummary" in "pyspark.ml.connect.classification";
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
    PyMultilayerPerceptronClassificationTrainingSummary =
        "MultilayerPerceptronClassificationTrainingSummary"
        in "pyspark.ml.connect.classification";
    properties: [
        "accuracy" => (accuracy, "accuracy"),
        "falsePositiveRateByLabel" => (false_positive_rate_by_label, "falsePositiveRateByLabel"),
        "labelCol" => (label_col, "labelCol"),
        "labels" => (labels, "labels"),
        "objectiveHistory" => (objective_history, "objectiveHistory"),
        "precisionByLabel" => (precision_by_label, "precisionByLabel"),
        "predictionCol" => (prediction_col, "predictionCol"),
        "recallByLabel" => (recall_by_label, "recallByLabel"),
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
        "predictions" => (predictions, "predictions"),
    ];
    relation_methods: [];
}

/// The training summary object of a MultilayerPerceptronClassificationModel, as `_summaryCls`
/// makes it (of no fetch).
fn mlp_training_summary(
    py: Python<'_>,
    _model: &object::ModelCore,
    target: spark_connect::ml::MlTarget,
) -> PyResult<Py<PyAny>> {
    Ok(Py::new(
        py,
        PyMultilayerPerceptronClassificationTrainingSummary::from(target),
    )?
    .into_any())
}

/// Register the typed classes of the family.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMultilayerPerceptronClassificationModel>()?;
    m.add_class::<PyMultilayerPerceptronClassificationSummary>()?;
    m.add_class::<PyMultilayerPerceptronClassificationTrainingSummary>()?;
    Ok(())
}
