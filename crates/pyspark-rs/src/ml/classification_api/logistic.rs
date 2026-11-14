// SPDX-License-Identifier: Apache-2.0
//! The typed models and summaries of LogisticRegression: LogisticRegressionModel and its four
//! summaries, and the threshold methods that its estimator and its model share.
use super::params_api::ParamsHolder;
use super::*;
use pyo3::types::{PyFloat, PyString};

/// `_checkThresholdConsistency` of PySpark's `_LogisticRegressionParams`, which the constructor
/// and `setParams` of LogisticRegression make once they have set their params, and whose getters
/// make first: when both `threshold` and `thresholds` are set, `thresholds` must have two values,
/// whose equivalent threshold, `1.0 / (1.0 + thresholds[0] / thresholds[1])`, is `threshold`
/// within 1e-5, else PySpark's ValueError. The arithmetic is Python's, so a zero division raises
/// its ZeroDivisionError, and a None raises its TypeError. (The setters make no check, as
/// PySpark's do not.)
pub(in crate::ml) fn check_threshold_consistency<T: ParamsHolder>(
    py: Python<'_>,
    this: &T,
) -> PyResult<()> {
    let (Some(threshold), Some(thresholds)) = (
        this.set_value(py, "threshold")?,
        this.set_value(py, "thresholds")?,
    ) else {
        return Ok(());
    };
    let (threshold, thresholds) = (threshold.into_bound(py), thresholds.into_bound(py));
    if thresholds.len()? != 2 {
        return Err(PyValueError::new_err(format!(
            "Logistic Regression getThreshold only applies to binary classification, but \
             thresholds has length != 2. thresholds: {}",
            thresholds.str()?
        )));
    }
    let t = equivalent_threshold(py, &thresholds)?;
    let difference = threshold.sub(&t)?.abs()?;
    if difference.ge(1e-5)? {
        // Of PySpark's `%g`s, which Python formats.
        let message = PyString::new(
            py,
            "Logistic Regression getThreshold found inconsistent values for threshold (%g) and \
             thresholds (equivalent to %g)",
        )
        .call_method1("__mod__", ((threshold, t),))?;
        return Err(PyValueError::new_err(message.extract::<String>()?));
    }
    Ok(())
}

/// The threshold equivalent to `thresholds` of two values, `1.0 / (1.0 + thresholds[0] /
/// thresholds[1])`, in Python's arithmetic.
fn equivalent_threshold<'py>(
    py: Python<'py>,
    thresholds: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let ratio = thresholds.get_item(0)?.div(thresholds.get_item(1)?)?;
    let denominator = PyFloat::new(py, 1.0).into_any().add(ratio)?;
    PyFloat::new(py, 1.0).into_any().div(denominator)
}

/// `getThreshold()` of PySpark's `_LogisticRegressionParams`: the equivalent threshold of
/// `thresholds` when it is set (of two values, else PySpark's ValueError), else `threshold`.
pub(in crate::ml) fn threshold_of<T: ParamsHolder>(slf: &Bound<'_, T>) -> PyResult<Py<PyAny>> {
    let py = slf.py();
    check_threshold_consistency(py, &*slf.try_borrow()?)?;
    let thresholds = slf.try_borrow()?.set_value(py, "thresholds")?;
    let Some(thresholds) = thresholds else {
        return params_api::getter(slf, "threshold");
    };
    let thresholds = thresholds.into_bound(py);
    if thresholds.len()? != 2 {
        return Err(PyValueError::new_err(format!(
            "Logistic Regression getThreshold only applies to binary classification, but \
             thresholds has length != 2.  thresholds: {}",
            thresholds.str()?
        )));
    }
    Ok(equivalent_threshold(py, &thresholds)?.unbind())
}

/// `getThresholds()` of PySpark's `_LogisticRegressionParams`: `thresholds` when it is set, else
/// `[1.0 - threshold, threshold]` when `threshold` is set, else PySpark's KeyError.
pub(in crate::ml) fn thresholds_of<T: ParamsHolder>(slf: &Bound<'_, T>) -> PyResult<Py<PyAny>> {
    let py = slf.py();
    check_threshold_consistency(py, &*slf.try_borrow()?)?;
    let thresholds_set = slf.try_borrow()?.set_value(py, "thresholds")?.is_some();
    let threshold_set = slf.try_borrow()?.set_value(py, "threshold")?.is_some();
    if !thresholds_set && threshold_set {
        let threshold = params_api::getter(slf, "threshold")?.into_bound(py);
        let complement = PyFloat::new(py, 1.0).into_any().sub(&threshold)?;
        return Ok(PyList::new(py, [complement, threshold])?
            .into_any()
            .unbind());
    }
    params_api::getter(slf, "thresholds")
}

/// The number of classes of the model `model`, as its `numClasses` fetches it.
fn fetch_num_classes(py: Python<'_>, model: &spark_connect::ml::MlTarget) -> PyResult<i64> {
    let output = object::fetch_py(py, model, "numClasses", &[])?;
    object::fetch_output_to_py(py, output)?.bind(py).extract()
}

/// The training summary object of a LogisticRegressionModel, as its `_summaryCls` makes it: the
/// binary one of two classes or fewer, else the multinomial one. Its `numClasses` is fetched.
fn logistic_training_summary(
    py: Python<'_>,
    model: &object::ModelCore,
    target: spark_connect::ml::MlTarget,
) -> PyResult<Py<PyAny>> {
    if fetch_num_classes(py, &model.target()?)? <= 2 {
        Ok(Py::new(py, PyBinaryLogisticRegressionTrainingSummary::from(target))?.into_any())
    } else {
        Ok(Py::new(py, PyLogisticRegressionTrainingSummary::from(target))?.into_any())
    }
}

/// The summary of the model `model` on a dataset, of the evaluated summary `summary`, of the
/// class that `LogisticRegressionModel.evaluate` chooses by `numClasses`, fetched after the
/// evaluation.
fn evaluated_summary_of(
    py: Python<'_>,
    model: &spark_connect::ml::MlTarget,
    summary: spark_connect::ml::MlTarget,
) -> PyResult<Py<PyAny>> {
    if fetch_num_classes(py, model)? <= 2 {
        Ok(Py::new(py, PyBinaryLogisticRegressionSummary::from(summary))?.into_any())
    } else {
        Ok(Py::new(py, PyLogisticRegressionSummary::from(summary))?.into_any())
    }
}

native_model! {
    PyLogisticRegressionModel = "LogisticRegressionModel" in "pyspark.ml.connect.classification",
        jvm: "org.apache.spark.ml.classification.LogisticRegressionModel";
    params: [
        "aggregationDepth" => (
            "getAggregationDepth",
            get_aggregation_depth,
            param_aggregation_depth
        ),
        "elasticNetParam" => ("getElasticNetParam", get_elastic_net_param, param_elastic_net_param),
        "family" => ("getFamily", get_family, param_family),
        "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
        "fitIntercept" => ("getFitIntercept", get_fit_intercept, param_fit_intercept),
        "labelCol" => ("getLabelCol", get_label_col, param_label_col),
        "lowerBoundsOnCoefficients" => (
            "getLowerBoundsOnCoefficients",
            get_lower_bounds_on_coefficients,
            param_lower_bounds_on_coefficients
        ),
        "lowerBoundsOnIntercepts" => (
            "getLowerBoundsOnIntercepts",
            get_lower_bounds_on_intercepts,
            param_lower_bounds_on_intercepts
        ),
        "maxBlockSizeInMB" => (
            "getMaxBlockSizeInMB",
            get_max_block_size_in_mb,
            param_max_block_size_in_mb
        ),
        "maxIter" => ("getMaxIter", get_max_iter, param_max_iter),
        "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
        "probabilityCol" => ("getProbabilityCol", get_probability_col, param_probability_col),
        "rawPredictionCol" => (
            "getRawPredictionCol",
            get_raw_prediction_col,
            param_raw_prediction_col
        ),
        "regParam" => ("getRegParam", get_reg_param, param_reg_param),
        "standardization" => ("getStandardization", get_standardization, param_standardization),
        "tol" => ("getTol", get_tol, param_tol),
        "upperBoundsOnCoefficients" => (
            "getUpperBoundsOnCoefficients",
            get_upper_bounds_on_coefficients,
            param_upper_bounds_on_coefficients
        ),
        "upperBoundsOnIntercepts" => (
            "getUpperBoundsOnIntercepts",
            get_upper_bounds_on_intercepts,
            param_upper_bounds_on_intercepts
        ),
        "weightCol" => ("getWeightCol", get_weight_col, param_weight_col),
    ];
    params_only: ["threshold" => param_threshold, "thresholds" => param_thresholds];
    setters: [
        "setFeaturesCol" => (set_features_col, "featuresCol"),
        "setPredictionCol" => (set_prediction_col, "predictionCol"),
        "setProbabilityCol" => (set_probability_col, "probabilityCol"),
        "setRawPredictionCol" => (set_raw_prediction_col, "rawPredictionCol"),
        "setThreshold" => (set_threshold, "threshold"),
        "setThresholds" => (set_thresholds, "thresholds"),
    ];
    properties: [
        "coefficientMatrix" => (coefficient_matrix, "coefficientMatrix"),
        "coefficients" => (coefficients, "coefficients"),
        "intercept" => (intercept, "intercept"),
        "interceptVector" => (intercept_vector, "interceptVector"),
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
    summary: Some(logistic_training_summary);
    cached: [];

    /// `getThreshold()`, as `_LogisticRegressionParams` (see [`threshold_of`]).
    #[pyo3(name = "getThreshold")]
    fn get_threshold(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        threshold_of(slf)
    }

    /// `getThresholds()`, as `_LogisticRegressionParams` (see [`thresholds_of`]).
    #[pyo3(name = "getThresholds")]
    fn get_thresholds(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        thresholds_of(slf)
    }

    /// The training summary of the fit, as `HasTrainingSummary.summary` over Spark Connect: the
    /// same object on every call, of a model that this client fitted, else PySpark's RuntimeError.
    #[getter]
    fn summary(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        slf.try_borrow()?.core.summary_object(slf.as_any(), logistic_training_summary)
    }

    /// Whether the model has a training summary, as `HasTrainingSummary.hasSummary` over Spark
    /// Connect: of a model that this client fitted, not of a loaded one.
    #[getter]
    #[pyo3(name = "hasSummary")]
    fn has_summary(&self) -> bool {
        self.core.summary.is_some()
    }

    /// The summary of the model on `dataset`, as `LogisticRegressionModel.evaluate`: the binary
    /// summary of two classes or fewer, else the multinomial one, which this client does not
    /// release, as PySpark does not.
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
        evaluated_summary_of(py, &target, summary)
    }
}

// The summary of a LogisticRegressionModel of more than two classes, over its server-side
// summary object.
native_summary! {
    PyLogisticRegressionSummary =
        "LogisticRegressionSummary" in "pyspark.ml.connect.classification";
    properties: [
        "accuracy" => (accuracy, "accuracy"),
        "falsePositiveRateByLabel" => (false_positive_rate_by_label, "falsePositiveRateByLabel"),
        "featuresCol" => (features_col, "featuresCol"),
        "labelCol" => (label_col, "labelCol"),
        "labels" => (labels, "labels"),
        "precisionByLabel" => (precision_by_label, "precisionByLabel"),
        "predictionCol" => (prediction_col, "predictionCol"),
        "probabilityCol" => (probability_col, "probabilityCol"),
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

// The training summary of a LogisticRegressionModel of more than two classes.
native_summary! {
    PyLogisticRegressionTrainingSummary =
        "LogisticRegressionTrainingSummary" in "pyspark.ml.connect.classification";
    properties: [
        "accuracy" => (accuracy, "accuracy"),
        "falsePositiveRateByLabel" => (false_positive_rate_by_label, "falsePositiveRateByLabel"),
        "featuresCol" => (features_col, "featuresCol"),
        "labelCol" => (label_col, "labelCol"),
        "labels" => (labels, "labels"),
        "objectiveHistory" => (objective_history, "objectiveHistory"),
        "precisionByLabel" => (precision_by_label, "precisionByLabel"),
        "predictionCol" => (prediction_col, "predictionCol"),
        "probabilityCol" => (probability_col, "probabilityCol"),
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

// The summary of a LogisticRegressionModel of two classes or fewer, binary.
native_summary! {
    PyBinaryLogisticRegressionSummary =
        "BinaryLogisticRegressionSummary" in "pyspark.ml.connect.classification";
    properties: [
        "accuracy" => (accuracy, "accuracy"),
        "areaUnderROC" => (area_under_roc, "areaUnderROC"),
        "falsePositiveRateByLabel" => (false_positive_rate_by_label, "falsePositiveRateByLabel"),
        "featuresCol" => (features_col, "featuresCol"),
        "labelCol" => (label_col, "labelCol"),
        "labels" => (labels, "labels"),
        "precisionByLabel" => (precision_by_label, "precisionByLabel"),
        "predictionCol" => (prediction_col, "predictionCol"),
        "probabilityCol" => (probability_col, "probabilityCol"),
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

// The training summary of a LogisticRegressionModel of two classes or fewer, binary.
native_summary! {
    PyBinaryLogisticRegressionTrainingSummary =
        "BinaryLogisticRegressionTrainingSummary" in "pyspark.ml.connect.classification";
    properties: [
        "accuracy" => (accuracy, "accuracy"),
        "areaUnderROC" => (area_under_roc, "areaUnderROC"),
        "falsePositiveRateByLabel" => (false_positive_rate_by_label, "falsePositiveRateByLabel"),
        "featuresCol" => (features_col, "featuresCol"),
        "labelCol" => (label_col, "labelCol"),
        "labels" => (labels, "labels"),
        "objectiveHistory" => (objective_history, "objectiveHistory"),
        "precisionByLabel" => (precision_by_label, "precisionByLabel"),
        "predictionCol" => (prediction_col, "predictionCol"),
        "probabilityCol" => (probability_col, "probabilityCol"),
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

/// Register the typed LogisticRegression models and summaries.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLogisticRegressionModel>()?;
    m.add_class::<PyLogisticRegressionSummary>()?;
    m.add_class::<PyLogisticRegressionTrainingSummary>()?;
    m.add_class::<PyBinaryLogisticRegressionSummary>()?;
    m.add_class::<PyBinaryLogisticRegressionTrainingSummary>()?;
    Ok(())
}
