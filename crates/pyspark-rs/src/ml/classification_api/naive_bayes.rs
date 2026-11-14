// SPDX-License-Identifier: Apache-2.0
//! The typed models and summaries of NaiveBayes: NaiveBayesModel.
use super::*;

native_model! {
    PyNaiveBayesModel = "NaiveBayesModel" in "pyspark.ml.connect.classification",
        jvm: "org.apache.spark.ml.classification.NaiveBayesModel";
    params: [
        "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
        "labelCol" => ("getLabelCol", get_label_col, param_label_col),
        "modelType" => ("getModelType", get_model_type, param_model_type),
        "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
        "probabilityCol" => ("getProbabilityCol", get_probability_col, param_probability_col),
        "rawPredictionCol" => (
            "getRawPredictionCol",
            get_raw_prediction_col,
            param_raw_prediction_col
        ),
        "smoothing" => ("getSmoothing", get_smoothing, param_smoothing),
        "thresholds" => ("getThresholds", get_thresholds, param_thresholds),
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
        "numClasses" => (num_classes, "numClasses"),
        "numFeatures" => (num_features, "numFeatures"),
        "pi" => (pi, "pi"),
        "sigma" => (sigma, "sigma"),
        "theta" => (theta, "theta"),
    ];
    methods: [
        "predict" => (predict, "predict", (value)),
        "predictProbability" => (predict_probability, "predictProbability", (value)),
        "predictRaw" => (predict_raw, "predictRaw", (value)),
    ];
    relations: [];
    relation_methods: [];
    summary: None;
    cached: [];
}

/// Register the typed classes of the family.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNaiveBayesModel>()?;
    Ok(())
}
