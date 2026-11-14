// SPDX-License-Identifier: Apache-2.0
//! Native bindings for standard pyspark.ml.evaluation algorithms.
use super::*;
use spark_connect::ml::{
    BinaryClassificationEvaluator, ClusteringEvaluator, MulticlassClassificationEvaluator,
    MultilabelClassificationEvaluator, RankingEvaluator, RegressionEvaluator,
};

macro_rules! evaluator {
    ($wrapper:ident, $core:ident, $name:literal, $signature:literal,
     $(($method:ident, $setter:literal, $param:literal, $kind:ident)),* $(,)?) => {
        #[pyclass(name = $name, module = "pyspark.ml.connect.evaluation")]
        pub struct $wrapper { inner: $core, unsendable: Vec<&'static str> }
        impl Native for $wrapper {
            type Core = $core;
            fn inner(&self) -> &Self::Core { &self.inner }
            fn params(&self) -> &Params { self.inner.params() }
            fn params_mut(&mut self) -> &mut Params { self.inner.params_mut() }
            fn unsendable(&self) -> &[&'static str] { &self.unsendable }
            fn unsendable_mut(&mut self) -> &mut Vec<&'static str> { &mut self.unsendable }
        }
        #[pymethods]
        impl $wrapper {
            #[new]
            #[pyo3(signature = (*args, **kwargs), text_signature = $signature)]
            fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
                let mut this = Self { inner: $core::new(), unsendable: Vec::new() };
                init_params(&mut this, $name, &[], &[$(($param, Kind::$kind)),*], args, kwargs)?;
                Ok(this)
            }
            #[getter]
            fn uid(&self) -> &str { &self.inner.operator().uid }
            $(#[pyo3(name = $setter)]
            fn $method<'py>(slf: Bound<'py, Self>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
                set_param(slf, $param, Kind::$kind, value)
            })*
            #[pyo3(name = "isLargerBetter")]
            fn is_larger_better(&self) -> bool { self.inner.is_larger_better() }
            fn evaluate(slf: &Bound<'_, Self>, dataset: &Bound<'_, PyAny>) -> PyResult<f64> {
                // Reject pandas before parameter serialization. Spark itself validates column
                // types and names, just as it does for standard pyspark.ml.evaluation.
                let dataset = dataset_of(dataset, "evaluate", "select")?;
                let evaluator = snapshot(slf)?;
                interruptible(slf.py(), &dataset.spark_session(), || {
                    evaluator.evaluate(&dataset).map_err(Box::new)
                })
            }
        }
    };
}

evaluator!(PyBinaryClassificationEvaluator, BinaryClassificationEvaluator, "BinaryClassificationEvaluator", "(*, rawPredictionCol='rawPrediction', labelCol='label', metricName='areaUnderROC', weightCol=None, numBins=1000)",
    (set_raw_prediction_col, "setRawPredictionCol", "rawPredictionCol", Str),
    (set_label_col, "setLabelCol", "labelCol", Str),
    (set_metric_name, "setMetricName", "metricName", Str),
    (set_weight_col, "setWeightCol", "weightCol", Str),
    (set_num_bins, "setNumBins", "numBins", Int),
);

evaluator!(PyRegressionEvaluator, RegressionEvaluator, "RegressionEvaluator", "(*, predictionCol='prediction', labelCol='label', metricName='rmse', weightCol=None, throughOrigin=False)",
    (set_prediction_col, "setPredictionCol", "predictionCol", Str),
    (set_label_col, "setLabelCol", "labelCol", Str),
    (set_metric_name, "setMetricName", "metricName", Str),
    (set_weight_col, "setWeightCol", "weightCol", Str),
    (set_through_origin, "setThroughOrigin", "throughOrigin", Bool),
);

evaluator!(PyMulticlassClassificationEvaluator, MulticlassClassificationEvaluator, "MulticlassClassificationEvaluator", "(*, predictionCol='prediction', labelCol='label', metricName='f1', weightCol=None, metricLabel=0.0, beta=1.0, probabilityCol='probability', eps=1e-15)",
    (set_prediction_col, "setPredictionCol", "predictionCol", Str),
    (set_label_col, "setLabelCol", "labelCol", Str),
    (set_metric_name, "setMetricName", "metricName", Str),
    (set_weight_col, "setWeightCol", "weightCol", Str),
    (set_metric_label, "setMetricLabel", "metricLabel", Float),
    (set_beta, "setBeta", "beta", Float),
    (set_probability_col, "setProbabilityCol", "probabilityCol", Str),
    (set_eps, "setEps", "eps", Float),
);

evaluator!(
    PyMultilabelClassificationEvaluator,
    MultilabelClassificationEvaluator,
    "MultilabelClassificationEvaluator",
    "(*, predictionCol='prediction', labelCol='label', metricName='f1Measure', metricLabel=0.0)",
    (set_prediction_col, "setPredictionCol", "predictionCol", Str),
    (set_label_col, "setLabelCol", "labelCol", Str),
    (set_metric_name, "setMetricName", "metricName", Str),
    (set_metric_label, "setMetricLabel", "metricLabel", Float),
);

evaluator!(PyClusteringEvaluator, ClusteringEvaluator, "ClusteringEvaluator", "(*, predictionCol='prediction', featuresCol='features', metricName='silhouette', distanceMeasure='squaredEuclidean', weightCol=None)",
    (set_prediction_col, "setPredictionCol", "predictionCol", Str),
    (set_features_col, "setFeaturesCol", "featuresCol", Str),
    (set_metric_name, "setMetricName", "metricName", Str),
    (set_distance_measure, "setDistanceMeasure", "distanceMeasure", Str),
    (set_weight_col, "setWeightCol", "weightCol", Str),
);

evaluator!(
    PyRankingEvaluator,
    RankingEvaluator,
    "RankingEvaluator",
    "(*, predictionCol='prediction', labelCol='label', metricName='meanAveragePrecision', k=10)",
    (set_prediction_col, "setPredictionCol", "predictionCol", Str),
    (set_label_col, "setLabelCol", "labelCol", Str),
    (set_metric_name, "setMetricName", "metricName", Str),
    (set_k, "setK", "k", Int),
);
