// SPDX-License-Identifier: Apache-2.0
//! The native evaluators of `pyspark.ml.connect.evaluation`, as PySpark's `pyspark.ml.evaluation`
//! has them over Spark Connect: each with PySpark's Params API, its getters and setters, `evaluate`
//! (over a copy for a param map), `isLargerBetter`, and its persistence (see [`super::macros`]).
use super::*;
use spark_connect::ml::{
    BinaryClassificationEvaluator, ClusteringEvaluator, MulticlassClassificationEvaluator,
    MultilabelClassificationEvaluator, RankingEvaluator, RegressionEvaluator,
};

/// The struct of the native evaluator `$ty` over the core evaluator `$core`, of the Python class
/// `$name`, with the wrapper's [`Native`] and [`params_api::NativeInstance`] (its copy and uid).
macro_rules! evaluator_struct {
    ($ty:ident, $core:ident, $name:literal) => {
        #[pyclass(name = $name, module = "pyspark.ml.connect.evaluation")]
        pub struct $ty {
            inner: $core,
            unsendable: Vec<&'static str>,
        }

        impl Native for $ty {
            type Core = $core;
            fn inner(&self) -> &Self::Core {
                &self.inner
            }
            fn params(&self) -> &Params {
                self.inner.params()
            }
            fn params_mut(&mut self) -> &mut Params {
                self.inner.params_mut()
            }
            fn unsendable(&self) -> &[&'static str] {
                &self.unsendable
            }
            fn unsendable_mut(&mut self) -> &mut Vec<&'static str> {
                &mut self.unsendable
            }
        }

        impl params_api::NativeInstance for $ty {
            fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
                Py::new(
                    py,
                    Self {
                        inner: self.inner.clone(),
                        unsendable: self.unsendable.clone(),
                    },
                )
            }

            fn uid(&self) -> String {
                Evaluator::operator(&self.inner).uid.clone()
            }

            fn set_uid(&mut self, uid: &str) {
                self.inner.set_uid(uid);
            }
        }
    };
}

/// `Evaluator.evaluate(dataset, params)`, as PySpark's: a non-empty param map applies to a copy of
/// `slf`, which evaluates `dataset`; otherwise `slf` does. The dataset is checked before the
/// evaluation, and the evaluation waits for the server with the interrupt of the session.
fn evaluate_of<'py, T>(
    slf: &Bound<'py, T>,
    dataset: &Bound<'py, PyAny>,
    params: Option<&Bound<'py, PyAny>>,
) -> PyResult<f64>
where
    T: Native + params_api::NativeInstance + PyClass<Frozen = False>,
    T::Core: Evaluator,
{
    let map = params.filter(|p| p.cast::<PyDict>().is_ok_and(|d| !d.is_empty()));
    if let Some(map) = map {
        let copy = params_api::copy(slf, Some(map))?;
        return evaluate_of(&copy, dataset, None);
    }
    check_params(params)?;
    // Reject pandas before parameter serialization, and read `sparkSession` of another object
    // first, as `try_remote_evaluate` does. Spark itself validates column types and names, just as
    // it does for standard pyspark.ml.evaluation.
    let dataset = dataset_of(dataset, "evaluate")?;
    let evaluator = snapshot(slf)?;
    interruptible(slf.py(), &dataset.spark_session(), || {
        evaluator.evaluate(&dataset).map_err(Box::new)
    })
}

evaluator_struct!(
    PyBinaryClassificationEvaluator,
    BinaryClassificationEvaluator,
    "BinaryClassificationEvaluator"
);

const BINARY_CLASSIFICATION_PARAMS: &[(&str, Kind)] = &[
    ("rawPredictionCol", Kind::Str),
    ("labelCol", Kind::Str),
    ("metricName", Kind::Str),
    ("weightCol", Kind::Str),
    ("numBins", Kind::Int),
];

native_pymethods! {
    PyBinaryClassificationEvaluator;
    jvm: "org.apache.spark.ml.evaluation.BinaryClassificationEvaluator", op: Evaluator;
    params: [
        "labelCol" => ("getLabelCol", get_label_col, param_label_col),
        "metricName" => ("getMetricName", get_metric_name, param_metric_name),
        "numBins" => ("getNumBins", get_num_bins, param_num_bins),
        "rawPredictionCol" => (
            "getRawPredictionCol",
            get_raw_prediction_col,
            param_raw_prediction_col
        ),
        "weightCol" => ("getWeightCol", get_weight_col, param_weight_col),
    ];
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, rawPredictionCol='rawPrediction', labelCol='label', metricName='areaUnderROC', weightCol=None, numBins=1000)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: BinaryClassificationEvaluator::new(),
            unsendable: Vec::new(),
        };
        init_params(&mut this, "BinaryClassificationEvaluator", &[], BINARY_CLASSIFICATION_PARAMS, args, kwargs)?;
        Ok(this)
    }
    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, rawPredictionCol='rawPrediction', labelCol='label', metricName='areaUnderROC', weightCol=None, numBins=1000)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(slf, "BinaryClassificationEvaluator", BINARY_CLASSIFICATION_PARAMS, args, kwargs)
    }

    #[pyo3(name = "setLabelCol")]
    fn set_label_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "labelCol", Kind::Str, value)
    }

    #[pyo3(name = "setMetricName")]
    fn set_metric_name<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "metricName", Kind::Str, value)
    }

    #[pyo3(name = "setNumBins")]
    fn set_num_bins<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "numBins", Kind::Int, value)
    }

    #[pyo3(name = "setRawPredictionCol")]
    fn set_raw_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "rawPredictionCol", Kind::Str, value)
    }

    #[pyo3(name = "setWeightCol")]
    fn set_weight_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "weightCol", Kind::Str, value)
    }

    #[pyo3(name = "isLargerBetter")]
    fn is_larger_better(&self) -> bool {
        self.inner.is_larger_better()
    }

    /// The metric of `dataset`, as `Evaluator.evaluate`: of a copy of the evaluator of a non-empty
    /// param map `params`.
    #[pyo3(signature = (dataset, params=None))]
    fn evaluate<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        params: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<f64> {
        evaluate_of(slf, dataset, params)
    }
}

evaluator_struct!(
    PyRegressionEvaluator,
    RegressionEvaluator,
    "RegressionEvaluator"
);

const REGRESSION_PARAMS: &[(&str, Kind)] = &[
    ("predictionCol", Kind::Str),
    ("labelCol", Kind::Str),
    ("metricName", Kind::Str),
    ("weightCol", Kind::Str),
    ("throughOrigin", Kind::Bool),
];

native_pymethods! {
    PyRegressionEvaluator;
    jvm: "org.apache.spark.ml.evaluation.RegressionEvaluator", op: Evaluator;
    params: [
        "labelCol" => ("getLabelCol", get_label_col, param_label_col),
        "metricName" => ("getMetricName", get_metric_name, param_metric_name),
        "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
        "throughOrigin" => ("getThroughOrigin", get_through_origin, param_through_origin),
        "weightCol" => ("getWeightCol", get_weight_col, param_weight_col),
    ];
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, predictionCol='prediction', labelCol='label', metricName='rmse', weightCol=None, throughOrigin=False)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: RegressionEvaluator::new(),
            unsendable: Vec::new(),
        };
        init_params(&mut this, "RegressionEvaluator", &[], REGRESSION_PARAMS, args, kwargs)?;
        Ok(this)
    }
    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, predictionCol='prediction', labelCol='label', metricName='rmse', weightCol=None, throughOrigin=False)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(slf, "RegressionEvaluator", REGRESSION_PARAMS, args, kwargs)
    }

    #[pyo3(name = "setLabelCol")]
    fn set_label_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "labelCol", Kind::Str, value)
    }

    #[pyo3(name = "setMetricName")]
    fn set_metric_name<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "metricName", Kind::Str, value)
    }

    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "predictionCol", Kind::Str, value)
    }

    #[pyo3(name = "setThroughOrigin")]
    fn set_through_origin<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "throughOrigin", Kind::Bool, value)
    }

    #[pyo3(name = "setWeightCol")]
    fn set_weight_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "weightCol", Kind::Str, value)
    }

    #[pyo3(name = "isLargerBetter")]
    fn is_larger_better(&self) -> bool {
        self.inner.is_larger_better()
    }

    /// The metric of `dataset`, as `Evaluator.evaluate`: of a copy of the evaluator of a non-empty
    /// param map `params`.
    #[pyo3(signature = (dataset, params=None))]
    fn evaluate<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        params: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<f64> {
        evaluate_of(slf, dataset, params)
    }
}

evaluator_struct!(
    PyMulticlassClassificationEvaluator,
    MulticlassClassificationEvaluator,
    "MulticlassClassificationEvaluator"
);

const MULTICLASS_CLASSIFICATION_PARAMS: &[(&str, Kind)] = &[
    ("predictionCol", Kind::Str),
    ("labelCol", Kind::Str),
    ("metricName", Kind::Str),
    ("weightCol", Kind::Str),
    ("metricLabel", Kind::Float),
    ("beta", Kind::Float),
    ("probabilityCol", Kind::Str),
    ("eps", Kind::Float),
];

native_pymethods! {
    PyMulticlassClassificationEvaluator;
    jvm: "org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator", op: Evaluator;
    params: [
        "beta" => ("getBeta", get_beta, param_beta),
        "eps" => ("getEps", get_eps, param_eps),
        "labelCol" => ("getLabelCol", get_label_col, param_label_col),
        "metricLabel" => ("getMetricLabel", get_metric_label, param_metric_label),
        "metricName" => ("getMetricName", get_metric_name, param_metric_name),
        "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
        "probabilityCol" => ("getProbabilityCol", get_probability_col, param_probability_col),
        "weightCol" => ("getWeightCol", get_weight_col, param_weight_col),
    ];
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, predictionCol='prediction', labelCol='label', metricName='f1', weightCol=None, metricLabel=0.0, beta=1.0, probabilityCol='probability', eps=1e-15)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: MulticlassClassificationEvaluator::new(),
            unsendable: Vec::new(),
        };
        init_params(&mut this, "MulticlassClassificationEvaluator", &[], MULTICLASS_CLASSIFICATION_PARAMS, args, kwargs)?;
        Ok(this)
    }
    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, predictionCol='prediction', labelCol='label', metricName='f1', weightCol=None, metricLabel=0.0, beta=1.0, probabilityCol='probability', eps=1e-15)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(slf, "MulticlassClassificationEvaluator", MULTICLASS_CLASSIFICATION_PARAMS, args, kwargs)
    }

    #[pyo3(name = "setBeta")]
    fn set_beta<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "beta", Kind::Float, value)
    }

    #[pyo3(name = "setEps")]
    fn set_eps<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "eps", Kind::Float, value)
    }

    #[pyo3(name = "setLabelCol")]
    fn set_label_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "labelCol", Kind::Str, value)
    }

    #[pyo3(name = "setMetricLabel")]
    fn set_metric_label<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "metricLabel", Kind::Float, value)
    }

    #[pyo3(name = "setMetricName")]
    fn set_metric_name<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "metricName", Kind::Str, value)
    }

    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "predictionCol", Kind::Str, value)
    }

    #[pyo3(name = "setProbabilityCol")]
    fn set_probability_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "probabilityCol", Kind::Str, value)
    }

    #[pyo3(name = "setWeightCol")]
    fn set_weight_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "weightCol", Kind::Str, value)
    }

    #[pyo3(name = "isLargerBetter")]
    fn is_larger_better(&self) -> bool {
        self.inner.is_larger_better()
    }

    /// The metric of `dataset`, as `Evaluator.evaluate`: of a copy of the evaluator of a non-empty
    /// param map `params`.
    #[pyo3(signature = (dataset, params=None))]
    fn evaluate<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        params: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<f64> {
        evaluate_of(slf, dataset, params)
    }
}

evaluator_struct!(
    PyMultilabelClassificationEvaluator,
    MultilabelClassificationEvaluator,
    "MultilabelClassificationEvaluator"
);

const MULTILABEL_CLASSIFICATION_PARAMS: &[(&str, Kind)] = &[
    ("predictionCol", Kind::Str),
    ("labelCol", Kind::Str),
    ("metricName", Kind::Str),
    ("metricLabel", Kind::Float),
];

native_pymethods! {
    PyMultilabelClassificationEvaluator;
    jvm: "org.apache.spark.ml.evaluation.MultilabelClassificationEvaluator", op: Evaluator;
    params: [
        "labelCol" => ("getLabelCol", get_label_col, param_label_col),
        "metricLabel" => ("getMetricLabel", get_metric_label, param_metric_label),
        "metricName" => ("getMetricName", get_metric_name, param_metric_name),
        "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
    ];
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, predictionCol='prediction', labelCol='label', metricName='f1Measure', metricLabel=0.0)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: MultilabelClassificationEvaluator::new(),
            unsendable: Vec::new(),
        };
        init_params(&mut this, "MultilabelClassificationEvaluator", &[], MULTILABEL_CLASSIFICATION_PARAMS, args, kwargs)?;
        Ok(this)
    }
    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, predictionCol='prediction', labelCol='label', metricName='f1Measure', metricLabel=0.0)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(slf, "MultilabelClassificationEvaluator", MULTILABEL_CLASSIFICATION_PARAMS, args, kwargs)
    }

    #[pyo3(name = "setLabelCol")]
    fn set_label_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "labelCol", Kind::Str, value)
    }

    #[pyo3(name = "setMetricLabel")]
    fn set_metric_label<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "metricLabel", Kind::Float, value)
    }

    #[pyo3(name = "setMetricName")]
    fn set_metric_name<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "metricName", Kind::Str, value)
    }

    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "predictionCol", Kind::Str, value)
    }

    #[pyo3(name = "isLargerBetter")]
    fn is_larger_better(&self) -> bool {
        self.inner.is_larger_better()
    }

    /// The metric of `dataset`, as `Evaluator.evaluate`: of a copy of the evaluator of a non-empty
    /// param map `params`.
    #[pyo3(signature = (dataset, params=None))]
    fn evaluate<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        params: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<f64> {
        evaluate_of(slf, dataset, params)
    }
}

evaluator_struct!(
    PyClusteringEvaluator,
    ClusteringEvaluator,
    "ClusteringEvaluator"
);

const CLUSTERING_PARAMS: &[(&str, Kind)] = &[
    ("predictionCol", Kind::Str),
    ("featuresCol", Kind::Str),
    ("metricName", Kind::Str),
    ("distanceMeasure", Kind::Str),
    ("weightCol", Kind::Str),
];

native_pymethods! {
    PyClusteringEvaluator;
    jvm: "org.apache.spark.ml.evaluation.ClusteringEvaluator", op: Evaluator;
    params: [
        "distanceMeasure" => ("getDistanceMeasure", get_distance_measure, param_distance_measure),
        "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
        "metricName" => ("getMetricName", get_metric_name, param_metric_name),
        "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
        "weightCol" => ("getWeightCol", get_weight_col, param_weight_col),
    ];
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, predictionCol='prediction', featuresCol='features', metricName='silhouette', distanceMeasure='squaredEuclidean', weightCol=None)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: ClusteringEvaluator::new(),
            unsendable: Vec::new(),
        };
        init_params(&mut this, "ClusteringEvaluator", &[], CLUSTERING_PARAMS, args, kwargs)?;
        Ok(this)
    }
    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, predictionCol='prediction', featuresCol='features', metricName='silhouette', distanceMeasure='squaredEuclidean', weightCol=None)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(slf, "ClusteringEvaluator", CLUSTERING_PARAMS, args, kwargs)
    }

    #[pyo3(name = "setDistanceMeasure")]
    fn set_distance_measure<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "distanceMeasure", Kind::Str, value)
    }

    #[pyo3(name = "setFeaturesCol")]
    fn set_features_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "featuresCol", Kind::Str, value)
    }

    #[pyo3(name = "setMetricName")]
    fn set_metric_name<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "metricName", Kind::Str, value)
    }

    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "predictionCol", Kind::Str, value)
    }

    #[pyo3(name = "setWeightCol")]
    fn set_weight_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "weightCol", Kind::Str, value)
    }

    #[pyo3(name = "isLargerBetter")]
    fn is_larger_better(&self) -> bool {
        self.inner.is_larger_better()
    }

    /// The metric of `dataset`, as `Evaluator.evaluate`: of a copy of the evaluator of a non-empty
    /// param map `params`.
    #[pyo3(signature = (dataset, params=None))]
    fn evaluate<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        params: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<f64> {
        evaluate_of(slf, dataset, params)
    }
}

evaluator_struct!(PyRankingEvaluator, RankingEvaluator, "RankingEvaluator");

const RANKING_PARAMS: &[(&str, Kind)] = &[
    ("predictionCol", Kind::Str),
    ("labelCol", Kind::Str),
    ("metricName", Kind::Str),
    ("k", Kind::Int),
];

native_pymethods! {
    PyRankingEvaluator;
    jvm: "org.apache.spark.ml.evaluation.RankingEvaluator", op: Evaluator;
    params: [
        "k" => ("getK", get_k, param_k),
        "labelCol" => ("getLabelCol", get_label_col, param_label_col),
        "metricName" => ("getMetricName", get_metric_name, param_metric_name),
        "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
    ];
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, predictionCol='prediction', labelCol='label', metricName='meanAveragePrecision', k=10)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: RankingEvaluator::new(),
            unsendable: Vec::new(),
        };
        init_params(&mut this, "RankingEvaluator", &[], RANKING_PARAMS, args, kwargs)?;
        Ok(this)
    }
    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, predictionCol='prediction', labelCol='label', metricName='meanAveragePrecision', k=10)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(slf, "RankingEvaluator", RANKING_PARAMS, args, kwargs)
    }

    #[pyo3(name = "setK")]
    fn set_k<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "k", Kind::Int, value)
    }

    #[pyo3(name = "setLabelCol")]
    fn set_label_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "labelCol", Kind::Str, value)
    }

    #[pyo3(name = "setMetricName")]
    fn set_metric_name<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "metricName", Kind::Str, value)
    }

    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "predictionCol", Kind::Str, value)
    }

    #[pyo3(name = "isLargerBetter")]
    fn is_larger_better(&self) -> bool {
        self.inner.is_larger_better()
    }

    /// The metric of `dataset`, as `Evaluator.evaluate`: of a copy of the evaluator of a non-empty
    /// param map `params`.
    #[pyo3(signature = (dataset, params=None))]
    fn evaluate<'py>(
        slf: &Bound<'py, Self>,
        dataset: &Bound<'py, PyAny>,
        params: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<f64> {
        evaluate_of(slf, dataset, params)
    }
}
