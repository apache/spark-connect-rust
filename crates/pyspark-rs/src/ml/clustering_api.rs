// SPDX-License-Identifier: Apache-2.0
//! The typed models and summaries of the native clustering estimators, as PySpark's
//! `pyspark.ml.clustering` has them over Spark Connect (see [`super::object`]).
use super::*;

native_model! {
    PyKMeansModel = "KMeansModel" in "pyspark.ml.connect.clustering",
        jvm: "org.apache.spark.ml.clustering.KMeansModel";
    params: [
        "distanceMeasure" => ("getDistanceMeasure", get_distance_measure, param_distance_measure),
        "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
        "initMode" => ("getInitMode", get_init_mode, param_init_mode),
        "initSteps" => ("getInitSteps", get_init_steps, param_init_steps),
        "intermediateStorageLevel" => (
            "getIntermediateStorageLevel",
            get_intermediate_storage_level,
            param_intermediate_storage_level
        ),
        "k" => ("getK", get_k, param_k),
        "maxBlockSizeInMB" => (
            "getMaxBlockSizeInMB",
            get_max_block_size_in_mb,
            param_max_block_size_in_mb
        ),
        "maxIter" => ("getMaxIter", get_max_iter, param_max_iter),
        "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
        "seed" => ("getSeed", get_seed, param_seed),
        "solver" => ("getSolver", get_solver, param_solver),
        "tol" => ("getTol", get_tol, param_tol),
        "weightCol" => ("getWeightCol", get_weight_col, param_weight_col),
    ];
    setters: [
        "setFeaturesCol" => (set_features_col, "featuresCol"),
        "setPredictionCol" => (set_prediction_col, "predictionCol"),
    ];
    properties: [
        "numFeatures" => (num_features, "numFeatures"),
    ];
    methods: [
        "predict" => (predict, "predict", (value)),
    ];
    relations: [];
    relation_methods: [];
    summary: Some(kmeans_summary);
    cached: [];

    /// The cluster centers, as `KMeansModel.clusterCenters`: the rows of the server's
    /// `clusterCenterMatrix`, as NumPy arrays.
    #[pyo3(name = "clusterCenters")]
    fn cluster_centers(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let target = slf.try_borrow()?.core.target()?;
        let out = object::fetch_py(py, &target, "clusterCenterMatrix", &[])?;
        let matrix = object::fetch_output_to_py(py, out)?;
        let rows = matrix.bind(py).call_method0("toArray")?;
        Ok(PyList::new(py, rows.try_iter()?.collect::<PyResult<Vec<_>>>()?)?
            .into_any()
            .unbind())
    }

    /// The training summary of the fit, as `HasTrainingSummary.summary` over Spark Connect: the
    /// same object on every call, of a model that this client fitted, else PySpark's RuntimeError.
    #[getter]
    fn summary(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        slf.try_borrow()?.core.summary_object(slf.as_any(), kmeans_summary)
    }

    /// Whether the model has a training summary, as `HasTrainingSummary.hasSummary` over Spark
    /// Connect: of a model that this client fitted, not of a loaded one.
    #[getter]
    #[pyo3(name = "hasSummary")]
    fn has_summary(&self) -> bool {
        self.core.summary.is_some()
    }
}

native_summary! {
    PyKMeansSummary = "KMeansSummary" in "pyspark.ml.connect.clustering";
    properties: [
        "clusterSizes" => (cluster_sizes, "clusterSizes"),
        "featuresCol" => (features_col, "featuresCol"),
        "k" => (k, "k"),
        "numIter" => (num_iter, "numIter"),
        "predictionCol" => (prediction_col, "predictionCol"),
        "trainingCost" => (training_cost, "trainingCost"),
    ];
    methods: [];
    relations: [
        "cluster" => (cluster, "cluster"),
        "predictions" => (predictions, "predictions"),
    ];
    relation_methods: [];
}

/// The training summary object of a KMeansModel, as `_summaryCls` makes it (of no fetch).
fn kmeans_summary(
    py: Python<'_>,
    _model: &object::ModelCore,
    target: spark_connect::ml::MlTarget,
) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, PyKMeansSummary::from(target))?.into_any())
}

native_model! {
    PyBisectingKMeansModel = "BisectingKMeansModel" in "pyspark.ml.connect.clustering",
        jvm: "org.apache.spark.ml.clustering.BisectingKMeansModel";
    params: [
        "distanceMeasure" => ("getDistanceMeasure", get_distance_measure, param_distance_measure),
        "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
        "k" => ("getK", get_k, param_k),
        "maxIter" => ("getMaxIter", get_max_iter, param_max_iter),
        "minDivisibleClusterSize" => (
            "getMinDivisibleClusterSize",
            get_min_divisible_cluster_size,
            param_min_divisible_cluster_size
        ),
        "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
        "seed" => ("getSeed", get_seed, param_seed),
        "weightCol" => ("getWeightCol", get_weight_col, param_weight_col),
    ];
    setters: [
        "setFeaturesCol" => (set_features_col, "featuresCol"),
        "setPredictionCol" => (set_prediction_col, "predictionCol"),
    ];
    properties: [
        "numFeatures" => (num_features, "numFeatures"),
    ];
    methods: [
        "computeCost" => (compute_cost, "computeCost", (dataset)),
        "predict" => (predict, "predict", (value)),
    ];
    relations: [];
    relation_methods: [];
    summary: Some(bisecting_kmeans_summary);
    cached: [];

    /// The cluster centers, as `BisectingKMeansModel.clusterCenters`: the rows of the server's
    /// `clusterCenterMatrix`, as NumPy arrays.
    #[pyo3(name = "clusterCenters")]
    fn cluster_centers(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let target = slf.try_borrow()?.core.target()?;
        let out = object::fetch_py(py, &target, "clusterCenterMatrix", &[])?;
        let matrix = object::fetch_output_to_py(py, out)?;
        let rows = matrix.bind(py).call_method0("toArray")?;
        Ok(PyList::new(py, rows.try_iter()?.collect::<PyResult<Vec<_>>>()?)?
            .into_any()
            .unbind())
    }

    /// The training summary of the fit, as `HasTrainingSummary.summary` over Spark Connect: the
    /// same object on every call, of a model that this client fitted, else PySpark's RuntimeError.
    #[getter]
    fn summary(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        slf.try_borrow()?.core.summary_object(slf.as_any(), bisecting_kmeans_summary)
    }

    /// Whether the model has a training summary, as `HasTrainingSummary.hasSummary` over Spark
    /// Connect: of a model that this client fitted, not of a loaded one.
    #[getter]
    #[pyo3(name = "hasSummary")]
    fn has_summary(&self) -> bool {
        self.core.summary.is_some()
    }
}

/// The training summary object of a BisectingKMeansModel, as `_summaryCls` makes it (of no fetch).
fn bisecting_kmeans_summary(
    py: Python<'_>,
    _model: &object::ModelCore,
    target: spark_connect::ml::MlTarget,
) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, PyBisectingKMeansSummary::from(target))?.into_any())
}

native_summary! {
    PyBisectingKMeansSummary = "BisectingKMeansSummary" in "pyspark.ml.connect.clustering";
    properties: [
        "clusterSizes" => (cluster_sizes, "clusterSizes"),
        "featuresCol" => (features_col, "featuresCol"),
        "k" => (k, "k"),
        "numIter" => (num_iter, "numIter"),
        "predictionCol" => (prediction_col, "predictionCol"),
        "trainingCost" => (training_cost, "trainingCost"),
    ];
    methods: [];
    relations: [
        "cluster" => (cluster, "cluster"),
        "predictions" => (predictions, "predictions"),
    ];
    relation_methods: [];
}

native_model! {
    PyGaussianMixtureModel = "GaussianMixtureModel" in "pyspark.ml.connect.clustering",
        jvm: "org.apache.spark.ml.clustering.GaussianMixtureModel";
    params: [
        "aggregationDepth" => (
            "getAggregationDepth",
            get_aggregation_depth,
            param_aggregation_depth
        ),
        "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
        "k" => ("getK", get_k, param_k),
        "maxIter" => ("getMaxIter", get_max_iter, param_max_iter),
        "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
        "probabilityCol" => ("getProbabilityCol", get_probability_col, param_probability_col),
        "seed" => ("getSeed", get_seed, param_seed),
        "tol" => ("getTol", get_tol, param_tol),
        "weightCol" => ("getWeightCol", get_weight_col, param_weight_col),
    ];
    setters: [
        "setFeaturesCol" => (set_features_col, "featuresCol"),
        "setPredictionCol" => (set_prediction_col, "predictionCol"),
        "setProbabilityCol" => (set_probability_col, "probabilityCol"),
    ];
    properties: [
        "numFeatures" => (num_features, "numFeatures"),
        "weights" => (weights, "weights"),
    ];
    methods: [
        "predict" => (predict, "predict", (value)),
        "predictProbability" => (predict_probability, "predictProbability", (value)),
    ];
    relations: [
        "gaussiansDF" => (gaussians_df, "gaussiansDF"),
    ];
    relation_methods: [];
    summary: Some(gaussian_mixture_summary);
    cached: [];

    /// The Gaussian distributions of the model, as `GaussianMixtureModel.gaussians`: a
    /// `MultivariateGaussian` of the mean and the covariance of each row of `gaussiansDF`, which
    /// PySpark builds on the client too.
    #[getter]
    fn gaussians(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let gaussian = py.import("pyspark.ml.stat")?.getattr("MultivariateGaussian")?;
        let mut gaussians = Vec::new();
        for row in slf.getattr("gaussiansDF")?.call_method0("collect")?.try_iter()? {
            let row = row?;
            let mean = row.getattr("mean")?.call_method0("asML")?;
            let cov = row.getattr("cov")?.call_method0("asML")?;
            gaussians.push(gaussian.call1((mean, cov))?);
        }
        Ok(PyList::new(py, gaussians)?.into_any().unbind())
    }

    /// The training summary of the fit, as `HasTrainingSummary.summary` over Spark Connect: the
    /// same object on every call, of a model that this client fitted, else PySpark's RuntimeError.
    #[getter]
    fn summary(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        slf.try_borrow()?.core.summary_object(slf.as_any(), gaussian_mixture_summary)
    }

    /// Whether the model has a training summary, as `HasTrainingSummary.hasSummary` over Spark
    /// Connect: of a model that this client fitted, not of a loaded one.
    #[getter]
    #[pyo3(name = "hasSummary")]
    fn has_summary(&self) -> bool {
        self.core.summary.is_some()
    }
}

/// The training summary object of a GaussianMixtureModel, as `_summaryCls` makes it.
fn gaussian_mixture_summary(
    py: Python<'_>,
    _model: &object::ModelCore,
    target: spark_connect::ml::MlTarget,
) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, PyGaussianMixtureSummary::from(target))?.into_any())
}

native_summary! {
    PyGaussianMixtureSummary = "GaussianMixtureSummary" in "pyspark.ml.connect.clustering";
    properties: [
        "clusterSizes" => (cluster_sizes, "clusterSizes"),
        "featuresCol" => (features_col, "featuresCol"),
        "k" => (k, "k"),
        "logLikelihood" => (log_likelihood, "logLikelihood"),
        "numIter" => (num_iter, "numIter"),
        "predictionCol" => (prediction_col, "predictionCol"),
        "probabilityCol" => (probability_col, "probabilityCol"),
    ];
    methods: [];
    relations: [
        "cluster" => (cluster, "cluster"),
        "predictions" => (predictions, "predictions"),
        "probability" => (probability, "probability"),
    ];
    relation_methods: [];
}

/// The members that PySpark's `LDAModel` has, which its two subclasses have too: the model's
/// params, the server's methods, and `describeTopics` (an attribute relation of the topics).
macro_rules! lda_model {
    ($ty:ident = $name:literal, jvm: $jvm:literal, cached_methods: [ $($cm:tt)* ]; $($body:tt)*) => {
        lda_model! { @emit $ty = $name, jvm: $jvm; cached_methods: [ $($cm)* ]; $($body)* }
    };
    ($ty:ident = $name:literal, jvm: $jvm:literal; $($body:tt)*) => {
        lda_model! { @emit $ty = $name, jvm: $jvm; cached_methods: []; $($body)* }
    };
    (@emit $ty:ident = $name:literal, jvm: $jvm:literal; cached_methods: [ $($cm:tt)* ]; $($body:tt)*) => {
        native_model! {
            $ty = $name in "pyspark.ml.connect.clustering", jvm: $jvm;
            params: [
        "checkpointInterval" => (
            "getCheckpointInterval",
            get_checkpoint_interval,
            param_checkpoint_interval
        ),
        "docConcentration" => (
            "getDocConcentration",
            get_doc_concentration,
            param_doc_concentration
        ),
        "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
        "k" => ("getK", get_k, param_k),
        "keepLastCheckpoint" => (
            "getKeepLastCheckpoint",
            get_keep_last_checkpoint,
            param_keep_last_checkpoint
        ),
        "learningDecay" => ("getLearningDecay", get_learning_decay, param_learning_decay),
        "learningOffset" => ("getLearningOffset", get_learning_offset, param_learning_offset),
        "maxIter" => ("getMaxIter", get_max_iter, param_max_iter),
        "optimizeDocConcentration" => (
            "getOptimizeDocConcentration",
            get_optimize_doc_concentration,
            param_optimize_doc_concentration
        ),
        "optimizer" => ("getOptimizer", get_optimizer, param_optimizer),
        "seed" => ("getSeed", get_seed, param_seed),
        "subsamplingRate" => ("getSubsamplingRate", get_subsampling_rate, param_subsampling_rate),
        "topicConcentration" => (
            "getTopicConcentration",
            get_topic_concentration,
            param_topic_concentration
        ),
        "topicDistributionCol" => (
            "getTopicDistributionCol",
            get_topic_distribution_col,
            param_topic_distribution_col
        ),
            ];
            setters: [
                "setFeaturesCol" => (set_features_col, "featuresCol"),
                "setSeed" => (set_seed, "seed"),
                "setTopicDistributionCol" => (set_topic_distribution_col, "topicDistributionCol"),
            ];
            properties: [];
            methods: [
                "estimatedDocConcentration" => (
                    estimated_doc_concentration,
                    "estimatedDocConcentration",
                    ()
                ),
                "isDistributed" => (is_distributed, "isDistributed", ()),
                "logLikelihood" => (log_likelihood, "logLikelihood", (dataset)),
                "logPerplexity" => (log_perplexity, "logPerplexity", (dataset)),
                "topicsMatrix" => (topics_matrix, "topicsMatrix", ()),
                "vocabSize" => (vocab_size, "vocabSize", ()),
            ];
            relations: [];
            relation_methods: [
                "describeTopics" => (
                    describe_topics,
                    "describeTopics",
                    (maxTermsPerTopic = 10),
                    "($self, maxTermsPerTopic=10)"
                ),
            ];
            summary: None;
            cached_methods: [ $($cm)* ];
            cached: [];

            $($body)*
        }
    };
}

/// The LocalLDAModel that `DistributedLDAModel.toLocal` returns, of the server's object of the
/// model `model`: of a fresh uid and no params. The cache of `functools.cache` is kept by the
/// model (see [`object::ModelCore::cached_method`]).
fn local_model_of_distributed(py: Python<'_>, model: &object::ModelCore) -> PyResult<Py<PyAny>> {
    let target = model.target()?;
    let object = object::fetch_py(py, &target, "toLocal", &[])?
        .into_object()
        .to_pyerr()?;
    Ok(Py::new(py, PyLocalLDAModel::of_object(py, object)?)?.into_any())
}

lda_model! {
    PyLocalLDAModel = "LocalLDAModel", jvm: "org.apache.spark.ml.clustering.LocalLDAModel";
}

lda_model! {
    PyDistributedLDAModel = "DistributedLDAModel",
        jvm: "org.apache.spark.ml.clustering.DistributedLDAModel",
        cached_methods: ["toLocal" => (to_local, local_model_of_distributed)];

    /// The log likelihood of the training set, as `DistributedLDAModel.trainingLogLikelihood`.
    #[pyo3(name = "trainingLogLikelihood")]
    fn training_log_likelihood(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let target = slf.try_borrow()?.core.target()?;
        let out = object::fetch_py(py, &target, "trainingLogLikelihood", &[])?;
        object::fetch_output_to_py(py, out)
    }

    /// The log probability of the current parameter estimate, as `DistributedLDAModel.logPrior`.
    #[pyo3(name = "logPrior")]
    fn log_prior(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let target = slf.try_borrow()?.core.target()?;
        let out = object::fetch_py(py, &target, "logPrior", &[])?;
        object::fetch_output_to_py(py, out)
    }

    /// The checkpoint files of the training, as `DistributedLDAModel.getCheckpointFiles`.
    #[pyo3(name = "getCheckpointFiles")]
    fn get_checkpoint_files(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let target = slf.try_borrow()?.core.target()?;
        let out = object::fetch_py(py, &target, "getCheckpointFiles", &[])?;
        object::fetch_output_to_py(py, out)
    }
}

/// Register the typed clustering models and summaries.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKMeansModel>()?;
    m.add_class::<PyKMeansSummary>()?;
    m.add_class::<PyBisectingKMeansModel>()?;
    m.add_class::<PyBisectingKMeansSummary>()?;
    m.add_class::<PyGaussianMixtureModel>()?;
    m.add_class::<PyGaussianMixtureSummary>()?;
    m.add_class::<PyLocalLDAModel>()?;
    m.add_class::<PyDistributedLDAModel>()?;
    Ok(())
}
