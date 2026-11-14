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
        "intermediateStorageLevel" => ("getIntermediateStorageLevel", get_intermediate_storage_level, param_intermediate_storage_level),
        "k" => ("getK", get_k, param_k),
        "maxBlockSizeInMB" => ("getMaxBlockSizeInMB", get_max_block_size_in_mb, param_max_block_size_in_mb),
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
        "predict" => (predict, "predict", (value: &Bound<'py, PyAny>)),
    ];
    relations: [];

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
        let py = slf.py();
        slf.try_borrow()?.core.summary_object(slf.as_any(), |target| {
            Ok(Py::new(py, PyKMeansSummary::from(target))?.into_any())
        })
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
}

/// Register the typed clustering models and summaries.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKMeansModel>()?;
    m.add_class::<PyKMeansSummary>()?;
    Ok(())
}
