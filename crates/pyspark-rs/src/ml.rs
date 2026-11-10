//! PyO3 wrappers for `spark_connect::ml` (mirrors `pyspark.ml.connect`).
//!
//! The core estimators/transformers/evaluators use consuming builder setters and
//! `&mut self` fit/transform; every concrete type is `Clone`, so each wrapper holds
//! the value and clones it for fit/transform/evaluate, keeping the Python objects
//! reusable. A setter replaces the value with the result of the core setter and
//! returns the wrapper itself, as PySpark's `Params._set` does.

use pyo3::prelude::*;
use spark_connect::ml::{
    BinaryClassificationEvaluator, CrossValidator, Estimator, Evaluator, KMeans, LinearRegression,
    LinearSVC, LogisticRegression, MaxAbsScaler, Model, MulticlassClassificationEvaluator,
    Pipeline, RandomForestClassifier, RegressionEvaluator, StandardScaler, StringIndexer,
    Transformer, VectorAssembler,
};

use crate::dataframe::PyDataFrame;
use crate::errors::ResultExt;
use crate::interrupt::interruptible;

/// A fitted ML model (result of `Estimator.fit`), wrapping a boxed core `Model`.
#[pyclass(name = "MLModel", module = "pyspark.ml.connect")]
pub struct PyMLModel {
    model: Box<dyn Model>,
}

#[pymethods]
impl PyMLModel {
    /// Apply the model to a DataFrame.
    fn transform(&self, py: Python<'_>, df: &PyDataFrame) -> PyResult<PyDataFrame> {
        let mut m = self.model.clone_box();
        let out = py
            .detach(|| m.transform(&df.dataframe).map_err(Box::new))
            .to_pyerr()?;
        Ok(PyDataFrame::new(out))
    }
}

/// StandardScaler estimator.
#[pyclass(name = "StandardScaler", module = "pyspark.ml.connect.feature")]
pub struct PyStandardScaler {
    inner: StandardScaler,
}

#[pymethods]
impl PyStandardScaler {
    #[new]
    #[pyo3(signature = (inputCol=None, outputCol=None))]
    #[allow(non_snake_case)]
    fn new(inputCol: Option<String>, outputCol: Option<String>) -> Self {
        let mut e = StandardScaler::new();
        if let Some(c) = inputCol {
            e = e.set_input_col(&c);
        }
        if let Some(c) = outputCol {
            e = e.set_output_col(&c);
        }
        PyStandardScaler { inner: e }
    }
    #[pyo3(name = "setInputCol")]
    fn set_input_col<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_input_col(value);
        slf
    }
    #[pyo3(name = "setOutputCol")]
    fn set_output_col<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_output_col(value);
        slf
    }
    fn fit(&self, py: Python<'_>, df: &PyDataFrame) -> PyResult<PyMLModel> {
        let mut e = self.inner.clone();
        let model = interruptible(py, &df.dataframe.spark_session(), || {
            e.fit(&df.dataframe).map_err(Box::new)
        })?;
        Ok(PyMLModel { model })
    }
}

/// MaxAbsScaler estimator.
#[pyclass(name = "MaxAbsScaler", module = "pyspark.ml.connect.feature")]
pub struct PyMaxAbsScaler {
    inner: MaxAbsScaler,
}

#[pymethods]
impl PyMaxAbsScaler {
    #[new]
    #[pyo3(signature = (inputCol=None, outputCol=None))]
    #[allow(non_snake_case)]
    fn new(inputCol: Option<String>, outputCol: Option<String>) -> Self {
        let mut e = MaxAbsScaler::new();
        if let Some(c) = inputCol {
            e = e.set_input_col(&c);
        }
        if let Some(c) = outputCol {
            e = e.set_output_col(&c);
        }
        PyMaxAbsScaler { inner: e }
    }
    #[pyo3(name = "setInputCol")]
    fn set_input_col<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_input_col(value);
        slf
    }
    #[pyo3(name = "setOutputCol")]
    fn set_output_col<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_output_col(value);
        slf
    }
    fn fit(&self, py: Python<'_>, df: &PyDataFrame) -> PyResult<PyMLModel> {
        let mut e = self.inner.clone();
        let model = interruptible(py, &df.dataframe.spark_session(), || {
            e.fit(&df.dataframe).map_err(Box::new)
        })?;
        Ok(PyMLModel { model })
    }
}

/// StringIndexer estimator.
#[pyclass(name = "StringIndexer", module = "pyspark.ml.connect.feature")]
pub struct PyStringIndexer {
    inner: StringIndexer,
}

#[pymethods]
impl PyStringIndexer {
    #[new]
    #[pyo3(signature = (inputCol=None, outputCol=None))]
    #[allow(non_snake_case)]
    fn new(inputCol: Option<String>, outputCol: Option<String>) -> Self {
        let mut e = StringIndexer::new();
        if let Some(c) = inputCol {
            e = e.set_input_col(&c);
        }
        if let Some(c) = outputCol {
            e = e.set_output_col(&c);
        }
        PyStringIndexer { inner: e }
    }
    #[pyo3(name = "setInputCol")]
    fn set_input_col<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_input_col(value);
        slf
    }
    #[pyo3(name = "setOutputCol")]
    fn set_output_col<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_output_col(value);
        slf
    }
    fn fit(&self, py: Python<'_>, df: &PyDataFrame) -> PyResult<PyMLModel> {
        let mut e = self.inner.clone();
        let model = interruptible(py, &df.dataframe.spark_session(), || {
            e.fit(&df.dataframe).map_err(Box::new)
        })?;
        Ok(PyMLModel { model })
    }
}

/// VectorAssembler transformer.
#[pyclass(name = "VectorAssembler", module = "pyspark.ml.connect.feature")]
pub struct PyVectorAssembler {
    inner: VectorAssembler,
}

#[pymethods]
impl PyVectorAssembler {
    #[new]
    #[pyo3(signature = (inputCols=None, outputCol=None))]
    #[allow(non_snake_case)]
    fn new(inputCols: Option<Vec<String>>, outputCol: Option<String>) -> Self {
        let mut t = VectorAssembler::new();
        if let Some(cols) = inputCols {
            let refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
            t = t.set_input_cols(refs);
        }
        if let Some(c) = outputCol {
            t = t.set_output_col(&c);
        }
        PyVectorAssembler { inner: t }
    }
    #[pyo3(name = "setInputCols")]
    fn set_input_cols(mut slf: PyRefMut<'_, Self>, value: Vec<String>) -> PyRefMut<'_, Self> {
        let refs: Vec<&str> = value.iter().map(|s| s.as_str()).collect();
        slf.inner = slf.inner.clone().set_input_cols(refs);
        slf
    }
    #[pyo3(name = "setOutputCol")]
    fn set_output_col<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_output_col(value);
        slf
    }
    fn transform(&self, py: Python<'_>, df: &PyDataFrame) -> PyResult<PyDataFrame> {
        let mut t = self.inner.clone();
        let out = py
            .detach(|| t.transform(&df.dataframe).map_err(Box::new))
            .to_pyerr()?;
        Ok(PyDataFrame::new(out))
    }
}

/// LogisticRegression estimator.
#[pyclass(
    name = "LogisticRegression",
    module = "pyspark.ml.connect.classification"
)]
pub struct PyLogisticRegression {
    inner: LogisticRegression,
}

#[pymethods]
impl PyLogisticRegression {
    #[new]
    #[pyo3(signature = (featuresCol=None, labelCol=None, predictionCol=None, maxIter=None))]
    #[allow(non_snake_case)]
    fn new(
        featuresCol: Option<String>,
        labelCol: Option<String>,
        predictionCol: Option<String>,
        maxIter: Option<i64>,
    ) -> Self {
        let mut e = LogisticRegression::new();
        if let Some(c) = featuresCol {
            e = e.set_feature_col(&c);
        }
        if let Some(c) = labelCol {
            e = e.set_label_col(&c);
        }
        if let Some(c) = predictionCol {
            e = e.set_prediction_col(&c);
        }
        if let Some(n) = maxIter {
            e = e.set_max_iter(n);
        }
        PyLogisticRegression { inner: e }
    }
    #[pyo3(name = "setFeaturesCol")]
    fn set_features_col<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_feature_col(value);
        slf
    }
    #[pyo3(name = "setLabelCol")]
    fn set_label_col<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_label_col(value);
        slf
    }
    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_prediction_col(value);
        slf
    }
    #[pyo3(name = "setMaxIter")]
    fn set_max_iter(mut slf: PyRefMut<'_, Self>, value: i64) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_max_iter(value);
        slf
    }
    fn fit(&self, py: Python<'_>, df: &PyDataFrame) -> PyResult<PyMLModel> {
        let mut e = self.inner.clone();
        let model = interruptible(py, &df.dataframe.spark_session(), || {
            e.fit(&df.dataframe).map_err(Box::new)
        })?;
        Ok(PyMLModel { model })
    }
}

/// Linear support vector classifier backed by Spark Connect's JVM ML API.
#[pyclass(name = "LinearSVC", module = "pyspark.ml.connect.classification")]
pub struct PyLinearSVC {
    inner: LinearSVC,
}

#[pymethods]
impl PyLinearSVC {
    #[new]
    #[pyo3(signature = (*, featuresCol="features", labelCol="label", predictionCol="prediction", maxIter=100, regParam=0.0, tol=1e-6, rawPredictionCol="rawPrediction", fitIntercept=true, standardization=true, threshold=0.0, weightCol=None, aggregationDepth=2, maxBlockSizeInMB=0.0))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn new(
        featuresCol: &str,
        labelCol: &str,
        predictionCol: &str,
        maxIter: i64,
        regParam: f64,
        tol: f64,
        rawPredictionCol: &str,
        fitIntercept: bool,
        standardization: bool,
        threshold: f64,
        weightCol: Option<&str>,
        aggregationDepth: i64,
        maxBlockSizeInMB: f64,
    ) -> Self {
        let mut inner = LinearSVC::new()
            .set_feature_col(featuresCol)
            .set_label_col(labelCol)
            .set_prediction_col(predictionCol)
            .set_max_iter(maxIter)
            .set_reg_param(regParam)
            .set_tol(tol)
            .set_raw_prediction_col(rawPredictionCol)
            .set_fit_intercept(fitIntercept)
            .set_standardization(standardization)
            .set_threshold(threshold)
            .set_aggregation_depth(aggregationDepth)
            .set_max_block_size_in_mb(maxBlockSizeInMB);
        if let Some(col) = weightCol {
            inner = inner.set_weight_col(col);
        }
        Self { inner }
    }

    #[pyo3(name = "setFeaturesCol")]
    fn set_feature_col<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_feature_col(value);
        slf
    }

    #[pyo3(name = "setLabelCol")]
    fn set_label_col<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_label_col(value);
        slf
    }

    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_prediction_col(value);
        slf
    }

    #[pyo3(name = "setMaxIter")]
    fn set_max_iter(mut slf: PyRefMut<'_, Self>, value: i64) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_max_iter(value);
        slf
    }

    #[pyo3(name = "setRegParam")]
    fn set_reg_param(mut slf: PyRefMut<'_, Self>, value: f64) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_reg_param(value);
        slf
    }

    #[pyo3(name = "setTol")]
    fn set_tol(mut slf: PyRefMut<'_, Self>, value: f64) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_tol(value);
        slf
    }

    #[pyo3(name = "setRawPredictionCol")]
    fn set_raw_prediction_col<'py>(
        mut slf: PyRefMut<'py, Self>,
        value: &str,
    ) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_raw_prediction_col(value);
        slf
    }

    #[pyo3(name = "setFitIntercept")]
    fn set_fit_intercept(mut slf: PyRefMut<'_, Self>, value: bool) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_fit_intercept(value);
        slf
    }

    #[pyo3(name = "setStandardization")]
    fn set_standardization(mut slf: PyRefMut<'_, Self>, value: bool) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_standardization(value);
        slf
    }

    #[pyo3(name = "setThreshold")]
    fn set_threshold(mut slf: PyRefMut<'_, Self>, value: f64) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_threshold(value);
        slf
    }

    #[pyo3(name = "setWeightCol")]
    fn set_weight_col<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_weight_col(value);
        slf
    }

    #[pyo3(name = "setAggregationDepth")]
    fn set_aggregation_depth(mut slf: PyRefMut<'_, Self>, value: i64) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_aggregation_depth(value);
        slf
    }

    #[pyo3(name = "setMaxBlockSizeInMB")]
    fn set_max_block_size_in_mb(mut slf: PyRefMut<'_, Self>, value: f64) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_max_block_size_in_mb(value);
        slf
    }

    fn fit(&self, py: Python<'_>, df: &PyDataFrame) -> PyResult<PyMLModel> {
        let mut estimator = self.inner.clone();
        let model = interruptible(py, &df.dataframe.spark_session(), || {
            estimator.fit(&df.dataframe).map_err(Box::new)
        })?;
        Ok(PyMLModel { model })
    }
}

/// Linear regression backed by Spark Connect's JVM ML API.
#[pyclass(name = "LinearRegression", module = "pyspark.ml.connect.regression")]
pub struct PyLinearRegression {
    inner: LinearRegression,
}

#[pymethods]
impl PyLinearRegression {
    #[new]
    #[pyo3(signature = (*, featuresCol="features", labelCol="label", predictionCol="prediction", maxIter=100, regParam=0.0, elasticNetParam=0.0, tol=1e-6, fitIntercept=true, standardization=true, solver="auto", weightCol=None, aggregationDepth=2, loss="squaredError", epsilon=1.35, maxBlockSizeInMB=0.0))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn new(
        featuresCol: &str,
        labelCol: &str,
        predictionCol: &str,
        maxIter: i32,
        regParam: f64,
        elasticNetParam: f64,
        tol: f64,
        fitIntercept: bool,
        standardization: bool,
        solver: &str,
        weightCol: Option<&str>,
        aggregationDepth: i32,
        loss: &str,
        epsilon: f64,
        maxBlockSizeInMB: f64,
    ) -> Self {
        let mut inner = LinearRegression::new()
            .set_feature_col(featuresCol)
            .set_label_col(labelCol)
            .set_prediction_col(predictionCol)
            .set_max_iter(maxIter)
            .set_reg_param(regParam)
            .set_elastic_net_param(elasticNetParam)
            .set_tol(tol)
            .set_fit_intercept(fitIntercept)
            .set_standardization(standardization)
            .set_solver(solver)
            .set_aggregation_depth(aggregationDepth)
            .set_loss(loss)
            .set_epsilon(epsilon)
            .set_max_block_size_in_mb(maxBlockSizeInMB);
        if let Some(col) = weightCol {
            inner = inner.set_weight_col(col);
        }
        Self { inner }
    }

    #[pyo3(name = "setFeaturesCol")]
    fn set_feature_col<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_feature_col(value);
        slf
    }

    #[pyo3(name = "setLabelCol")]
    fn set_label_col<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_label_col(value);
        slf
    }

    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_prediction_col(value);
        slf
    }

    #[pyo3(name = "setMaxIter")]
    fn set_max_iter(mut slf: PyRefMut<'_, Self>, value: i32) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_max_iter(value);
        slf
    }

    #[pyo3(name = "setRegParam")]
    fn set_reg_param(mut slf: PyRefMut<'_, Self>, value: f64) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_reg_param(value);
        slf
    }

    #[pyo3(name = "setElasticNetParam")]
    fn set_elastic_net_param(mut slf: PyRefMut<'_, Self>, value: f64) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_elastic_net_param(value);
        slf
    }

    #[pyo3(name = "setTol")]
    fn set_tol(mut slf: PyRefMut<'_, Self>, value: f64) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_tol(value);
        slf
    }

    #[pyo3(name = "setFitIntercept")]
    fn set_fit_intercept(mut slf: PyRefMut<'_, Self>, value: bool) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_fit_intercept(value);
        slf
    }

    #[pyo3(name = "setStandardization")]
    fn set_standardization(mut slf: PyRefMut<'_, Self>, value: bool) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_standardization(value);
        slf
    }

    #[pyo3(name = "setSolver")]
    fn set_solver<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_solver(value);
        slf
    }

    #[pyo3(name = "setWeightCol")]
    fn set_weight_col<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_weight_col(value);
        slf
    }

    #[pyo3(name = "setAggregationDepth")]
    fn set_aggregation_depth(mut slf: PyRefMut<'_, Self>, value: i32) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_aggregation_depth(value);
        slf
    }

    #[pyo3(name = "setLoss")]
    fn set_loss<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_loss(value);
        slf
    }

    #[pyo3(name = "setEpsilon")]
    fn set_epsilon(mut slf: PyRefMut<'_, Self>, value: f64) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_epsilon(value);
        slf
    }

    #[pyo3(name = "setMaxBlockSizeInMB")]
    fn set_max_block_size_in_mb(mut slf: PyRefMut<'_, Self>, value: f64) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_max_block_size_in_mb(value);
        slf
    }

    fn fit(&self, py: Python<'_>, df: &PyDataFrame) -> PyResult<PyMLModel> {
        let mut estimator = self.inner.clone();
        let model = interruptible(py, &df.dataframe.spark_session(), || {
            estimator.fit(&df.dataframe).map_err(Box::new)
        })?;
        Ok(PyMLModel { model })
    }
}

/// K-means clustering backed by Spark Connect's JVM ML API.
/// Random forest classifier fitted and cached by Spark Connect.
#[pyclass(
    name = "RandomForestClassifier",
    module = "pyspark.ml.connect.classification"
)]
pub struct PyRandomForestClassifier {
    inner: RandomForestClassifier,
}

#[pymethods]
impl PyRandomForestClassifier {
    #[new]
    #[pyo3(signature = (*, featuresCol="features", labelCol="label", predictionCol="prediction", probabilityCol="probability", rawPredictionCol="rawPrediction", maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, pruneTree=true, maxMemoryInMB=256, cacheNodeIds=false, checkpointInterval=10, impurity="gini", numTrees=20, featureSubsetStrategy="auto", seed=None, subsamplingRate=1.0, leafCol="", minWeightFractionPerNode=0.0, weightCol=None, bootstrap=true))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn new(
        featuresCol: &str,
        labelCol: &str,
        predictionCol: &str,
        probabilityCol: &str,
        rawPredictionCol: &str,
        maxDepth: i32,
        maxBins: i32,
        minInstancesPerNode: i32,
        minInfoGain: f64,
        pruneTree: bool,
        maxMemoryInMB: i32,
        cacheNodeIds: bool,
        checkpointInterval: i32,
        impurity: &str,
        numTrees: i32,
        featureSubsetStrategy: &str,
        seed: Option<i64>,
        subsamplingRate: f64,
        leafCol: &str,
        minWeightFractionPerNode: f64,
        weightCol: Option<&str>,
        bootstrap: Option<bool>,
    ) -> Self {
        let mut inner = RandomForestClassifier::new()
            .set_feature_col(featuresCol)
            .set_label_col(labelCol)
            .set_prediction_col(predictionCol)
            .set_probability_col(probabilityCol)
            .set_raw_prediction_col(rawPredictionCol)
            .set_max_depth(maxDepth)
            .set_max_bins(maxBins)
            .set_min_instances_per_node(minInstancesPerNode)
            .set_min_info_gain(minInfoGain)
            .set_prune_tree(pruneTree)
            .set_max_memory_in_mb(maxMemoryInMB)
            .set_cache_node_ids(cacheNodeIds)
            .set_checkpoint_interval(checkpointInterval)
            .set_impurity(impurity)
            .set_num_trees(numTrees)
            .set_feature_subset_strategy(featureSubsetStrategy)
            .set_subsampling_rate(subsamplingRate)
            .set_leaf_col(leafCol)
            .set_min_weight_fraction_per_node(minWeightFractionPerNode);
        if let Some(value) = seed {
            inner = inner.set_seed(value);
        }
        if let Some(value) = weightCol {
            inner = inner.set_weight_col(value);
        }
        if let Some(value) = bootstrap {
            inner = inner.set_bootstrap(value);
        }
        Self { inner }
    }

    #[pyo3(name = "setFeaturesCol")]
    fn set_feature_col<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_feature_col(value);
        slf
    }

    #[pyo3(name = "setLabelCol")]
    fn set_label_col<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_label_col(value);
        slf
    }

    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_prediction_col(value);
        slf
    }

    #[pyo3(name = "setProbabilityCol")]
    fn set_probability_col<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_probability_col(value);
        slf
    }

    #[pyo3(name = "setRawPredictionCol")]
    fn set_raw_prediction_col<'py>(
        mut slf: PyRefMut<'py, Self>,
        value: &str,
    ) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_raw_prediction_col(value);
        slf
    }

    #[pyo3(name = "setMaxDepth")]
    fn set_max_depth(mut slf: PyRefMut<'_, Self>, value: i32) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_max_depth(value);
        slf
    }

    #[pyo3(name = "setMaxBins")]
    fn set_max_bins(mut slf: PyRefMut<'_, Self>, value: i32) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_max_bins(value);
        slf
    }

    #[pyo3(name = "setMinInstancesPerNode")]
    fn set_min_instances_per_node(mut slf: PyRefMut<'_, Self>, value: i32) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_min_instances_per_node(value);
        slf
    }

    #[pyo3(name = "setMinInfoGain")]
    fn set_min_info_gain(mut slf: PyRefMut<'_, Self>, value: f64) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_min_info_gain(value);
        slf
    }

    #[pyo3(name = "setPruneTree")]
    fn set_prune_tree(mut slf: PyRefMut<'_, Self>, value: bool) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_prune_tree(value);
        slf
    }

    #[pyo3(name = "setMaxMemoryInMB")]
    fn set_max_memory_in_mb(mut slf: PyRefMut<'_, Self>, value: i32) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_max_memory_in_mb(value);
        slf
    }

    #[pyo3(name = "setCacheNodeIds")]
    fn set_cache_node_ids(mut slf: PyRefMut<'_, Self>, value: bool) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_cache_node_ids(value);
        slf
    }

    #[pyo3(name = "setCheckpointInterval")]
    fn set_checkpoint_interval(mut slf: PyRefMut<'_, Self>, value: i32) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_checkpoint_interval(value);
        slf
    }

    #[pyo3(name = "setImpurity")]
    fn set_impurity<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_impurity(value);
        slf
    }

    #[pyo3(name = "setNumTrees")]
    fn set_num_trees(mut slf: PyRefMut<'_, Self>, value: i32) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_num_trees(value);
        slf
    }

    #[pyo3(name = "setFeatureSubsetStrategy")]
    fn set_feature_subset_strategy<'py>(
        mut slf: PyRefMut<'py, Self>,
        value: &str,
    ) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_feature_subset_strategy(value);
        slf
    }

    #[pyo3(name = "setSeed")]
    fn set_seed(mut slf: PyRefMut<'_, Self>, value: i64) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_seed(value);
        slf
    }

    #[pyo3(name = "setSubsamplingRate")]
    fn set_subsampling_rate(mut slf: PyRefMut<'_, Self>, value: f64) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_subsampling_rate(value);
        slf
    }

    #[pyo3(name = "setLeafCol")]
    fn set_leaf_col<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_leaf_col(value);
        slf
    }

    #[pyo3(name = "setMinWeightFractionPerNode")]
    fn set_min_weight_fraction_per_node(
        mut slf: PyRefMut<'_, Self>,
        value: f64,
    ) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_min_weight_fraction_per_node(value);
        slf
    }

    #[pyo3(name = "setWeightCol")]
    fn set_weight_col<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_weight_col(value);
        slf
    }

    #[pyo3(name = "setBootstrap")]
    fn set_bootstrap(mut slf: PyRefMut<'_, Self>, value: bool) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_bootstrap(value);
        slf
    }

    #[pyo3(name = "setThresholds")]
    fn set_thresholds(mut slf: PyRefMut<'_, Self>, value: Vec<f64>) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_thresholds(value);
        slf
    }

    fn fit(&self, py: Python<'_>, df: &PyDataFrame) -> PyResult<PyMLModel> {
        let mut estimator = self.inner.clone();
        let model = interruptible(py, &df.dataframe.spark_session(), || {
            estimator.fit(&df.dataframe).map_err(Box::new)
        })?;
        Ok(PyMLModel { model })
    }
}

#[pyclass(name = "KMeans", module = "pyspark.ml.connect.clustering")]
pub struct PyKMeans {
    inner: KMeans,
}

#[pymethods]
impl PyKMeans {
    #[new]
    #[pyo3(signature = (*, featuresCol="features", predictionCol="prediction", k=2, initMode="k-means||", initSteps=2, tol=1e-4, maxIter=20, seed=None, distanceMeasure="euclidean", weightCol=None, solver="auto", maxBlockSizeInMB=0.0))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn new(
        featuresCol: &str,
        predictionCol: &str,
        k: i32,
        initMode: &str,
        initSteps: i32,
        tol: f64,
        maxIter: i32,
        seed: Option<i64>,
        distanceMeasure: &str,
        weightCol: Option<&str>,
        solver: &str,
        maxBlockSizeInMB: f64,
    ) -> Self {
        let mut inner = KMeans::new()
            .set_feature_col(featuresCol)
            .set_prediction_col(predictionCol)
            .set_k(k)
            .set_init_mode(initMode)
            .set_init_steps(initSteps)
            .set_tol(tol)
            .set_max_iter(maxIter)
            .set_distance_measure(distanceMeasure)
            .set_solver(solver)
            .set_max_block_size_in_mb(maxBlockSizeInMB);
        if let Some(seed) = seed {
            inner = inner.set_seed(seed);
        }
        if let Some(col) = weightCol {
            inner = inner.set_weight_col(col);
        }
        Self { inner }
    }

    #[pyo3(name = "setFeaturesCol")]
    fn set_feature_col<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_feature_col(value);
        slf
    }

    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_prediction_col(value);
        slf
    }

    #[pyo3(name = "setK")]
    fn set_k(mut slf: PyRefMut<'_, Self>, value: i32) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_k(value);
        slf
    }

    #[pyo3(name = "setInitMode")]
    fn set_init_mode<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_init_mode(value);
        slf
    }

    #[pyo3(name = "setInitSteps")]
    fn set_init_steps(mut slf: PyRefMut<'_, Self>, value: i32) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_init_steps(value);
        slf
    }

    #[pyo3(name = "setTol")]
    fn set_tol(mut slf: PyRefMut<'_, Self>, value: f64) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_tol(value);
        slf
    }

    #[pyo3(name = "setMaxIter")]
    fn set_max_iter(mut slf: PyRefMut<'_, Self>, value: i32) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_max_iter(value);
        slf
    }

    #[pyo3(name = "setSeed")]
    fn set_seed(mut slf: PyRefMut<'_, Self>, value: i64) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_seed(value);
        slf
    }

    #[pyo3(name = "setDistanceMeasure")]
    fn set_distance_measure<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_distance_measure(value);
        slf
    }

    #[pyo3(name = "setWeightCol")]
    fn set_weight_col<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_weight_col(value);
        slf
    }

    #[pyo3(name = "setSolver")]
    fn set_solver<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_solver(value);
        slf
    }

    #[pyo3(name = "setMaxBlockSizeInMB")]
    fn set_max_block_size_in_mb(mut slf: PyRefMut<'_, Self>, value: f64) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_max_block_size_in_mb(value);
        slf
    }

    fn fit(&self, py: Python<'_>, df: &PyDataFrame) -> PyResult<PyMLModel> {
        let mut estimator = self.inner.clone();
        let model = interruptible(py, &df.dataframe.spark_session(), || {
            estimator.fit(&df.dataframe).map_err(Box::new)
        })?;
        Ok(PyMLModel { model })
    }
}

/// RegressionEvaluator.
#[pyclass(name = "RegressionEvaluator", module = "pyspark.ml.connect.evaluation")]
pub struct PyRegressionEvaluator {
    inner: RegressionEvaluator,
}

#[pymethods]
impl PyRegressionEvaluator {
    #[new]
    #[pyo3(signature = (labelCol=None, predictionCol=None, metricName=None))]
    #[allow(non_snake_case)]
    fn new(
        labelCol: Option<String>,
        predictionCol: Option<String>,
        metricName: Option<String>,
    ) -> Self {
        let mut e = RegressionEvaluator::new();
        if let Some(c) = labelCol {
            e = e.set_label_col(&c);
        }
        if let Some(c) = predictionCol {
            e = e.set_prediction_col(&c);
        }
        if let Some(m) = metricName {
            e = e.set_metric_name(&m);
        }
        PyRegressionEvaluator { inner: e }
    }
    #[pyo3(name = "setLabelCol")]
    fn set_label_col<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_label_col(value);
        slf
    }
    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_prediction_col(value);
        slf
    }
    #[pyo3(name = "setMetricName")]
    fn set_metric_name<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_metric_name(value);
        slf
    }
    fn evaluate(&self, py: Python<'_>, df: &PyDataFrame) -> PyResult<f64> {
        interruptible(py, &df.dataframe.spark_session(), || {
            self.inner.evaluate(&df.dataframe).map_err(Box::new)
        })
    }
}

/// BinaryClassificationEvaluator.
#[pyclass(
    name = "BinaryClassificationEvaluator",
    module = "pyspark.ml.connect.evaluation"
)]
pub struct PyBinaryClassificationEvaluator {
    inner: BinaryClassificationEvaluator,
}

#[pymethods]
impl PyBinaryClassificationEvaluator {
    #[new]
    #[pyo3(signature = (labelCol=None, rawPredictionCol=None, metricName=None))]
    #[allow(non_snake_case)]
    fn new(
        labelCol: Option<String>,
        rawPredictionCol: Option<String>,
        metricName: Option<String>,
    ) -> Self {
        let mut e = BinaryClassificationEvaluator::new();
        if let Some(c) = labelCol {
            e = e.set_label_col(&c);
        }
        if let Some(c) = rawPredictionCol {
            e = e.set_raw_prediction_col(&c);
        }
        if let Some(m) = metricName {
            e = e.set_metric_name(&m);
        }
        PyBinaryClassificationEvaluator { inner: e }
    }
    #[pyo3(name = "setLabelCol")]
    fn set_label_col<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_label_col(value);
        slf
    }
    #[pyo3(name = "setRawPredictionCol")]
    fn set_raw_prediction_col<'py>(
        mut slf: PyRefMut<'py, Self>,
        value: &str,
    ) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_raw_prediction_col(value);
        slf
    }
    #[pyo3(name = "setMetricName")]
    fn set_metric_name<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_metric_name(value);
        slf
    }
    fn evaluate(&self, py: Python<'_>, df: &PyDataFrame) -> PyResult<f64> {
        interruptible(py, &df.dataframe.spark_session(), || {
            self.inner.evaluate(&df.dataframe).map_err(Box::new)
        })
    }
}

/// Pipeline estimator (chains stages by name).
#[pyclass(name = "Pipeline", module = "pyspark.ml.connect.pipeline")]
pub struct PyPipeline {
    inner: Pipeline,
}

#[pymethods]
impl PyPipeline {
    #[new]
    #[pyo3(signature = (stages=None))]
    fn new(stages: Option<Vec<String>>) -> Self {
        let mut p = Pipeline::new();
        if let Some(s) = stages {
            let refs: Vec<&str> = s.iter().map(|x| x.as_str()).collect();
            p = p.set_stages(refs);
        }
        PyPipeline { inner: p }
    }
    #[pyo3(name = "setStages")]
    fn set_stages(mut slf: PyRefMut<'_, Self>, value: Vec<String>) -> PyRefMut<'_, Self> {
        let refs: Vec<&str> = value.iter().map(|x| x.as_str()).collect();
        slf.inner = slf.inner.clone().set_stages(refs);
        slf
    }
    fn fit(&self, py: Python<'_>, df: &PyDataFrame) -> PyResult<PyMLModel> {
        let mut e = self.inner.clone();
        let model = interruptible(py, &df.dataframe.spark_session(), || {
            e.fit(&df.dataframe).map_err(Box::new)
        })?;
        Ok(PyMLModel { model })
    }
}

/// MulticlassClassificationEvaluator.
#[pyclass(
    name = "MulticlassClassificationEvaluator",
    module = "pyspark.ml.connect.evaluation"
)]
pub struct PyMulticlassClassificationEvaluator {
    inner: MulticlassClassificationEvaluator,
}

#[pymethods]
impl PyMulticlassClassificationEvaluator {
    #[new]
    #[pyo3(signature = (labelCol=None, predictionCol=None, metricName=None))]
    #[allow(non_snake_case)]
    fn new(
        labelCol: Option<String>,
        predictionCol: Option<String>,
        metricName: Option<String>,
    ) -> Self {
        let mut e = MulticlassClassificationEvaluator::new();
        if let Some(c) = labelCol {
            e = e.set_label_col(&c);
        }
        if let Some(c) = predictionCol {
            e = e.set_prediction_col(&c);
        }
        if let Some(m) = metricName {
            e = e.set_metric_name(&m);
        }
        PyMulticlassClassificationEvaluator { inner: e }
    }
    #[pyo3(name = "setLabelCol")]
    fn set_label_col<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_label_col(value);
        slf
    }
    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_prediction_col(value);
        slf
    }
    #[pyo3(name = "setMetricName")]
    fn set_metric_name<'py>(mut slf: PyRefMut<'py, Self>, value: &str) -> PyRefMut<'py, Self> {
        slf.inner = slf.inner.clone().set_metric_name(value);
        slf
    }
    fn evaluate(&self, py: Python<'_>, df: &PyDataFrame) -> PyResult<f64> {
        interruptible(py, &df.dataframe.spark_session(), || {
            self.inner.evaluate(&df.dataframe).map_err(Box::new)
        })
    }
}

/// CrossValidator (k-fold tuning).
#[pyclass(name = "CrossValidator", module = "pyspark.ml.connect.tuning")]
pub struct PyCrossValidator {
    inner: CrossValidator,
}

#[pymethods]
impl PyCrossValidator {
    #[new]
    #[pyo3(signature = (numFolds=None, parallelism=None, seed=None))]
    #[allow(non_snake_case)]
    fn new(numFolds: Option<i32>, parallelism: Option<i32>, seed: Option<i64>) -> Self {
        let mut e = CrossValidator::new();
        if let Some(n) = numFolds {
            e = e.set_num_folds(n);
        }
        if let Some(p) = parallelism {
            e = e.set_parallelism(p);
        }
        if let Some(s) = seed {
            e = e.set_seed(s);
        }
        PyCrossValidator { inner: e }
    }
    #[pyo3(name = "setNumFolds")]
    fn set_num_folds(mut slf: PyRefMut<'_, Self>, value: i32) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_num_folds(value);
        slf
    }
    #[pyo3(name = "setParallelism")]
    fn set_parallelism(mut slf: PyRefMut<'_, Self>, value: i32) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_parallelism(value);
        slf
    }
    #[pyo3(name = "setSeed")]
    fn set_seed(mut slf: PyRefMut<'_, Self>, value: i64) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().set_seed(value);
        slf
    }
    fn fit(&self, py: Python<'_>, df: &PyDataFrame) -> PyResult<PyMLModel> {
        let mut e = self.inner.clone();
        let model = interruptible(py, &df.dataframe.spark_session(), || {
            e.fit(&df.dataframe).map_err(Box::new)
        })?;
        Ok(PyMLModel { model })
    }
}
