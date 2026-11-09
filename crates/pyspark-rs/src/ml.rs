//! PyO3 wrappers for `spark_connect::ml` (mirrors `pyspark.ml.connect`).
//!
//! The core estimators/transformers/evaluators use consuming builder setters and
//! `&mut self` fit/transform; every concrete type is `Clone`, so each wrapper holds
//! the value and clones it for the fluent setters and for fit/transform/evaluate,
//! keeping the Python objects reusable.

use pyo3::prelude::*;
use spark_connect::ml::{
    BinaryClassificationEvaluator, CrossValidator, Estimator, Evaluator, KMeans, LinearRegression,
    LinearSVC, LogisticRegression, MaxAbsScaler, Model, MulticlassClassificationEvaluator,
    Pipeline, RegressionEvaluator, StandardScaler, StringIndexer, Transformer, VectorAssembler,
};

use crate::dataframe::PyDataFrame;
use crate::errors::ResultExt;

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
        let out = py.detach(|| m.transform(&df.dataframe)).to_pyerr()?;
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
    fn set_input_col(&self, col: &str) -> Self {
        PyStandardScaler {
            inner: self.inner.clone().set_input_col(col),
        }
    }
    #[pyo3(name = "setOutputCol")]
    fn set_output_col(&self, col: &str) -> Self {
        PyStandardScaler {
            inner: self.inner.clone().set_output_col(col),
        }
    }
    fn fit(&self, py: Python<'_>, df: &PyDataFrame) -> PyResult<PyMLModel> {
        let mut e = self.inner.clone();
        let model = py.detach(|| e.fit(&df.dataframe)).to_pyerr()?;
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
    fn set_input_col(&self, col: &str) -> Self {
        PyMaxAbsScaler {
            inner: self.inner.clone().set_input_col(col),
        }
    }
    #[pyo3(name = "setOutputCol")]
    fn set_output_col(&self, col: &str) -> Self {
        PyMaxAbsScaler {
            inner: self.inner.clone().set_output_col(col),
        }
    }
    fn fit(&self, py: Python<'_>, df: &PyDataFrame) -> PyResult<PyMLModel> {
        let mut e = self.inner.clone();
        let model = py.detach(|| e.fit(&df.dataframe)).to_pyerr()?;
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
    fn set_input_col(&self, col: &str) -> Self {
        PyStringIndexer {
            inner: self.inner.clone().set_input_col(col),
        }
    }
    #[pyo3(name = "setOutputCol")]
    fn set_output_col(&self, col: &str) -> Self {
        PyStringIndexer {
            inner: self.inner.clone().set_output_col(col),
        }
    }
    fn fit(&self, py: Python<'_>, df: &PyDataFrame) -> PyResult<PyMLModel> {
        let mut e = self.inner.clone();
        let model = py.detach(|| e.fit(&df.dataframe)).to_pyerr()?;
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
    fn set_input_cols(&self, cols: Vec<String>) -> Self {
        let refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        PyVectorAssembler {
            inner: self.inner.clone().set_input_cols(refs),
        }
    }
    #[pyo3(name = "setOutputCol")]
    fn set_output_col(&self, col: &str) -> Self {
        PyVectorAssembler {
            inner: self.inner.clone().set_output_col(col),
        }
    }
    fn transform(&self, py: Python<'_>, df: &PyDataFrame) -> PyResult<PyDataFrame> {
        let mut t = self.inner.clone();
        let out = py.detach(|| t.transform(&df.dataframe)).to_pyerr()?;
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
    fn set_features_col(&self, col: &str) -> Self {
        PyLogisticRegression {
            inner: self.inner.clone().set_feature_col(col),
        }
    }
    #[pyo3(name = "setLabelCol")]
    fn set_label_col(&self, col: &str) -> Self {
        PyLogisticRegression {
            inner: self.inner.clone().set_label_col(col),
        }
    }
    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col(&self, col: &str) -> Self {
        PyLogisticRegression {
            inner: self.inner.clone().set_prediction_col(col),
        }
    }
    #[pyo3(name = "setMaxIter")]
    fn set_max_iter(&self, max_iter: i64) -> Self {
        PyLogisticRegression {
            inner: self.inner.clone().set_max_iter(max_iter),
        }
    }
    fn fit(&self, py: Python<'_>, df: &PyDataFrame) -> PyResult<PyMLModel> {
        let mut e = self.inner.clone();
        let model = py.detach(|| e.fit(&df.dataframe)).to_pyerr()?;
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
        let model = py
            .detach(|| estimator.fit(&df.dataframe).map_err(Box::new))
            .to_pyerr()?;
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
        let model = py
            .detach(|| estimator.fit(&df.dataframe).map_err(Box::new))
            .to_pyerr()?;
        Ok(PyMLModel { model })
    }
}

/// K-means clustering backed by Spark Connect's JVM ML API.
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
        let model = py
            .detach(|| estimator.fit(&df.dataframe).map_err(Box::new))
            .to_pyerr()?;
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
    fn set_label_col(&self, col: &str) -> Self {
        PyRegressionEvaluator {
            inner: self.inner.clone().set_label_col(col),
        }
    }
    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col(&self, col: &str) -> Self {
        PyRegressionEvaluator {
            inner: self.inner.clone().set_prediction_col(col),
        }
    }
    #[pyo3(name = "setMetricName")]
    fn set_metric_name(&self, metric: &str) -> Self {
        PyRegressionEvaluator {
            inner: self.inner.clone().set_metric_name(metric),
        }
    }
    fn evaluate(&self, py: Python<'_>, df: &PyDataFrame) -> PyResult<f64> {
        py.detach(|| self.inner.evaluate(&df.dataframe)).to_pyerr()
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
    fn set_label_col(&self, col: &str) -> Self {
        PyBinaryClassificationEvaluator {
            inner: self.inner.clone().set_label_col(col),
        }
    }
    #[pyo3(name = "setRawPredictionCol")]
    fn set_raw_prediction_col(&self, col: &str) -> Self {
        PyBinaryClassificationEvaluator {
            inner: self.inner.clone().set_raw_prediction_col(col),
        }
    }
    #[pyo3(name = "setMetricName")]
    fn set_metric_name(&self, metric: &str) -> Self {
        PyBinaryClassificationEvaluator {
            inner: self.inner.clone().set_metric_name(metric),
        }
    }
    fn evaluate(&self, py: Python<'_>, df: &PyDataFrame) -> PyResult<f64> {
        py.detach(|| self.inner.evaluate(&df.dataframe)).to_pyerr()
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
    fn set_stages(&self, stages: Vec<String>) -> Self {
        let refs: Vec<&str> = stages.iter().map(|x| x.as_str()).collect();
        PyPipeline {
            inner: self.inner.clone().set_stages(refs),
        }
    }
    fn fit(&self, py: Python<'_>, df: &PyDataFrame) -> PyResult<PyMLModel> {
        let mut e = self.inner.clone();
        let model = py.detach(|| e.fit(&df.dataframe)).to_pyerr()?;
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
    fn set_label_col(&self, col: &str) -> Self {
        PyMulticlassClassificationEvaluator {
            inner: self.inner.clone().set_label_col(col),
        }
    }
    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col(&self, col: &str) -> Self {
        PyMulticlassClassificationEvaluator {
            inner: self.inner.clone().set_prediction_col(col),
        }
    }
    #[pyo3(name = "setMetricName")]
    fn set_metric_name(&self, metric: &str) -> Self {
        PyMulticlassClassificationEvaluator {
            inner: self.inner.clone().set_metric_name(metric),
        }
    }
    fn evaluate(&self, py: Python<'_>, df: &PyDataFrame) -> PyResult<f64> {
        py.detach(|| self.inner.evaluate(&df.dataframe)).to_pyerr()
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
    fn set_num_folds(&self, num_folds: i32) -> Self {
        PyCrossValidator {
            inner: self.inner.clone().set_num_folds(num_folds),
        }
    }
    #[pyo3(name = "setParallelism")]
    fn set_parallelism(&self, parallelism: i32) -> Self {
        PyCrossValidator {
            inner: self.inner.clone().set_parallelism(parallelism),
        }
    }
    #[pyo3(name = "setSeed")]
    fn set_seed(&self, seed: i64) -> Self {
        PyCrossValidator {
            inner: self.inner.clone().set_seed(seed),
        }
    }
    fn fit(&self, py: Python<'_>, df: &PyDataFrame) -> PyResult<PyMLModel> {
        let mut e = self.inner.clone();
        let model = py.detach(|| e.fit(&df.dataframe)).to_pyerr()?;
        Ok(PyMLModel { model })
    }
}
