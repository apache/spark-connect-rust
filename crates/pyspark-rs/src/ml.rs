//! PyO3 wrappers for `spark_connect::ml` (mirrors `pyspark.ml.connect`).
//!
//! The core estimators/transformers/evaluators use consuming builder setters and
//! `&mut self` fit/transform; every concrete type is `Clone`, so each wrapper holds
//! the value and clones it for the fluent setters and for fit/transform/evaluate,
//! keeping the Python objects reusable.

use pyo3::prelude::*;
use spark_connect::ml::{
    BinaryClassificationEvaluator, CrossValidator, Estimator, Evaluator, LogisticRegression,
    MaxAbsScaler, Model, MulticlassClassificationEvaluator, Pipeline, RegressionEvaluator,
    StandardScaler, StringIndexer, Transformer, VectorAssembler,
};

use crate::dataframe::PyDataFrame;
use crate::errors::ResultExt;

/// A fitted ML model (result of `Estimator.fit`), wrapping a boxed core `Model`.
#[pyclass(name = "MLModel")]
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
#[pyclass(name = "StandardScaler")]
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
#[pyclass(name = "MaxAbsScaler")]
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
#[pyclass(name = "StringIndexer")]
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
#[pyclass(name = "VectorAssembler")]
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
#[pyclass(name = "LogisticRegression")]
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

/// RegressionEvaluator.
#[pyclass(name = "RegressionEvaluator")]
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
#[pyclass(name = "BinaryClassificationEvaluator")]
pub struct PyBinaryClassificationEvaluator {
    inner: BinaryClassificationEvaluator,
}

#[pymethods]
impl PyBinaryClassificationEvaluator {
    #[new]
    #[pyo3(signature = (labelCol=None, scoreCol=None, metricName=None))]
    #[allow(non_snake_case)]
    fn new(labelCol: Option<String>, scoreCol: Option<String>, metricName: Option<String>) -> Self {
        let mut e = BinaryClassificationEvaluator::new();
        if let Some(c) = labelCol {
            e = e.set_label_col(&c);
        }
        if let Some(c) = scoreCol {
            e = e.set_score_col(&c);
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
    #[pyo3(name = "setScoreCol")]
    fn set_score_col(&self, col: &str) -> Self {
        PyBinaryClassificationEvaluator {
            inner: self.inner.clone().set_score_col(col),
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
#[pyclass(name = "Pipeline")]
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
#[pyclass(name = "MulticlassClassificationEvaluator")]
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
#[pyclass(name = "CrossValidator")]
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
