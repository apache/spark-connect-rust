// SPDX-License-Identifier: Apache-2.0
//! Native bindings for standard pyspark.ml.stat over Spark Connect.
use super::*;
use crate::column::PyColumn;
use pyo3::exceptions::PyAssertionError;
use spark_connect::ml::{
    ChiSquareTest, Correlation, KolmogorovSmirnovTest, Summarizer, SummaryBuilder,
};

#[pyclass(name = "ChiSquareTest", module = "pyspark.ml.stat")]
pub struct PyChiSquareTest;
#[pymethods]
impl PyChiSquareTest {
    #[staticmethod]
    #[pyo3(signature = (dataset, featuresCol, labelCol, flatten=false))]
    #[allow(non_snake_case)]
    fn test(
        dataset: &Bound<'_, PyAny>,
        featuresCol: &Bound<'_, PyAny>,
        labelCol: &Bound<'_, PyAny>,
        flatten: bool,
    ) -> PyResult<PyDataFrame> {
        let df = dataset_of(dataset, "test", "_plan")?;
        Ok(PyDataFrame::new(
            ChiSquareTest::test(
                &df,
                &featuresCol.extract::<String>()?,
                &labelCol.extract::<String>()?,
                flatten,
            )
            .to_pyerr()?,
        ))
    }
}

#[pyclass(name = "Correlation", module = "pyspark.ml.stat")]
pub struct PyCorrelation;
#[pymethods]
impl PyCorrelation {
    #[staticmethod]
    #[pyo3(signature = (dataset, column, method="pearson"))]
    fn corr(
        dataset: &Bound<'_, PyAny>,
        column: &Bound<'_, PyAny>,
        method: &str,
    ) -> PyResult<PyDataFrame> {
        let df = dataset_of(dataset, "corr", "_plan")?;
        Ok(PyDataFrame::new(
            Correlation::corr(&df, &column.extract::<String>()?, method).to_pyerr()?,
        ))
    }
}

#[pyclass(name = "KolmogorovSmirnovTest", module = "pyspark.ml.stat")]
pub struct PyKolmogorovSmirnovTest;
#[pymethods]
impl PyKolmogorovSmirnovTest {
    #[staticmethod]
    #[pyo3(signature = (dataset, sampleCol, distName, *params))]
    #[allow(non_snake_case)]
    fn test(
        dataset: &Bound<'_, PyAny>,
        sampleCol: &Bound<'_, PyAny>,
        distName: &Bound<'_, PyAny>,
        params: &Bound<'_, PyTuple>,
    ) -> PyResult<PyDataFrame> {
        let df = dataset_of(dataset, "test", "_plan")?;
        // PySpark deliberately uses float(), accepting numeric strings as well as numbers.
        let float = params.py().import("builtins")?.getattr("float")?;
        let values = params
            .iter()
            .map(|v| float.call1((v,))?.extract::<f64>())
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyDataFrame::new(
            KolmogorovSmirnovTest::test(
                &df,
                &sampleCol.extract::<String>()?,
                &distName.extract::<String>()?,
                values,
            )
            .to_pyerr()?,
        ))
    }
}

#[pyclass(name = "SummaryBuilder", module = "pyspark.ml.stat")]
pub struct PySummaryBuilder {
    metrics: Vec<Py<PyAny>>,
}
#[pymethods]
impl PySummaryBuilder {
    #[pyo3(signature = (featuresCol, weightCol=None))]
    #[allow(non_snake_case)]
    fn summary(
        &self,
        py: Python<'_>,
        featuresCol: &Bound<'_, PyAny>,
        weightCol: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyColumn> {
        // As in the remote SummaryBuilder, metric literals are converted at summary(),
        // and non-Column arguments fail the _invoke_function assertion.
        let metrics = self
            .metrics
            .iter()
            .map(|v| crate::functions::to_value_column(v.bind(py)))
            .collect::<PyResult<Vec<_>>>()?;
        let features = featuresCol
            .extract::<PyRef<'_, PyColumn>>()
            .map_err(|_| PyAssertionError::new_err(()))?
            .column
            .clone();
        let weight = weightCol
            .map(|v| {
                v.extract::<PyRef<'_, PyColumn>>()
                    .map(|c| c.column.clone())
                    .map_err(|_| PyAssertionError::new_err(()))
            })
            .transpose()?;
        Ok(PyColumn::new(
            SummaryBuilder::from_metric_columns(metrics).summary(features, weight),
        ))
    }
}

#[pyclass(name = "Summarizer", module = "pyspark.ml.stat")]
pub struct PySummarizer;
fn columns(
    col: &Bound<'_, PyAny>,
    weight: Option<&Bound<'_, PyAny>>,
) -> PyResult<(spark_connect::Column, Option<spark_connect::Column>)> {
    let error = || PyTypeError::new_err("featureCol and weightCol should be a Column");
    let col = col
        .extract::<PyRef<'_, PyColumn>>()
        .map_err(|_| error())?
        .column
        .clone();
    let weight = weight
        .map(|v| {
            v.extract::<PyRef<'_, PyColumn>>()
                .map(|c| c.column.clone())
                .map_err(|_| error())
        })
        .transpose()?;
    Ok((col, weight))
}
macro_rules! metrics {
    ($(($name:ident, $python:literal)),* $(,)?) => { #[pymethods] impl PySummarizer {
        #[staticmethod]
        #[pyo3(signature = (*metrics))]
        fn metrics(metrics: &Bound<'_, PyTuple>) -> PySummaryBuilder {
            PySummaryBuilder { metrics: metrics.iter().map(Bound::unbind).collect() }
        }
        $(
        #[staticmethod]
        #[pyo3(name=$python, signature=(col, weightCol=None))]
        #[allow(non_snake_case)]
        fn $name(col: &Bound<'_, PyAny>, weightCol: Option<&Bound<'_, PyAny>>) -> PyResult<PyColumn> {
            let (col, weight) = columns(col, weightCol)?;
            Ok(PyColumn::new(Summarizer::$name(col, weight)))
        }
    )* } };
}
metrics!(
    (mean, "mean"),
    (sum, "sum"),
    (variance, "variance"),
    (std, "std"),
    (count, "count"),
    (num_non_zeros, "numNonZeros"),
    (max, "max"),
    (min, "min"),
    (norm_l1, "normL1"),
    (norm_l2, "normL2")
);

// PySpark's holder keeps the original Python objects, with no type/shape validation.
#[pyclass(name = "MultivariateGaussian", module = "pyspark.ml.stat", dict)]
pub struct PyMultivariateGaussian {
    #[pyo3(get, set)]
    mean: Py<PyAny>,
    #[pyo3(get, set)]
    cov: Py<PyAny>,
}
#[pymethods]
impl PyMultivariateGaussian {
    #[new]
    fn new(mean: Py<PyAny>, cov: Py<PyAny>) -> Self {
        Self { mean, cov }
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyChiSquareTest>()?;
    m.add_class::<PyCorrelation>()?;
    m.add_class::<PyKolmogorovSmirnovTest>()?;
    m.add_class::<PySummarizer>()?;
    m.add_class::<PySummaryBuilder>()?;
    m.add_class::<PyMultivariateGaussian>()?;
    Ok(())
}
