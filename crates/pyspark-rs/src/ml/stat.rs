// SPDX-License-Identifier: Apache-2.0
//! Native bindings for standard pyspark.ml.stat over Spark Connect.
use super::*;
use crate::column::PyColumn;
use pyo3::exceptions::PyAssertionError;
use pyo3::types::PyString;
use spark_connect::ml::{
    ChiSquareTest, Correlation, KolmogorovSmirnovTest, Summarizer, SummaryBuilder,
};

/// The str argument `name` of the helper call of `method`, which `serialize` of
/// `pyspark.ml.connect.serialize` sends as a literal after the dataset (see
/// [`helper_dataset_of`]). Another value raises PySpark's error of its conversion when it cannot be
/// a literal, and otherwise a TypeError before any request, where PySpark sends it for the server
/// to reject.
fn helper_str(value: &Bound<'_, PyAny>, method: &str, name: &str) -> PyResult<String> {
    if value.is_instance_of::<PyString>() {
        return value.extract();
    }
    if !serialized_as_is(value)? {
        crate::functions::to_value_column(value)?;
    }
    Err(PyTypeError::new_err(format!(
        "{method}() takes a str {name}, not '{}'",
        value.get_type().name()?
    )))
}

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
        // In the order of PySpark's serialization of the arguments of the helper call.
        let df = helper_dataset_of(dataset, "test")?;
        let features = helper_str(featuresCol, "test", "featuresCol")?;
        let label = helper_str(labelCol, "test", "labelCol")?;
        Ok(PyDataFrame::new(
            ChiSquareTest::test(&df, &features, &label, flatten).to_pyerr()?,
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
        let df = helper_dataset_of(dataset, "corr")?;
        let column = helper_str(column, "corr", "column")?;
        Ok(PyDataFrame::new(
            Correlation::corr(&df, &column, method).to_pyerr()?,
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
        // PySpark deliberately uses float(), accepting numeric strings as well as numbers, and
        // converts the params first: its list of them is an argument of the helper call, whose
        // serialization then raises its errors of the dataset and the other arguments.
        let float = params.py().import("builtins")?.getattr("float")?;
        let values = params
            .iter()
            .map(|v| float.call1((v,))?.extract::<f64>())
            .collect::<PyResult<Vec<_>>>()?;
        let df = helper_dataset_of(dataset, "test")?;
        let sample = helper_str(sampleCol, "test", "sampleCol")?;
        let dist = helper_str(distName, "test", "distName")?;
        Ok(PyDataFrame::new(
            KolmogorovSmirnovTest::test(&df, &sample, &dist, values).to_pyerr()?,
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

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyChiSquareTest>()?;
    m.add_class::<PyCorrelation>()?;
    m.add_class::<PyKolmogorovSmirnovTest>()?;
    m.add_class::<PySummarizer>()?;
    m.add_class::<PySummaryBuilder>()?;
    Ok(())
}
