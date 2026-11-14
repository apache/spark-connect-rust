// SPDX-License-Identifier: Apache-2.0
//! Native bindings for standard pyspark.ml.stat over Spark Connect.
use super::object::fetch_arg_of;
use super::*;
use crate::column::PyColumn;
use pyo3::exceptions::PyAssertionError;
use spark_connect::ml::{attribute_relation, FetchArg, MlTarget, Summarizer, SummaryBuilder};
use spark_connect_proto as proto;

use crate::options::DefaultedOption;

/// The DataFrame of the helper call `method` of `pyspark.ml.stat`, of the arguments `args` (the
/// dataset first) and `typed`, as `invoke_helper_relation` of `pyspark.ml.util` builds it: of the
/// active session of the thread (`SparkSession.getActiveSession()`, whose absence is the
/// AssertionError of its `assert`, before the arguments), also when the dataset is a DataFrame of
/// another session, and of each argument as `serialize` of `pyspark.ml.connect.serialize` sends
/// it, in their order: a DataFrame as its relation, and another value as the literal of
/// `LiteralExpression._from_value`, which raises PySpark's error of a value that cannot be one
/// (`UNSUPPORTED_DATA_TYPE`, also of a pandas DataFrame, before any request). As PySpark does, a
/// literal of another type than the helper takes, such as a dataset that is not a DataFrame or a
/// column name that is not a str, is sent for the server to reject when the result is analyzed or
/// collected. `typed` is the literal of the argument that PySpark sends as a `(value, DataType)`
/// pair.
fn helper_relation(
    method: &str,
    args: &[&Bound<'_, PyAny>],
    typed: Option<proto::expression::Literal>,
) -> PyResult<PyDataFrame> {
    let py = args[0].py();
    let session = crate::session::active_session(py)
        .ok_or_else(|| PyAssertionError::new_err(()))?
        .borrow(py)
        .session
        .clone();
    let mut args = args
        .iter()
        .map(|arg| fetch_arg_of(arg))
        .collect::<PyResult<Vec<_>>>()?;
    args.extend(typed.map(FetchArg::Literal));
    Ok(PyDataFrame::new(
        attribute_relation(&MlTarget::helper(&session), method, args).to_pyerr()?,
    ))
}

#[pyclass(name = "ChiSquareTest", module = "pyspark.ml.stat")]
pub struct PyChiSquareTest;
#[pymethods]
impl PyChiSquareTest {
    #[staticmethod]
    #[pyo3(
        signature = (dataset, featuresCol, labelCol, flatten=DefaultedOption::Omitted),
        text_signature = "(dataset, featuresCol, labelCol, flatten=False)"
    )]
    #[allow(non_snake_case)]
    fn test(
        py: Python<'_>,
        dataset: &Bound<'_, PyAny>,
        featuresCol: &Bound<'_, PyAny>,
        labelCol: &Bound<'_, PyAny>,
        flatten: DefaultedOption,
    ) -> PyResult<PyDataFrame> {
        let flatten = flatten.or_bool(py, false);
        helper_relation(
            "chiSquareTest",
            &[dataset, featuresCol, labelCol, &flatten],
            None,
        )
    }
}

#[pyclass(name = "Correlation", module = "pyspark.ml.stat")]
pub struct PyCorrelation;
#[pymethods]
impl PyCorrelation {
    #[staticmethod]
    #[pyo3(
        signature = (dataset, column, method=DefaultedOption::Omitted),
        text_signature = "(dataset, column, method='pearson')"
    )]
    fn corr(
        py: Python<'_>,
        dataset: &Bound<'_, PyAny>,
        column: &Bound<'_, PyAny>,
        method: DefaultedOption,
    ) -> PyResult<PyDataFrame> {
        let method = match method {
            DefaultedOption::Omitted => pyo3::types::PyString::new(py, "pearson").into_any(),
            DefaultedOption::Given(method) => method.into_bound(py),
        };
        helper_relation("correlation", &[dataset, column, &method], None)
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
        use proto::expression::literal::{Array, LiteralType};
        // PySpark deliberately uses float(), accepting numeric strings as well as numbers, and
        // converts the params first: its list of them is an argument of the helper call, whose
        // serialization then raises its errors of the dataset and the other arguments. The list
        // is sent as an array<double>, also when it is empty, as PySpark's
        // `([float(p) for p in params], ArrayType(DoubleType()))`.
        let float = params.py().import("builtins")?.getattr("float")?;
        let values = params
            .iter()
            .map(|v| float.call1((v,))?.extract::<f64>())
            .collect::<PyResult<Vec<_>>>()?;
        #[allow(deprecated)] // PySpark's array literal of an explicit type.
        let params = proto::expression::Literal {
            literal_type: Some(LiteralType::Array(Array {
                element_type: Some(spark_connect::types::DataType::Double.to_proto()),
                elements: values
                    .into_iter()
                    .map(|v| proto::expression::Literal {
                        literal_type: Some(LiteralType::Double(v)),
                        ..Default::default()
                    })
                    .collect(),
            })),
            ..Default::default()
        };
        helper_relation(
            "kolmogorovSmirnovTest",
            &[dataset, sampleCol, distName],
            Some(params),
        )
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
        // As in the remote SummaryBuilder, the metrics are converted at summary(), each with
        // `lit` (`F.array([F.lit(m) for m in self._metrics])`), so that a list is the array of
        // `lit`, and non-Column arguments fail the _invoke_function assertion.
        let metrics = self
            .metrics
            .iter()
            .map(|v| crate::functions::to_column(v.bind(py)))
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
