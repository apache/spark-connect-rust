//! PyO3-backed data holders for polymorphic UDTF `analyze`, mirroring
//! `pyspark.sql.udtf.{AnalyzeArgument,PartitioningColumn,OrderingColumn,SelectedColumn,
//! AnalyzeResult}` and `SkipRestOfInputTableException`. They carry the analyze
//! specification (constructed in the user's `analyze` staticmethod, cloudpickled to the
//! server worker), so the fields are stored as opaque Python objects where typed.

use pyo3::prelude::*;

pyo3::create_exception!(
    _pyspark,
    SkipRestOfInputTableException,
    pyo3::exceptions::PyException
);

/// `pyspark.sql.udtf.AnalyzeArgument`.
#[pyclass(name = "AnalyzeArgument", get_all)]
pub struct PyAnalyzeArgument {
    pub dataType: Py<PyAny>,
    pub value: Py<PyAny>,
    pub isTable: bool,
    pub isConstantExpression: bool,
}
#[pymethods]
impl PyAnalyzeArgument {
    #[new]
    #[allow(non_snake_case)]
    fn new(dataType: Py<PyAny>, value: Py<PyAny>, isTable: bool, isConstantExpression: bool) -> Self {
        PyAnalyzeArgument { dataType, value, isTable, isConstantExpression }
    }
}

/// `pyspark.sql.udtf.PartitioningColumn` (frozen).
#[pyclass(name = "PartitioningColumn", get_all, frozen)]
pub struct PyPartitioningColumn {
    pub name: String,
}
#[pymethods]
impl PyPartitioningColumn {
    #[new]
    fn new(name: String) -> Self {
        PyPartitioningColumn { name }
    }
}

/// `pyspark.sql.udtf.OrderingColumn` (frozen).
#[pyclass(name = "OrderingColumn", get_all, frozen)]
pub struct PyOrderingColumn {
    pub name: String,
    pub ascending: bool,
    pub overrideNullsFirst: Option<bool>,
}
#[pymethods]
impl PyOrderingColumn {
    #[new]
    #[pyo3(signature = (name, ascending=true, overrideNullsFirst=None))]
    #[allow(non_snake_case)]
    fn new(name: String, ascending: bool, overrideNullsFirst: Option<bool>) -> Self {
        PyOrderingColumn { name, ascending, overrideNullsFirst }
    }
}

/// `pyspark.sql.udtf.SelectedColumn` (frozen).
#[pyclass(name = "SelectedColumn", get_all, frozen)]
pub struct PySelectedColumn {
    pub name: String,
    pub alias: String,
}
#[pymethods]
impl PySelectedColumn {
    #[new]
    #[pyo3(signature = (name, alias=String::new()))]
    fn new(name: String, alias: String) -> Self {
        PySelectedColumn { name, alias }
    }
}

/// `pyspark.sql.udtf.AnalyzeResult` (mutable dataclass).
#[pyclass(name = "AnalyzeResult", get_all, set_all)]
pub struct PyAnalyzeResult {
    pub schema: Py<PyAny>,
    pub withSinglePartition: bool,
    pub partitionBy: Py<PyAny>,
    pub orderBy: Py<PyAny>,
    pub select: Py<PyAny>,
}
#[pymethods]
impl PyAnalyzeResult {
    #[new]
    #[pyo3(signature = (schema, withSinglePartition=false, partitionBy=None, orderBy=None, select=None))]
    #[allow(non_snake_case)]
    fn new(
        py: Python<'_>,
        schema: Py<PyAny>,
        withSinglePartition: bool,
        partitionBy: Option<Py<PyAny>>,
        orderBy: Option<Py<PyAny>>,
        select: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let empty = || pyo3::types::PyTuple::empty(py).into_any().unbind();
        Ok(PyAnalyzeResult {
            schema,
            withSinglePartition,
            partitionBy: partitionBy.unwrap_or_else(empty),
            orderBy: orderBy.unwrap_or_else(empty),
            select: select.unwrap_or_else(empty),
        })
    }
}
