//! PyO3 bindings for the Rust Spark Connect client.

mod catalog;
mod column;
mod dataframe;
mod errors;
mod functions;
mod group;
mod row;
mod session;
mod transport;
mod types;
mod window;

use catalog::PyCatalog;
use column::PyColumn;
use dataframe::PyDataFrame;
use group::PyGroupedData;
use pyo3::prelude::*;
use row::PyRow;
use session::{PySparkSession, PySparkSessionBuilder};
use types::{
    PyArrayType, PyBinaryType, PyBooleanType, PyByteType, PyDataType, PyDateType, PyDecimalType,
    PyDoubleType, PyFloatType, PyIntegerType, PyLongType, PyMapType, PyNullType, PyShortType,
    PyStringType, PyStructField, PyStructType, PyTimestampNTZType, PyTimestampType,
};
use window::{PyFrameBound, PyWindow, PyWindowSpec};

#[pymodule]
fn _pyspark(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySparkSession>()?;
    m.add_class::<PySparkSessionBuilder>()?;
    m.add_class::<PyDataFrame>()?;
    m.add_class::<PyColumn>()?;
    m.add_class::<PyRow>()?;
    m.add_class::<PyGroupedData>()?;
    m.add_class::<transport::RustConnectStub>()?;
    m.add_class::<transport::ResponseStream>()?;
    m.add("RustRpcError", m.py().get_type::<transport::RustRpcError>())?;
    m.add_class::<PyCatalog>()?;

    // Register DataType classes
    m.add_class::<PyDataType>()?;
    m.add_class::<PyNullType>()?;
    m.add_class::<PyBooleanType>()?;
    m.add_class::<PyByteType>()?;
    m.add_class::<PyShortType>()?;
    m.add_class::<PyIntegerType>()?;
    m.add_class::<PyLongType>()?;
    m.add_class::<PyFloatType>()?;
    m.add_class::<PyDoubleType>()?;
    m.add_class::<PyDecimalType>()?;
    m.add_class::<PyStringType>()?;
    m.add_class::<PyBinaryType>()?;
    m.add_class::<PyDateType>()?;
    m.add_class::<PyTimestampType>()?;
    m.add_class::<PyTimestampNTZType>()?;
    m.add_class::<PyArrayType>()?;
    m.add_class::<PyMapType>()?;
    m.add_class::<PyStructField>()?;
    m.add_class::<PyStructType>()?;

    // Register Window classes
    m.add_class::<PyWindow>()?;
    m.add_class::<PyWindowSpec>()?;
    m.add_class::<PyFrameBound>()?;

    // Register functions as a submodule
    let functions_module = PyModule::new(_py, "functions")?;
    functions::register_functions(_py, &functions_module)?;
    m.add_submodule(&functions_module)?;

    Ok(())
}
