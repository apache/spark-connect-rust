//! PyO3 wrappers for Spark DataTypes.

use pyo3::prelude::*;
use spark_connect::types::{DataType, StructField};
use std::collections::BTreeMap;

/// Python wrapper for any DataType.
#[pyclass(name = "DataType")]
pub struct PyDataType {
    pub(crate) inner: DataType,
}

impl PyDataType {
    pub fn new(inner: DataType) -> Self {
        PyDataType { inner }
    }
}

#[pymethods]
impl PyDataType {
    fn __repr__(&self) -> String {
        self.inner.simple_string()
    }

    fn __str__(&self) -> String {
        self.inner.simple_string()
    }

    fn simpleString(&self) -> String {
        self.inner.simple_string()
    }
}

// Define concrete type classes
#[pyclass(name = "NullType")]
pub struct PyNullType;

#[pymethods]
impl PyNullType {
    #[new]
    fn new() -> Self {
        PyNullType
    }
    fn __repr__(&self) -> String {
        "NullType()".to_string()
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "void"
    }
    #[pyo3(name = "typeName")]
    fn type_name(&self) -> &'static str {
        "void"
    }
}

#[pyclass(name = "BooleanType")]
pub struct PyBooleanType;

#[pymethods]
impl PyBooleanType {
    #[new]
    fn new() -> Self {
        PyBooleanType
    }
    fn __repr__(&self) -> String {
        "BooleanType()".to_string()
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "boolean"
    }
    #[pyo3(name = "typeName")]
    fn type_name(&self) -> &'static str {
        "boolean"
    }
}

#[pyclass(name = "ByteType")]
pub struct PyByteType;

#[pymethods]
impl PyByteType {
    #[new]
    fn new() -> Self {
        PyByteType
    }
    fn __repr__(&self) -> String {
        "ByteType()".to_string()
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "tinyint"
    }
    #[pyo3(name = "typeName")]
    fn type_name(&self) -> &'static str {
        "byte"
    }
}

#[pyclass(name = "ShortType")]
pub struct PyShortType;

#[pymethods]
impl PyShortType {
    #[new]
    fn new() -> Self {
        PyShortType
    }
    fn __repr__(&self) -> String {
        "ShortType()".to_string()
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "smallint"
    }
    #[pyo3(name = "typeName")]
    fn type_name(&self) -> &'static str {
        "short"
    }
}

#[pyclass(name = "IntegerType")]
pub struct PyIntegerType;

#[pymethods]
impl PyIntegerType {
    #[new]
    fn new() -> Self {
        PyIntegerType
    }
    fn __repr__(&self) -> String {
        "IntegerType()".to_string()
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "int"
    }
    #[pyo3(name = "typeName")]
    fn type_name(&self) -> &'static str {
        "integer"
    }
}

#[pyclass(name = "LongType")]
pub struct PyLongType;

#[pymethods]
impl PyLongType {
    #[new]
    fn new() -> Self {
        PyLongType
    }
    fn __repr__(&self) -> String {
        "LongType()".to_string()
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "bigint"
    }
    #[pyo3(name = "typeName")]
    fn type_name(&self) -> &'static str {
        "long"
    }
}

#[pyclass(name = "FloatType")]
pub struct PyFloatType;

#[pymethods]
impl PyFloatType {
    #[new]
    fn new() -> Self {
        PyFloatType
    }
    fn __repr__(&self) -> String {
        "FloatType()".to_string()
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "float"
    }
    #[pyo3(name = "typeName")]
    fn type_name(&self) -> &'static str {
        "float"
    }
}

#[pyclass(name = "DoubleType")]
pub struct PyDoubleType;

#[pymethods]
impl PyDoubleType {
    #[new]
    fn new() -> Self {
        PyDoubleType
    }
    fn __repr__(&self) -> String {
        "DoubleType()".to_string()
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "double"
    }
    #[pyo3(name = "typeName")]
    fn type_name(&self) -> &'static str {
        "double"
    }
}

#[pyclass(name = "DecimalType")]
pub struct PyDecimalType {
    pub precision: i32,
    pub scale: i32,
}

#[pymethods]
impl PyDecimalType {
    /// `DecimalType(precision=10, scale=0)` - defaults match pyspark.
    #[new]
    #[pyo3(signature = (precision=10, scale=0))]
    fn new(precision: i32, scale: i32) -> Self {
        PyDecimalType { precision, scale }
    }
}

#[pyclass(name = "StringType")]
pub struct PyStringType;

#[pymethods]
impl PyStringType {
    #[new]
    fn new() -> Self {
        PyStringType
    }
    fn __repr__(&self) -> String {
        "StringType()".to_string()
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "string"
    }
    #[pyo3(name = "typeName")]
    fn type_name(&self) -> &'static str {
        "string"
    }
}

#[pyclass(name = "BinaryType")]
pub struct PyBinaryType;

#[pymethods]
impl PyBinaryType {
    #[new]
    fn new() -> Self {
        PyBinaryType
    }
    fn __repr__(&self) -> String {
        "BinaryType()".to_string()
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "binary"
    }
    #[pyo3(name = "typeName")]
    fn type_name(&self) -> &'static str {
        "binary"
    }
}

#[pyclass(name = "DateType")]
pub struct PyDateType;

#[pymethods]
impl PyDateType {
    #[new]
    fn new() -> Self {
        PyDateType
    }
    fn __repr__(&self) -> String {
        "DateType()".to_string()
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "date"
    }
    #[pyo3(name = "typeName")]
    fn type_name(&self) -> &'static str {
        "date"
    }
}

#[pyclass(name = "TimestampType")]
pub struct PyTimestampType;

#[pymethods]
impl PyTimestampType {
    #[new]
    fn new() -> Self {
        PyTimestampType
    }
    fn __repr__(&self) -> String {
        "TimestampType()".to_string()
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "timestamp"
    }
    #[pyo3(name = "typeName")]
    fn type_name(&self) -> &'static str {
        "timestamp"
    }
}

#[pyclass(name = "TimestampNTZType")]
pub struct PyTimestampNTZType;

#[pymethods]
impl PyTimestampNTZType {
    #[new]
    fn new() -> Self {
        PyTimestampNTZType
    }
    fn __repr__(&self) -> String {
        "TimestampNTZType()".to_string()
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "timestamp_ntz"
    }
    #[pyo3(name = "typeName")]
    fn type_name(&self) -> &'static str {
        "timestamp_ntz"
    }
}

#[pyclass(name = "ArrayType")]
pub struct PyArrayType {
    pub element_type_str: String,
    pub contains_null: bool,
}

#[pymethods]
impl PyArrayType {
    #[new]
    fn new(element_type: String, contains_null: Option<bool>) -> Self {
        PyArrayType {
            element_type_str: element_type,
            contains_null: contains_null.unwrap_or(true),
        }
    }
}

#[pyclass(name = "MapType")]
pub struct PyMapType {
    pub key_type_str: String,
    pub value_type_str: String,
    pub value_contains_null: bool,
}

#[pymethods]
impl PyMapType {
    #[new]
    fn new(key_type: String, value_type: String, value_contains_null: Option<bool>) -> Self {
        PyMapType {
            key_type_str: key_type,
            value_type_str: value_type,
            value_contains_null: value_contains_null.unwrap_or(true),
        }
    }
}

#[pyclass(name = "StructField")]
pub struct PyStructField {
    pub name: String,
    pub data_type_str: String,
    pub nullable: bool,
}

#[pymethods]
impl PyStructField {
    #[new]
    fn new(name: String, data_type_str: String, nullable: Option<bool>) -> Self {
        PyStructField {
            name,
            data_type_str,
            nullable: nullable.unwrap_or(true),
        }
    }
}

#[pyclass(name = "StructType")]
pub struct PyStructType {
    pub fields_str: String,
}

#[pymethods]
impl PyStructType {
    #[new]
    fn new(fields: Option<Vec<String>>) -> Self {
        let fields_str = if let Some(f) = fields {
            format!("struct<{}>", f.join(","))
        } else {
            "struct<>".to_string()
        };
        PyStructType { fields_str }
    }

    fn __repr__(&self) -> String {
        self.fields_str.clone()
    }
}
