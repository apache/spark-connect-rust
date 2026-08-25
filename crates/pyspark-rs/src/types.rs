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

#[pyclass(name = "BooleanType")]
pub struct PyBooleanType;

#[pyclass(name = "ByteType")]
pub struct PyByteType;

#[pyclass(name = "ShortType")]
pub struct PyShortType;

#[pyclass(name = "IntegerType")]
pub struct PyIntegerType;

#[pyclass(name = "LongType")]
pub struct PyLongType;

#[pyclass(name = "FloatType")]
pub struct PyFloatType;

#[pyclass(name = "DoubleType")]
pub struct PyDoubleType;

#[pyclass(name = "DecimalType")]
pub struct PyDecimalType {
    pub precision: i32,
    pub scale: i32,
}

#[pymethods]
impl PyDecimalType {
    #[new]
    fn new(precision: i32, scale: i32) -> Self {
        PyDecimalType { precision, scale }
    }
}

#[pyclass(name = "StringType")]
pub struct PyStringType;

#[pyclass(name = "BinaryType")]
pub struct PyBinaryType;

#[pyclass(name = "DateType")]
pub struct PyDateType;

#[pyclass(name = "TimestampType")]
pub struct PyTimestampType;

#[pyclass(name = "TimestampNTZType")]
pub struct PyTimestampNTZType;

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
