//! PyO3 wrappers for Spark DataTypes.

use pyo3::prelude::*;
use spark_connect::types::{DataType, StructField};
use std::collections::BTreeMap;

/// Make a DataType picklable so it round-trips to an official-pyspark UDF worker:
/// pickle reconstructs it via `pyspark.sql.types._parse_datatype_json_string(json)`
/// (session-free, present in both our shim and the worker's official pyspark).
fn type_reduce(py: Python<'_>, json: &str) -> PyResult<(Py<PyAny>, (String,))> {
    let f = py
        .import("pyspark.sql.types")?
        .getattr("_parse_datatype_json_string")?;
    Ok((f.unbind(), (json.to_string(),)))
}

/// Python wrapper for any DataType. This is the base of the type-class hierarchy
/// (mirroring pyspark's `DataType`); concrete type classes extend it (directly or via
/// the intermediate abstract bases below), so `isinstance(dt, DataType)` and the object
/// model (simpleString/typeName/json/...) are inherited from here via `self.inner`.
#[pyclass(subclass, name = "DataType")]
pub struct PyDataType {
    pub(crate) inner: DataType,
}

/// The abstract intermediate base classes of the type hierarchy (mirror pyspark's
/// AtomicType/NumericType/IntegralType/... ). They carry no data — a concrete type's
/// `__new__` builds the initializer chain `PyDataType{inner} -> ... -> Concrete` — but
/// existing so `isinstance(dt, NumericType)` etc. work with the reference MRO.
macro_rules! abstract_type {
    ($ty:ident, $name:literal, $parent:ty) => {
        #[pyclass(subclass, name = $name, extends = $parent)]
        pub struct $ty;
    };
}
abstract_type!(PyAtomicType, "AtomicType", PyDataType);
abstract_type!(PyNumericType, "NumericType", PyAtomicType);
abstract_type!(PyIntegralType, "IntegralType", PyNumericType);
abstract_type!(PyFractionalType, "FractionalType", PyNumericType);
abstract_type!(PyDatetimeType, "DatetimeType", PyAtomicType);
abstract_type!(PyAnyTimeType, "AnyTimeType", PyDatetimeType);
abstract_type!(PyAnsiIntervalType, "AnsiIntervalType", PyAtomicType);
abstract_type!(PySpatialType, "SpatialType", PyAtomicType);

/// Build the `PyClassInitializer` chain for a concrete type: the base `PyDataType`
/// carries `inner`; each intermediate abstract base and finally the concrete unit
/// value are stacked on top. Usage: `init_chain!(inner_expr, Concrete, [Mid1, Mid2, ...])`.
macro_rules! init_chain {
    ($inner:expr, $concrete:ident, [$($mid:ident),*]) => {
        pyo3::PyClassInitializer::from(PyDataType { inner: $inner })
            $(.add_subclass($mid))*
            .add_subclass($concrete)
    };
    // Variant carrying a concrete value (parameterized types).
    ($inner:expr, $concrete_val:expr, [$($mid:ident),*], value) => {
        pyo3::PyClassInitializer::from(PyDataType { inner: $inner })
            $(.add_subclass($mid))*
            .add_subclass($concrete_val)
    };
}

impl PyDataType {
    pub fn new(inner: DataType) -> Self {
        PyDataType { inner }
    }
}

/// Construct a `StructType` Python object (with its full base chain) from fields.
fn py_new_struct(py: Python<'_>, fields: Vec<StructField>) -> PyResult<Py<PyStructType>> {
    Py::new(
        py,
        init_chain!(
            DataType::Struct {
                fields: fields.clone(),
            },
            PyStructType { fields },
            [],
            value
        ),
    )
}

#[pymethods]
impl PyDataType {
    /// Allow instantiating the base and, crucially, pure-Python subclasses of it
    /// (e.g. a user's `UserDefinedType`). The sentinel `inner` is never consulted for
    /// such subclasses — they override the object-model methods in Python, and
    /// `py_to_data_type` special-cases `UserDefinedType` before reading `inner`.
    #[new]
    fn __new__() -> Self {
        PyDataType {
            inner: DataType::Null,
        }
    }

    // --- DataType object-model methods (v4.2.0 parity) ---
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py.import("json")?.getattr("loads")?.call1((v.to_string(),))
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(py_to_data_type(slf.as_any())?.need_conversion())
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[classmethod]
    #[pyo3(name = "fromDDL")]
    fn __obj_from_ddl(_cls: &Bound<'_, pyo3::types::PyType>, ddl: &str) -> PyResult<PyDataType> {
        DataType::from_ddl(ddl)
            .map(PyDataType::new)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    #[pyo3(name = "typeName")]
    fn __dt_type_name(&self) -> String {
        self.inner.type_name()
    }

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
#[pyclass(name = "NullType", extends = PyDataType)]
pub struct PyNullType;

#[pymethods]
impl PyNullType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py.import("json")?.getattr("loads")?.call1((v.to_string(),))
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(py_to_data_type(slf.as_any())?.need_conversion())
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[classmethod]
    #[pyo3(name = "fromDDL")]
    fn __obj_from_ddl(_cls: &Bound<'_, pyo3::types::PyType>, ddl: &str) -> PyResult<PyDataType> {
        DataType::from_ddl(ddl)
            .map(PyDataType::new)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        type_reduce(py, "\"void\"")
    }
    #[new]
    fn new() -> pyo3::PyClassInitializer<Self> {
        init_chain!(DataType::Null, PyNullType, [])
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

#[pyclass(name = "BooleanType", extends = PyAtomicType)]
pub struct PyBooleanType;

#[pymethods]
impl PyBooleanType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py.import("json")?.getattr("loads")?.call1((v.to_string(),))
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(py_to_data_type(slf.as_any())?.need_conversion())
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[classmethod]
    #[pyo3(name = "fromDDL")]
    fn __obj_from_ddl(_cls: &Bound<'_, pyo3::types::PyType>, ddl: &str) -> PyResult<PyDataType> {
        DataType::from_ddl(ddl)
            .map(PyDataType::new)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        type_reduce(py, "\"boolean\"")
    }
    #[new]
    fn new() -> pyo3::PyClassInitializer<Self> {
        init_chain!(DataType::Boolean, PyBooleanType, [PyAtomicType])
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

#[pyclass(name = "ByteType", extends = PyIntegralType)]
pub struct PyByteType;

#[pymethods]
impl PyByteType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py.import("json")?.getattr("loads")?.call1((v.to_string(),))
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(py_to_data_type(slf.as_any())?.need_conversion())
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[classmethod]
    #[pyo3(name = "fromDDL")]
    fn __obj_from_ddl(_cls: &Bound<'_, pyo3::types::PyType>, ddl: &str) -> PyResult<PyDataType> {
        DataType::from_ddl(ddl)
            .map(PyDataType::new)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        type_reduce(py, "\"byte\"")
    }
    #[new]
    fn new() -> pyo3::PyClassInitializer<Self> {
        init_chain!(
            DataType::Byte,
            PyByteType,
            [PyAtomicType, PyNumericType, PyIntegralType]
        )
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

#[pyclass(name = "ShortType", extends = PyIntegralType)]
pub struct PyShortType;

#[pymethods]
impl PyShortType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py.import("json")?.getattr("loads")?.call1((v.to_string(),))
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(py_to_data_type(slf.as_any())?.need_conversion())
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[classmethod]
    #[pyo3(name = "fromDDL")]
    fn __obj_from_ddl(_cls: &Bound<'_, pyo3::types::PyType>, ddl: &str) -> PyResult<PyDataType> {
        DataType::from_ddl(ddl)
            .map(PyDataType::new)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        type_reduce(py, "\"short\"")
    }
    #[new]
    fn new() -> pyo3::PyClassInitializer<Self> {
        init_chain!(
            DataType::Short,
            PyShortType,
            [PyAtomicType, PyNumericType, PyIntegralType]
        )
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

#[pyclass(name = "IntegerType", extends = PyIntegralType)]
pub struct PyIntegerType;

#[pymethods]
impl PyIntegerType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py.import("json")?.getattr("loads")?.call1((v.to_string(),))
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(py_to_data_type(slf.as_any())?.need_conversion())
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[classmethod]
    #[pyo3(name = "fromDDL")]
    fn __obj_from_ddl(_cls: &Bound<'_, pyo3::types::PyType>, ddl: &str) -> PyResult<PyDataType> {
        DataType::from_ddl(ddl)
            .map(PyDataType::new)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        type_reduce(py, "\"integer\"")
    }
    #[new]
    fn new() -> pyo3::PyClassInitializer<Self> {
        init_chain!(
            DataType::Integer,
            PyIntegerType,
            [PyAtomicType, PyNumericType, PyIntegralType]
        )
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

#[pyclass(name = "LongType", extends = PyIntegralType)]
pub struct PyLongType;

#[pymethods]
impl PyLongType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py.import("json")?.getattr("loads")?.call1((v.to_string(),))
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(py_to_data_type(slf.as_any())?.need_conversion())
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[classmethod]
    #[pyo3(name = "fromDDL")]
    fn __obj_from_ddl(_cls: &Bound<'_, pyo3::types::PyType>, ddl: &str) -> PyResult<PyDataType> {
        DataType::from_ddl(ddl)
            .map(PyDataType::new)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        type_reduce(py, "\"long\"")
    }
    #[new]
    fn new() -> pyo3::PyClassInitializer<Self> {
        init_chain!(
            DataType::Long,
            PyLongType,
            [PyAtomicType, PyNumericType, PyIntegralType]
        )
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

#[pyclass(name = "FloatType", extends = PyFractionalType)]
pub struct PyFloatType;

#[pymethods]
impl PyFloatType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py.import("json")?.getattr("loads")?.call1((v.to_string(),))
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(py_to_data_type(slf.as_any())?.need_conversion())
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[classmethod]
    #[pyo3(name = "fromDDL")]
    fn __obj_from_ddl(_cls: &Bound<'_, pyo3::types::PyType>, ddl: &str) -> PyResult<PyDataType> {
        DataType::from_ddl(ddl)
            .map(PyDataType::new)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        type_reduce(py, "\"float\"")
    }
    #[new]
    fn new() -> pyo3::PyClassInitializer<Self> {
        init_chain!(
            DataType::Float,
            PyFloatType,
            [PyAtomicType, PyNumericType, PyFractionalType]
        )
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

#[pyclass(name = "DoubleType", extends = PyFractionalType)]
pub struct PyDoubleType;

#[pymethods]
impl PyDoubleType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py.import("json")?.getattr("loads")?.call1((v.to_string(),))
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(py_to_data_type(slf.as_any())?.need_conversion())
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[classmethod]
    #[pyo3(name = "fromDDL")]
    fn __obj_from_ddl(_cls: &Bound<'_, pyo3::types::PyType>, ddl: &str) -> PyResult<PyDataType> {
        DataType::from_ddl(ddl)
            .map(PyDataType::new)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        type_reduce(py, "\"double\"")
    }
    #[new]
    fn new() -> pyo3::PyClassInitializer<Self> {
        init_chain!(
            DataType::Double,
            PyDoubleType,
            [PyAtomicType, PyNumericType, PyFractionalType]
        )
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

#[pyclass(name = "DecimalType", extends = PyFractionalType)]
pub struct PyDecimalType {
    pub precision: i32,
    pub scale: i32,
}

#[pymethods]
impl PyDecimalType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py.import("json")?.getattr("loads")?.call1((v.to_string(),))
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(py_to_data_type(slf.as_any())?.need_conversion())
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[classmethod]
    #[pyo3(name = "fromDDL")]
    fn __obj_from_ddl(_cls: &Bound<'_, pyo3::types::PyType>, ddl: &str) -> PyResult<PyDataType> {
        DataType::from_ddl(ddl)
            .map(PyDataType::new)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        let dt = DataType::Decimal {
            precision: self.precision,
            scale: self.scale,
        };
        type_reduce(py, &dt.json())
    }
    /// `DecimalType(precision=10, scale=0)` - defaults match pyspark.
    #[new]
    #[pyo3(signature = (precision=10, scale=0))]
    fn new(precision: i32, scale: i32) -> pyo3::PyClassInitializer<Self> {
        init_chain!(
            DataType::Decimal { precision, scale },
            PyDecimalType { precision, scale },
            [PyAtomicType, PyNumericType, PyFractionalType],
            value
        )
    }
    #[getter]
    fn precision(&self) -> i32 {
        self.precision
    }
    #[getter]
    fn scale(&self) -> i32 {
        self.scale
    }
    fn __repr__(&self) -> String {
        format!("DecimalType({},{})", self.precision, self.scale)
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> String {
        format!("decimal({},{})", self.precision, self.scale)
    }
    #[pyo3(name = "typeName")]
    fn type_name(&self) -> &'static str {
        "decimal"
    }
}

#[pyclass(name = "StringType", extends = PyAtomicType)]
pub struct PyStringType;

#[pymethods]
impl PyStringType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py.import("json")?.getattr("loads")?.call1((v.to_string(),))
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(py_to_data_type(slf.as_any())?.need_conversion())
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[classmethod]
    #[pyo3(name = "fromDDL")]
    fn __obj_from_ddl(_cls: &Bound<'_, pyo3::types::PyType>, ddl: &str) -> PyResult<PyDataType> {
        DataType::from_ddl(ddl)
            .map(PyDataType::new)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        type_reduce(py, "\"string\"")
    }
    #[new]
    fn new() -> pyo3::PyClassInitializer<Self> {
        init_chain!(
            DataType::String {
                collation: "UTF8_BINARY".to_string()
            },
            PyStringType,
            [PyAtomicType]
        )
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

#[pyclass(name = "BinaryType", extends = PyAtomicType)]
pub struct PyBinaryType;

#[pymethods]
impl PyBinaryType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py.import("json")?.getattr("loads")?.call1((v.to_string(),))
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(py_to_data_type(slf.as_any())?.need_conversion())
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[classmethod]
    #[pyo3(name = "fromDDL")]
    fn __obj_from_ddl(_cls: &Bound<'_, pyo3::types::PyType>, ddl: &str) -> PyResult<PyDataType> {
        DataType::from_ddl(ddl)
            .map(PyDataType::new)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        type_reduce(py, "\"binary\"")
    }
    #[new]
    fn new() -> pyo3::PyClassInitializer<Self> {
        init_chain!(DataType::Binary, PyBinaryType, [PyAtomicType])
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

#[pyclass(name = "DateType", extends = PyDatetimeType)]
pub struct PyDateType;

#[pymethods]
impl PyDateType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py.import("json")?.getattr("loads")?.call1((v.to_string(),))
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(py_to_data_type(slf.as_any())?.need_conversion())
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[classmethod]
    #[pyo3(name = "fromDDL")]
    fn __obj_from_ddl(_cls: &Bound<'_, pyo3::types::PyType>, ddl: &str) -> PyResult<PyDataType> {
        DataType::from_ddl(ddl)
            .map(PyDataType::new)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        type_reduce(py, "\"date\"")
    }
    #[new]
    fn new() -> pyo3::PyClassInitializer<Self> {
        init_chain!(DataType::Date, PyDateType, [PyAtomicType, PyDatetimeType])
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

#[pyclass(name = "TimestampType", extends = PyDatetimeType)]
pub struct PyTimestampType;

#[pymethods]
impl PyTimestampType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py.import("json")?.getattr("loads")?.call1((v.to_string(),))
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(py_to_data_type(slf.as_any())?.need_conversion())
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[classmethod]
    #[pyo3(name = "fromDDL")]
    fn __obj_from_ddl(_cls: &Bound<'_, pyo3::types::PyType>, ddl: &str) -> PyResult<PyDataType> {
        DataType::from_ddl(ddl)
            .map(PyDataType::new)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        type_reduce(py, "\"timestamp\"")
    }
    #[new]
    fn new() -> pyo3::PyClassInitializer<Self> {
        init_chain!(
            DataType::Timestamp,
            PyTimestampType,
            [PyAtomicType, PyDatetimeType]
        )
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

#[pyclass(name = "TimestampNTZType", extends = PyDatetimeType)]
pub struct PyTimestampNTZType;

#[pymethods]
impl PyTimestampNTZType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py.import("json")?.getattr("loads")?.call1((v.to_string(),))
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(py_to_data_type(slf.as_any())?.need_conversion())
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[classmethod]
    #[pyo3(name = "fromDDL")]
    fn __obj_from_ddl(_cls: &Bound<'_, pyo3::types::PyType>, ddl: &str) -> PyResult<PyDataType> {
        DataType::from_ddl(ddl)
            .map(PyDataType::new)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        type_reduce(py, "\"timestamp_ntz\"")
    }
    #[new]
    fn new() -> pyo3::PyClassInitializer<Self> {
        init_chain!(
            DataType::TimestampNtz,
            PyTimestampNTZType,
            [PyAtomicType, PyDatetimeType]
        )
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

#[pyclass(name = "ArrayType", extends = PyDataType)]
pub struct PyArrayType {
    pub element_type: DataType,
    pub contains_null: bool,
}

#[pymethods]
impl PyArrayType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py.import("json")?.getattr("loads")?.call1((v.to_string(),))
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(py_to_data_type(slf.as_any())?.need_conversion())
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[classmethod]
    #[pyo3(name = "fromDDL")]
    fn __obj_from_ddl(_cls: &Bound<'_, pyo3::types::PyType>, ddl: &str) -> PyResult<PyDataType> {
        DataType::from_ddl(ddl)
            .map(PyDataType::new)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        let dt = DataType::Array {
            element_type: Box::new(self.element_type.clone()),
            contains_null: self.contains_null,
        };
        type_reduce(py, &dt.json())
    }
    #[new]
    #[pyo3(signature = (element_type, contains_null=true))]
    fn new(
        element_type: &Bound<'_, PyAny>,
        contains_null: bool,
    ) -> PyResult<pyo3::PyClassInitializer<Self>> {
        let et = py_to_data_type(element_type)?;
        Ok(init_chain!(
            DataType::Array {
                element_type: Box::new(et.clone()),
                contains_null,
            },
            PyArrayType {
                element_type: et,
                contains_null,
            },
            [],
            value
        ))
    }
    fn __repr__(&self) -> String {
        format!("ArrayType({})", self.element_type.simple_string())
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> String {
        format!("array<{}>", self.element_type.simple_string())
    }
    #[pyo3(name = "typeName")]
    fn type_name(&self) -> &'static str {
        "array"
    }
}

#[pyclass(name = "MapType", extends = PyDataType)]
pub struct PyMapType {
    pub key_type: DataType,
    pub value_type: DataType,
    pub value_contains_null: bool,
}

#[pymethods]
impl PyMapType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py.import("json")?.getattr("loads")?.call1((v.to_string(),))
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(py_to_data_type(slf.as_any())?.need_conversion())
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[classmethod]
    #[pyo3(name = "fromDDL")]
    fn __obj_from_ddl(_cls: &Bound<'_, pyo3::types::PyType>, ddl: &str) -> PyResult<PyDataType> {
        DataType::from_ddl(ddl)
            .map(PyDataType::new)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        let dt = DataType::Map {
            key_type: Box::new(self.key_type.clone()),
            value_type: Box::new(self.value_type.clone()),
            value_contains_null: self.value_contains_null,
        };
        type_reduce(py, &dt.json())
    }
    #[new]
    #[pyo3(signature = (key_type, value_type, value_contains_null=true))]
    fn new(
        key_type: &Bound<'_, PyAny>,
        value_type: &Bound<'_, PyAny>,
        value_contains_null: bool,
    ) -> PyResult<pyo3::PyClassInitializer<Self>> {
        let kt = py_to_data_type(key_type)?;
        let vt = py_to_data_type(value_type)?;
        Ok(init_chain!(
            DataType::Map {
                key_type: Box::new(kt.clone()),
                value_type: Box::new(vt.clone()),
                value_contains_null,
            },
            PyMapType {
                key_type: kt,
                value_type: vt,
                value_contains_null,
            },
            [],
            value
        ))
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> String {
        format!(
            "map<{},{}>",
            self.key_type.simple_string(),
            self.value_type.simple_string()
        )
    }
    #[pyo3(name = "typeName")]
    fn type_name(&self) -> &'static str {
        "map"
    }
}

#[pyclass(name = "StructField")]
pub struct PyStructField {
    pub(crate) field: StructField,
}

#[pymethods]
impl PyStructField {
    // --- StructField object-model methods (v4.2.0 parity), operating on the field
    // itself (a StructField is not a DataType, so these do not go through
    // py_to_data_type). ---
    #[pyo3(name = "simpleString")]
    fn __sf_simple_string(&self) -> String {
        self.field.simple_string()
    }
    #[pyo3(name = "typeName")]
    fn __sf_type_name(&self) -> PyResult<String> {
        // Mirrors pyspark: StructField.typeName raises; use typeName on the type.
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "StructField does not have typeName. Use typeName on its type explicitly instead.",
        ))
    }
    #[pyo3(name = "json")]
    fn __sf_json(&self) -> String {
        self.field.json_value().to_string()
    }
    #[pyo3(name = "jsonValue")]
    fn __sf_json_value<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let v = self.field.json_value();
        py.import("json")?.getattr("loads")?.call1((v.to_string(),))
    }
    #[pyo3(name = "needConversion")]
    fn __sf_need_conversion(&self) -> bool {
        self.field.data_type.need_conversion()
    }
    #[pyo3(name = "fromInternal")]
    fn __sf_from_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[pyo3(name = "toInternal")]
    fn __sf_to_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    /// Collation metadata for the field (empty when the field has no collations).
    #[pyo3(name = "getCollationMetadata")]
    fn __sf_get_collation_metadata<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        Ok(pyo3::types::PyDict::new(py))
    }

    /// Map of collated-field path -> collation name, parsed from the field's
    /// `__COLLATIONS` metadata (empty when absent). Mirrors `StructField.getCollationsMap`.
    #[pyo3(name = "getCollationsMap")]
    fn __sf_get_collations_map<'py>(
        &self,
        py: Python<'py>,
        metadata: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let out = pyo3::types::PyDict::new(py);
        if let Ok(dict) = metadata.downcast::<pyo3::types::PyDict>() {
            if let Some(coll) = dict.get_item("__COLLATIONS")? {
                if let Ok(cdict) = coll.downcast::<pyo3::types::PyDict>() {
                    for (k, v) in cdict.iter() {
                        let val: String = v.extract()?;
                        // Value is "provider.name"; the map holds just the name.
                        let name = val.rsplit('.').next().unwrap_or(&val).to_string();
                        out.set_item(k, name)?;
                    }
                }
            }
        }
        Ok(out)
    }

    /// The "provider.collation" value for a collated string type. Mirrors
    /// `StructField.schemaCollationValue` (UTF8_BINARY/UTF8_LCASE are the "spark"
    /// provider, everything else "icu").
    #[pyo3(name = "schemaCollationValue")]
    fn __sf_schema_collation_value(&self, dt: &Bound<'_, PyAny>) -> PyResult<String> {
        let collation = match py_to_data_type(dt)? {
            DataType::String { collation } => collation,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "schemaCollationValue expects a StringType",
                ))
            }
        };
        let provider = if collation == "UTF8_BINARY" || collation == "UTF8_LCASE" {
            "spark"
        } else {
            "icu"
        };
        Ok(format!("{provider}.{collation}"))
    }
    /// Parse a single-field DDL string (e.g. "a INT" or "a: int") into a StructField.
    /// Mirrors `StructField.fromDDL`.
    #[classmethod]
    #[pyo3(name = "fromDDL")]
    fn __sf_from_ddl(_cls: &Bound<'_, pyo3::types::PyType>, ddl: &str) -> PyResult<PyStructField> {
        let trimmed = ddl.trim();
        // Split "name type" at the first run of whitespace; tolerate a "name:" colon.
        let split = trimmed.find(char::is_whitespace).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "fromDDL expects 'name type', e.g. 'a INT'",
            )
        })?;
        let name = trimmed[..split].trim_end_matches(':').to_string();
        let type_str = trimmed[split..].trim();
        let data_type = DataType::from_ddl(type_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyStructField {
            field: StructField {
                name,
                data_type,
                nullable: true,
                metadata: BTreeMap::new(),
            },
        })
    }

    #[classmethod]
    #[pyo3(name = "fromJson")]
    fn __sf_from_json(
        _cls: &Bound<'_, pyo3::types::PyType>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<PyStructField> {
        let s: String = py
            .import("json")?
            .getattr("dumps")?
            .call1((data,))?
            .extract()?;
        let field = StructField::from_json_str(&s)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyStructField { field })
    }
    #[new]
    #[pyo3(signature = (name, dataType, nullable=true, metadata=None))]
    #[allow(non_snake_case)]
    fn new(
        name: String,
        dataType: &Bound<'_, PyAny>,
        nullable: bool,
        metadata: Option<std::collections::HashMap<String, String>>,
    ) -> PyResult<Self> {
        let md: BTreeMap<String, String> = metadata.unwrap_or_default().into_iter().collect();
        Ok(PyStructField {
            field: StructField {
                name,
                data_type: py_to_data_type(dataType)?,
                nullable,
                metadata: md,
            },
        })
    }
    #[getter]
    fn name(&self) -> String {
        self.field.name.clone()
    }
    #[getter]
    fn nullable(&self) -> bool {
        self.field.nullable
    }
    /// The field's data type as its concrete Python type object (mirrors `StructField.dataType`).
    #[getter]
    #[pyo3(name = "dataType")]
    fn data_type(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        data_type_to_py(py, &self.field.data_type)
    }
    /// The field's metadata dict (mirrors `StructField.metadata`).
    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let out = pyo3::types::PyDict::new(py);
        for (k, v) in &self.field.metadata {
            out.set_item(k, v)?;
        }
        Ok(out)
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        // Mirror pyspark: StructField('name', <DataType repr>, nullable).
        let dt = data_type_to_py(py, &self.field.data_type)?;
        let dt_repr: String = dt.bind(py).repr()?.extract()?;
        Ok(format!(
            "StructField('{}', {}, {})",
            self.field.name,
            dt_repr,
            if self.field.nullable { "True" } else { "False" }
        ))
    }
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        if let Ok(o) = other.extract::<PyRef<PyStructField>>() {
            self.field == o.field
        } else {
            false
        }
    }
}

#[pyclass(name = "StructType", extends = PyDataType)]
pub struct PyStructType {
    pub(crate) fields: Vec<StructField>,
}

#[pymethods]
impl PyStructType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py.import("json")?.getattr("loads")?.call1((v.to_string(),))
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(py_to_data_type(slf.as_any())?.need_conversion())
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[classmethod]
    #[pyo3(name = "fromDDL")]
    fn __obj_from_ddl(_cls: &Bound<'_, pyo3::types::PyType>, ddl: &str) -> PyResult<PyDataType> {
        DataType::from_ddl(ddl)
            .map(PyDataType::new)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        let dt = DataType::Struct {
            fields: self.fields.clone(),
        };
        type_reduce(py, &dt.json())
    }
    /// `StructType(fields=None)` where each field is a `StructField` (matching
    /// pyspark). A list of DDL field strings is also accepted for convenience.
    #[new]
    #[pyo3(signature = (fields=None))]
    fn new(fields: Option<Vec<Bound<'_, PyAny>>>) -> PyResult<pyo3::PyClassInitializer<Self>> {
        let mut out = Vec::new();
        if let Some(fs) = fields {
            for f in fs {
                if let Ok(sf) = f.extract::<PyRef<PyStructField>>() {
                    out.push(sf.field.clone());
                } else if let Ok(s) = f.extract::<String>() {
                    // "name type" DDL fragment.
                    let dt = DataType::from_ddl(&format!("struct<{s}>")).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "invalid struct field '{s}': {e:?}"
                        ))
                    })?;
                    if let DataType::Struct { fields } = dt {
                        out.extend(fields);
                    }
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "StructType fields must be StructField objects or DDL strings",
                    ));
                }
            }
        }
        Ok(init_chain!(
            DataType::Struct {
                fields: out.clone(),
            },
            PyStructType { fields: out },
            [],
            value
        ))
    }

    /// Append a field and return `self` (chainable, mirroring `StructType.add`):
    /// `StructType().add("a", "int").add("b", StringType())`.
    #[pyo3(signature = (field, data_type=None, nullable=true, metadata=None))]
    fn add<'py>(
        slf: Bound<'py, Self>,
        field: &Bound<'_, PyAny>,
        data_type: Option<&Bound<'_, PyAny>>,
        nullable: bool,
        metadata: Option<std::collections::HashMap<String, String>>,
    ) -> PyResult<Bound<'py, Self>> {
        let new_field = if let Ok(sf) = field.extract::<PyRef<PyStructField>>() {
            sf.field.clone()
        } else {
            let name: String = field.extract()?;
            let dt = match data_type {
                Some(d) => py_to_data_type(d)?,
                None => DataType::String {
                    collation: "UTF8_BINARY".to_string(),
                },
            };
            StructField {
                name,
                data_type: dt,
                nullable,
                metadata: metadata.unwrap_or_default().into_iter().collect(),
            }
        };
        {
            let mut me = slf.borrow_mut();
            me.fields.push(new_field);
        }
        Ok(slf)
    }

    /// The fields as `StructField` objects (mirrors `StructType.fields`).
    #[getter]
    fn fields(&self, py: Python<'_>) -> PyResult<Vec<Py<PyStructField>>> {
        self.fields
            .iter()
            .map(|f| Py::new(py, PyStructField { field: f.clone() }))
            .collect()
    }

    /// The field names in order (mirrors `StructType.names`).
    #[getter]
    fn names(&self) -> Vec<String> {
        self.fields.iter().map(|f| f.name.clone()).collect()
    }

    fn __len__(&self) -> usize {
        self.fields.len()
    }

    /// Index by position (`st[0]`) or by name (`st["a"]`), returning a `StructField`
    /// (mirrors `StructType.__getitem__`; a slice yields a new `StructType`).
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(name) = key.extract::<String>() {
            for f in &self.fields {
                if f.name == name {
                    return Ok(Py::new(py, PyStructField { field: f.clone() })?.into_any());
                }
            }
            return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "No StructField named {name}"
            )));
        }
        if let Ok(slice) = key.downcast::<pyo3::types::PySlice>() {
            let idx = slice.indices(self.fields.len() as isize)?;
            let mut out = Vec::new();
            let (mut i, stop, step) = (idx.start, idx.stop, idx.step);
            while (step > 0 && i < stop) || (step < 0 && i > stop) {
                out.push(self.fields[i as usize].clone());
                i += step;
            }
            return Ok(py_new_struct(py, out)?.into_any());
        }
        let i: isize = key.extract().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "StructType indices must be integers, slices, or field names",
            )
        })?;
        let n = self.fields.len() as isize;
        let idx = if i < 0 { i + n } else { i };
        if idx < 0 || idx >= n {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "StructType index out of range",
            ));
        }
        Ok(Py::new(
            py,
            PyStructField {
                field: self.fields[idx as usize].clone(),
            },
        )?
        .into_any())
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let items: Vec<Py<PyStructField>> = self
            .fields
            .iter()
            .map(|f| Py::new(py, PyStructField { field: f.clone() }))
            .collect::<PyResult<_>>()?;
        let list = pyo3::types::PyList::new(py, items)?;
        Ok(list.as_any().call_method0("__iter__")?.unbind())
    }

    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> String {
        DataType::Struct {
            fields: self.fields.clone(),
        }
        .simple_string()
    }

    /// `StructType.typeName()` == "struct".
    #[pyo3(name = "typeName")]
    fn __st_type_name(&self) -> &'static str {
        "struct"
    }

    /// Field names in order. Mirrors `StructType.fieldNames()`.
    #[pyo3(name = "fieldNames")]
    fn __st_field_names(&self) -> Vec<String> {
        self.fields.iter().map(|f| f.name.clone()).collect()
    }

    /// DDL string. Mirrors `StructType.toDDL()`.
    #[pyo3(name = "toDDL")]
    fn __st_to_ddl(&self) -> PyResult<String> {
        DataType::Struct {
            fields: self.fields.clone(),
        }
        .to_ddl()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// Tree string. Mirrors `StructType.treeString()`.
    #[pyo3(name = "treeString")]
    fn __st_tree_string(&self) -> PyResult<String> {
        DataType::Struct {
            fields: self.fields.clone(),
        }
        .tree_string()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// A copy with all fields made nullable (recursively). Mirrors `StructType.toNullable()`.
    #[pyo3(name = "toNullable")]
    fn __st_to_nullable(&self, py: Python<'_>) -> PyResult<Py<PyStructType>> {
        let fields = match (DataType::Struct {
            fields: self.fields.clone(),
        })
        .to_nullable()
        {
            DataType::Struct { fields } => fields,
            _ => self.fields.clone(),
        };
        py_new_struct(py, fields)
    }

    /// Build a StructType from its JSON value. Mirrors `StructType.fromJson`.
    #[classmethod]
    #[pyo3(name = "fromJson")]
    fn __st_from_json(
        _cls: &Bound<'_, pyo3::types::PyType>,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyStructType>> {
        let s: String = py
            .import("json")?
            .getattr("dumps")?
            .call1((data,))?
            .extract()?;
        match DataType::from_json_str(&s)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
        {
            DataType::Struct { fields } => py_new_struct(py, fields),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "fromJson did not produce a StructType",
            )),
        }
    }

    fn __repr__(&self) -> String {
        self.simple_string()
    }
}

#[pyclass(name = "CharType", extends = PyAtomicType)]
pub struct PyCharType {
    pub length: i32,
}

#[pymethods]
impl PyCharType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py.import("json")?.getattr("loads")?.call1((v.to_string(),))
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(py_to_data_type(slf.as_any())?.need_conversion())
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[classmethod]
    #[pyo3(name = "fromDDL")]
    fn __obj_from_ddl(_cls: &Bound<'_, pyo3::types::PyType>, ddl: &str) -> PyResult<PyDataType> {
        DataType::from_ddl(ddl)
            .map(PyDataType::new)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        let dt = DataType::Char {
            length: self.length,
        };
        type_reduce(py, &dt.json())
    }
    #[new]
    fn new(length: i32) -> pyo3::PyClassInitializer<Self> {
        init_chain!(
            DataType::Char { length },
            PyCharType { length },
            [PyAtomicType],
            value
        )
    }
    fn __repr__(&self) -> String {
        format!("CharType({})", self.length)
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> String {
        format!("char({})", self.length)
    }
    #[pyo3(name = "typeName")]
    fn type_name(&self) -> &'static str {
        "char"
    }
}

#[pyclass(name = "VarcharType", extends = PyAtomicType)]
pub struct PyVarcharType {
    pub length: i32,
}

#[pymethods]
impl PyVarcharType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py.import("json")?.getattr("loads")?.call1((v.to_string(),))
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(py_to_data_type(slf.as_any())?.need_conversion())
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[classmethod]
    #[pyo3(name = "fromDDL")]
    fn __obj_from_ddl(_cls: &Bound<'_, pyo3::types::PyType>, ddl: &str) -> PyResult<PyDataType> {
        DataType::from_ddl(ddl)
            .map(PyDataType::new)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        let dt = DataType::Varchar {
            length: self.length,
        };
        type_reduce(py, &dt.json())
    }
    #[new]
    fn new(length: i32) -> pyo3::PyClassInitializer<Self> {
        init_chain!(
            DataType::Varchar { length },
            PyVarcharType { length },
            [PyAtomicType],
            value
        )
    }
    fn __repr__(&self) -> String {
        format!("VarcharType({})", self.length)
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> String {
        format!("varchar({})", self.length)
    }
    #[pyo3(name = "typeName")]
    fn type_name(&self) -> &'static str {
        "varchar"
    }
}

#[pyclass(name = "TimeType", extends = PyAnyTimeType)]
pub struct PyTimeType {
    pub precision: i32,
}

#[pymethods]
impl PyTimeType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py.import("json")?.getattr("loads")?.call1((v.to_string(),))
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(py_to_data_type(slf.as_any())?.need_conversion())
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[classmethod]
    #[pyo3(name = "fromDDL")]
    fn __obj_from_ddl(_cls: &Bound<'_, pyo3::types::PyType>, ddl: &str) -> PyResult<PyDataType> {
        DataType::from_ddl(ddl)
            .map(PyDataType::new)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        let dt = DataType::Time {
            precision: self.precision,
        };
        type_reduce(py, &dt.json())
    }
    /// `TimeType(precision=6)` (microsecond precision default), matching pyspark 4.2.
    #[new]
    #[pyo3(signature = (precision=6))]
    fn new(precision: i32) -> pyo3::PyClassInitializer<Self> {
        init_chain!(
            DataType::Time { precision },
            PyTimeType { precision },
            [PyAtomicType, PyDatetimeType, PyAnyTimeType],
            value
        )
    }
    fn __repr__(&self) -> String {
        format!("TimeType({})", self.precision)
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> String {
        format!("time({})", self.precision)
    }
    #[pyo3(name = "typeName")]
    fn type_name(&self) -> &'static str {
        "time"
    }
}

#[pyclass(name = "CalendarIntervalType", extends = PyDataType)]
pub struct PyCalendarIntervalType;

#[pymethods]
impl PyCalendarIntervalType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py.import("json")?.getattr("loads")?.call1((v.to_string(),))
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(py_to_data_type(slf.as_any())?.need_conversion())
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[classmethod]
    #[pyo3(name = "fromDDL")]
    fn __obj_from_ddl(_cls: &Bound<'_, pyo3::types::PyType>, ddl: &str) -> PyResult<PyDataType> {
        DataType::from_ddl(ddl)
            .map(PyDataType::new)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        type_reduce(py, "\"calendarinterval\"")
    }
    #[new]
    fn new() -> pyo3::PyClassInitializer<Self> {
        init_chain!(DataType::CalendarInterval, PyCalendarIntervalType, [])
    }
    fn __repr__(&self) -> String {
        "CalendarIntervalType()".to_string()
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "interval"
    }
    #[pyo3(name = "typeName")]
    fn type_name(&self) -> &'static str {
        "calendarinterval"
    }
}

#[pyclass(name = "YearMonthIntervalType", extends = PyAnsiIntervalType)]
pub struct PyYearMonthIntervalType {
    pub start_field: i32,
    pub end_field: i32,
}

#[pymethods]
impl PyYearMonthIntervalType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py.import("json")?.getattr("loads")?.call1((v.to_string(),))
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(py_to_data_type(slf.as_any())?.need_conversion())
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[classmethod]
    #[pyo3(name = "fromDDL")]
    fn __obj_from_ddl(_cls: &Bound<'_, pyo3::types::PyType>, ddl: &str) -> PyResult<PyDataType> {
        DataType::from_ddl(ddl)
            .map(PyDataType::new)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        let dt = DataType::YearMonthInterval {
            start_field: self.start_field,
            end_field: self.end_field,
        };
        type_reduce(py, &dt.json())
    }
    /// Fields: YEAR=0, MONTH=1. Defaults to the full YEAR..MONTH range.
    #[new]
    #[pyo3(signature = (startField=None, endField=None))]
    #[allow(non_snake_case)]
    fn new(startField: Option<i32>, endField: Option<i32>) -> pyo3::PyClassInitializer<Self> {
        let start = startField.unwrap_or(0);
        let end = endField.unwrap_or(if startField.is_some() { start } else { 1 });
        init_chain!(
            DataType::YearMonthInterval {
                start_field: start,
                end_field: end,
            },
            PyYearMonthIntervalType {
                start_field: start,
                end_field: end,
            },
            [PyAtomicType, PyAnsiIntervalType],
            value
        )
    }
    fn __repr__(&self) -> String {
        format!(
            "YearMonthIntervalType({}, {})",
            self.start_field, self.end_field
        )
    }
    #[pyo3(name = "typeName")]
    fn type_name(&self) -> &'static str {
        "yearmonthinterval"
    }
}

#[pyclass(name = "DayTimeIntervalType", extends = PyAnsiIntervalType)]
pub struct PyDayTimeIntervalType {
    pub start_field: i32,
    pub end_field: i32,
}

#[pymethods]
impl PyDayTimeIntervalType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py.import("json")?.getattr("loads")?.call1((v.to_string(),))
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(py_to_data_type(slf.as_any())?.need_conversion())
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[classmethod]
    #[pyo3(name = "fromDDL")]
    fn __obj_from_ddl(_cls: &Bound<'_, pyo3::types::PyType>, ddl: &str) -> PyResult<PyDataType> {
        DataType::from_ddl(ddl)
            .map(PyDataType::new)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        let dt = DataType::DayTimeInterval {
            start_field: self.start_field,
            end_field: self.end_field,
        };
        type_reduce(py, &dt.json())
    }
    /// Fields: DAY=0, HOUR=1, MINUTE=2, SECOND=3. Defaults to the full DAY..SECOND range.
    #[new]
    #[pyo3(signature = (startField=None, endField=None))]
    #[allow(non_snake_case)]
    fn new(startField: Option<i32>, endField: Option<i32>) -> pyo3::PyClassInitializer<Self> {
        let start = startField.unwrap_or(0);
        let end = endField.unwrap_or(if startField.is_some() { start } else { 3 });
        init_chain!(
            DataType::DayTimeInterval {
                start_field: start,
                end_field: end,
            },
            PyDayTimeIntervalType {
                start_field: start,
                end_field: end,
            },
            [PyAtomicType, PyAnsiIntervalType],
            value
        )
    }
    fn __repr__(&self) -> String {
        format!(
            "DayTimeIntervalType({}, {})",
            self.start_field, self.end_field
        )
    }
    #[pyo3(name = "typeName")]
    fn type_name(&self) -> &'static str {
        "daytimeinterval"
    }
}

#[pyclass(name = "VariantType", extends = PyAtomicType)]
pub struct PyVariantType;

#[pymethods]
impl PyVariantType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py.import("json")?.getattr("loads")?.call1((v.to_string(),))
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(py_to_data_type(slf.as_any())?.need_conversion())
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(&self, obj: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        obj
    }
    #[classmethod]
    #[pyo3(name = "fromDDL")]
    fn __obj_from_ddl(_cls: &Bound<'_, pyo3::types::PyType>, ddl: &str) -> PyResult<PyDataType> {
        DataType::from_ddl(ddl)
            .map(PyDataType::new)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        type_reduce(py, "\"variant\"")
    }
    #[new]
    fn new() -> pyo3::PyClassInitializer<Self> {
        init_chain!(DataType::Variant, PyVariantType, [PyAtomicType])
    }
    fn __repr__(&self) -> String {
        "VariantType()".to_string()
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "variant"
    }
    #[pyo3(name = "typeName")]
    fn type_name(&self) -> &'static str {
        "variant"
    }
}

#[pyclass(name = "GeometryType", extends = PySpatialType)]
pub struct PyGeometryType {
    pub srid: i32,
}

#[pymethods]
impl PyGeometryType {
    #[new]
    #[pyo3(signature = (srid=0))]
    fn new(srid: i32) -> pyo3::PyClassInitializer<Self> {
        init_chain!(
            DataType::Geometry { srid },
            PyGeometryType { srid },
            [PyAtomicType, PySpatialType],
            value
        )
    }
    #[getter]
    fn srid(&self) -> i32 {
        self.srid
    }
    fn __repr__(&self) -> String {
        format!("GeometryType({})", self.srid)
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> String {
        DataType::Geometry { srid: self.srid }.simple_string()
    }
    #[pyo3(name = "typeName")]
    fn type_name(&self) -> &'static str {
        "geometry"
    }
}

#[pyclass(name = "GeographyType", extends = PySpatialType)]
pub struct PyGeographyType {
    pub srid: i32,
}

#[pymethods]
impl PyGeographyType {
    #[new]
    #[pyo3(signature = (srid=0))]
    fn new(srid: i32) -> pyo3::PyClassInitializer<Self> {
        init_chain!(
            DataType::Geography { srid },
            PyGeographyType { srid },
            [PyAtomicType, PySpatialType],
            value
        )
    }
    #[getter]
    fn srid(&self) -> i32 {
        self.srid
    }
    fn __repr__(&self) -> String {
        format!("GeographyType({})", self.srid)
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> String {
        DataType::Geography { srid: self.srid }.simple_string()
    }
    #[pyo3(name = "typeName")]
    fn type_name(&self) -> &'static str {
        "geography"
    }
}

/// Convert a Python DataType object (any of the type classes) or a DDL string into
/// the core `DataType`. Mirrors pyspark accepting `Union[DataType, str]` everywhere.
/// Materialize a core [`DataType`] into its concrete Python type object — a real
/// `StructType`/`IntegerType`/... with the correct MRO — the inverse of
/// [`py_to_data_type`]. Used by `df.schema`, `StructField.dataType`, and `StructType`
/// field access so schema introspection yields the proper classes (not a bare `DataType`).
pub(crate) fn data_type_to_py(py: Python<'_>, dt: &DataType) -> PyResult<Py<PyAny>> {
    macro_rules! obj {
        ($init:expr) => {
            Py::new(py, $init)?.into_any()
        };
    }
    let o = match dt {
        DataType::Null => obj!(init_chain!(DataType::Null, PyNullType, [])),
        DataType::Boolean => obj!(init_chain!(
            DataType::Boolean,
            PyBooleanType,
            [PyAtomicType]
        )),
        DataType::Byte => obj!(init_chain!(
            DataType::Byte,
            PyByteType,
            [PyAtomicType, PyNumericType, PyIntegralType]
        )),
        DataType::Short => obj!(init_chain!(
            DataType::Short,
            PyShortType,
            [PyAtomicType, PyNumericType, PyIntegralType]
        )),
        DataType::Integer => obj!(init_chain!(
            DataType::Integer,
            PyIntegerType,
            [PyAtomicType, PyNumericType, PyIntegralType]
        )),
        DataType::Long => obj!(init_chain!(
            DataType::Long,
            PyLongType,
            [PyAtomicType, PyNumericType, PyIntegralType]
        )),
        DataType::Float => obj!(init_chain!(
            DataType::Float,
            PyFloatType,
            [PyAtomicType, PyNumericType, PyFractionalType]
        )),
        DataType::Double => obj!(init_chain!(
            DataType::Double,
            PyDoubleType,
            [PyAtomicType, PyNumericType, PyFractionalType]
        )),
        DataType::Decimal { precision, scale } => {
            let (precision, scale) = (*precision, *scale);
            obj!(init_chain!(
                DataType::Decimal { precision, scale },
                PyDecimalType { precision, scale },
                [PyAtomicType, PyNumericType, PyFractionalType],
                value
            ))
        }
        DataType::String { collation } => obj!(init_chain!(
            DataType::String {
                collation: collation.clone()
            },
            PyStringType,
            [PyAtomicType]
        )),
        DataType::Binary => obj!(init_chain!(DataType::Binary, PyBinaryType, [PyAtomicType])),
        DataType::Date => obj!(init_chain!(
            DataType::Date,
            PyDateType,
            [PyAtomicType, PyDatetimeType]
        )),
        DataType::Timestamp => obj!(init_chain!(
            DataType::Timestamp,
            PyTimestampType,
            [PyAtomicType, PyDatetimeType]
        )),
        DataType::TimestampNtz => obj!(init_chain!(
            DataType::TimestampNtz,
            PyTimestampNTZType,
            [PyAtomicType, PyDatetimeType]
        )),
        DataType::Array {
            element_type,
            contains_null,
        } => {
            let (et, contains_null) = ((**element_type).clone(), *contains_null);
            obj!(init_chain!(
                DataType::Array {
                    element_type: Box::new(et.clone()),
                    contains_null,
                },
                PyArrayType {
                    element_type: et,
                    contains_null,
                },
                [],
                value
            ))
        }
        DataType::Map {
            key_type,
            value_type,
            value_contains_null,
        } => {
            let (kt, vt, value_contains_null) = (
                (**key_type).clone(),
                (**value_type).clone(),
                *value_contains_null,
            );
            obj!(init_chain!(
                DataType::Map {
                    key_type: Box::new(kt.clone()),
                    value_type: Box::new(vt.clone()),
                    value_contains_null,
                },
                PyMapType {
                    key_type: kt,
                    value_type: vt,
                    value_contains_null,
                },
                [],
                value
            ))
        }
        DataType::Struct { fields } => py_new_struct(py, fields.clone())?.into_any(),
        DataType::Char { length } => {
            let length = *length;
            obj!(init_chain!(
                DataType::Char { length },
                PyCharType { length },
                [PyAtomicType],
                value
            ))
        }
        DataType::Varchar { length } => {
            let length = *length;
            obj!(init_chain!(
                DataType::Varchar { length },
                PyVarcharType { length },
                [PyAtomicType],
                value
            ))
        }
        DataType::Time { precision } => {
            let precision = *precision;
            obj!(init_chain!(
                DataType::Time { precision },
                PyTimeType { precision },
                [PyAtomicType, PyDatetimeType, PyAnyTimeType],
                value
            ))
        }
        DataType::CalendarInterval => obj!(init_chain!(
            DataType::CalendarInterval,
            PyCalendarIntervalType,
            []
        )),
        DataType::YearMonthInterval {
            start_field,
            end_field,
        } => {
            let (start_field, end_field) = (*start_field, *end_field);
            obj!(init_chain!(
                DataType::YearMonthInterval {
                    start_field,
                    end_field,
                },
                PyYearMonthIntervalType {
                    start_field,
                    end_field,
                },
                [PyAtomicType, PyAnsiIntervalType],
                value
            ))
        }
        DataType::DayTimeInterval {
            start_field,
            end_field,
        } => {
            let (start_field, end_field) = (*start_field, *end_field);
            obj!(init_chain!(
                DataType::DayTimeInterval {
                    start_field,
                    end_field,
                },
                PyDayTimeIntervalType {
                    start_field,
                    end_field,
                },
                [PyAtomicType, PyAnsiIntervalType],
                value
            ))
        }
        DataType::Variant => obj!(init_chain!(
            DataType::Variant,
            PyVariantType,
            [PyAtomicType]
        )),
        DataType::Geometry { srid } => {
            let srid = *srid;
            obj!(init_chain!(
                DataType::Geometry { srid },
                PyGeometryType { srid },
                [PyAtomicType, PySpatialType],
                value
            ))
        }
        DataType::Geography { srid } => {
            let srid = *srid;
            obj!(init_chain!(
                DataType::Geography { srid },
                PyGeographyType { srid },
                [PyAtomicType, PySpatialType],
                value
            ))
        }
        DataType::Udt { .. } | DataType::Unparsed { .. } => {
            // A UDT reconstructs by re-importing its Python class; delegate to the Python
            // parser (which also handles the unparsed-DDL fallback).
            py.import("pyspark.sql.types")?
                .getattr("_parse_datatype_json_string")?
                .call1((dt.json(),))?
                .unbind()
        }
    };
    Ok(o)
}

pub(crate) fn py_to_data_type(obj: &Bound<'_, PyAny>) -> PyResult<DataType> {
    if let Ok(g) = obj.extract::<PyRef<PyGeometryType>>() {
        return Ok(DataType::Geometry { srid: g.srid });
    }
    if let Ok(g) = obj.extract::<PyRef<PyGeographyType>>() {
        return Ok(DataType::Geography { srid: g.srid });
    }
    if obj.extract::<PyRef<PyNullType>>().is_ok() {
        return Ok(DataType::Null);
    }
    if obj.extract::<PyRef<PyBooleanType>>().is_ok() {
        return Ok(DataType::Boolean);
    }
    if obj.extract::<PyRef<PyByteType>>().is_ok() {
        return Ok(DataType::Byte);
    }
    if obj.extract::<PyRef<PyShortType>>().is_ok() {
        return Ok(DataType::Short);
    }
    if obj.extract::<PyRef<PyIntegerType>>().is_ok() {
        return Ok(DataType::Integer);
    }
    if obj.extract::<PyRef<PyLongType>>().is_ok() {
        return Ok(DataType::Long);
    }
    if obj.extract::<PyRef<PyFloatType>>().is_ok() {
        return Ok(DataType::Float);
    }
    if obj.extract::<PyRef<PyDoubleType>>().is_ok() {
        return Ok(DataType::Double);
    }
    if obj.extract::<PyRef<PyStringType>>().is_ok() {
        return Ok(DataType::String {
            collation: "UTF8_BINARY".to_string(),
        });
    }
    if obj.extract::<PyRef<PyBinaryType>>().is_ok() {
        return Ok(DataType::Binary);
    }
    if obj.extract::<PyRef<PyDateType>>().is_ok() {
        return Ok(DataType::Date);
    }
    if obj.extract::<PyRef<PyTimestampType>>().is_ok() {
        return Ok(DataType::Timestamp);
    }
    if obj.extract::<PyRef<PyTimestampNTZType>>().is_ok() {
        return Ok(DataType::TimestampNtz);
    }
    if let Ok(d) = obj.extract::<PyRef<PyDecimalType>>() {
        return Ok(DataType::Decimal {
            precision: d.precision,
            scale: d.scale,
        });
    }
    if let Ok(a) = obj.extract::<PyRef<PyArrayType>>() {
        return Ok(DataType::Array {
            element_type: Box::new(a.element_type.clone()),
            contains_null: a.contains_null,
        });
    }
    if let Ok(m) = obj.extract::<PyRef<PyMapType>>() {
        return Ok(DataType::Map {
            key_type: Box::new(m.key_type.clone()),
            value_type: Box::new(m.value_type.clone()),
            value_contains_null: m.value_contains_null,
        });
    }
    if let Ok(st) = obj.extract::<PyRef<PyStructType>>() {
        return Ok(DataType::Struct {
            fields: st.fields.clone(),
        });
    }
    if let Ok(c) = obj.extract::<PyRef<PyCharType>>() {
        return Ok(DataType::Char { length: c.length });
    }
    if let Ok(v) = obj.extract::<PyRef<PyVarcharType>>() {
        return Ok(DataType::Varchar { length: v.length });
    }
    if let Ok(t) = obj.extract::<PyRef<PyTimeType>>() {
        return Ok(DataType::Time {
            precision: t.precision,
        });
    }
    if obj.extract::<PyRef<PyCalendarIntervalType>>().is_ok() {
        return Ok(DataType::CalendarInterval);
    }
    if let Ok(i) = obj.extract::<PyRef<PyYearMonthIntervalType>>() {
        return Ok(DataType::YearMonthInterval {
            start_field: i.start_field,
            end_field: i.end_field,
        });
    }
    if let Ok(i) = obj.extract::<PyRef<PyDayTimeIntervalType>>() {
        return Ok(DataType::DayTimeInterval {
            start_field: i.start_field,
            end_field: i.end_field,
        });
    }
    if obj.extract::<PyRef<PyVariantType>>().is_ok() {
        return Ok(DataType::Variant);
    }
    // `UserDefinedType` is a pure-Python subclass of the base `DataType` (its jsonValue
    // cloudpickles the concrete class, so it can't be lowered into Rust). Detect it before
    // the bare-base fallback and build the core `Udt` from its jsonValue + sqlType().
    if let Ok(udt_cls) = obj
        .py()
        .import("pyspark.sql.types")
        .and_then(|m| m.getattr("UserDefinedType"))
    {
        if obj.is_instance(&udt_cls)? {
            let jv = obj.call_method0("jsonValue")?;
            let get = |k: &str| -> PyResult<Option<String>> {
                let v = jv.call_method1("get", (k,))?;
                if v.is_none() {
                    Ok(None)
                } else {
                    Ok(Some(v.extract()?))
                }
            };
            let sql_type = py_to_data_type(&obj.call_method0("sqlType")?)?;
            return Ok(DataType::Udt {
                type_str: "udt".to_string(),
                jvm_class: get("class")?,
                python_class: get("pyClass")?,
                serialized_python_class: get("serializedClass")?,
                sql_type: Some(Box::new(sql_type)),
            });
        }
    }
    // The bare base `DataType` (e.g. from `fromDDL`) is matched LAST: concrete subclasses
    // (which all share the PyDataType base) are handled above via their own fields, so a
    // mutated StructType reflects its live fields rather than the base's snapshot.
    if let Ok(d) = obj.extract::<PyRef<PyDataType>>() {
        return Ok(d.inner.clone());
    }
    if let Ok(s) = obj.extract::<String>() {
        return DataType::from_ddl(&s).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "invalid type string '{s}': {e:?}"
            ))
        });
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "expected a DataType or a DDL type string",
    ))
}
