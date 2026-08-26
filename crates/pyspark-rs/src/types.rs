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
    pub element_type: DataType,
    pub contains_null: bool,
}

#[pymethods]
impl PyArrayType {
    #[new]
    #[pyo3(signature = (element_type, contains_null=true))]
    fn new(element_type: &Bound<'_, PyAny>, contains_null: bool) -> PyResult<Self> {
        Ok(PyArrayType {
            element_type: py_to_data_type(element_type)?,
            contains_null,
        })
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

#[pyclass(name = "MapType")]
pub struct PyMapType {
    pub key_type: DataType,
    pub value_type: DataType,
    pub value_contains_null: bool,
}

#[pymethods]
impl PyMapType {
    #[new]
    #[pyo3(signature = (key_type, value_type, value_contains_null=true))]
    fn new(
        key_type: &Bound<'_, PyAny>,
        value_type: &Bound<'_, PyAny>,
        value_contains_null: bool,
    ) -> PyResult<Self> {
        Ok(PyMapType {
            key_type: py_to_data_type(key_type)?,
            value_type: py_to_data_type(value_type)?,
            value_contains_null,
        })
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
    fn __repr__(&self) -> String {
        format!(
            "StructField('{}', {}, {})",
            self.field.name,
            self.field.data_type.simple_string(),
            self.field.nullable
        )
    }
}

#[pyclass(name = "StructType")]
pub struct PyStructType {
    pub(crate) fields: Vec<StructField>,
}

#[pymethods]
impl PyStructType {
    /// `StructType(fields=None)` where each field is a `StructField` (matching
    /// pyspark). A list of DDL field strings is also accepted for convenience.
    #[new]
    #[pyo3(signature = (fields=None))]
    fn new(fields: Option<Vec<Bound<'_, PyAny>>>) -> PyResult<Self> {
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
        Ok(PyStructType { fields: out })
    }

    /// Append a field (pyspark `StructType.add`). Chainable is not required here.
    #[pyo3(signature = (field, data_type=None, nullable=true))]
    fn add(
        &mut self,
        field: &Bound<'_, PyAny>,
        data_type: Option<&Bound<'_, PyAny>>,
        nullable: bool,
    ) -> PyResult<()> {
        if let Ok(sf) = field.extract::<PyRef<PyStructField>>() {
            self.fields.push(sf.field.clone());
        } else {
            let name: String = field.extract()?;
            let dt = match data_type {
                Some(d) => py_to_data_type(d)?,
                None => DataType::String {
                    collation: "UTF8_BINARY".to_string(),
                },
            };
            self.fields.push(StructField {
                name,
                data_type: dt,
                nullable,
                metadata: BTreeMap::new(),
            });
        }
        Ok(())
    }

    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> String {
        DataType::Struct {
            fields: self.fields.clone(),
        }
        .simple_string()
    }

    fn __repr__(&self) -> String {
        self.simple_string()
    }
}

#[pyclass(name = "CharType")]
pub struct PyCharType {
    pub length: i32,
}

#[pymethods]
impl PyCharType {
    #[new]
    fn new(length: i32) -> Self {
        PyCharType { length }
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

#[pyclass(name = "VarcharType")]
pub struct PyVarcharType {
    pub length: i32,
}

#[pymethods]
impl PyVarcharType {
    #[new]
    fn new(length: i32) -> Self {
        PyVarcharType { length }
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

#[pyclass(name = "TimeType")]
pub struct PyTimeType {
    pub precision: i32,
}

#[pymethods]
impl PyTimeType {
    /// `TimeType(precision=6)` (microsecond precision default), matching pyspark 4.2.
    #[new]
    #[pyo3(signature = (precision=6))]
    fn new(precision: i32) -> Self {
        PyTimeType { precision }
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

#[pyclass(name = "CalendarIntervalType")]
pub struct PyCalendarIntervalType;

#[pymethods]
impl PyCalendarIntervalType {
    #[new]
    fn new() -> Self {
        PyCalendarIntervalType
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

#[pyclass(name = "YearMonthIntervalType")]
pub struct PyYearMonthIntervalType {
    pub start_field: i32,
    pub end_field: i32,
}

#[pymethods]
impl PyYearMonthIntervalType {
    /// Fields: YEAR=0, MONTH=1. Defaults to the full YEAR..MONTH range.
    #[new]
    #[pyo3(signature = (startField=None, endField=None))]
    #[allow(non_snake_case)]
    fn new(startField: Option<i32>, endField: Option<i32>) -> Self {
        let start = startField.unwrap_or(0);
        PyYearMonthIntervalType {
            start_field: start,
            end_field: endField.unwrap_or(if startField.is_some() { start } else { 1 }),
        }
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

#[pyclass(name = "DayTimeIntervalType")]
pub struct PyDayTimeIntervalType {
    pub start_field: i32,
    pub end_field: i32,
}

#[pymethods]
impl PyDayTimeIntervalType {
    /// Fields: DAY=0, HOUR=1, MINUTE=2, SECOND=3. Defaults to the full DAY..SECOND range.
    #[new]
    #[pyo3(signature = (startField=None, endField=None))]
    #[allow(non_snake_case)]
    fn new(startField: Option<i32>, endField: Option<i32>) -> Self {
        let start = startField.unwrap_or(0);
        PyDayTimeIntervalType {
            start_field: start,
            end_field: endField.unwrap_or(if startField.is_some() { start } else { 3 }),
        }
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

#[pyclass(name = "VariantType")]
pub struct PyVariantType;

#[pymethods]
impl PyVariantType {
    #[new]
    fn new() -> Self {
        PyVariantType
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

/// Convert a Python DataType object (any of the type classes) or a DDL string into
/// the core `DataType`. Mirrors pyspark accepting `Union[DataType, str]` everywhere.
pub(crate) fn py_to_data_type(obj: &Bound<'_, PyAny>) -> PyResult<DataType> {
    if let Ok(d) = obj.extract::<PyRef<PyDataType>>() {
        return Ok(d.inner.clone());
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
