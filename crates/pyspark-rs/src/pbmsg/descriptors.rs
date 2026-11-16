//! The descriptors of the runtime's messages, fields, oneofs and enums, as protobuf's Python
//! descriptors (of upb 7, which have no `label`) give them to code that reads a message
//! generically: `ListFields()` gives field descriptors with `name`, `type`, `is_repeated` and the
//! `TYPE_*` and `LABEL_*` constants, and a
//! message class has the `DESCRIPTOR` of its message. One object stands for each descriptor, so
//! descriptors compare by identity as protobuf's do.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use super::message::registered_class;
use super::pool::{pool, EnumDesc, Kind, MessageDesc};
use super::scalar::Scalar;

#[pyclass(frozen, module = "pyspark.sql.connect.proto", name = "Descriptor")]
pub struct PyDescriptor {
    pub(crate) desc: Arc<MessageDesc>,
}

#[pyclass(frozen, module = "pyspark.sql.connect.proto", name = "FieldDescriptor")]
pub struct PyFieldDescriptor {
    message: Arc<MessageDesc>,
    index: usize,
}

#[pyclass(frozen, module = "pyspark.sql.connect.proto", name = "OneofDescriptor")]
pub struct PyOneofDescriptor {
    message: Arc<MessageDesc>,
    index: usize,
}

#[pyclass(frozen, module = "pyspark.sql.connect.proto", name = "EnumDescriptor")]
pub struct PyEnumDescriptor {
    desc: Arc<EnumDesc>,
}

#[pyclass(
    frozen,
    module = "pyspark.sql.connect.proto",
    name = "EnumValueDescriptor"
)]
pub struct PyEnumValueDescriptor {
    desc: Arc<EnumDesc>,
    index: usize,
}

#[pyclass(frozen, module = "pyspark.sql.connect.proto", name = "FileDescriptor")]
pub struct PyFileDescriptor {
    name: String,
}

type Cache<K, V> = OnceLock<Mutex<HashMap<K, Py<V>>>>;

fn cached<K, V, F>(py: Python<'_>, cache: &'static Cache<K, V>, key: K, make: F) -> PyResult<Py<V>>
where
    K: std::hash::Hash + Eq,
    V: pyo3::PyClass,
    F: FnOnce() -> PyResult<Py<V>>,
{
    let map = cache.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(found) = map.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        return Ok(found.clone_ref(py));
    }
    let made = make()?;
    let mut map = map.lock().unwrap_or_else(|e| e.into_inner());
    Ok(map.entry(key).or_insert(made).clone_ref(py))
}

/// The descriptor of the message type `full_name`.
pub(crate) fn descriptor_object(py: Python<'_>, full_name: &str) -> PyResult<Py<PyDescriptor>> {
    static CACHE: Cache<String, PyDescriptor> = OnceLock::new();
    cached(py, &CACHE, full_name.to_string(), || {
        let desc = pool()
            .message(full_name)
            .ok_or_else(|| PyKeyError::new_err(full_name.to_string()))?
            .clone();
        Py::new(py, PyDescriptor { desc })
    })
}

/// The descriptor of the field `index` of `message`.
pub(crate) fn field_descriptor_object(
    py: Python<'_>,
    message: &Arc<MessageDesc>,
    index: usize,
) -> PyResult<Py<PyAny>> {
    static CACHE: Cache<(String, usize), PyFieldDescriptor> = OnceLock::new();
    let field = cached(py, &CACHE, (message.full_name.clone(), index), || {
        Py::new(
            py,
            PyFieldDescriptor {
                message: message.clone(),
                index,
            },
        )
    })?;
    Ok(field.into_any())
}

fn oneof_descriptor_object(
    py: Python<'_>,
    message: &Arc<MessageDesc>,
    index: usize,
) -> PyResult<Py<PyOneofDescriptor>> {
    static CACHE: Cache<(String, usize), PyOneofDescriptor> = OnceLock::new();
    cached(py, &CACHE, (message.full_name.clone(), index), || {
        Py::new(
            py,
            PyOneofDescriptor {
                message: message.clone(),
                index,
            },
        )
    })
}

/// The descriptor of the enum type `full_name`.
pub(crate) fn enum_descriptor_object(
    py: Python<'_>,
    full_name: &str,
) -> PyResult<Py<PyEnumDescriptor>> {
    static CACHE: Cache<String, PyEnumDescriptor> = OnceLock::new();
    cached(py, &CACHE, full_name.to_string(), || {
        let desc = pool()
            .enumeration(full_name)
            .ok_or_else(|| PyKeyError::new_err(full_name.to_string()))?
            .clone();
        Py::new(py, PyEnumDescriptor { desc })
    })
}

/// The descriptor of the file `name`, such as `spark/connect/base.proto`.
pub(crate) fn file_descriptor_object(py: Python<'_>, name: &str) -> PyResult<Py<PyFileDescriptor>> {
    static CACHE: Cache<String, PyFileDescriptor> = OnceLock::new();
    cached(py, &CACHE, name.to_string(), || {
        if pool().file(name).is_none() {
            return Err(PyKeyError::new_err(name.to_string()));
        }
        Py::new(
            py,
            PyFileDescriptor {
                name: name.to_string(),
            },
        )
    })
}

fn empty_dict(py: Python<'_>) -> Bound<'_, PyDict> {
    PyDict::new(py)
}

fn enum_value_object(
    py: Python<'_>,
    desc: &Arc<EnumDesc>,
    index: usize,
) -> PyResult<Py<PyEnumValueDescriptor>> {
    static CACHE: Cache<(String, usize), PyEnumValueDescriptor> = OnceLock::new();
    cached(py, &CACHE, (desc.full_name.clone(), index), || {
        Py::new(
            py,
            PyEnumValueDescriptor {
                desc: desc.clone(),
                index,
            },
        )
    })
}

#[pymethods]
impl PyDescriptor {
    #[getter]
    fn name(&self) -> &str {
        &self.desc.name
    }

    #[getter]
    fn full_name(&self) -> &str {
        &self.desc.full_name
    }

    #[getter]
    fn fields<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let fields = (0..self.desc.fields.len())
            .map(|i| field_descriptor_object(py, &self.desc, i))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, fields)
    }

    #[getter]
    fn fields_by_name<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (i, field) in self.desc.fields.iter().enumerate() {
            dict.set_item(&field.name, field_descriptor_object(py, &self.desc, i)?)?;
        }
        Ok(dict)
    }

    #[getter]
    fn fields_by_number<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (i, field) in self.desc.fields.iter().enumerate() {
            dict.set_item(field.number, field_descriptor_object(py, &self.desc, i)?)?;
        }
        Ok(dict)
    }

    #[getter]
    fn oneofs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let oneofs = (0..self.desc.oneofs.len())
            .map(|i| oneof_descriptor_object(py, &self.desc, i))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, oneofs)
    }

    #[getter]
    fn oneofs_by_name<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (i, oneof) in self.desc.oneofs.iter().enumerate() {
            dict.set_item(&oneof.name, oneof_descriptor_object(py, &self.desc, i)?)?;
        }
        Ok(dict)
    }

    #[getter]
    fn nested_types<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let nested = self
            .desc
            .nested_messages
            .iter()
            .map(|n| descriptor_object(py, n))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, nested)
    }

    #[getter]
    fn nested_types_by_name<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for name in &self.desc.nested_messages {
            let short = name.rsplit('.').next().unwrap_or(name);
            dict.set_item(short, descriptor_object(py, name)?)?;
        }
        Ok(dict)
    }

    #[getter]
    fn enum_types<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let enums = self
            .desc
            .nested_enums
            .iter()
            .map(|n| enum_descriptor_object(py, n))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, enums)
    }

    #[getter]
    fn enum_types_by_name<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for name in &self.desc.nested_enums {
            let short = name.rsplit('.').next().unwrap_or(name);
            dict.set_item(short, enum_descriptor_object(py, name)?)?;
        }
        Ok(dict)
    }

    #[getter]
    fn containing_type(&self, py: Python<'_>) -> PyResult<Option<Py<PyDescriptor>>> {
        self.desc
            .containing
            .as_deref()
            .map(|c| descriptor_object(py, c))
            .transpose()
    }

    #[getter]
    fn file_name(&self) -> &str {
        &self.desc.file
    }

    #[getter]
    fn file(&self, py: Python<'_>) -> PyResult<Py<PyFileDescriptor>> {
        file_descriptor_object(py, &self.desc.file)
    }

    #[getter]
    fn package(&self) -> &str {
        &self.desc.package
    }

    #[getter]
    fn has_options(&self) -> bool {
        self.desc.has_options
    }

    #[getter]
    fn is_extendable(&self) -> bool {
        false
    }

    #[getter]
    fn fields_by_camelcase_name<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (i, field) in self.desc.fields.iter().enumerate() {
            dict.set_item(
                &field.json_name,
                field_descriptor_object(py, &self.desc, i)?,
            )?;
        }
        Ok(dict)
    }

    /// The values of the enums of the message, by name, as protobuf's `enum_values_by_name`.
    #[getter]
    fn enum_values_by_name<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for name in &self.desc.nested_enums {
            let enumeration = enum_descriptor_object(py, name)?;
            for (value_name, value) in enumeration
                .bind(py)
                .getattr("values_by_name")?
                .cast::<PyDict>()?
                .iter()
            {
                dict.set_item(value_name, value)?;
            }
        }
        Ok(dict)
    }

    /// The name of the value `value` of the enum `enum_type` of the message, as protobuf's
    /// `EnumValueName`.
    #[pyo3(name = "EnumValueName")]
    fn enum_value_name(&self, enum_type: &str, value: i32) -> PyResult<String> {
        let full_name = format!("{}.{enum_type}", self.desc.full_name);
        pool()
            .enumeration(&full_name)
            .and_then(|e| e.name_of(value))
            .map(str::to_string)
            .ok_or_else(|| PyKeyError::new_err(value))
    }

    #[getter]
    fn extension_ranges<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        PyList::empty(py)
    }

    #[getter]
    fn extensions<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        PyList::empty(py)
    }

    #[getter]
    fn extensions_by_name<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        empty_dict(py)
    }

    /// The class of the message, as protobuf's `_concrete_class` gives it.
    #[getter]
    fn _concrete_class(&self, py: Python<'_>) -> Option<Py<pyo3::types::PyType>> {
        registered_class(py, &self.desc.full_name)
    }

    fn __repr__(&self) -> String {
        format!("<Descriptor {}>", self.desc.full_name)
    }
}

#[pymethods]
impl PyFieldDescriptor {
    #[classattr]
    const TYPE_DOUBLE: i32 = 1;
    #[classattr]
    const TYPE_FLOAT: i32 = 2;
    #[classattr]
    const TYPE_INT64: i32 = 3;
    #[classattr]
    const TYPE_UINT64: i32 = 4;
    #[classattr]
    const TYPE_INT32: i32 = 5;
    #[classattr]
    const TYPE_FIXED64: i32 = 6;
    #[classattr]
    const TYPE_FIXED32: i32 = 7;
    #[classattr]
    const TYPE_BOOL: i32 = 8;
    #[classattr]
    const TYPE_STRING: i32 = 9;
    #[classattr]
    const TYPE_GROUP: i32 = 10;
    #[classattr]
    const TYPE_MESSAGE: i32 = 11;
    #[classattr]
    const TYPE_BYTES: i32 = 12;
    #[classattr]
    const TYPE_UINT32: i32 = 13;
    #[classattr]
    const TYPE_ENUM: i32 = 14;
    #[classattr]
    const TYPE_SFIXED32: i32 = 15;
    #[classattr]
    const TYPE_SFIXED64: i32 = 16;
    #[classattr]
    const TYPE_SINT32: i32 = 17;
    #[classattr]
    const TYPE_SINT64: i32 = 18;
    #[classattr]
    const MAX_TYPE: i32 = 18;
    #[classattr]
    const CPPTYPE_INT32: i32 = 1;
    #[classattr]
    const CPPTYPE_INT64: i32 = 2;
    #[classattr]
    const CPPTYPE_UINT32: i32 = 3;
    #[classattr]
    const CPPTYPE_UINT64: i32 = 4;
    #[classattr]
    const CPPTYPE_DOUBLE: i32 = 5;
    #[classattr]
    const CPPTYPE_FLOAT: i32 = 6;
    #[classattr]
    const CPPTYPE_BOOL: i32 = 7;
    #[classattr]
    const CPPTYPE_ENUM: i32 = 8;
    #[classattr]
    const CPPTYPE_STRING: i32 = 9;
    #[classattr]
    const CPPTYPE_MESSAGE: i32 = 10;
    #[classattr]
    const MAX_CPPTYPE: i32 = 10;
    #[classattr]
    const LABEL_OPTIONAL: i32 = 1;
    #[classattr]
    const LABEL_REQUIRED: i32 = 2;
    #[classattr]
    const LABEL_REPEATED: i32 = 3;
    #[classattr]
    const MAX_LABEL: i32 = 3;
    #[classattr]
    const MAX_FIELD_NUMBER: i32 = 536_870_911;

    #[getter]
    fn name(&self) -> &str {
        &self.message.fields[self.index].name
    }

    #[getter]
    fn full_name(&self) -> &str {
        &self.message.fields[self.index].full_name
    }

    #[getter]
    fn number(&self) -> u32 {
        self.message.fields[self.index].number
    }

    #[getter]
    fn index(&self) -> usize {
        self.index
    }

    #[getter(r#type)]
    fn type_(&self) -> i32 {
        self.message.fields[self.index].kind.proto_type()
    }

    #[getter]
    fn cpp_type(&self) -> i32 {
        self.message.fields[self.index].kind.cpp_type()
    }

    #[getter]
    fn is_repeated(&self) -> bool {
        self.message.fields[self.index].repeated
    }

    #[getter]
    fn has_presence(&self) -> bool {
        self.message.fields[self.index].has_presence()
    }

    #[getter]
    fn json_name(&self) -> &str {
        &self.message.fields[self.index].json_name
    }

    #[getter]
    fn camelcase_name(&self) -> &str {
        &self.message.fields[self.index].json_name
    }

    #[getter]
    fn message_type(&self, py: Python<'_>) -> PyResult<Option<Py<PyDescriptor>>> {
        let field = &self.message.fields[self.index];
        if field.kind != Kind::Message {
            return Ok(None);
        }
        descriptor_object(py, &field.type_name).map(Some)
    }

    #[getter]
    fn enum_type(&self, py: Python<'_>) -> PyResult<Option<Py<PyEnumDescriptor>>> {
        let field = &self.message.fields[self.index];
        if field.kind != Kind::Enum {
            return Ok(None);
        }
        enum_descriptor_object(py, &field.type_name).map(Some)
    }

    #[getter]
    fn containing_type(&self, py: Python<'_>) -> PyResult<Py<PyDescriptor>> {
        descriptor_object(py, &self.message.full_name)
    }

    #[getter]
    fn containing_oneof(&self, py: Python<'_>) -> PyResult<Option<Py<PyOneofDescriptor>>> {
        self.message.fields[self.index]
            .oneof
            .map(|o| oneof_descriptor_object(py, &self.message, o))
            .transpose()
    }

    #[getter]
    fn default_value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let field = &self.message.fields[self.index];
        if field.repeated {
            return Ok(PyList::empty(py).into_any().unbind());
        }
        if field.kind == Kind::Message {
            return Ok(py.None());
        }
        Scalar::default_of(field.kind).to_py(py)
    }

    #[getter]
    fn has_default_value(&self) -> bool {
        false
    }

    #[getter]
    fn has_options(&self) -> bool {
        self.message.fields[self.index].has_options
    }

    #[getter]
    fn is_packed(&self) -> bool {
        self.message.fields[self.index].packed
    }

    #[getter]
    fn is_required(&self) -> bool {
        false
    }

    #[getter]
    fn file(&self, py: Python<'_>) -> PyResult<Py<PyFileDescriptor>> {
        file_descriptor_object(py, &self.message.file)
    }

    #[getter]
    fn is_extension(&self) -> bool {
        false
    }

    #[getter]
    fn extension_scope(&self, py: Python<'_>) -> Py<PyAny> {
        py.None()
    }

    fn __repr__(&self) -> String {
        format!(
            "<FieldDescriptor {}>",
            self.message.fields[self.index].full_name
        )
    }
}

#[pymethods]
impl PyOneofDescriptor {
    #[getter]
    fn name(&self) -> &str {
        &self.message.oneofs[self.index].name
    }

    #[getter]
    fn full_name(&self) -> &str {
        &self.message.oneofs[self.index].full_name
    }

    #[getter]
    fn index(&self) -> usize {
        self.index
    }

    #[getter]
    fn fields<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let fields = self.message.oneofs[self.index]
            .fields
            .iter()
            .map(|&i| field_descriptor_object(py, &self.message, i))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, fields)
    }

    #[getter]
    fn containing_type(&self, py: Python<'_>) -> PyResult<Py<PyDescriptor>> {
        descriptor_object(py, &self.message.full_name)
    }

    fn __repr__(&self) -> String {
        format!(
            "<OneofDescriptor {}>",
            self.message.oneofs[self.index].full_name
        )
    }
}

#[pymethods]
impl PyEnumDescriptor {
    #[getter]
    fn name(&self) -> &str {
        &self.desc.name
    }

    #[getter]
    fn full_name(&self) -> &str {
        &self.desc.full_name
    }

    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let values = (0..self.desc.values.len())
            .map(|i| enum_value_object(py, &self.desc, i))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, values)
    }

    #[getter]
    fn values_by_name<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (i, (name, _)) in self.desc.values.iter().enumerate() {
            dict.set_item(name, enum_value_object(py, &self.desc, i)?)?;
        }
        Ok(dict)
    }

    #[getter]
    fn values_by_number<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (i, (_, number)) in self.desc.values.iter().enumerate() {
            if !dict.contains(*number)? {
                dict.set_item(*number, enum_value_object(py, &self.desc, i)?)?;
            }
        }
        Ok(dict)
    }

    #[getter]
    fn containing_type(&self, py: Python<'_>) -> PyResult<Option<Py<PyDescriptor>>> {
        self.desc
            .containing
            .as_deref()
            .map(|c| descriptor_object(py, c))
            .transpose()
    }

    #[getter]
    fn file_name(&self) -> &str {
        &self.desc.file
    }

    #[getter]
    fn file(&self, py: Python<'_>) -> PyResult<Py<PyFileDescriptor>> {
        file_descriptor_object(py, &self.desc.file)
    }

    #[getter]
    fn has_options(&self) -> bool {
        self.desc.has_options
    }

    fn __repr__(&self) -> String {
        format!("<EnumDescriptor {}>", self.desc.full_name)
    }
}

#[pymethods]
impl PyEnumValueDescriptor {
    #[getter]
    fn name(&self) -> &str {
        &self.desc.values[self.index].0
    }

    #[getter]
    fn number(&self) -> i32 {
        self.desc.values[self.index].1
    }

    #[getter]
    fn index(&self) -> usize {
        self.index
    }

    #[getter(r#type)]
    fn type_(&self, py: Python<'_>) -> PyResult<Py<PyEnumDescriptor>> {
        enum_descriptor_object(py, &self.desc.full_name)
    }

    fn __repr__(&self) -> String {
        format!(
            "<EnumValueDescriptor {}.{}>",
            self.desc.full_name, self.desc.values[self.index].0
        )
    }
}

#[pymethods]
impl PyFileDescriptor {
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[getter]
    fn package(&self) -> PyResult<&'static str> {
        Ok(&self.file()?.package)
    }

    /// The `FileDescriptorProto` of the file, serialized without its source info, as protobuf's
    /// generated modules embed it.
    #[getter]
    fn serialized_pb<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &self.file()?.serialized))
    }

    #[getter]
    fn dependencies<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let files = self
            .file()?
            .dependencies
            .iter()
            .map(|d| file_descriptor_object(py, d))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, files)
    }

    #[getter]
    fn public_dependencies<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let files = self
            .file()?
            .public_dependencies
            .iter()
            .map(|d| file_descriptor_object(py, d))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, files)
    }

    #[getter]
    fn message_types_by_name<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for name in &self.file()?.messages {
            let short = name.rsplit('.').next().unwrap_or(name);
            dict.set_item(short, descriptor_object(py, name)?)?;
        }
        Ok(dict)
    }

    #[getter]
    fn enum_types_by_name<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for name in &self.file()?.enums {
            let short = name.rsplit('.').next().unwrap_or(name);
            dict.set_item(short, enum_descriptor_object(py, name)?)?;
        }
        Ok(dict)
    }

    #[getter]
    fn extensions_by_name<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        empty_dict(py)
    }

    #[getter]
    fn services_by_name<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        empty_dict(py)
    }

    #[getter]
    fn has_options(&self) -> PyResult<bool> {
        Ok(self.file()?.has_options)
    }

    fn __repr__(&self) -> String {
        format!("<FileDescriptor {}>", self.name)
    }
}

impl PyFileDescriptor {
    fn file(&self) -> PyResult<&'static super::pool::FileDesc> {
        pool()
            .file(&self.name)
            .ok_or_else(|| PyValueError::new_err(format!("no file {}", self.name)))
    }
}
