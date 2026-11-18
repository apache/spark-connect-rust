//! The descriptors of the runtime's messages, fields, oneofs, enums, files, services and methods,
//! and the pool that finds them, as protobuf's Python descriptors (of upb 7, which have no
//! `label`) give them to code that reads a message generically: `ListFields()` gives field
//! descriptors with `name`, `type`, `is_repeated` and the `TYPE_*` and `LABEL_*` constants, and a
//! message class has the `DESCRIPTOR` of its message. One object stands for each descriptor, so
//! descriptors compare by identity as protobuf's do.
//!
//! `GetOptions()` gives the options of a descriptor as upb's does, as a message of protobuf's
//! `google.protobuf.descriptor_pb2` (which `json_format` and `text_format` read the `map_entry` of
//! a map field of), made once for each descriptor; `CopyToProto(proto)` merges the descriptor's
//! proto, as its file declares it, into `proto`, a message of that type.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use prost::Message as _;
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyDict, PyList};

use super::message::registered_class;
use super::pool::{pool, EnumDesc, Kind, MessageDesc, ServiceDesc};
use super::scalar::Scalar;

#[pyclass(frozen, module = "pyspark.sql.connect.proto", name = "Descriptor")]
pub struct PyDescriptor {
    pub(crate) desc: Arc<MessageDesc>,
    options: PyOnceLock<Py<PyAny>>,
}

#[pyclass(frozen, module = "pyspark.sql.connect.proto", name = "FieldDescriptor")]
pub struct PyFieldDescriptor {
    message: Arc<MessageDesc>,
    index: usize,
    options: PyOnceLock<Py<PyAny>>,
}

#[pyclass(frozen, module = "pyspark.sql.connect.proto", name = "OneofDescriptor")]
pub struct PyOneofDescriptor {
    message: Arc<MessageDesc>,
    index: usize,
    options: PyOnceLock<Py<PyAny>>,
}

#[pyclass(frozen, module = "pyspark.sql.connect.proto", name = "EnumDescriptor")]
pub struct PyEnumDescriptor {
    desc: Arc<EnumDesc>,
    options: PyOnceLock<Py<PyAny>>,
}

#[pyclass(
    frozen,
    module = "pyspark.sql.connect.proto",
    name = "EnumValueDescriptor"
)]
pub struct PyEnumValueDescriptor {
    desc: Arc<EnumDesc>,
    index: usize,
    options: PyOnceLock<Py<PyAny>>,
}

#[pyclass(frozen, module = "pyspark.sql.connect.proto", name = "FileDescriptor")]
pub struct PyFileDescriptor {
    name: String,
    options: PyOnceLock<Py<PyAny>>,
}

#[pyclass(
    frozen,
    module = "pyspark.sql.connect.proto",
    name = "ServiceDescriptor"
)]
pub struct PyServiceDescriptor {
    desc: Arc<ServiceDesc>,
    options: PyOnceLock<Py<PyAny>>,
}

#[pyclass(
    frozen,
    module = "pyspark.sql.connect.proto",
    name = "MethodDescriptor"
)]
pub struct PyMethodDescriptor {
    service: Arc<ServiceDesc>,
    index: usize,
    options: PyOnceLock<Py<PyAny>>,
}

/// The pool of the descriptors of the runtime, as protobuf's `DescriptorPool` finds them: the
/// `pool` of a `FileDescriptor`.
#[pyclass(frozen, module = "pyspark.sql.connect.proto", name = "DescriptorPool")]
pub struct PyDescriptorPool;

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
        Py::new(
            py,
            PyDescriptor {
                desc,
                options: PyOnceLock::new(),
            },
        )
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
                options: PyOnceLock::new(),
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
                options: PyOnceLock::new(),
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
        Py::new(
            py,
            PyEnumDescriptor {
                desc,
                options: PyOnceLock::new(),
            },
        )
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
                options: PyOnceLock::new(),
            },
        )
    })
}

/// The descriptor of the service `full_name`, such as `spark.connect.SparkConnectService`.
pub(crate) fn service_descriptor_object(
    py: Python<'_>,
    full_name: &str,
) -> PyResult<Py<PyServiceDescriptor>> {
    static CACHE: Cache<String, PyServiceDescriptor> = OnceLock::new();
    cached(py, &CACHE, full_name.to_string(), || {
        let desc = pool()
            .service(full_name)
            .ok_or_else(|| PyKeyError::new_err(full_name.to_string()))?
            .clone();
        Py::new(
            py,
            PyServiceDescriptor {
                desc,
                options: PyOnceLock::new(),
            },
        )
    })
}

fn method_descriptor_object(
    py: Python<'_>,
    service: &Arc<ServiceDesc>,
    index: usize,
) -> PyResult<Py<PyMethodDescriptor>> {
    static CACHE: Cache<(String, usize), PyMethodDescriptor> = OnceLock::new();
    cached(py, &CACHE, (service.full_name.clone(), index), || {
        Py::new(
            py,
            PyMethodDescriptor {
                service: service.clone(),
                index,
                options: PyOnceLock::new(),
            },
        )
    })
}

/// The pool of the runtime, one object, as a `FileDescriptor` gives it.
fn pool_object(py: Python<'_>) -> PyResult<Py<PyDescriptorPool>> {
    static POOL: PyOnceLock<Py<PyDescriptorPool>> = PyOnceLock::new();
    POOL.get_or_try_init(py, || Py::new(py, PyDescriptorPool))
        .map(|p| p.clone_ref(py))
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
                options: PyOnceLock::new(),
            },
        )
    })
}

/// The options of a descriptor, as upb's `GetOptions()` gives them: the message of the class
/// `class` of protobuf's `descriptor_pb2` (such as `MessageOptions`) of the encoded options that
/// `encoded` gives, made once, in `cell`, so that every call gives the same object.
fn options_of(
    py: Python<'_>,
    cell: &PyOnceLock<Py<PyAny>>,
    class: &str,
    encoded: impl FnOnce() -> Vec<u8>,
) -> PyResult<Py<PyAny>> {
    cell.get_or_try_init(py, || {
        let options = py
            .import("google.protobuf.descriptor_pb2")?
            .getattr(class)?
            .call_method1("FromString", (PyBytes::new(py, &encoded()),))?;
        Ok::<_, PyErr>(options.unbind())
    })
    .map(|options| options.clone_ref(py))
}

/// The encoded `options` of a descriptor's proto, or none when it has none.
fn encoded<M: prost::Message>(options: Option<M>) -> Vec<u8> {
    options.map(|o| o.encode_to_vec()).unwrap_or_default()
}

/// `CopyToProto(proto)`, as upb's: merge `bytes`, the encoded proto of a descriptor, which is of
/// the type `expected` (such as `google.protobuf.DescriptorProto`), into `proto`, a message of
/// that type, of upb's errors of an object that is not a message and of a message of another
/// type. The result is that of the message's `MergeFromString`.
fn copy_to_proto(
    py: Python<'_>,
    proto: &Bound<'_, PyAny>,
    expected: &str,
    bytes: Option<Vec<u8>>,
) -> PyResult<Py<PyAny>> {
    let full_name = proto
        .getattr("DESCRIPTOR")
        .and_then(|d| d.getattr("full_name"))
        .and_then(|n| n.extract::<String>());
    let Ok(full_name) = full_name else {
        return Err(PyTypeError::new_err(format!(
            "Expected a message object, but got {}.",
            proto.repr()?
        )));
    };
    if full_name != expected {
        return Err(PyTypeError::new_err(format!(
            "CopyToProto: message is of incorrect type '{full_name}' (expected '{expected}'"
        )));
    }
    let bytes = bytes.ok_or_else(|| {
        PyValueError::new_err(format!("the proto of a {expected} is not in its file"))
    })?;
    Ok(proto
        .call_method1("MergeFromString", (PyBytes::new(py, &bytes),))?
        .unbind())
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
    fn file(&self, py: Python<'_>) -> PyResult<Py<PyFileDescriptor>> {
        file_descriptor_object(py, &self.desc.file)
    }

    #[getter]
    fn has_options(&self) -> bool {
        self.desc.has_options
    }

    /// The `google.protobuf.MessageOptions` of the message, such as the `map_entry` of a map
    /// entry.
    #[pyo3(name = "GetOptions")]
    fn get_options(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        options_of(py, &self.options, "MessageOptions", || {
            encoded(pool().message_proto(&self.desc).and_then(|p| p.options))
        })
    }

    /// Merge the `google.protobuf.DescriptorProto` of the message into `proto`.
    #[pyo3(name = "CopyToProto")]
    fn copy_to_proto(&self, py: Python<'_>, proto: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let bytes = pool().message_proto(&self.desc).map(|p| p.encode_to_vec());
        copy_to_proto(py, proto, "google.protobuf.DescriptorProto", bytes)
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
    /// As upb's: the C++ type of a bytes field is that of a string.
    #[classattr]
    const CPPTYPE_BYTES: i32 = 9;
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

    /// The `google.protobuf.FieldOptions` of the field, such as its `deprecated`.
    #[pyo3(name = "GetOptions")]
    fn get_options(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        options_of(py, &self.options, "FieldOptions", || {
            encoded(
                pool()
                    .message_proto(&self.message)
                    .and_then(|p| p.field.into_iter().nth(self.index))
                    .and_then(|f| f.options),
            )
        })
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

    #[getter]
    fn has_options(&self) -> bool {
        self.message.oneofs[self.index].has_options
    }

    /// The `google.protobuf.OneofOptions` of the oneof.
    #[pyo3(name = "GetOptions")]
    fn get_options(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        options_of(py, &self.options, "OneofOptions", || {
            encoded(
                pool()
                    .message_proto(&self.message)
                    .and_then(|p| p.oneof_decl.into_iter().nth(self.index))
                    .and_then(|o| o.options),
            )
        })
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
    fn file(&self, py: Python<'_>) -> PyResult<Py<PyFileDescriptor>> {
        file_descriptor_object(py, &self.desc.file)
    }

    #[getter]
    fn has_options(&self) -> bool {
        self.desc.has_options
    }

    /// Whether the enum is closed, as an enum of a proto2 file is; those of Spark Connect, of
    /// proto3, are open.
    #[getter]
    fn is_closed(&self) -> bool {
        self.desc.closed
    }

    /// The `google.protobuf.EnumOptions` of the enum.
    #[pyo3(name = "GetOptions")]
    fn get_options(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        options_of(py, &self.options, "EnumOptions", || {
            encoded(pool().enum_proto(&self.desc).and_then(|p| p.options))
        })
    }

    /// Merge the `google.protobuf.EnumDescriptorProto` of the enum into `proto`.
    #[pyo3(name = "CopyToProto")]
    fn copy_to_proto(&self, py: Python<'_>, proto: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let bytes = pool().enum_proto(&self.desc).map(|p| p.encode_to_vec());
        copy_to_proto(py, proto, "google.protobuf.EnumDescriptorProto", bytes)
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

    #[getter]
    fn has_options(&self) -> bool {
        self.desc.value_has_options[self.index]
    }

    /// The `google.protobuf.EnumValueOptions` of the value.
    #[pyo3(name = "GetOptions")]
    fn get_options(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        options_of(py, &self.options, "EnumValueOptions", || {
            encoded(
                pool()
                    .enum_proto(&self.desc)
                    .and_then(|p| p.value.into_iter().nth(self.index))
                    .and_then(|v| v.options),
            )
        })
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

    /// The pool of the file: that of the runtime, which finds its descriptors.
    #[getter]
    fn pool(&self, py: Python<'_>) -> PyResult<Py<PyDescriptorPool>> {
        pool_object(py)
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
    fn services_by_name<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for name in &self.file()?.services {
            let short = name.rsplit('.').next().unwrap_or(name);
            dict.set_item(short, service_descriptor_object(py, name)?)?;
        }
        Ok(dict)
    }

    #[getter]
    fn has_options(&self) -> PyResult<bool> {
        Ok(self.file()?.has_options)
    }

    /// The `google.protobuf.FileOptions` of the file, such as its `java_package`.
    #[pyo3(name = "GetOptions")]
    fn get_options(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        options_of(py, &self.options, "FileOptions", || {
            encoded(pool().file_proto(&self.name).and_then(|p| p.options))
        })
    }

    /// Merge the `google.protobuf.FileDescriptorProto` of the file into `proto`.
    #[pyo3(name = "CopyToProto")]
    fn copy_to_proto(&self, py: Python<'_>, proto: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let bytes = Some(self.file()?.serialized.clone());
        copy_to_proto(py, proto, "google.protobuf.FileDescriptorProto", bytes)
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

#[pymethods]
impl PyServiceDescriptor {
    #[getter]
    fn name(&self) -> &str {
        &self.desc.name
    }

    #[getter]
    fn full_name(&self) -> &str {
        &self.desc.full_name
    }

    #[getter]
    fn index(&self) -> usize {
        self.desc.index
    }

    #[getter]
    fn file(&self, py: Python<'_>) -> PyResult<Py<PyFileDescriptor>> {
        file_descriptor_object(py, &self.desc.file)
    }

    #[getter]
    fn methods<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let methods = (0..self.desc.methods.len())
            .map(|i| method_descriptor_object(py, &self.desc, i))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, methods)
    }

    #[getter]
    fn methods_by_name<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (i, method) in self.desc.methods.iter().enumerate() {
            dict.set_item(&method.name, method_descriptor_object(py, &self.desc, i)?)?;
        }
        Ok(dict)
    }

    /// The method `name` of the service, as upb's `FindMethodByName`.
    #[pyo3(name = "FindMethodByName")]
    fn find_method_by_name(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyMethodDescriptor>> {
        let index = self
            .desc
            .methods
            .iter()
            .position(|m| m.name == name)
            .ok_or_else(|| PyKeyError::new_err(format!("Couldn't find method {name}")))?;
        method_descriptor_object(py, &self.desc, index)
    }

    #[getter]
    fn has_options(&self) -> bool {
        self.desc.has_options
    }

    /// The `google.protobuf.ServiceOptions` of the service.
    #[pyo3(name = "GetOptions")]
    fn get_options(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        options_of(py, &self.options, "ServiceOptions", || {
            encoded(pool().service_proto(&self.desc).and_then(|p| p.options))
        })
    }

    /// Merge the `google.protobuf.ServiceDescriptorProto` of the service into `proto`.
    #[pyo3(name = "CopyToProto")]
    fn copy_to_proto(&self, py: Python<'_>, proto: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let bytes = pool().service_proto(&self.desc).map(|p| p.encode_to_vec());
        copy_to_proto(py, proto, "google.protobuf.ServiceDescriptorProto", bytes)
    }

    fn __repr__(&self) -> String {
        format!("<ServiceDescriptor {}>", self.desc.full_name)
    }
}

#[pymethods]
impl PyMethodDescriptor {
    #[getter]
    fn name(&self) -> &str {
        &self.service.methods[self.index].name
    }

    #[getter]
    fn full_name(&self) -> &str {
        &self.service.methods[self.index].full_name
    }

    #[getter]
    fn index(&self) -> usize {
        self.index
    }

    #[getter]
    fn containing_service(&self, py: Python<'_>) -> PyResult<Py<PyServiceDescriptor>> {
        service_descriptor_object(py, &self.service.full_name)
    }

    #[getter]
    fn input_type(&self, py: Python<'_>) -> PyResult<Py<PyDescriptor>> {
        descriptor_object(py, &self.service.methods[self.index].input_type)
    }

    #[getter]
    fn output_type(&self, py: Python<'_>) -> PyResult<Py<PyDescriptor>> {
        descriptor_object(py, &self.service.methods[self.index].output_type)
    }

    #[getter]
    fn client_streaming(&self) -> bool {
        self.service.methods[self.index].client_streaming
    }

    #[getter]
    fn server_streaming(&self) -> bool {
        self.service.methods[self.index].server_streaming
    }

    #[getter]
    fn has_options(&self) -> bool {
        self.service.methods[self.index].has_options
    }

    /// The `google.protobuf.MethodOptions` of the method.
    #[pyo3(name = "GetOptions")]
    fn get_options(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        options_of(py, &self.options, "MethodOptions", || {
            encoded(
                pool()
                    .service_proto(&self.service)
                    .and_then(|p| p.method.into_iter().nth(self.index))
                    .and_then(|m| m.options),
            )
        })
    }

    /// Merge the `google.protobuf.MethodDescriptorProto` of the method into `proto`.
    #[pyo3(name = "CopyToProto")]
    fn copy_to_proto(&self, py: Python<'_>, proto: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let bytes = pool()
            .service_proto(&self.service)
            .and_then(|p| p.method.into_iter().nth(self.index))
            .map(|m| m.encode_to_vec());
        copy_to_proto(py, proto, "google.protobuf.MethodDescriptorProto", bytes)
    }

    fn __repr__(&self) -> String {
        format!(
            "<MethodDescriptor {}>",
            self.service.methods[self.index].full_name
        )
    }
}

/// The file of `symbol`, as upb's `FindFileContainingSymbol` finds it: of a message, an enum, a
/// service, a field or oneof of a message, or a value of an enum (in the scope that declares the
/// enum), but not of a method of a service, which upb's does not find either.
fn file_of_symbol(symbol: &str) -> Option<&'static str> {
    let pool = pool();
    if let Some(m) = pool.message(symbol) {
        return Some(&m.file);
    }
    if let Some(e) = pool.enumeration(symbol) {
        return Some(&e.file);
    }
    if let Some(s) = pool.service(symbol) {
        return Some(&s.file);
    }
    let (scope, name) = symbol.rsplit_once('.')?;
    let has_value = |enums: &[String]| {
        enums.iter().any(|e| {
            pool.enumeration(e)
                .is_some_and(|e| e.number_of(name).is_some())
        })
    };
    if let Some(m) = pool.message(scope) {
        let member = m.by_name.contains_key(name) || m.oneof_by_name.contains_key(name);
        return (member || has_value(&m.nested_enums)).then_some(m.file.as_str());
    }
    pool.all_files()
        .into_iter()
        .find(|f| f.package == scope && has_value(&f.enums))
        .map(|f| f.name.as_str())
}

/// The message type of `full_name` and the index of its member `name` that `index_of` finds.
fn member_of(
    full_name: &str,
    index_of: impl Fn(&MessageDesc, &str) -> Option<usize>,
) -> Option<(Arc<MessageDesc>, usize)> {
    let (scope, name) = full_name.rsplit_once('.')?;
    let message = pool().message(scope)?;
    Some((message.clone(), index_of(message, name)?))
}

#[pymethods]
impl PyDescriptorPool {
    #[pyo3(name = "FindFileByName")]
    fn find_file_by_name(&self, py: Python<'_>, file_name: &str) -> PyResult<Py<PyFileDescriptor>> {
        if pool().file(file_name).is_none() {
            return Err(PyKeyError::new_err(format!(
                "Couldn't find file {file_name}"
            )));
        }
        file_descriptor_object(py, file_name)
    }

    #[pyo3(name = "FindMessageTypeByName")]
    fn find_message_type_by_name(
        &self,
        py: Python<'_>,
        full_name: &str,
    ) -> PyResult<Py<PyDescriptor>> {
        if pool().message(full_name).is_none() {
            return Err(PyKeyError::new_err(format!(
                "Couldn't find message {full_name}"
            )));
        }
        descriptor_object(py, full_name)
    }

    #[pyo3(name = "FindEnumTypeByName")]
    fn find_enum_type_by_name(
        &self,
        py: Python<'_>,
        full_name: &str,
    ) -> PyResult<Py<PyEnumDescriptor>> {
        if pool().enumeration(full_name).is_none() {
            return Err(PyKeyError::new_err(format!(
                "Couldn't find enum {full_name}"
            )));
        }
        enum_descriptor_object(py, full_name)
    }

    #[pyo3(name = "FindFieldByName")]
    fn find_field_by_name(&self, py: Python<'_>, full_name: &str) -> PyResult<Py<PyAny>> {
        let (message, index) = member_of(full_name, |m, name| m.by_name.get(name).copied())
            .ok_or_else(|| PyKeyError::new_err(format!("Couldn't find field {full_name}")))?;
        field_descriptor_object(py, &message, index)
    }

    #[pyo3(name = "FindOneofByName")]
    fn find_oneof_by_name(
        &self,
        py: Python<'_>,
        full_name: &str,
    ) -> PyResult<Py<PyOneofDescriptor>> {
        let (message, index) =
            member_of(full_name, |m, name| m.oneof_by_name.get(name).copied())
                .ok_or_else(|| PyKeyError::new_err(format!("Couldn't find oneof {full_name}")))?;
        oneof_descriptor_object(py, &message, index)
    }

    #[pyo3(name = "FindServiceByName")]
    fn find_service_by_name(
        &self,
        py: Python<'_>,
        full_name: &str,
    ) -> PyResult<Py<PyServiceDescriptor>> {
        if pool().service(full_name).is_none() {
            return Err(PyKeyError::new_err(format!(
                "Couldn't find service {full_name}"
            )));
        }
        service_descriptor_object(py, full_name)
    }

    #[pyo3(name = "FindMethodByName")]
    fn find_method_by_name(
        &self,
        py: Python<'_>,
        full_name: &str,
    ) -> PyResult<Py<PyMethodDescriptor>> {
        let not_found = || PyKeyError::new_err(format!("Couldn't find method {full_name}"));
        let (scope, name) = full_name.rsplit_once('.').ok_or_else(not_found)?;
        let service = pool().service(scope).ok_or_else(not_found)?;
        let index = service
            .methods
            .iter()
            .position(|m| m.name == name)
            .ok_or_else(not_found)?;
        method_descriptor_object(py, service, index)
    }

    /// As upb's: the protos of Spark Connect declare no extensions.
    #[pyo3(name = "FindExtensionByName")]
    fn find_extension_by_name(&self, full_name: &str) -> PyResult<Py<PyAny>> {
        Err(PyKeyError::new_err(format!(
            "Couldn't find extension {full_name}"
        )))
    }

    #[pyo3(name = "FindExtensionByNumber")]
    fn find_extension_by_number(
        &self,
        message_descriptor: &Bound<'_, PyAny>,
        number: i64,
    ) -> PyResult<Py<PyAny>> {
        let _ = message_descriptor;
        Err(PyKeyError::new_err(format!(
            "Couldn't find Extension {number}"
        )))
    }

    #[pyo3(name = "FindAllExtensions")]
    fn find_all_extensions<'py>(
        &self,
        py: Python<'py>,
        message_descriptor: &Bound<'_, PyAny>,
    ) -> Bound<'py, PyList> {
        let _ = message_descriptor;
        PyList::empty(py)
    }

    #[pyo3(name = "FindFileContainingSymbol")]
    fn find_file_containing_symbol(
        &self,
        py: Python<'_>,
        symbol: &str,
    ) -> PyResult<Py<PyFileDescriptor>> {
        let file = file_of_symbol(symbol)
            .ok_or_else(|| PyKeyError::new_err(format!("Couldn't find symbol {symbol}")))?;
        file_descriptor_object(py, file)
    }

    fn __repr__(&self) -> &'static str {
        "<DescriptorPool of pyspark.sql.connect.proto>"
    }
}
