//! A protobuf runtime for the messages of Spark Connect, the messages of
//! `pyspark.sql.connect.proto` in place of those of the modules that protoc generates for the
//! `protobuf` package, which the drop-in does not need.
//!
//! The vendored modules of PySpark that build their own Spark Connect messages, such as
//! `pyspark.ml`, the client's `execute_command` and plan logging, the request hooks of a session,
//! `Column._expr` and the pipelines, run unchanged on it: the messages hold their fields in Rust
//! ([`message`]), read the descriptors of the protos that the proto crate compiles ([`pool`]),
//! and encode and decode the wire format here ([`codec`]), with the Python API of protobuf's
//! messages, containers, descriptors and descriptor pool. The messages of a plugin's protos that
//! import those of Spark Connect run on it too: the files of the modules that protoc generates of
//! them are added to its pool when protobuf's builder builds them ([`pool::register`]).

mod codec;
mod containers;
mod descriptors;
pub(crate) mod message;
pub(crate) mod pool;
mod scalar;
mod text;

use std::sync::Mutex;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;

pyo3::create_exception!(
    pyspark.sql.connect.proto,
    DecodeError,
    pyo3::exceptions::PyException,
    "The error of parsing a protocol message, as protobuf's `google.protobuf.message.DecodeError`."
);

/// The class of the error of a parse that fails, which `_proto_set_decode_error` set:
/// `pyspark.sql.connect.proto._rust.DecodeError`, a subclass of [`DecodeError`] and, where
/// protobuf imports, of protobuf's `google.protobuf.message.DecodeError`, which upb raises, so
/// that `except google.protobuf.message.DecodeError` catches it.
static DECODE_ERROR_CLASS: Mutex<Option<Py<PyType>>> = Mutex::new(None);

/// Set the class of the error of a parse that fails, a subclass of `ProtoDecodeError` (see
/// [`decode_error_of`]), in place of the one set before.
#[pyfunction]
fn _proto_set_decode_error(class: &Bound<'_, PyType>) -> PyResult<()> {
    if !class.is_subclass_of::<DecodeError>()? {
        return Err(PyTypeError::new_err(format!(
            "{} is not a subclass of ProtoDecodeError",
            class.str()?
        )));
    }
    let previous = DECODE_ERROR_CLASS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .replace(class.clone().unbind());
    // Released once the class is no longer locked.
    drop(previous);
    Ok(())
}

/// The error of a parse that fails, of upb's `message`: of the class that
/// `_proto_set_decode_error` set, else (before `pyspark.sql.connect.proto` sets it) of
/// [`DecodeError`].
pub(crate) fn decode_error_of(py: Python<'_>, message: String) -> PyErr {
    let class = DECODE_ERROR_CLASS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map(|class| class.clone_ref(py));
    match class {
        Some(class) => PyErr::from_type(class.into_bound(py), (message,)),
        None => DecodeError::new_err(message),
    }
}

/// Run `f`, one level of a recursion over a message, on a stack with room for it: deeply nested
/// plans recurse once per level, and a new segment of stack is taken when this one runs low.
pub(crate) fn grow<T>(f: impl FnOnce() -> T) -> T {
    stacker::maybe_grow(128 * 1024, 2 << 20, f)
}

/// A file as `_proto_files` gives it: its name and package, and the full names of its top-level
/// messages and enums.
type FileSummary = (String, String, Vec<String>, Vec<String>);

/// The value of a bool field of the Python value `value`, as the messages take it (see
/// `scalar`): a bool, a bool of numpy, or an int that a C long holds, true but of 0, with upb's
/// TypeError of another value (`'str' object cannot be interpreted as an integer`). The field's
/// name, which only the errors of an integer field name, is not needed.
pub(crate) fn bool_value(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    let value = scalar::Scalar::from_py(pool::Kind::Bool, "", "", value)?;
    Ok(value == scalar::Scalar::Bool(true))
}

fn summary_of(file: &pool::FileDesc) -> FileSummary {
    (
        file.name.clone(),
        file.package.clone(),
        file.messages.clone(),
        file.enums.clone(),
    )
}

/// The files of the descriptor set: the name and package of each, and the full names of its
/// top-level messages and enums.
#[pyfunction]
fn _proto_files() -> Vec<FileSummary> {
    pool::pool().files.iter().map(summary_of).collect()
}

/// Add the files `files`, serialized `FileDescriptorProto`s (protobuf's
/// `FileDescriptor.serialized_pb`), to the pool of the runtime, as [`pool::register`] adds them:
/// the name and package of each, and the full names of its top-level messages and enums, as
/// `_proto_files` gives those of the descriptor set. A `ValueError` of files that the runtime
/// does not hold adds none of them.
#[pyfunction]
fn _proto_register_files(files: &Bound<'_, PyAny>) -> PyResult<Vec<FileSummary>> {
    let files = files
        .try_iter()?
        .map(|file| message::bytes_of(&file?))
        .collect::<PyResult<Vec<_>>>()?;
    let registered = pool::register(&files).map_err(PyValueError::new_err)?;
    Ok(registered.into_iter().map(summary_of).collect())
}

/// The descriptor of the message type `full_name`.
#[pyfunction]
fn _proto_message_descriptor(
    py: Python<'_>,
    full_name: &str,
) -> PyResult<Py<descriptors::PyDescriptor>> {
    descriptors::descriptor_object(py, full_name)
}

/// The descriptor of the enum type `full_name`.
#[pyfunction]
fn _proto_enum_descriptor(
    py: Python<'_>,
    full_name: &str,
) -> PyResult<Py<descriptors::PyEnumDescriptor>> {
    descriptors::enum_descriptor_object(py, full_name)
}

/// The descriptor of the file `name`, such as `spark/connect/base.proto`, the `DESCRIPTOR` of its
/// module.
#[pyfunction]
fn _proto_file_descriptor(
    py: Python<'_>,
    name: &str,
) -> PyResult<Py<descriptors::PyFileDescriptor>> {
    descriptors::file_descriptor_object(py, name)
}

/// The text format of `message`, as protobuf's `text_format.MessageToString` writes it.
#[pyfunction]
#[pyo3(signature = (message, as_one_line=false))]
fn _proto_message_to_string(
    message: &Bound<'_, message::PyMessage>,
    as_one_line: bool,
) -> PyResult<String> {
    text::to_text(message.py(), message, text::Style::TextFormat, as_one_line)
}

/// The encoded fields that `message` read that its type has no field for, in the order in which it
/// read them, as upb keeps them: the fields of protobuf's `unknown_fields.UnknownFieldSet` of the
/// message (see `pyspark.sql.connect.proto._rust`).
#[pyfunction]
fn _proto_unknown_fields<'py>(
    message: &Bound<'py, message::PyMessage>,
) -> Bound<'py, pyo3::types::PyBytes> {
    pyo3::types::PyBytes::new(message.py(), &message.borrow().unknown)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("ProtoMessage", py.get_type::<message::PyMessage>())?;
    m.add(
        "ProtoRepeatedScalarContainer",
        py.get_type::<containers::RepeatedScalar>(),
    )?;
    m.add(
        "ProtoRepeatedCompositeContainer",
        py.get_type::<containers::RepeatedComposite>(),
    )?;
    m.add(
        "ProtoMapContainer",
        py.get_type::<containers::MapContainer>(),
    )?;
    m.add(
        "ProtoMessageMapContainer",
        py.get_type::<containers::MessageMapContainer>(),
    )?;
    m.add("ProtoMapIterator", py.get_type::<containers::MapIterator>())?;
    m.add(
        "ProtoDescriptor",
        py.get_type::<descriptors::PyDescriptor>(),
    )?;
    m.add(
        "ProtoFieldDescriptor",
        py.get_type::<descriptors::PyFieldDescriptor>(),
    )?;
    m.add(
        "ProtoOneofDescriptor",
        py.get_type::<descriptors::PyOneofDescriptor>(),
    )?;
    m.add(
        "ProtoEnumDescriptor",
        py.get_type::<descriptors::PyEnumDescriptor>(),
    )?;
    m.add(
        "ProtoEnumValueDescriptor",
        py.get_type::<descriptors::PyEnumValueDescriptor>(),
    )?;
    m.add(
        "ProtoFileDescriptor",
        py.get_type::<descriptors::PyFileDescriptor>(),
    )?;
    m.add(
        "ProtoServiceDescriptor",
        py.get_type::<descriptors::PyServiceDescriptor>(),
    )?;
    m.add(
        "ProtoMethodDescriptor",
        py.get_type::<descriptors::PyMethodDescriptor>(),
    )?;
    m.add(
        "ProtoDescriptorPool",
        py.get_type::<descriptors::PyDescriptorPool>(),
    )?;
    m.add("ProtoDecodeError", py.get_type::<DecodeError>())?;
    m.add_function(wrap_pyfunction!(_proto_set_decode_error, m)?)?;
    m.add_function(wrap_pyfunction!(_proto_files, m)?)?;
    m.add_function(wrap_pyfunction!(_proto_register_files, m)?)?;
    m.add_function(wrap_pyfunction!(_proto_message_descriptor, m)?)?;
    m.add_function(wrap_pyfunction!(_proto_enum_descriptor, m)?)?;
    m.add_function(wrap_pyfunction!(_proto_file_descriptor, m)?)?;
    m.add_function(wrap_pyfunction!(message::_proto_register_classes, m)?)?;
    m.add_function(wrap_pyfunction!(_proto_message_to_string, m)?)?;
    m.add_function(wrap_pyfunction!(_proto_unknown_fields, m)?)?;
    Ok(())
}
