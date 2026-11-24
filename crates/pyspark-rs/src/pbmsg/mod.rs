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
//! import those of Spark Connect (of proto2, proto3 or editions) run on it too: the files of the
//! modules that protoc generates of them are added to its pool when protobuf's metaclass builds
//! their classes ([`pool::register`]).

mod arena;
mod codec;
mod containers;
mod descriptors;
pub(crate) mod message;
mod native;
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

pyo3::create_exception!(
    pyspark.sql.connect.proto,
    EncodeError,
    pyo3::exceptions::PyException,
    "The error of serializing a protocol message, as protobuf's `google.protobuf.message.EncodeError`."
);

/// The class of the error of a serialization of a message that lacks a required field, which
/// `_proto_set_encode_error` set: `protobuf`'s `google.protobuf.message.EncodeError`, which upb
/// raises, where `protobuf` imports, else [`EncodeError`].
static ENCODE_ERROR_CLASS: Mutex<Option<Py<PyType>>> = Mutex::new(None);

/// Set the class of the error of a serialization that fails, an exception class.
#[pyfunction]
fn _proto_set_encode_error(class: &Bound<'_, PyType>) -> PyResult<()> {
    if !class.is_subclass_of::<pyo3::exceptions::PyException>()? {
        return Err(PyTypeError::new_err(format!(
            "{} is not an exception class",
            class.str()?
        )));
    }
    let previous = ENCODE_ERROR_CLASS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .replace(class.clone().unbind());
    drop(previous);
    Ok(())
}

/// The error of a serialization that fails, of upb's `message`, of the class that
/// `_proto_set_encode_error` set, else of [`EncodeError`].
pub(crate) fn encode_error_of(py: Python<'_>, message: String) -> PyErr {
    let class = ENCODE_ERROR_CLASS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map(|class| class.clone_ref(py));
    match class {
        Some(class) => PyErr::from_type(class.into_bound(py), (message,)),
        None => EncodeError::new_err(message),
    }
}

/// The class of the error of a parse that fails, which `_proto_set_decode_error` set:
/// `protobuf`'s `google.protobuf.message.DecodeError`, which upb raises, where `protobuf`
/// imports, else `pyspark.sql.connect.proto._rust.DecodeError`, a subclass of [`DecodeError`].
static DECODE_ERROR_CLASS: Mutex<Option<Py<PyType>>> = Mutex::new(None);

/// Set the class of the error of a parse that fails, an exception class (see
/// [`decode_error_of`]), in place of the one set before.
#[pyfunction]
fn _proto_set_decode_error(class: &Bound<'_, PyType>) -> PyResult<()> {
    if !class.is_subclass_of::<pyo3::exceptions::PyException>()? {
        return Err(PyTypeError::new_err(format!(
            "{} is not an exception class",
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

/// protobuf's `google.protobuf.message._InternalConstructMessage`, which the pickle of a message
/// of a nested type names where `protobuf` imports, as protobuf's `Message.__reduce__` names it,
/// which `_proto_set_internal_construct` set; none without `protobuf`, so that such a pickle names
/// the class of the message by its qualified name.
static INTERNAL_CONSTRUCT: Mutex<Option<Py<PyAny>>> = Mutex::new(None);

/// Set the function of the pickles of the messages of nested types (see [`INTERNAL_CONSTRUCT`]),
/// or none.
#[pyfunction]
fn _proto_set_internal_construct(function: Option<Py<PyAny>>) {
    let previous = std::mem::replace(
        &mut *INTERNAL_CONSTRUCT.lock().unwrap_or_else(|e| e.into_inner()),
        function,
    );
    // Released once the function is no longer locked.
    drop(previous);
}

/// The function of the pickles of the messages of nested types, if any (see
/// [`INTERNAL_CONSTRUCT`]).
pub(crate) fn internal_construct(py: Python<'_>) -> Option<Bound<'_, PyAny>> {
    INTERNAL_CONSTRUCT
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map(|function| function.bind(py).clone())
}

/// Set the limit of the levels of messages that a parse reads below the message that it parses, as
/// the parsers of the `protobuf` of the process have it (see `codec::LIMIT`).
#[pyfunction]
fn _proto_set_depth_limit(levels: u32) {
    codec::set_depth_limit(levels);
}

/// Set the function of Python that gives the limit of the parsers of `protobuf`, which a parse
/// that reaches its limit calls (see `codec::DEPTH_PROBE`), or none.
#[pyfunction]
fn _proto_set_depth_probe(probe: Option<Py<PyAny>>) {
    codec::set_depth_probe(probe);
}

/// Run `f`, one level of a recursion over a message, on a stack with room for it: deeply nested
/// plans recurse once per level, and a new segment of stack is taken when this one runs low.
/// Inlined, so that each recursion (the encoder's, of a level of a message) takes `f` in its own
/// code, of the unit of code of its caller.
#[inline]
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

/// A message of `class`, a class of the runtime, of the bytes `data`, as its `FromString` gives
/// it, but of any depth, where upb's parse stops at 100 levels: the drop-in's own parse of the
/// bytes of a message that PySpark builds in Python of any depth (the expression of a Column, a
/// schema, the deep plans of `parse_deep`), whose bytes the core encodes.
pub(crate) fn parse_deep<'py>(
    class: &Bound<'py, PyAny>,
    data: &[u8],
) -> PyResult<Bound<'py, PyAny>> {
    // A message class of the runtime makes its message as its `FromString` does, without the
    // call of the class and of its `__init__`, which takes no keyword argument here; any other
    // callable is called.
    let message = match message::new_message_of_class(class)? {
        Some(message) => message.into_any(),
        None => class.call0()?,
    };
    codec::merge_unlimited(class.py(), message.cast::<message::PyMessage>()?, data)?;
    Ok(message)
}

/// The attribute `name` of the module `pyspark.sql.connect.proto`, of the module imported once,
/// as PySpark's modules hold the module that they imported (`from pyspark.sql.connect import
/// proto`), so that a class of it that a test patches is the one given.
pub(crate) fn proto_attr<'py>(
    py: Python<'py>,
    name: &Bound<'py, pyo3::types::PyString>,
) -> PyResult<Bound<'py, PyAny>> {
    static PROTO: pyo3::sync::PyOnceLock<Py<PyModule>> = pyo3::sync::PyOnceLock::new();
    let module = PROTO.get_or_try_init(py, || -> PyResult<_> {
        Ok(py.import("pyspark.sql.connect.proto")?.unbind())
    })?;
    module.bind(py).getattr(name)
}

/// [`parse_deep`] of Python: a message of `class` of the bytes `data`, of any depth.
#[pyfunction]
fn _proto_parse_deep<'py>(
    class: &Bound<'py, PyAny>,
    data: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    parse_deep(class, &message::bytes_of(data)?)
}

/// Raise the `RecursionError` that the core raises of a response of the message type `name` of the
/// bytes `data` whose kept messages nest past `spark_connect_proto::MAX_KEPT_DEPTH` levels, within
/// the limit of the parses (see `spark_connect_proto::ClientCodec`), for the responses that
/// `SparkSession.client` parses; of any other bytes, nothing, as its parse checks the limit.
#[pyfunction]
fn _proto_check_response_depth(name: &str, data: &[u8]) -> PyResult<()> {
    match spark_connect_proto::response_depth_of(name, data) {
        Err(too_deep) if too_deep.is_kept() => Err(pyo3::exceptions::PyRecursionError::new_err(
            too_deep.to_string(),
        )),
        _ => Ok(()),
    }
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

/// The unknown fields of `message` as protobuf's pure-Python messages hold them
/// (`_unknown_fields`): a list of the bytes of the tag and the bytes of the value of each, in the
/// order in which the message read them (see [`codec::unknown_field_pairs`]).
#[pyfunction]
fn _proto_unknown_field_pairs<'py>(
    message: &Bound<'py, message::PyMessage>,
) -> PyResult<Bound<'py, pyo3::types::PyList>> {
    let py = message.py();
    // A copy of the bytes, of no borrow of the message held while the list is made, whose
    // allocations may run the Python code of a collection of the garbage collector.
    let unknown = message.borrow().unknown.clone();
    let pairs = codec::unknown_field_pairs(&unknown)?;
    let list = pyo3::types::PyList::empty(py);
    for (tag, value) in pairs {
        list.append((
            pyo3::types::PyBytes::new(py, tag),
            pyo3::types::PyBytes::new(py, value),
        ))?;
    }
    Ok(list)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    spark_connect_proto::set_depth_probe(codec::response_depth_probe);
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
        "ProtoScalarMapContainer",
        py.get_type::<containers::ScalarMapContainer>(),
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
    m.add("ProtoEncodeError", py.get_type::<EncodeError>())?;
    m.add_function(wrap_pyfunction!(_proto_set_decode_error, m)?)?;
    m.add_function(wrap_pyfunction!(_proto_set_encode_error, m)?)?;
    m.add_function(wrap_pyfunction!(_proto_set_internal_construct, m)?)?;
    m.add_function(wrap_pyfunction!(_proto_set_depth_limit, m)?)?;
    m.add_function(wrap_pyfunction!(_proto_set_depth_probe, m)?)?;
    m.add_function(wrap_pyfunction!(_proto_unknown_field_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(_proto_files, m)?)?;
    m.add_function(wrap_pyfunction!(_proto_register_files, m)?)?;
    m.add_function(wrap_pyfunction!(_proto_message_descriptor, m)?)?;
    m.add_function(wrap_pyfunction!(_proto_enum_descriptor, m)?)?;
    m.add_function(wrap_pyfunction!(_proto_file_descriptor, m)?)?;
    m.add_function(wrap_pyfunction!(message::_proto_register_classes, m)?)?;
    m.add_function(wrap_pyfunction!(native::_proto_whole_limit, m)?)?;
    m.add_function(wrap_pyfunction!(native::_proto_slab_messages, m)?)?;
    m.add_function(wrap_pyfunction!(_proto_message_to_string, m)?)?;
    m.add_function(wrap_pyfunction!(_proto_parse_deep, m)?)?;
    m.add_function(wrap_pyfunction!(_proto_check_response_depth, m)?)?;
    m.add_function(wrap_pyfunction!(_proto_unknown_fields, m)?)?;
    Ok(())
}
