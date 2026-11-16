//! A protobuf runtime for the messages of Spark Connect, which `pyspark.sql.connect.proto` uses
//! in place of the `protobuf` package when that is not installed (or is too old for the
//! generated `*_pb2` modules, or `PYSPARK_CONNECT_PROTO_IMPLEMENTATION=rust` asks for it).
//!
//! The vendored modules of PySpark that build their own Spark Connect messages, such as
//! `pyspark.ml`, the client's `execute_command` and plan logging, the request hooks of a session,
//! `Column._expr` and the pipelines, then run unchanged without `protobuf`: the messages hold
//! their fields in Rust ([`message`]), read the descriptors of the protos that the proto crate
//! compiles ([`pool`]), and encode and decode the wire format here ([`codec`]), with the Python
//! API of protobuf's messages, containers and descriptors.

mod codec;
mod containers;
mod descriptors;
pub(crate) mod message;
pub(crate) mod pool;
mod scalar;
mod text;

use pyo3::prelude::*;

pyo3::create_exception!(
    pyspark.sql.connect.proto,
    DecodeError,
    pyo3::exceptions::PyException,
    "The error of parsing a protocol message, as protobuf's `google.protobuf.message.DecodeError`."
);

/// Run `f`, one level of a recursion over a message, on a stack with room for it: deeply nested
/// plans recurse once per level, and a new segment of stack is taken when this one runs low.
pub(crate) fn grow<T>(f: impl FnOnce() -> T) -> T {
    stacker::maybe_grow(128 * 1024, 2 << 20, f)
}

/// The files of the descriptor set: the name and package of each, and the full names of its
/// top-level messages and enums.
#[pyfunction]
fn _proto_files() -> Vec<(String, String, Vec<String>, Vec<String>)> {
    pool::pool()
        .files
        .iter()
        .map(|f| {
            (
                f.name.clone(),
                f.package.clone(),
                f.messages.clone(),
                f.enums.clone(),
            )
        })
        .collect()
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

/// The text format of `message`, as protobuf's `text_format.MessageToString` writes it.
#[pyfunction]
#[pyo3(signature = (message, as_one_line=false))]
fn _proto_message_to_string(
    message: &Bound<'_, message::PyMessage>,
    as_one_line: bool,
) -> PyResult<String> {
    text::to_text(message.py(), message, text::Style::TextFormat, as_one_line)
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
    m.add("ProtoDecodeError", py.get_type::<DecodeError>())?;
    m.add_function(wrap_pyfunction!(_proto_files, m)?)?;
    m.add_function(wrap_pyfunction!(_proto_message_descriptor, m)?)?;
    m.add_function(wrap_pyfunction!(_proto_enum_descriptor, m)?)?;
    m.add_function(wrap_pyfunction!(message::_proto_register_classes, m)?)?;
    m.add_function(wrap_pyfunction!(_proto_message_to_string, m)?)?;
    Ok(())
}
