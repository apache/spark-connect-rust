//! `Message`, the base class of the message classes of the runtime, which holds the fields of a
//! message in Rust and gives them to Python as protobuf's Python messages do:
//!
//! - A field is an attribute: a scalar gives its value or its default; a message gives the
//!   message it holds, or an unset one that the field then holds once one of its fields is set
//!   (as `command.ml_command.fit.CopyFrom(fit)` sets `ml_command` and `fit`); a repeated field
//!   gives its container, and a map field its map.
//! - Setting one member of a oneof clears the others.
//! - Only a scalar field may be assigned; a message, repeated or map field changes through its
//!   value (`CopyFrom`, `append`, `add`, `msg.map[key] = value`).
//!
//! The classes of the messages of Spark Connect subclass it (see `pyspark.sql.connect.proto`'s
//! `_rust`), each with the `DESCRIPTOR` of its message, and the runtime makes their instances by
//! calling the class that `_proto_register_classes` gave it for the full name of the message.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use pyo3::basic::CompareOp;
use pyo3::exceptions::{
    PyAttributeError, PyKeyError, PyNotImplementedError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::pyclass_init::PyClassInitializer;
use pyo3::types::{
    PyByteArray, PyBytes, PyDict, PyMemoryView, PyString, PyTuple, PyType, PyWeakrefMethods,
    PyWeakrefReference,
};
use pyo3::IntoPyObjectExt;

use super::containers::{MapContainer, MapEntries, MapValue, RepeatedComposite, RepeatedScalar};
use super::descriptors::{field_descriptor_object, PyDescriptor};
use super::pool::{pool, FieldDesc, Kind, MessageDesc};
use super::scalar::Scalar;
use super::{codec, grow, text};
use crate::interrupt::ConversionBudget;

/// The maximum depth of upb's `DiscardUnknownFields`, which passes it to
/// `upb_Message_DiscardUnknown`.
const DISCARD_UNKNOWN_DEPTH: u32 = 64;

/// The value of a set field.
pub(crate) enum Slot {
    Scalar(Scalar),
    Message(Py<PyMessage>),
    Repeated(Py<RepeatedScalar>),
    Composite(Py<RepeatedComposite>),
    Map(Py<MapContainer>),
}

/// The set fields of a message, each with the index of its field, in the order of the numbers of
/// the fields, in which the message writes, prints and lists them. A message holds the slots of
/// its set fields only, as a message of Spark Connect sets few of its fields (a literal, one of
/// its 25), so that a message of many messages takes the room of those it sets; one set field is
/// held in place, without an allocation of its own.
#[derive(Default)]
pub(crate) enum Slots {
    #[default]
    Empty,
    One((usize, Slot)),
    Many(Vec<(usize, Slot)>),
}

impl Slots {
    /// The slots of `entries`, which are in the order of the numbers of their fields.
    pub(crate) fn from_vec(mut entries: Vec<(usize, Slot)>) -> Slots {
        match entries.len() {
            0 => Slots::Empty,
            1 => Slots::One(entries.pop().expect("one entry")),
            _ => Slots::Many(entries),
        }
    }

    fn as_slice(&self) -> &[(usize, Slot)] {
        match self {
            Slots::Empty => &[],
            Slots::One(entry) => std::slice::from_ref(entry),
            Slots::Many(entries) => entries,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.as_slice().len()
    }

    pub(crate) fn get(&self, index: usize) -> Option<&Slot> {
        self.as_slice()
            .iter()
            .find(|(i, _)| *i == index)
            .map(|(_, slot)| slot)
    }

    pub(crate) fn contains(&self, index: usize) -> bool {
        self.as_slice().iter().any(|(i, _)| *i == index)
    }

    fn get_mut(&mut self, index: usize) -> Option<&mut Slot> {
        match self {
            Slots::Empty => None,
            Slots::One((i, slot)) => (*i == index).then_some(slot),
            Slots::Many(entries) => entries
                .iter_mut()
                .find(|(i, _)| *i == index)
                .map(|(_, slot)| slot),
        }
    }

    /// Set the slot of the field `index` of `desc`, and give back the slot that it replaces.
    pub(crate) fn set(&mut self, desc: &MessageDesc, index: usize, slot: Slot) -> Option<Slot> {
        if let Some(old) = self.get_mut(index) {
            return Some(std::mem::replace(old, slot));
        }
        if let Slots::Empty = self {
            *self = Slots::One((index, slot));
            return None;
        }
        let mut entries = std::mem::take(self).into_vec();
        let number = desc.fields[index].number;
        let at = entries
            .iter()
            .rposition(|(i, _)| desc.fields[*i].number < number)
            .map_or(0, |at| at + 1);
        entries.insert(at, (index, slot));
        *self = Slots::Many(entries);
        None
    }

    /// Set the slot of the field `index`, whose number is greater than those of the set fields.
    fn push_last(&mut self, index: usize, slot: Slot) {
        *self = match std::mem::take(self) {
            Slots::Empty => Slots::One((index, slot)),
            Slots::One(first) => Slots::Many(vec![first, (index, slot)]),
            Slots::Many(mut entries) => {
                entries.push((index, slot));
                Slots::Many(entries)
            }
        };
    }

    /// Clear the field `index`, and give back its slot.
    pub(crate) fn remove(&mut self, index: usize) -> Option<Slot> {
        let at = self.as_slice().iter().position(|(i, _)| *i == index)?;
        let mut entries = std::mem::take(self).into_vec();
        let (_, slot) = entries.remove(at);
        *self = Slots::from_vec(entries);
        Some(slot)
    }

    /// Keep the slots that `keep` takes of the index of their field.
    pub(crate) fn retain(&mut self, keep: impl Fn(usize) -> bool) {
        match self {
            Slots::One((i, _)) if !keep(*i) => *self = Slots::Empty,
            Slots::Many(entries) => entries.retain(|(i, _)| keep(*i)),
            _ => {}
        }
    }

    /// The set fields, by the index of their field, in the order of their numbers.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (usize, &Slot)> {
        self.as_slice().iter().map(|(i, slot)| (*i, slot))
    }

    pub(crate) fn into_vec(self) -> Vec<(usize, Slot)> {
        match self {
            Slots::Empty => Vec::new(),
            Slots::One(entry) => vec![entry],
            Slots::Many(entries) => entries,
        }
    }
}

/// The message, and its field, that an unset message of `__getattr__` sets itself in when one of
/// its fields is set.
struct Parent {
    message: Py<PyWeakrefReference>,
    field: usize,
}

/// A weak reference to `message`, which a container or an unset child holds.
pub(crate) fn weak(message: &Bound<'_, PyMessage>) -> PyResult<Py<PyWeakrefReference>> {
    Ok(PyWeakrefReference::new(message.as_any())?.unbind())
}

#[pyclass(
    subclass,
    weakref,
    module = "pyspark.sql.connect.proto",
    name = "Message"
)]
pub struct PyMessage {
    pub(crate) desc: Arc<MessageDesc>,
    /// The set fields.
    pub(crate) slots: Slots,
    /// The unset children that `__getattr__` gave, each with the index of its field.
    lazy: Vec<(usize, Py<PyMessage>)>,
    /// The parent of an unset child.
    parent: Option<Parent>,
    /// The encoded fields that the message type has no field for, as the message read them.
    pub(crate) unknown: Vec<u8>,
}

/// A class that `_proto_register_classes` set, with its message type and, of a class whose
/// `DESCRIPTOR` is `protobuf`'s (as the classes of the runtime have where `protobuf` imports, as
/// upb's do), `protobuf`'s descriptors of its fields, in the order of the fields of the message
/// type, which `ListFields` gives: read of the `DESCRIPTOR` once `ListFields` needs them (see
/// `registered_fields`), not as the class registers, as many classes register at once.
struct Registered {
    class: Py<PyType>,
    desc: Arc<MessageDesc>,
    fields: Arc<OnceLock<ProtobufFields>>,
}

/// `protobuf`'s descriptors of the fields of a registered class, of its `DESCRIPTOR` when it is
/// not the runtime's.
type ProtobufFields = Option<Arc<Vec<Py<PyAny>>>>;

/// The classes that `_proto_register_classes` set, by the full name of their message type, and
/// their message types by the address of the class, which `__new__` reads.
#[derive(Default)]
struct Registry {
    by_name: HashMap<String, Registered>,
    by_type: HashMap<usize, Arc<MessageDesc>>,
}

fn classes() -> &'static Mutex<Registry> {
    static CLASSES: OnceLock<Mutex<Registry>> = OnceLock::new();
    CLASSES.get_or_init(|| Mutex::new(Registry::default()))
}

/// Set the class of each message, by full name, that the runtime makes the instances of: a
/// subclass of `Message` of the message type of the name, whose `DESCRIPTOR` is the runtime's
/// descriptor of the type or `protobuf`'s, of the same fields (which `ListFields` reads, and
/// checks, once it needs them).
#[pyfunction]
pub(crate) fn _proto_register_classes(classes_by_name: &Bound<'_, PyDict>) -> PyResult<()> {
    let mut found = Vec::new();
    for (name, class) in classes_by_name.iter() {
        let class = class.cast_into::<PyType>()?;
        if !class.is_subclass_of::<PyMessage>()? {
            return Err(PyTypeError::new_err(format!(
                "{} is not a class of protocol messages",
                class.str()?
            )));
        }
        let name = name.extract::<String>()?;
        let desc = pool()
            .message(&name)
            .ok_or_else(|| PyKeyError::new_err(name.clone()))?
            .clone();
        found.push((name, class.unbind(), desc));
    }
    let mut registered = classes().lock().unwrap_or_else(|e| e.into_inner());
    for (name, class, desc) in found {
        registered
            .by_type
            .insert(class.as_ptr() as usize, desc.clone());
        registered.by_name.insert(
            name,
            Registered {
                class,
                desc,
                fields: Arc::default(),
            },
        );
    }
    Ok(())
}

/// `protobuf`'s descriptors of the fields of `desc`, of the `DESCRIPTOR` of `class` when it is
/// not the runtime's: those of the same names, in the same order.
fn protobuf_fields_of(class: &Bound<'_, PyType>, desc: &MessageDesc) -> PyResult<ProtobufFields> {
    let descriptor = class.getattr("DESCRIPTOR")?;
    if descriptor.cast::<PyDescriptor>().is_ok() {
        return Ok(None);
    }
    let fields = descriptor
        .getattr("fields")?
        .try_iter()?
        .map(|field| field.map(Bound::unbind))
        .collect::<PyResult<Vec<_>>>()?;
    let py = class.py();
    let same = fields.len() == desc.fields.len()
        && fields.iter().zip(&desc.fields).all(|(field, ours)| {
            field
                .bind(py)
                .getattr("name")
                .and_then(|name| name.extract::<String>())
                .is_ok_and(|name| name == ours.name)
        });
    if !same {
        return Err(PyTypeError::new_err(format!(
            "the DESCRIPTOR of {} has other fields than {}",
            class.str()?,
            desc.full_name
        )));
    }
    Ok(Some(Arc::new(fields)))
}

/// `protobuf`'s descriptors of the fields of the message type `full_name`, of its registered
/// class, if they are its `DESCRIPTOR`'s: read of the class once (with no lock held, as the read
/// runs Python).
fn registered_fields(py: Python<'_>, full_name: &str) -> PyResult<ProtobufFields> {
    let (class, desc, cell) = {
        let registered = classes().lock().unwrap_or_else(|e| e.into_inner());
        let Some(r) = registered.by_name.get(full_name) else {
            return Ok(None);
        };
        if let Some(fields) = r.fields.get() {
            return Ok(fields.clone());
        }
        (r.class.clone_ref(py), r.desc.clone(), r.fields.clone())
    };
    let fields = protobuf_fields_of(class.bind(py), &desc)?;
    Ok(cell.get_or_init(|| fields).clone())
}

/// The class of the message type `full_name` that `_proto_register_classes` set, if any.
pub(crate) fn registered_class(py: Python<'_>, full_name: &str) -> Option<Py<PyType>> {
    classes()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .by_name
        .get(full_name)
        .map(|r| r.class.clone_ref(py))
}

/// A maker of the messages of a message type: of its registered class, of the message type of
/// that class's `DESCRIPTOR`, or of the message type of the pool when it has no class.
pub(crate) struct MessageClass {
    class: Option<Py<PyType>>,
    desc: Arc<MessageDesc>,
}

impl MessageClass {
    pub(crate) fn clone_ref(&self, py: Python<'_>) -> MessageClass {
        MessageClass {
            class: self.class.as_ref().map(|c| c.clone_ref(py)),
            desc: self.desc.clone(),
        }
    }

    pub(crate) fn of(py: Python<'_>, full_name: &str) -> PyResult<MessageClass> {
        if let Some(r) = classes()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .by_name
            .get(full_name)
        {
            return Ok(MessageClass {
                class: Some(r.class.clone_ref(py)),
                desc: r.desc.clone(),
            });
        }
        let desc = pool()
            .message(full_name)
            .ok_or_else(|| PyTypeError::new_err(format!("no message type named {full_name}")))?
            .clone();
        Ok(MessageClass { class: None, desc })
    }

    /// A new, empty message, made as the `tp_new` of the class makes one, without the call of
    /// the class (whose `__init__` is `object`'s) or the read of its `DESCRIPTOR`.
    ///
    /// The cyclic garbage collector does not track it. A registered class has `__slots__ = ()`,
    /// so its object holds no Python reference that the collector can see (the messages and
    /// containers that it holds are Rust's): tracking it finds no cycle, while the collections
    /// that the tracked objects of a message of many messages start, every 700 of them, each read
    /// all of them, which took more than half of a parse. (A message that Python code makes by
    /// calling its class is tracked, as Python makes it.)
    pub(crate) fn new_message<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyMessage>> {
        let message = PyMessage::empty(self.desc.clone());
        let Some(class) = &self.class else {
            return Bound::new(py, message);
        };
        // SAFETY: `class` is a subclass of `PyMessage` (see `_proto_register_classes`), of which
        // PyO3's `tp_new_impl`, the `tp_new` of a `#[new]`, makes an object of an initializer.
        // The object is untracked once, before any Python code sees it; its deallocation
        // untracks it only if it is tracked.
        unsafe {
            let ptr = pyo3::impl_::pymethods::tp_new_impl::<
                PyClassInitializer<PyMessage>,
                PyMessage,
            >(py, message.into(), class.bind(py).as_type_ptr())?;
            if pyo3::ffi::PyObject_GC_IsTracked(ptr) != 0 {
                pyo3::ffi::PyObject_GC_UnTrack(ptr.cast());
            }
            Ok(Bound::from_owned_ptr(py, ptr).cast_into_unchecked())
        }
    }
}

/// Make the garbage collector track `message`, a message that [`MessageClass::new_message`] made,
/// which it does not track, as Python tracks one of a call of its class (see `FromString`).
fn track(message: &Bound<'_, PyMessage>) {
    // SAFETY: `message` is a live object of a type of the garbage collector, made in full; an
    // object that is tracked is not tracked again.
    unsafe {
        let ptr = message.as_ptr();
        if pyo3::ffi::PyObject_GC_IsTracked(ptr) == 0 {
            pyo3::ffi::PyObject_GC_Track(ptr.cast());
        }
    }
}

/// A new, empty message of the message type `full_name`, of its registered class.
pub(crate) fn new_instance<'py>(
    py: Python<'py>,
    full_name: &str,
) -> PyResult<Bound<'py, PyMessage>> {
    MessageClass::of(py, full_name)?.new_message(py)
}

/// A new, empty message of the message type `desc`, of its registered class.
fn new_instance_of<'py>(py: Python<'py>, desc: &MessageDesc) -> PyResult<Bound<'py, PyMessage>> {
    new_instance(py, &desc.full_name)
}

/// The message type of the class `cls`: of a registered class, the one that the registry holds,
/// else (a class that subclasses `Message` itself) of its `DESCRIPTOR`, the runtime's descriptor
/// of a message type or `protobuf`'s of one of the pool of the runtime.
fn message_type_of(cls: &Bound<'_, PyType>) -> PyResult<Arc<MessageDesc>> {
    let registered = classes()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .by_type
        .get(&(cls.as_ptr() as usize))
        .cloned();
    match registered {
        Some(desc) => Ok(desc),
        None => class_descriptor(cls),
    }
}

/// The message type of the class `cls`, of its `DESCRIPTOR`.
fn class_descriptor(cls: &Bound<'_, PyType>) -> PyResult<Arc<MessageDesc>> {
    let abstract_base =
        || PyTypeError::new_err("Message is the abstract base of the message classes");
    let descriptor = cls.getattr("DESCRIPTOR").map_err(|_| abstract_base())?;
    if let Ok(descriptor) = descriptor.cast::<PyDescriptor>() {
        return Ok(descriptor.get().desc.clone());
    }
    let full_name = descriptor
        .getattr("full_name")
        .and_then(|name| name.extract::<String>())
        .map_err(|_| abstract_base())?;
    pool().message(&full_name).cloned().ok_or_else(|| {
        PyTypeError::new_err(format!(
            "the DESCRIPTOR of a message class is of {full_name}, which the runtime does not hold"
        ))
    })
}

/// The bytes of `value`: bytes, a bytearray or a memoryview, as protobuf's parsers take.
pub(crate) fn bytes_of(value: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(b) = value.cast::<PyBytes>() {
        return Ok(b.as_bytes().to_vec());
    }
    if let Ok(b) = value.cast::<PyByteArray>() {
        return Ok(b.to_vec());
    }
    if value.cast::<PyMemoryView>().is_ok() {
        let bytes = value.py().get_type::<PyBytes>().call1((value,))?;
        return Ok(bytes.cast::<PyBytes>()?.as_bytes().to_vec());
    }
    // upb's error of any other value: "expected bytes, str found".
    Err(super::scalar::Scalar::from_py(Kind::Bytes, "", "", value)
        .err()
        .unwrap_or_else(|| PyTypeError::new_err("expected bytes")))
}

/// The name of a field (or a oneof) that `HasField`, `ClearField` and `WhichOneof` take, as upb
/// reads it: the UTF-8 of a str, or the bytes of a bytes object, a name if they are UTF-8. Another
/// object, or a str of no UTF-8 (of a lone surrogate), is upb's error, of `str()` of the object.
struct FieldName<'py>(Bound<'py, PyBytes>);

impl<'py> FieldName<'py> {
    fn of(name: &Bound<'py, PyAny>) -> PyResult<FieldName<'py>> {
        if let Ok(text) = name.cast::<PyString>() {
            return text
                .encode_utf8()
                .map(FieldName)
                .map_err(|_| not_a_field_name(name));
        }
        match name.cast::<PyBytes>() {
            Ok(bytes) => Ok(FieldName(bytes.clone())),
            Err(_) => Err(not_a_field_name(name)),
        }
    }

    /// The name to look up: none of a field, of bytes that are not UTF-8.
    fn text(&self) -> &str {
        std::str::from_utf8(self.0.as_bytes()).unwrap_or("")
    }

    /// The name as upb's errors print it: up to its first NUL, of U+FFFD for bytes that are not
    /// UTF-8.
    fn display(&self) -> String {
        let bytes = self.0.as_bytes();
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        String::from_utf8_lossy(&bytes[..end]).into_owned()
    }
}

fn not_a_field_name(name: &Bound<'_, PyAny>) -> PyErr {
    let py = name.py();
    let message = name.str().and_then(|text| {
        PyString::new(py, "Expected a field name, but got non-string argument ")
            .add(text)?
            .add(".")
    });
    match message {
        Ok(message) => PyValueError::new_err(message.unbind()),
        Err(err) => err,
    }
}

/// upb's error of `CopyFrom` and `MergeFrom` of a value that is not a message of the class.
fn not_same_class(method: &str, message: &Bound<'_, PyAny>, other: &Bound<'_, PyAny>) -> PyErr {
    let class_of = |v: &Bound<'_, PyAny>| {
        v.get_type()
            .str()
            .map(|s| s.to_string())
            .unwrap_or_default()
    };
    PyTypeError::new_err(format!(
        "Parameter to {method}() must be instance of same class: expected {} got {}.",
        class_of(message),
        class_of(other)
    ))
}

/// The arguments of `SerializeToString`: upb's takes `deterministic`, and nothing else.
fn check_serialize_arguments(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    if args.len() > 1 {
        return Err(PyTypeError::new_err(format!(
            "function takes at most 1 argument ({} given)",
            args.len()
        )));
    }
    for key in kwargs.iter().flat_map(|k| k.keys()) {
        let key = key.str()?.to_string();
        if key != "deterministic" {
            return Err(PyTypeError::new_err(format!(
                "'{key}' is an invalid keyword argument for this function"
            )));
        }
    }
    Ok(())
}

/// The full name and the bytes of `message`, a message of this runtime or one of protobuf's.
pub(crate) fn name_and_bytes(message: &Bound<'_, PyAny>) -> PyResult<(String, Vec<u8>)> {
    let py = message.py();
    if let Ok(m) = message.cast::<PyMessage>() {
        let m = m.borrow();
        return Ok((m.desc.full_name.clone(), codec::encode(py, &m)?));
    }
    let name = message
        .getattr("DESCRIPTOR")
        .and_then(|d| d.getattr("full_name"))
        .and_then(|n| n.extract::<String>())
        .map_err(|_| {
            PyTypeError::new_err(format!(
                "{} is not a protocol message",
                message.repr().map(|r| r.to_string()).unwrap_or_default()
            ))
        })?;
    let bytes = bytes_of(&message.call_method0("SerializeToString")?)?;
    Ok((name, bytes))
}

/// Whether `value`, an object other than a message of the runtime, is a message of protobuf's
/// (of upb, or of protobuf's pure-Python implementation) of the type `full_name` of protobuf's
/// default pool: the pool whose files the runtime holds, so that the message is of the type of the
/// runtime of that name, as upb has one class of each type of a pool. A message of a type of
/// another pool, of another class for upb too, is not, nor is an object that only has a
/// `DESCRIPTOR` and a `SerializeToString`, which upb's messages refuse too.
fn is_protobufs_message_of(value: &Bound<'_, PyAny>, full_name: &str) -> bool {
    let py = value.py();
    let of_the_type = || -> PyResult<bool> {
        let message = py.import("google.protobuf.message")?.getattr("Message")?;
        if !value.is_instance(&message)? {
            return Ok(false);
        }
        let descriptor = value.getattr("DESCRIPTOR")?;
        if descriptor.getattr("full_name")?.extract::<String>()? != full_name {
            return Ok(false);
        }
        let default = py
            .import("google.protobuf.descriptor_pool")?
            .call_method0("Default")?;
        Ok(descriptor.getattr("file")?.getattr("pool")?.is(&default))
    };
    of_the_type().unwrap_or(false)
}

/// A new message of the type `full_name` of `value`: a copy of a message of that type, of this
/// runtime or of protobuf's, or a message whose fields are the items of a mapping, each message
/// that it copies or builds a step of `budget`. A message of protobuf's of the type is taken as a
/// message before any mapping but a dict, as it can have a field named `keys`
/// (`Expression.Literal.Map` has one), which made it a mapping without `items`.
pub(crate) fn message_from_value<'py>(
    py: Python<'py>,
    full_name: &str,
    value: &Bound<'py, PyAny>,
    budget: &ConversionBudget,
) -> PyResult<Bound<'py, PyMessage>> {
    message_from_value_of(py, full_name, value, budget, false)
}

/// [`message_from_value`], whose copy of a message of protobuf's of the type is of its bytes of
/// any depth when `unlimited`, as upb's `==` compares messages of any depth, else of upb's limit
/// of a parse, as upb merges a message into a field.
fn message_from_value_of<'py>(
    py: Python<'py>,
    full_name: &str,
    value: &Bound<'py, PyAny>,
    budget: &ConversionBudget,
    unlimited: bool,
) -> PyResult<Bound<'py, PyMessage>> {
    if let Ok(m) = value.cast::<PyMessage>() {
        if m.borrow().desc.full_name != full_name {
            return Err(not_initializing(py, full_name, value));
        }
        return duplicate_with(py, m, budget);
    }
    let from_mapping = || -> PyResult<Bound<'py, PyMessage>> {
        let child = new_instance(py, full_name)?;
        child
            .borrow_mut()
            .init_from_mapping_with(py, value, budget)?;
        Ok(child)
    };
    if value.cast::<PyDict>().is_ok() {
        return from_mapping();
    }
    if is_protobufs_message_of(value, full_name) {
        if let Ok((_, bytes)) = name_and_bytes(value) {
            let child = new_instance(py, full_name)?;
            if unlimited {
                codec::merge_unlimited_in(py, &child, &bytes, budget)?;
            } else {
                codec::merge_in(py, &child, &bytes, budget)?;
            }
            return Ok(child);
        }
    }
    if value.hasattr("keys").unwrap_or(false) {
        return from_mapping();
    }
    Err(not_initializing(py, full_name, value))
}

/// `other`, an object that `==` compares a message of the runtime with, as a message of the
/// runtime of the type of `message`, when it is a message of protobuf's of that type that
/// protobuf's default pool describes, the pool whose files the runtime holds: as a message field
/// of the runtime takes it (see [`message_from_value`]), of its bytes, each message that it
/// builds a step of a full budget, as for a parse. None of any other object, such as a class, or a
/// message of a type of another pool, which upb's `==` gives `NotImplemented` of, as a message of
/// another class: upb has one class of each type of a pool, so its messages of a type compare by
/// their contents, where the runtime has a class of the type besides protobuf's.
fn as_message_of_type<'py>(
    py: Python<'py>,
    message: &Bound<'py, PyMessage>,
    other: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyMessage>>> {
    if other.is_none() || other.is_instance_of::<PyType>() {
        return Ok(None);
    }
    let full_name = message.borrow().desc.full_name.clone();
    if !is_protobufs_message_of(other, &full_name) {
        return Ok(None);
    }
    message_from_value_of(py, &full_name, other, &ConversionBudget::full(), true).map(Some)
}

/// A new message of the type `full_name` of `value`, as a message field of a constructor takes
/// it: as [`message_from_value`] does, and, as upb's, through the `_internal_assign` of a new
/// message of a well-known type whose class assigns values (a `Timestamp`, of a `datetime`), of a
/// value that is neither a mapping nor a message of the type. A message of the runtime of the
/// type is copied at once, as upb copies a message, without the lookup of a `keys` of it, which
/// raised and dropped an `AttributeError` for each message of a list of many.
fn message_for_field<'py>(
    py: Python<'py>,
    full_name: &str,
    value: &Bound<'py, PyAny>,
    budget: &ConversionBudget,
) -> PyResult<Bound<'py, PyMessage>> {
    let runtime_message = value.cast::<PyMessage>().ok();
    if let Some(m) = runtime_message {
        if m.borrow().desc.full_name == full_name {
            // As upb's, which copies the message of a field of a constructor by its bytes.
            if deeper_than(py, &m.borrow(), codec::depth_limit()) {
                return copy_of_bytes(py, full_name, m, budget);
            }
            return duplicate_with(py, m, budget);
        }
    }
    let is_dict = value.cast::<PyDict>().is_ok();
    // As upb's: a dict of a Struct is its items, of `_internal_assign`, else, of a dict of only
    // the key `fields` that it refuses, the fields of the message.
    if is_dict && full_name == STRUCT && assigns_values(py, full_name) {
        let child = new_instance(py, full_name)?;
        match child.call_method1("_internal_assign", (value,)) {
            Ok(_) => return Ok(child),
            Err(err) => {
                let dict = value.cast::<PyDict>()?;
                if !(dict.len() == 1 && dict.contains("fields")?) {
                    return Err(err);
                }
            }
        }
        return message_from_value(py, full_name, value, budget);
    }
    let is_mapping = is_dict || value.hasattr("keys").unwrap_or(false);
    let is_of_type = runtime_message.is_none() && is_protobufs_message_of(value, full_name);
    if !is_mapping && !is_of_type && assigns_values(py, full_name) {
        let child = new_instance(py, full_name)?;
        child.call_method1("_internal_assign", (value,))?;
        return Ok(child);
    }
    message_from_value(py, full_name, value, budget)
}

/// The full names of the well-known types of which upb's messages compare with, and find a
/// value in, a dict and a list: `Struct` and `ListValue`.
const STRUCT: &str = "google.protobuf.Struct";
const LIST_VALUE: &str = "google.protobuf.ListValue";

/// Whether `message` nests more than `levels` levels of messages below it, as upb's parse counts
/// them: a message of a field or of a repeated field, a map entry, and the message of the value of
/// a map entry each a level. The walk stops past `levels`.
pub(crate) fn deeper_than(py: Python<'_>, message: &PyMessage, levels: u32) -> bool {
    // A level of the walk on a stack with room for it, as the limit of upb's
    // `SetAllowOversizeProtos(True)` is of 65535 levels.
    grow(|| {
        let below = |child: &Py<PyMessage>, levels: u32| -> bool {
            levels == 0 || deeper_than(py, &child.bind(py).borrow(), levels - 1)
        };
        message.slots.iter().any(|(_, slot)| match slot {
            Slot::Message(child) => below(child, levels),
            Slot::Composite(c) => c.bind(py).borrow().items.iter().any(|m| below(m, levels)),
            Slot::Map(map) => {
                let map = map.bind(py).borrow();
                !map.entries.is_empty()
                    && (levels == 0
                        || map.entries.iter().any(|(_, value)| match value {
                            MapValue::Message(child) => below(child, levels - 1),
                            MapValue::Scalar(_) => false,
                        }))
            }
            Slot::Scalar(_) | Slot::Repeated(_) => false,
        })
    })
}

/// A copy of `message` of the type `full_name` of its bytes, as upb copies a message by a parse
/// of its bytes, so of upb's error of a message of more levels than its limit
/// ([`codec::depth_limit`]).
fn copy_of_bytes<'py>(
    py: Python<'py>,
    full_name: &str,
    message: &Bound<'py, PyMessage>,
    budget: &ConversionBudget,
) -> PyResult<Bound<'py, PyMessage>> {
    let bytes = codec::encode(py, &message.borrow())?;
    let child = new_instance(py, full_name)?;
    codec::merge_in(py, &child, &bytes, budget)?;
    Ok(child)
}

/// upb's error of a value of a message field of a constructor that is neither a dict nor a
/// message of the type `full_name`.
fn not_initializing(py: Python<'_>, full_name: &str, value: &Bound<'_, PyAny>) -> PyErr {
    let expected = registered_class(py, full_name)
        .and_then(|c| c.bind(py).str().ok().map(|s| s.to_string()))
        .unwrap_or_else(|| full_name.to_string());
    let got = value
        .get_type()
        .str()
        .map(|s| s.to_string())
        .unwrap_or_default();
    PyTypeError::new_err(format!(
        "Parameter to initialize message field must be dict or instance of same class: expected \
         {expected} got {got}."
    ))
}

/// A slot of a message, read for its copy: its scalars copied, and the messages that it holds,
/// which the copy copies once the message is no longer borrowed.
enum Shallow {
    Copied(Slot),
    Message(Py<PyMessage>),
    Composite(MessageClass, Vec<Py<PyMessage>>),
    Map(MapEntries),
}

/// A copy of `message`, of the class of its type, whose messages and containers are copies too,
/// a step of `budget` for each message copied:
/// each message is read under a short borrow, and the messages that it holds are copied after
/// it, so that the checkpoints of the copy of a message of many messages, whose Python code may
/// use the message, find no borrow of it held.
pub(crate) fn duplicate_with<'py>(
    py: Python<'py>,
    message: &Bound<'py, PyMessage>,
    budget: &ConversionBudget,
) -> PyResult<Bound<'py, PyMessage>> {
    copy_as(py, message, None, budget)
}

/// A copy of `message`, as [`duplicate_with`] makes it, of `class` when it is given: that of the
/// messages of a container, which the copy of each of them takes, rather than the class of the
/// message's type.
fn copy_as<'py>(
    py: Python<'py>,
    message: &Bound<'py, PyMessage>,
    class: Option<&MessageClass>,
    budget: &ConversionBudget,
) -> PyResult<Bound<'py, PyMessage>> {
    grow(|| {
        budget.step(py)?;
        let (slots, unknown) = copy_fields_of(py, message, budget)?;
        let copy = match class {
            Some(class) => class.new_message(py)?,
            None => {
                let desc = message.borrow().desc.clone();
                new_instance_of(py, &desc)?
            }
        };
        {
            let mut c = copy.borrow_mut();
            c.slots = slots;
            c.unknown = unknown;
        }
        Ok(copy)
    })
}

/// Copies of the fields and the unknown fields of `message`, read under a short borrow, whose
/// messages are copied after it.
fn copy_fields_of(
    py: Python<'_>,
    message: &Bound<'_, PyMessage>,
    budget: &ConversionBudget,
) -> PyResult<(Slots, Vec<u8>)> {
    // The copied slots are set in the order of the fields at once; those of messages, after the
    // borrow, in their places.
    let (mut slots, held, desc, unknown) = {
        let m = message.borrow();
        let mut slots = Slots::default();
        let mut held = Vec::new();
        for (index, slot) in m.slots.iter() {
            match shallow_slot(py, slot)? {
                Shallow::Copied(slot) => slots.push_last(index, slot),
                shallow => held.push((index, shallow)),
            }
        }
        let desc = (!held.is_empty()).then(|| m.desc.clone());
        (slots, held, desc, m.unknown.clone())
    };
    if let Some(desc) = desc {
        for (index, shallow) in held {
            let slot = deep_slot(py, shallow, budget)?;
            slots.set(&desc, index, slot);
        }
    }
    Ok((slots, unknown))
}

fn shallow_slot(py: Python<'_>, slot: &Slot) -> PyResult<Shallow> {
    Ok(match slot {
        Slot::Scalar(v) => Shallow::Copied(Slot::Scalar(v.clone())),
        Slot::Message(c) => Shallow::Message(c.clone_ref(py)),
        Slot::Repeated(c) => {
            Shallow::Copied(Slot::Repeated(Py::new(py, c.bind(py).borrow().copy())?))
        }
        Slot::Composite(c) => {
            let c = c.bind(py).borrow();
            Shallow::Composite(
                c.class.clone_ref(py),
                c.items.iter().map(|m| m.clone_ref(py)).collect(),
            )
        }
        Slot::Map(c) => Shallow::Map(c.bind(py).borrow().entries_for_copy(py)),
    })
}

fn deep_slot(py: Python<'_>, slot: Shallow, budget: &ConversionBudget) -> PyResult<Slot> {
    Ok(match slot {
        Shallow::Copied(slot) => slot,
        Shallow::Message(c) => Slot::Message(duplicate_with(py, c.bind(py), budget)?.unbind()),
        Shallow::Composite(class, items) => {
            let mut copies = Vec::with_capacity(items.len());
            for item in &items {
                copies.push(copy_as(py, item.bind(py), Some(&class), budget)?.unbind());
            }
            Slot::Composite(Py::new(py, RepeatedComposite::new(class, copies, None))?)
        }
        Shallow::Map(entries) => Slot::Map(entries.copy(py, budget)?.into_object(py)?),
    })
}

/// Set `message` in its parent if it is an unset child, and the parent in its own, as setting a
/// field of an unset child does.
pub(crate) fn mark_present(message: &Bound<'_, PyMessage>) -> PyResult<()> {
    let py = message.py();
    let mut current = message.clone();
    loop {
        let parent = current.borrow_mut().parent.take();
        let Some(parent) = parent else {
            return Ok(());
        };
        let Some(owner) = parent.message.bind(py).upgrade_as::<PyMessage>()? else {
            return Ok(());
        };
        {
            let mut p = owner.borrow_mut();
            let still_unset = p
                .lazy_child(parent.field)
                .is_some_and(|c| c.bind(py).is(&current));
            if !still_unset {
                return Ok(());
            }
            p.take_lazy_child(parent.field);
            p.clear_oneof_siblings(py, parent.field);
            p.set_slot(parent.field, Slot::Message(current.clone().unbind()));
        }
        current = owner;
    }
}

/// Merge `source` into `target`, as protobuf's `MergeFrom` does: a set scalar replaces the value,
/// a message merges into the target's, and repeated fields and maps add their values. Each
/// message merged or copied is a step of `budget`, with no borrow held between the fields.
fn merge_into_with(
    py: Python<'_>,
    target: &Bound<'_, PyMessage>,
    source: &Bound<'_, PyMessage>,
    budget: &ConversionBudget,
) -> PyResult<()> {
    if target.is(source) {
        let copy = duplicate_with(py, source, budget)?;
        return merge_into_with(py, target, &copy, budget);
    }
    enum Merge {
        Scalar(Scalar),
        Message(Py<PyMessage>),
        Repeated(Vec<Scalar>),
        Composite(Vec<Py<PyMessage>>),
        Map(MapEntries),
    }
    grow(|| {
        budget.step(py)?;
        let (desc, indices): (Arc<MessageDesc>, Vec<usize>) = {
            let s = source.borrow();
            (
                s.desc.clone(),
                s.slots.iter().map(|(index, _)| index).collect(),
            )
        };
        for index in indices {
            let merge = {
                let s = source.borrow();
                match s.slots.get(index) {
                    None => continue,
                    Some(Slot::Scalar(v)) => {
                        if !desc.fields[index].has_presence() && v.is_default() {
                            continue;
                        }
                        Merge::Scalar(v.clone())
                    }
                    Some(Slot::Message(c)) => Merge::Message(c.clone_ref(py)),
                    Some(Slot::Repeated(c)) => Merge::Repeated(c.bind(py).borrow().items.clone()),
                    Some(Slot::Composite(c)) => Merge::Composite(
                        c.bind(py)
                            .borrow()
                            .items
                            .iter()
                            .map(|m| m.clone_ref(py))
                            .collect(),
                    ),
                    Some(Slot::Map(c)) => Merge::Map(c.bind(py).borrow().entries_for_copy(py)),
                }
            };
            match merge {
                Merge::Scalar(v) => target.borrow_mut().set_scalar(py, index, v),
                Merge::Message(child) => {
                    let into = target.borrow_mut().child_for_merge(py, index)?;
                    merge_into_with(py, &into, child.bind(py), budget)?;
                }
                Merge::Repeated(items) => {
                    let container = PyMessage::repeated_scalar_of(target, index)?;
                    container.bind(py).borrow_mut().items.extend(items);
                }
                Merge::Composite(items) => {
                    let container = PyMessage::composite_of(target, index)?;
                    let class = container.bind(py).borrow().class.clone_ref(py);
                    for item in items {
                        let copy = copy_as(py, item.bind(py), Some(&class), budget)?;
                        container.bind(py).borrow_mut().items.push(copy.unbind());
                    }
                }
                Merge::Map(entries) => {
                    let container = PyMessage::map_of_field(target, index)?;
                    MapContainer::merge_entries(py, container.bind(py), entries, budget)?;
                }
            }
        }
        let unknown = source.borrow().unknown.clone();
        target.borrow_mut().unknown.extend(unknown);
        Ok(())
    })
}

/// Whether `a` and `b` hold the same fields, as protobuf's `==` compares two messages: of the
/// same type, with the same set fields of the same values. It reads the set fields of the two
/// only, in the order of their numbers, as a field that neither sets is equal, and compares the
/// values where they are: it runs no Python code, so the messages and containers that it reads
/// stay borrowed meanwhile. It holds the GIL until it ends, as upb's `==` does.
pub(crate) fn content_eq(
    py: Python<'_>,
    a: &Bound<'_, PyMessage>,
    b: &Bound<'_, PyMessage>,
) -> PyResult<bool> {
    if a.is(b) {
        return Ok(true);
    }
    grow(|| {
        let (x, y) = (a.borrow(), b.borrow());
        if x.desc.full_name != y.desc.full_name {
            return Ok(false);
        }
        let fields = &x.desc.fields;
        let (mut xs, mut ys) = (x.slots.iter().peekable(), y.slots.iter().peekable());
        loop {
            // The next field that either sets.
            let index = match (xs.peek(), ys.peek()) {
                (None, None) => break,
                (Some(&(i, _)), None) | (None, Some(&(i, _))) => i,
                (Some(&(i, _)), Some(&(j, _))) => {
                    if fields[i].number <= fields[j].number {
                        i
                    } else {
                        j
                    }
                }
            };
            let in_a = xs.next_if(|&(i, _)| i == index).map(|(_, slot)| slot);
            let in_b = ys.next_if(|&(j, _)| j == index).map(|(_, slot)| slot);
            if !slot_eq(py, &fields[index], in_a, in_b)? {
                return Ok(false);
            }
        }
        Ok(codec::unknown_fields_eq(&x.unknown, &y.unknown))
    })
}

fn slot_eq(
    py: Python<'_>,
    field: &FieldDesc,
    a: Option<&Slot>,
    b: Option<&Slot>,
) -> PyResult<bool> {
    if field.map.is_some() {
        let (a, b) = (map_of(a), map_of(b));
        return match (a, b) {
            (None, None) => Ok(true),
            (Some(m), None) | (None, Some(m)) => Ok(m.bind(py).borrow().entries.is_empty()),
            (Some(x), Some(y)) => MapContainer::equal(py, x.bind(py), y.bind(py)),
        };
    }
    if field.repeated && field.kind == Kind::Message {
        let composite = |slot: Option<&Slot>| match slot {
            Some(Slot::Composite(c)) => Some(c.bind(py).borrow()),
            _ => None,
        };
        let (x, y) = (composite(a), composite(b));
        let (x, y) = (
            x.as_ref().map_or(&[][..], |c| &c.items[..]),
            y.as_ref().map_or(&[][..], |c| &c.items[..]),
        );
        if x.len() != y.len() {
            return Ok(false);
        }
        for (m, n) in x.iter().zip(y) {
            if !content_eq(py, m.bind(py), n.bind(py))? {
                return Ok(false);
            }
        }
        return Ok(true);
    }
    if field.repeated {
        let scalars = |slot: Option<&Slot>| match slot {
            Some(Slot::Repeated(c)) => Some(c.bind(py).borrow()),
            _ => None,
        };
        let (x, y) = (scalars(a), scalars(b));
        let (x, y) = (
            x.as_ref().map_or(&[][..], |c| &c.items[..]),
            y.as_ref().map_or(&[][..], |c| &c.items[..]),
        );
        return Ok(x.len() == y.len() && x.iter().zip(y).all(|(v, w)| v.eq_as_upb(w)));
    }
    if field.kind == Kind::Message {
        return match (a, b) {
            (None, None) => Ok(true),
            (Some(Slot::Message(x)), Some(Slot::Message(y))) => {
                content_eq(py, x.bind(py), y.bind(py))
            }
            _ => Ok(false),
        };
    }
    fn value(slot: Option<&Slot>) -> Option<&Scalar> {
        match slot {
            Some(Slot::Scalar(v)) => Some(v),
            _ => None,
        }
    }
    // As upb's, of the bits of a float or a double. A scalar that one sets is equal to the
    // other's default, of a field without presence: of the bits of 0.0, which -0.0 is not.
    Ok(match (value(a), value(b)) {
        (Some(v), Some(w)) => v.eq_as_upb(w),
        (Some(v), None) | (None, Some(v)) => !field.has_presence() && v.is_default(),
        (None, None) => true,
    })
}

fn map_of(slot: Option<&Slot>) -> Option<&Py<MapContainer>> {
    match slot {
        Some(Slot::Map(m)) => Some(m),
        _ => None,
    }
}

impl PyMessage {
    /// The maker of the messages of the class of `slf` (see [`MessageClass`]), of which its copies
    /// are, as upb's.
    fn maker_of(slf: &Bound<'_, Self>) -> MessageClass {
        MessageClass {
            class: Some(slf.get_type().unbind()),
            desc: slf.borrow().desc.clone(),
        }
    }

    pub(crate) fn empty(desc: Arc<MessageDesc>) -> Self {
        PyMessage {
            desc,
            slots: Slots::default(),
            lazy: Vec::new(),
            parent: None,
            unknown: Vec::new(),
        }
    }

    /// Set the slot of the field `index`, giving back the one that it replaces.
    pub(crate) fn set_slot(&mut self, index: usize, slot: Slot) -> Option<Slot> {
        self.slots.set(&self.desc, index, slot)
    }

    /// The unset child of the field `index` that `__getattr__` gave, if any.
    fn lazy_child(&self, index: usize) -> Option<&Py<PyMessage>> {
        self.lazy.iter().find(|(i, _)| *i == index).map(|(_, c)| c)
    }

    fn take_lazy_child(&mut self, index: usize) -> Option<Py<PyMessage>> {
        let at = self.lazy.iter().position(|(i, _)| *i == index)?;
        Some(self.lazy.swap_remove(at).1)
    }

    /// Detach the unset child of the field `index` that `__getattr__` gave, so that setting its
    /// fields no longer sets the field.
    fn sever(&mut self, py: Python<'_>, index: usize) {
        if let Some(child) = self.take_lazy_child(index) {
            if let Ok(mut c) = child.bind(py).try_borrow_mut() {
                c.parent = None;
            }
        }
    }

    /// Clear the other members of the oneof of the field `index`. An unset child that a member
    /// gave out stays the member's, as upb's does: setting a field of it later sets the member.
    pub(crate) fn clear_oneof_siblings(&mut self, py: Python<'_>, index: usize) {
        let _ = py;
        let Some(oneof) = self.desc.fields[index].oneof else {
            return;
        };
        // Of the set fields, which are few, not of the members, which may be many.
        let desc = &self.desc;
        self.slots
            .retain(|member| member == index || desc.fields[member].oneof != Some(oneof));
    }

    pub(crate) fn set_scalar(&mut self, py: Python<'_>, index: usize, value: Scalar) {
        self.clear_oneof_siblings(py, index);
        self.set_slot(index, Slot::Scalar(value));
    }

    pub(crate) fn set_message(&mut self, py: Python<'_>, index: usize, child: Py<PyMessage>) {
        self.clear_oneof_siblings(py, index);
        self.sever(py, index);
        self.set_slot(index, Slot::Message(child));
    }

    fn clear_index(&mut self, py: Python<'_>, index: usize) {
        self.slots.remove(index);
        self.sever(py, index);
    }

    fn clear_all(&mut self, py: Python<'_>) {
        self.slots = Slots::default();
        for (_, child) in std::mem::take(&mut self.lazy) {
            if let Ok(mut c) = child.bind(py).try_borrow_mut() {
                c.parent = None;
            }
        }
        self.unknown.clear();
    }

    /// Replace the fields with `slots` and the unknown fields with `unknown`, as upb's `CopyFrom`
    /// copies a message: an unset child that the message gave out becomes the child of its field
    /// when the copy sets that field, and stays unset (and the field's) when it does not.
    fn replace_slots(&mut self, py: Python<'_>, slots: Slots, unknown: Vec<u8>) -> PyResult<()> {
        let mut replaced = Vec::with_capacity(slots.len());
        for (index, slot) in slots.into_vec() {
            let slot = match (slot, self.take_lazy_child(index)) {
                (Slot::Message(copied), Some(stub)) => {
                    let (fields, unknown) = {
                        let mut c = copied.bind(py).borrow_mut();
                        (std::mem::take(&mut c.slots), std::mem::take(&mut c.unknown))
                    };
                    {
                        let mut child = stub.bind(py).try_borrow_mut()?;
                        child.parent = None;
                        child.slots = fields;
                        child.unknown = unknown;
                    }
                    Slot::Message(stub)
                }
                (slot, stub) => {
                    if let Some(stub) = stub {
                        self.lazy.push((index, stub));
                    }
                    slot
                }
            };
            replaced.push((index, slot));
        }
        self.slots = Slots::from_vec(replaced);
        self.unknown = unknown;
        Ok(())
    }

    /// The child of the singular message field `index` to merge into: the set one, else a new
    /// one, set now.
    pub(crate) fn child_for_merge<'py>(
        &mut self,
        py: Python<'py>,
        index: usize,
    ) -> PyResult<Bound<'py, PyMessage>> {
        if let Some(Slot::Message(c)) = self.slots.get(index) {
            return Ok(c.bind(py).clone());
        }
        // As upb's: an unset child that the message gave out becomes the child that the merge
        // sets, so that the object it gave out holds the merged fields.
        let child = match self.take_lazy_child(index) {
            Some(stub) => {
                let stub = stub.into_bound(py);
                stub.try_borrow_mut()?.parent = None;
                stub
            }
            None => new_instance(py, &self.desc.fields[index].type_name)?,
        };
        self.clear_oneof_siblings(py, index);
        self.set_slot(index, Slot::Message(child.clone().unbind()));
        Ok(child)
    }

    /// The container of the repeated scalar field `index`, made if absent.
    pub(crate) fn repeated_scalar(
        &mut self,
        py: Python<'_>,
        index: usize,
        owner: Option<Py<PyWeakrefReference>>,
    ) -> PyResult<Py<RepeatedScalar>> {
        if let Some(Slot::Repeated(c)) = self.slots.get(index) {
            return Ok(c.clone_ref(py));
        }
        let field = &self.desc.fields[index];
        let container = Py::new(
            py,
            RepeatedScalar::new(
                field.kind,
                field.full_name.clone(),
                field.type_name.clone(),
                Vec::new(),
                owner,
            ),
        )?;
        self.set_slot(index, Slot::Repeated(container.clone_ref(py)));
        Ok(container)
    }

    /// The container of the repeated message field `index`, made if absent.
    pub(crate) fn composite(
        &mut self,
        py: Python<'_>,
        index: usize,
        owner: Option<Py<PyWeakrefReference>>,
    ) -> PyResult<Py<RepeatedComposite>> {
        if let Some(Slot::Composite(c)) = self.slots.get(index) {
            return Ok(c.clone_ref(py));
        }
        let field = &self.desc.fields[index];
        let class = MessageClass::of(py, &field.type_name)?;
        let container = Py::new(py, RepeatedComposite::new(class, Vec::new(), owner))?;
        self.set_slot(index, Slot::Composite(container.clone_ref(py)));
        Ok(container)
    }

    /// The map of the map field `index`, made if absent.
    pub(crate) fn map(
        &mut self,
        py: Python<'_>,
        index: usize,
        owner: Option<Py<PyWeakrefReference>>,
    ) -> PyResult<Py<MapContainer>> {
        if let Some(Slot::Map(c)) = self.slots.get(index) {
            return Ok(c.clone_ref(py));
        }
        let field = &self.desc.fields[index];
        let map = field
            .map
            .as_ref()
            .ok_or_else(|| PyTypeError::new_err(format!("{} is not a map", field.full_name)))?;
        let container =
            MapContainer::new(field.name.clone(), map.clone(), owner).into_object(py)?;
        self.set_slot(index, Slot::Map(container.clone_ref(py)));
        Ok(container)
    }

    /// The container of the repeated scalar field `index` of `message`, made if absent, which
    /// holds `message` weakly.
    pub(crate) fn repeated_scalar_of(
        message: &Bound<'_, PyMessage>,
        index: usize,
    ) -> PyResult<Py<RepeatedScalar>> {
        let py = message.py();
        if let Some(Slot::Repeated(c)) = message.borrow().slots.get(index) {
            return Ok(c.clone_ref(py));
        }
        let owner = weak(message)?;
        message.borrow_mut().repeated_scalar(py, index, Some(owner))
    }

    /// The container of the repeated message field `index` of `message`, made if absent.
    pub(crate) fn composite_of(
        message: &Bound<'_, PyMessage>,
        index: usize,
    ) -> PyResult<Py<RepeatedComposite>> {
        let py = message.py();
        if let Some(Slot::Composite(c)) = message.borrow().slots.get(index) {
            return Ok(c.clone_ref(py));
        }
        let owner = weak(message)?;
        message.borrow_mut().composite(py, index, Some(owner))
    }

    /// The map of the map field `index` of `message`, made if absent.
    pub(crate) fn map_of_field(
        message: &Bound<'_, PyMessage>,
        index: usize,
    ) -> PyResult<Py<MapContainer>> {
        let py = message.py();
        if let Some(Slot::Map(c)) = message.borrow().slots.get(index) {
            return Ok(c.clone_ref(py));
        }
        let owner = weak(message)?;
        message.borrow_mut().map(py, index, Some(owner))
    }

    /// Set the fields of the items of `mapping`, as the constructor and a mapping for a message
    /// field do. None leaves a field unset. A mapping of a message field sets the fields of its
    /// message through here too, one level of a recursion as deep as the mappings nest, on a stack
    /// with room for it.
    ///
    /// One interrupt budget counts a step for each message that the fields copy or build, and for
    /// each value of a repeated field and each key of a map of messages, of every level, so that a
    /// constructor of many values (`Literal.Array(elements=[...])`, of a million messages) lets
    /// another thread take the GIL, and a signal interrupt it, as the copies of the runtime do. A
    /// map of scalars takes its value in Python, as upb's does (see [`MapContainer::update_from`]).
    pub(crate) fn init_from_mapping(
        &mut self,
        py: Python<'_>,
        mapping: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.init_from_mapping_with(py, mapping, &ConversionBudget::full())
    }

    /// Set the fields of the items of `mapping`, as [`PyMessage::init_from_mapping`] does, of
    /// the steps of `budget`.
    pub(crate) fn init_from_mapping_with(
        &mut self,
        py: Python<'_>,
        mapping: &Bound<'_, PyAny>,
        budget: &ConversionBudget,
    ) -> PyResult<()> {
        grow(|| self.set_fields_of(py, mapping, budget))
    }

    fn set_fields_of(
        &mut self,
        py: Python<'_>,
        mapping: &Bound<'_, PyAny>,
        budget: &ConversionBudget,
    ) -> PyResult<()> {
        budget.step(py)?;
        let desc = self.desc.clone();
        let items = mapping.call_method0("items")?;
        for item in items.try_iter()? {
            let (key, value): (String, Bound<'_, PyAny>) = item?.extract()?;
            if value.is_none() {
                continue;
            }
            let Some(&index) = desc.by_name.get(&key) else {
                return Err(PyValueError::new_err(format!(
                    "Protocol message {} has no \"{key}\" field.",
                    desc.name
                )));
            };
            let field = &desc.fields[index];
            let not_iterable = || {
                PyTypeError::new_err(format!(
                    "Argument for field {} is not iterable",
                    field.full_name
                ))
            };
            if field.map.is_some() {
                let container = self.map(py, index, None)?.into_bound(py);
                MapContainer::update_from(py, &container, &value, &field.full_name, budget)?;
            } else if field.repeated && field.kind == Kind::Message {
                let mut items = Vec::new();
                for element in value.try_iter().map_err(|_| not_iterable())? {
                    let element = element?;
                    budget.step(py)?;
                    items.push(message_for_field(py, &field.type_name, &element, budget)?.unbind());
                }
                let container = self.composite(py, index, None)?;
                container.bind(py).borrow_mut().items.extend(items);
            } else if field.repeated {
                let mut items = Vec::new();
                for element in value.try_iter().map_err(|_| not_iterable())? {
                    let element = element?;
                    budget.step(py)?;
                    items.push(Scalar::from_py(
                        field.kind,
                        &field.full_name,
                        &field.type_name,
                        &element,
                    )?);
                }
                let container = self.repeated_scalar(py, index, None)?;
                container.bind(py).borrow_mut().items.extend(items);
            } else if field.kind == Kind::Message {
                let child = message_for_field(py, &field.type_name, &value, budget)?;
                self.set_message(py, index, child.unbind());
            } else {
                let scalar =
                    Scalar::from_py(field.kind, &field.full_name, &field.type_name, &value)?;
                self.set_scalar(py, index, scalar);
            }
        }
        Ok(())
    }

    /// Discard the unknown fields of the message and of its messages, as upb's
    /// `_upb_Message_DiscardUnknown` does at `depth`: it gives up once `depth`, less one for
    /// each level, is 0, so the messages of that level and below keep theirs.
    fn discard_unknown(&mut self, py: Python<'_>, depth: u32) {
        let depth = depth.saturating_sub(1);
        if depth == 0 {
            return;
        }
        grow(|| {
            self.unknown.clear();
            for (_, slot) in self.slots.iter() {
                match slot {
                    Slot::Message(child) => child.bind(py).borrow_mut().discard_unknown(py, depth),
                    Slot::Composite(c) => {
                        for child in &c.bind(py).borrow().items {
                            child.bind(py).borrow_mut().discard_unknown(py, depth);
                        }
                    }
                    Slot::Map(c) => {
                        for (_, value) in c.bind(py).borrow().entries.iter() {
                            if let MapValue::Message(child) = value {
                                child.bind(py).borrow_mut().discard_unknown(py, depth);
                            }
                        }
                    }
                    Slot::Scalar(_) | Slot::Repeated(_) => {}
                }
            }
        })
    }

    /// `MergeFrom` of `other_msg`, a message of the runtime or of protobuf's, each message that it
    /// merges or copies a step of `budget`.
    pub(crate) fn merge_from_with(
        slf: &Bound<'_, Self>,
        other_msg: &Bound<'_, PyAny>,
        budget: &ConversionBudget,
    ) -> PyResult<()> {
        let py = slf.py();
        let full_name = slf.borrow().desc.full_name.clone();
        if let Ok(other) = other_msg.cast::<PyMessage>() {
            if other.borrow().desc.full_name != full_name {
                return Err(not_same_class("MergeFrom", slf.as_any(), other_msg));
            }
            mark_present(slf)?;
            // As upb's, which merges a message by a parse of its bytes: one of more than 100
            // levels fails, of what the parse of its bytes keeps.
            if deeper_than(py, &other.borrow(), codec::depth_limit()) {
                let bytes = codec::encode(py, &other.borrow())?;
                return codec::merge_in(py, slf, &bytes, budget);
            }
            merge_into_with(py, slf, other, budget)
        } else {
            // As upb's, which takes a message of its class of the type only: protobuf's message
            // of the type of its default pool, of its bytes.
            if !is_protobufs_message_of(other_msg, &full_name) {
                return Err(not_same_class("MergeFrom", slf.as_any(), other_msg));
            }
            let (_, bytes) = name_and_bytes(other_msg)
                .map_err(|_| not_same_class("MergeFrom", slf.as_any(), other_msg))?;
            mark_present(slf)?;
            codec::merge_in(py, slf, &bytes, budget)
        }
    }

    /// The value of the field `index`, which `__getattribute__` gives.
    fn field_value(slf: &Bound<'_, Self>, index: usize) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        {
            let me = slf.borrow();
            let field = &me.desc.fields[index];
            if !field.repeated && field.kind != Kind::Message {
                return match me.slots.get(index) {
                    Some(Slot::Scalar(v)) => v.to_py(py),
                    _ => match &field.default {
                        Some(default) => default.to_py(py),
                        None => Scalar::default_of(field.kind).to_py(py),
                    },
                };
            }
        }
        let desc = slf.borrow().desc.clone();
        let field = &desc.fields[index];
        if field.map.is_some() {
            let map = slf.borrow_mut().map(py, index, Some(weak(slf)?))?;
            return Ok(map.into_any());
        }
        if field.repeated && field.kind == Kind::Message {
            let container = slf.borrow_mut().composite(py, index, Some(weak(slf)?))?;
            return Ok(container.into_any());
        }
        if field.repeated {
            let container = slf
                .borrow_mut()
                .repeated_scalar(py, index, Some(weak(slf)?))?;
            return Ok(container.into_any());
        }
        // A message field.
        {
            let me = slf.borrow();
            if let Some(Slot::Message(c)) = me.slots.get(index) {
                return Ok(c.clone_ref(py).into_any());
            }
            if let Some(c) = me.lazy_child(index) {
                return Ok(c.clone_ref(py).into_any());
            }
        }
        let child = new_instance(py, &field.type_name)?;
        child.borrow_mut().parent = Some(Parent {
            message: weak(slf)?,
            field: index,
        });
        slf.borrow_mut().lazy.push((index, child.clone().unbind()));
        Ok(child.into_any().unbind())
    }

    fn no_field(&self, name: &str) -> PyErr {
        PyValueError::new_err(format!(
            "Protocol message {} has no \"{name}\" field.",
            self.desc.name
        ))
    }

    /// The index of the field `name`, with upb's error of a name of no field.
    fn field_index_of(&self, name: &FieldName) -> PyResult<usize> {
        self.desc
            .by_name
            .get(name.text())
            .copied()
            .ok_or_else(|| self.no_field(&name.display()))
    }

    /// upb's `KeyError` of `HasExtension` and `ClearExtension` of `field`, of a message that
    /// declares no extension ranges: of an object that is not a field descriptor, of a field that
    /// is not an extension, and of an extension, which extends another message.
    fn not_an_extension(&self, field: &Bound<'_, PyAny>) -> PyErr {
        let full_name = field
            .getattr("full_name")
            .and_then(|n| n.extract::<String>());
        let is_extension = field.getattr("is_extension").and_then(|e| e.is_truthy());
        match (full_name, is_extension) {
            (Ok(name), Ok(false)) => {
                PyKeyError::new_err(format!("Field {name} is not an extension\n"))
            }
            (Ok(name), Ok(true)) => PyKeyError::new_err(format!(
                "Extension doesn't match ({} vs {name})",
                self.desc.full_name
            )),
            _ => PyKeyError::new_err(format!(
                "Object {} is not a field descriptor\n",
                field.repr().map(|r| r.to_string()).unwrap_or_default()
            )),
        }
    }
}

/// The paths of the required fields that `message` and the messages that it holds lack (see
/// `FindInitializationErrors`): its own, in the order of their declaration, then those of each
/// message of its fields, in the order of their numbers.
fn missing_fields(py: Python<'_>, message: &PyMessage) -> Vec<String> {
    let mut missing = Vec::new();
    if may_lack_required(&message.desc) {
        collect_missing(py, message, "", &mut missing);
    }
    missing
}

/// Whether a message of the type `desc` can lack a required field: whether the type, or a type of
/// a message that it can hold (of a field or of the value of a map), declares one. Of a type of
/// proto3, as all of Spark Connect are, it is false, so that the check of a message of many
/// messages reads none of them. Known of each type once.
fn may_lack_required(desc: &MessageDesc) -> bool {
    *desc
        .may_lack_required
        .get_or_init(|| holds_required_fields(desc))
}

/// Whether the message type `desc`, or a type of a message that it can hold, declares a required
/// field, of a walk of their message fields.
fn holds_required_fields(desc: &MessageDesc) -> bool {
    // The types that the type can hold, of a walk of their message fields.
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![desc.full_name.clone()];
    let mut found = false;
    while let Some(name) = stack.pop() {
        if !seen.insert(name.clone()) {
            continue;
        }
        let Some(desc) = pool().message(&name) else {
            continue;
        };
        if !desc.required.is_empty() {
            found = true;
            break;
        }
        for field in &desc.fields {
            match &field.map {
                Some(map) if map.value == Kind::Message => stack.push(map.value_type.clone()),
                Some(_) => {}
                None if field.kind == Kind::Message => stack.push(field.type_name.clone()),
                None => {}
            }
        }
    }
    found
}

fn collect_missing(py: Python<'_>, message: &PyMessage, prefix: &str, out: &mut Vec<String>) {
    grow(|| {
        for &index in &message.desc.required {
            if !message.slots.contains(index) {
                out.push(format!("{prefix}{}", message.desc.fields[index].name));
            }
        }
        for (index, slot) in message.slots.iter() {
            let name = &message.desc.fields[index].name;
            match slot {
                Slot::Message(child) => {
                    let prefix = format!("{prefix}{name}.");
                    collect_missing(py, &child.bind(py).borrow(), &prefix, out);
                }
                Slot::Composite(c) => {
                    for (at, child) in c.bind(py).borrow().items.iter().enumerate() {
                        let prefix = format!("{prefix}{name}[{at}].");
                        collect_missing(py, &child.bind(py).borrow(), &prefix, out);
                    }
                }
                Slot::Map(map) => {
                    for (key, value) in map.bind(py).borrow().entries.iter() {
                        if let MapValue::Message(child) = value {
                            let key = text::map_key_text(key);
                            let prefix = format!("{prefix}{name}[{key}].");
                            collect_missing(py, &child.bind(py).borrow(), &prefix, out);
                        }
                    }
                }
                Slot::Scalar(_) | Slot::Repeated(_) => {}
            }
        }
    })
}

/// upb's `EncodeError` of a message that lacks required fields, which it does not serialize.
fn check_initialized(message: &Bound<'_, PyMessage>) -> PyResult<()> {
    let py = message.py();
    let missing = missing_fields(py, &message.borrow());
    if missing.is_empty() {
        return Ok(());
    }
    let full_name = message.borrow().desc.full_name.clone();
    Err(super::encode_error_of(
        py,
        format!(
            "Message {full_name} is missing required fields: {}",
            missing.join(",")
        ),
    ))
}

/// Whether the message type `full_name` is a well-known type whose class assigns a value of
/// another type, as protobuf's classes of the well-known types do (`Timestamp` takes a
/// `datetime`): the class has `_internal_assign`.
fn assigns_values(py: Python<'_>, full_name: &str) -> bool {
    full_name.starts_with("google.protobuf.")
        && registered_class(py, full_name)
            .is_some_and(|class| class.bind(py).hasattr("_internal_assign").unwrap_or(false))
}

#[pymethods]
impl PyMessage {
    /// An empty message of the class, as upb's `__new__` makes one: of no fields, whatever its
    /// arguments, which `__init__` takes.
    #[new]
    #[classmethod]
    #[pyo3(signature = (*args, **kwargs))]
    fn py_new(
        cls: &Bound<'_, PyType>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let _ = (args, kwargs);
        Ok(PyMessage::empty(message_type_of(cls)?))
    }

    /// As upb's: set the fields of the keyword arguments, over those that the message has (so
    /// `message.__init__(**fields)` merges them, as a constructor sets them of a new message).
    #[pyo3(signature = (*args, **kwargs))]
    fn __init__(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        if !args.is_empty() {
            return Err(PyTypeError::new_err("No positional arguments allowed"));
        }
        match kwargs {
            Some(kwargs) if !kwargs.is_empty() => {
                let py = slf.py();
                slf.borrow_mut().init_from_mapping(py, kwargs.as_any())
            }
            _ => Ok(()),
        }
    }

    /// As upb's: whether the caller is a generated module, which no caller of the runtime is.
    #[staticmethod]
    #[pyo3(name = "_CheckCalledFromGeneratedFile")]
    fn check_called_from_generated_file() -> bool {
        false
    }

    /// As upb's: the key of an item of `ListFields` that sorts the items by the numbers of their
    /// fields.
    #[staticmethod]
    #[pyo3(name = "_ListFieldsItemKey")]
    fn list_fields_item_key(item: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Ok(item.get_item(0)?.getattr("number")?.unbind())
    }

    /// As upb's: a field of the name, else an attribute of the class (a method, a nested type),
    /// else (`__getattr__`) one of its metaclass. The field is looked up first, as upb's is, so
    /// that the read of a field raises no `AttributeError` to look it up, which would take longer
    /// than the read.
    fn __getattribute__(slf: &Bound<'_, Self>, name: &Bound<'_, PyString>) -> PyResult<Py<PyAny>> {
        let index = {
            // The UTF-8 of the name, of one allocation (the str's own is not in the stable ABI
            // of Python 3.9).
            let utf8 = name.encode_utf8()?;
            let me = slf.borrow();
            std::str::from_utf8(utf8.as_bytes())
                .ok()
                .and_then(|text| me.desc.by_name.get(text).copied())
        };
        if let Some(index) = index {
            return Self::field_value(slf, index);
        }
        // SAFETY: `slf` and `name` are live objects, and the result is a new reference, or null
        // with an exception set.
        unsafe {
            Bound::from_owned_ptr_or_err(
                slf.py(),
                pyo3::ffi::PyObject_GenericGetAttr(slf.as_ptr(), name.as_ptr()),
            )
        }
        .map(Bound::unbind)
    }

    /// The attribute `name` of no field nor attribute of the class: as upb's, an attribute of its
    /// metaclass (`__name__`, `mro`), else `AttributeError(name)`, but of `Extensions`, whose
    /// `AttributeError` (of a message that declares no extension ranges) has no message.
    fn __getattr__(slf: &Bound<'_, Self>, name: &Bound<'_, PyString>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if name.to_cow()? == "Extensions" {
            return Err(PyAttributeError::new_err(()));
        }
        let cls = slf.get_type();
        match cls.as_any().getattr(name) {
            Ok(value) => {
                // A descriptor of the class itself (of a class of its MRO), such as a property
                // whose getter raised the `AttributeError` that brought the lookup here, is not
                // given as the object of the class: its error stays, of the name. Another value
                // of the class is given, as one that the lookup of the class made of the name
                // (`__annotations__`, which a class makes on its first read), as upb gives it.
                for base in cls.mro().iter() {
                    let held = base
                        .getattr(pyo3::intern!(py, "__dict__"))
                        .and_then(|namespace| namespace.get_item(name));
                    if let Ok(held) = held {
                        if held.get_type().hasattr(pyo3::intern!(py, "__get__"))? {
                            return Err(PyAttributeError::new_err(name.clone().unbind()));
                        }
                        break;
                    }
                }
                Ok(value.unbind())
            }
            Err(error) if error.is_instance_of::<PyAttributeError>(py) => {
                Err(PyAttributeError::new_err(name.clone().unbind()))
            }
            Err(error) => Err(error),
        }
    }

    fn __setattr__(slf: &Bound<'_, Self>, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        let desc = slf.borrow().desc.clone();
        let Some(&index) = desc.by_name.get(name) else {
            return Err(PyAttributeError::new_err(format!(
                "Protocol message {} has no \"{name}\" field.",
                desc.name
            )));
        };
        let field = &desc.fields[index];
        if field.repeated {
            return Err(PyAttributeError::new_err(format!(
                "Assignment not allowed to map, or repeated field \"{name}\" in protocol message object."
            )));
        }
        // As upb's: the message is set in its parent before the value is checked, so a value that
        // fails still sets it.
        mark_present(slf)?;
        if field.kind == Kind::Message {
            // As upb's: a field of a well-known type whose class assigns values (a Timestamp, of
            // a datetime) takes the value through the `_internal_assign` of its message.
            if assigns_values(py, &field.type_name) {
                let child = Self::field_value(slf, index)?;
                child.bind(py).call_method1("_internal_assign", (value,))?;
                return Ok(());
            }
            return Err(PyAttributeError::new_err(format!(
                "Assignment not allowed to message field \"{name}\" in protocol message object."
            )));
        }
        let scalar = Scalar::from_py(field.kind, &field.full_name, &field.type_name, value)?;
        slf.borrow_mut().set_scalar(py, index, scalar);
        Ok(())
    }

    fn __delattr__(&self, name: &str) -> PyResult<()> {
        let _ = name;
        Err(PyAttributeError::new_err("Cannot delete field attribute"))
    }

    #[pyo3(name = "CopyFrom")]
    fn copy_from(slf: &Bound<'_, Self>, other_msg: &Bound<'_, PyAny>) -> PyResult<()> {
        if other_msg.is(slf) {
            return Ok(());
        }
        let py = slf.py();
        let full_name = slf.borrow().desc.full_name.clone();
        if let Ok(other) = other_msg.cast::<PyMessage>() {
            let unset = {
                let o = other.borrow();
                if o.desc.full_name != full_name {
                    return Err(not_same_class("CopyFrom", slf.as_any(), other_msg));
                }
                o.parent.is_some()
            };
            // An unset child (of no fields) copies as upb's: the message is cleared.
            let source = if unset {
                None
            } else {
                Some(copy_fields_of(py, other, &ConversionBudget::full())?)
            };
            mark_present(slf)?;
            let mut me = slf.borrow_mut();
            match source {
                Some((slots, unknown)) => me.replace_slots(py, slots, unknown)?,
                None => me.clear_all(py),
            }
            Ok(())
        } else {
            // As upb's, which takes a message of its class of the type only: protobuf's message
            // of the type of its default pool, of its bytes, of any depth, as upb copies one.
            if !is_protobufs_message_of(other_msg, &full_name) {
                return Err(not_same_class("CopyFrom", slf.as_any(), other_msg));
            }
            let (_, bytes) = name_and_bytes(other_msg)
                .map_err(|_| not_same_class("CopyFrom", slf.as_any(), other_msg))?;
            mark_present(slf)?;
            slf.borrow_mut()
                .replace_slots(py, Slots::default(), Vec::new())?;
            codec::merge_unlimited(py, slf, &bytes)
        }
    }

    #[pyo3(name = "MergeFrom")]
    pub(crate) fn merge_from(slf: &Bound<'_, Self>, other_msg: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::merge_from_with(slf, other_msg, &ConversionBudget::full())
    }

    /// As upb's: the message is set in its parent, and the unset children it gave out become
    /// messages of their own.
    #[pyo3(name = "Clear")]
    fn clear(slf: &Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        mark_present(slf)?;
        slf.borrow_mut().clear_all(py);
        Ok(())
    }

    /// As upb's: the message is set in its parent (even for a name it has no field of), a oneof
    /// clears its set member, if it has one, and an unset child that the field gave out becomes a
    /// message of its own.
    #[pyo3(name = "ClearField")]
    fn clear_field(slf: &Bound<'_, Self>, field_name: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        mark_present(slf)?;
        let field_name = FieldName::of(field_name)?;
        let index = {
            let me = slf.borrow();
            match me.desc.oneof_by_name.get(field_name.text()) {
                Some(&oneof) => {
                    let set = me.desc.oneofs[oneof]
                        .fields
                        .iter()
                        .find(|&&i| me.slots.contains(i));
                    match set {
                        Some(&index) => index,
                        None => return Ok(()),
                    }
                }
                None => me.field_index_of(&field_name)?,
            }
        };
        // As upb's, whose map of the field takes a new version, so that an iteration of it in
        // progress fails; the map itself keeps its entries, held by no message.
        let map = match slf.borrow().slots.get(index) {
            Some(Slot::Map(map)) => Some(map.clone_ref(py)),
            _ => None,
        };
        if let Some(map) = map {
            map.bind(py).borrow_mut().new_version();
        }
        slf.borrow_mut().clear_index(py, index);
        Ok(())
    }

    #[pyo3(name = "HasField")]
    fn has_field(slf: &Bound<'_, Self>, field_name: &Bound<'_, PyAny>) -> PyResult<bool> {
        let field_name = FieldName::of(field_name)?;
        let me = slf.borrow();
        if let Some(&oneof) = me.desc.oneof_by_name.get(field_name.text()) {
            return Ok(me.desc.oneofs[oneof]
                .fields
                .iter()
                .any(|&i| me.slots.contains(i)));
        }
        let index = me.field_index_of(&field_name)?;
        let field = &me.desc.fields[index];
        if !field.has_presence() {
            return Err(PyValueError::new_err(format!(
                "Field {} does not have presence.",
                field.full_name
            )));
        }
        Ok(me.slots.contains(index))
    }

    /// `name in message`, as upb's: whether the field `name` is set, as `HasField` tells it.
    /// As upb's: of a `Struct`, whether its fields have the key; of a `ListValue`, whether its
    /// items have the value; of any other message, whether the field `name` is set, as `HasField`
    /// tells it.
    fn __contains__(slf: &Bound<'_, Self>, name: &Bound<'_, PyAny>) -> PyResult<bool> {
        let full_name = slf.borrow().desc.full_name.clone();
        match full_name.as_str() {
            STRUCT => slf.getattr("fields")?.contains(name),
            LIST_VALUE => slf.call_method0("items")?.contains(name),
            _ => Self::has_field(slf, name),
        }
    }

    #[pyo3(name = "WhichOneof")]
    fn which_oneof(
        slf: &Bound<'_, Self>,
        oneof_group: &Bound<'_, PyAny>,
    ) -> PyResult<Option<String>> {
        let oneof_group = FieldName::of(oneof_group)?;
        let me = slf.borrow();
        let Some(&oneof) = me.desc.oneof_by_name.get(oneof_group.text()) else {
            if me.desc.by_name.contains_key(oneof_group.text()) {
                return Err(PyValueError::new_err(format!(
                    "Expected a oneof name, but got field name {}.",
                    oneof_group.display()
                )));
            }
            return Err(me.no_field(&oneof_group.display()));
        };
        Ok(me.desc.oneofs[oneof]
            .fields
            .iter()
            .find(|&&i| me.slots.contains(i))
            .map(|&i| me.desc.fields[i].name.clone()))
    }

    /// The set fields and their values, in the order of their numbers, each of the descriptor of
    /// its field of the class's `DESCRIPTOR`: `protobuf`'s, where `protobuf` imports, as upb's.
    #[pyo3(name = "ListFields")]
    fn list_fields(slf: &Bound<'_, Self>) -> PyResult<Vec<(Py<PyAny>, Py<PyAny>)>> {
        let py = slf.py();
        let me = slf.borrow();
        let protobuf_fields = registered_fields(py, &me.desc.full_name)?;
        let mut out = Vec::new();
        for (index, slot) in me.slots.iter() {
            let field = &me.desc.fields[index];
            let value: Option<Py<PyAny>> = match slot {
                Slot::Scalar(v) => (field.has_presence() || !v.is_default())
                    .then(|| v.to_py(py))
                    .transpose()?,
                Slot::Message(c) => Some(c.clone_ref(py).into_any()),
                Slot::Repeated(c) => {
                    (!c.bind(py).borrow().items.is_empty()).then(|| c.clone_ref(py).into_any())
                }
                Slot::Composite(c) => {
                    (!c.bind(py).borrow().items.is_empty()).then(|| c.clone_ref(py).into_any())
                }
                Slot::Map(c) => {
                    (!c.bind(py).borrow().entries.is_empty()).then(|| c.clone_ref(py).into_any())
                }
            };
            if let Some(value) = value {
                let descriptor = match &protobuf_fields {
                    Some(fields) => fields[index].clone_ref(py),
                    None => field_descriptor_object(py, &me.desc, index)?,
                };
                out.push((descriptor, value));
            }
        }
        Ok(out)
    }

    /// The bytes of the message, of its map entries in the order of their keys (as upb's
    /// `deterministic=True` gives them, so either way here).
    /// The bytes of the message, as upb's: of a message that lacks a required field (of proto2),
    /// upb's `EncodeError` of the fields that it lacks.
    #[pyo3(name = "SerializeToString", signature = (*args, **kwargs))]
    fn serialize_to_string<'py>(
        slf: &Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        check_serialize_arguments(args, kwargs)?;
        let py = slf.py();
        {
            // A message of a type that holds no required field, as all of Spark Connect's, is
            // encoded under the one borrow of the check.
            let m = slf.borrow();
            if !may_lack_required(&m.desc) {
                return Ok(PyBytes::new(py, &codec::encode(py, &m)?));
            }
        }
        check_initialized(slf)?;
        Ok(PyBytes::new(py, &codec::encode(py, &slf.borrow())?))
    }

    #[pyo3(name = "SerializePartialToString", signature = (*args, **kwargs))]
    fn serialize_partial_to_string<'py>(
        slf: &Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        check_serialize_arguments(args, kwargs)?;
        let py = slf.py();
        let bytes = codec::encode(py, &slf.borrow())?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// The length of the bytes of the message, as upb's: of upb's `EncodeError` of a message that
    /// lacks a required field.
    #[pyo3(name = "ByteSize")]
    fn byte_size(slf: &Bound<'_, Self>) -> PyResult<usize> {
        let py = slf.py();
        {
            let m = slf.borrow();
            if !may_lack_required(&m.desc) {
                return codec::encoded_len(py, &m);
            }
        }
        check_initialized(slf)?;
        codec::encoded_len(py, &slf.borrow())
    }

    /// As upb's: `Clear()`, then `MergeFromString`, so that a value that is not bytes leaves the
    /// message cleared.
    #[pyo3(name = "ParseFromString")]
    fn parse_from_string(slf: &Bound<'_, Self>, serialized: &Bound<'_, PyAny>) -> PyResult<usize> {
        Self::clear(slf)?;
        Self::merge_from_string(slf, serialized)
    }

    #[pyo3(name = "MergeFromString")]
    fn merge_from_string(slf: &Bound<'_, Self>, serialized: &Bound<'_, PyAny>) -> PyResult<usize> {
        let py = slf.py();
        mark_present(slf)?;
        let bytes = bytes_of(serialized)?;
        codec::merge(py, slf, &bytes)?;
        Ok(bytes.len())
    }

    /// A message of the class of the bytes `s`, made as a parse makes the messages that it reads,
    /// without the call of the class and of its `__init__`.
    #[classmethod]
    #[pyo3(name = "FromString")]
    fn from_string<'py>(
        cls: &Bound<'py, PyType>,
        s: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyMessage>> {
        let py = cls.py();
        let maker = MessageClass {
            class: Some(cls.clone().unbind()),
            desc: message_type_of(cls)?,
        };
        let message = maker.new_message(py)?;
        // As upb's message of `FromString`, which the garbage collector tracks, as of a call of
        // its class.
        track(&message);
        let bytes = bytes_of(s)?;
        codec::merge(py, &message, &bytes)?;
        Ok(message)
    }

    /// Whether the message sets every required field (of proto2, and of `LEGACY_REQUIRED`), its
    /// messages too, as upb's: the paths of the fields that it lacks are added to `errors`, when it
    /// is given, by its `extend`.
    #[pyo3(name = "IsInitialized", signature = (errors=None))]
    fn is_initialized(slf: &Bound<'_, Self>, errors: Option<&Bound<'_, PyAny>>) -> PyResult<bool> {
        let missing = missing_fields(slf.py(), &slf.borrow());
        if missing.is_empty() {
            return Ok(true);
        }
        if let Some(errors) = errors.filter(|e| !e.is_none()) {
            errors.call_method1("extend", (missing,))?;
        }
        Ok(false)
    }

    /// The paths of the required fields that the message and its messages lack, as upb's:
    /// `name`, `field.name`, `field[0].name` and `map[key].name`.
    #[pyo3(name = "FindInitializationErrors")]
    fn find_initialization_errors(slf: &Bound<'_, Self>) -> Vec<String> {
        missing_fields(slf.py(), &slf.borrow())
    }

    /// As upb's, which discards the unknown fields of the message and of its messages down to
    /// 62 levels below it (`upb_Message_DiscardUnknown` of a maximum depth of 64): those of the
    /// deeper messages stay.
    #[pyo3(name = "DiscardUnknownFields")]
    fn discard_unknown_fields(slf: &Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        mark_present(slf)?;
        slf.borrow_mut().discard_unknown(py, DISCARD_UNKNOWN_DEPTH);
        Ok(())
    }

    #[pyo3(name = "SetInParent")]
    fn set_in_parent(slf: &Bound<'_, Self>) -> PyResult<()> {
        mark_present(slf)
    }

    /// As upb's of a message that declares no extension ranges, as no message of Spark Connect
    /// does: the field is not one of its extensions.
    #[pyo3(name = "HasExtension")]
    fn has_extension(&self, field_descriptor: &Bound<'_, PyAny>) -> PyResult<bool> {
        Err(self.not_an_extension(field_descriptor))
    }

    /// As `HasExtension`: the field is not one of its extensions.
    #[pyo3(name = "ClearExtension")]
    fn clear_extension(&self, field_descriptor: &Bound<'_, PyAny>) -> PyResult<()> {
        Err(self.not_an_extension(field_descriptor))
    }

    /// The extensions of the message: as upb's of a message that declares no extension ranges,
    /// an `AttributeError` of no message (which `__getattr__` keeps).
    #[getter(Extensions)]
    fn extensions(&self) -> PyResult<Py<PyAny>> {
        Err(PyAttributeError::new_err(()))
    }

    fn __richcmp__(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        // As upb's: a `Struct` and a dict, or a `ListValue` and a list, compare of the
        // `_internal_compare` of their well-known type, whose result `!=` gives too.
        if matches!(op, CompareOp::Eq | CompareOp::Ne) {
            let (is_struct, is_list_value) = {
                let m = slf.borrow();
                (m.desc.full_name == STRUCT, m.desc.full_name == LIST_VALUE)
            };
            let compares = (is_struct && other.cast::<PyDict>().is_ok())
                || (is_list_value && other.cast::<pyo3::types::PyList>().is_ok());
            if compares && slf.hasattr("_internal_compare")? {
                return Ok(slf.call_method1("_internal_compare", (other,))?.unbind());
            }
        }
        match op {
            CompareOp::Eq | CompareOp::Ne => {
                // A message of protobuf's of the type, which upb's `==` of its one class of the
                // type compares by its contents: protobuf's class of the type of a plugin's file
                // that keeps it (`relation.scan.table == base_pb2.DeltaTable(...)`), of
                // `message_factory` or `symbol_database`, or of a pickle of upb. It is compared
                // this way when it is on either side, as Python reflects the `==` of protobuf's
                // message, which gives `NotImplemented` of the runtime's.
                let equal = match other.cast::<PyMessage>() {
                    Ok(other) => content_eq(py, slf, other)?,
                    Err(_) => match as_message_of_type(py, slf, other)? {
                        Some(other) => content_eq(py, slf, &other)?,
                        None => return Ok(py.NotImplemented()),
                    },
                };
                (if matches!(op, CompareOp::Eq) {
                    equal
                } else {
                    !equal
                })
                .into_py_any(py)
            }
            _ => Ok(py.NotImplemented()),
        }
    }

    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        text::to_text(slf.py(), slf, text::Style::Upb, false)
    }

    fn __str__(slf: &Bound<'_, Self>) -> PyResult<String> {
        text::to_text(slf.py(), slf, text::Style::Upb, false)
    }

    #[pyo3(name = "UnknownFields")]
    fn unknown_fields(&self) -> PyResult<()> {
        Err(PyNotImplementedError::new_err("unknown field accessor"))
    }

    /// `copy.deepcopy` of the message: a message of its class of the same fields, of any depth, as
    /// upb's.
    fn __deepcopy__<'py>(
        slf: &Bound<'py, Self>,
        memo: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyMessage>> {
        let _ = memo;
        copy_as(
            slf.py(),
            slf,
            Some(&Self::maker_of(slf)),
            &ConversionBudget::full(),
        )
    }

    /// `copy.copy` of the message: a message of its class of the same fields, as upb's copy of its
    /// bytes (of protobuf's `__reduce__`) gives, and so, of a message of more levels than upb's
    /// limit of a parse, upb's `DecodeError` of the parse.
    fn __copy__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyMessage>> {
        let py = slf.py();
        let maker = Self::maker_of(slf);
        if deeper_than(py, &slf.borrow(), codec::depth_limit()) {
            let copied = maker.new_message(py)?;
            track(&copied);
            let bytes = codec::encode(py, &slf.borrow())?;
            codec::merge(py, &copied, &bytes)?;
            return Ok(copied);
        }
        // Tracked by the garbage collector, as upb's copy of a call of its class is.
        let copied = copy_as(py, slf, Some(&maker), &ConversionBudget::full())?;
        track(&copied);
        Ok(copied)
    }

    /// The pickle of the message, as protobuf's `Message.__reduce__` gives it, of which upb's
    /// messages pickle: the class and the state of the bytes of the message, which `__setstate__`
    /// parses, and of a nested type, where protobuf imports, protobuf's
    /// `_InternalConstructMessage` of the full name of the type (see `internal_construct` of the
    /// module).
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let state = Self::__getstate__(slf)?;
        let desc = slf.borrow().desc.clone();
        if desc.containing.is_some() {
            if let Some(construct) = super::internal_construct(py) {
                let args = PyTuple::new(py, [desc.full_name.as_str()])?;
                return PyTuple::new(
                    py,
                    [construct.into_any(), args.into_any(), state.into_any()],
                );
            }
        }
        PyTuple::new(
            py,
            [
                slf.get_type().into_any(),
                PyTuple::empty(py).into_any(),
                state.into_any(),
            ],
        )
    }

    /// As protobuf's: the bytes of the message, of its fields as they are.
    fn __getstate__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDict>> {
        let py = slf.py();
        let state = PyDict::new(py);
        state.set_item(
            "serialized",
            PyBytes::new(py, &codec::encode(py, &slf.borrow())?),
        )?;
        Ok(state)
    }

    /// As protobuf's: the message of the bytes of the state, which `ParseFromString` parses (a
    /// str, of a pickle of Python 2, is latin-1).
    fn __setstate__(slf: &Bound<'_, Self>, state: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut serialized = state.get_item("serialized")?;
        if serialized.cast::<PyBytes>().is_err() {
            serialized = serialized.call_method1("encode", ("latin1",))?;
        }
        Self::parse_from_string(slf, &serialized).map(|_| ())
    }
}
