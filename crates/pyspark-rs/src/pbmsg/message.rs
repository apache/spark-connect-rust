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

/// A class that `_proto_register_classes` set, with the message type of its `DESCRIPTOR`.
struct Registered {
    class: Py<PyType>,
    desc: Arc<MessageDesc>,
}

/// The classes that `_proto_register_classes` set, by the full name of their message type, and
/// the message types of their `DESCRIPTOR`s by the address of the class, which `__new__` reads.
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
/// subclass of `Message`, of the `DESCRIPTOR` of its message type.
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
        let desc = class_descriptor(&class)?;
        found.push((name.extract::<String>()?, class.unbind(), desc));
    }
    let mut registered = classes().lock().unwrap_or_else(|e| e.into_inner());
    for (name, class, desc) in found {
        registered
            .by_type
            .insert(class.as_ptr() as usize, desc.clone());
        registered.by_name.insert(name, Registered { class, desc });
    }
    Ok(())
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

/// The message type of the class `cls`: of a registered class, the one of its `DESCRIPTOR` that
/// the registry holds, else (a subclass of one) of its `DESCRIPTOR`.
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
    let descriptor = cls
        .getattr("DESCRIPTOR")
        .map_err(|_| PyTypeError::new_err("Message is the abstract base of the message classes"))?;
    let descriptor = descriptor.cast::<PyDescriptor>().map_err(|_| {
        PyTypeError::new_err("the DESCRIPTOR of a message class is not one of this runtime")
    })?;
    Ok(descriptor.get().desc.clone())
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

/// A new message of the type `full_name` of `value`: a copy of a message of that type, of this
/// runtime or of protobuf's, or a message whose fields are the items of a mapping, each message
/// that it copies or builds a step of `budget`.
pub(crate) fn message_from_value<'py>(
    py: Python<'py>,
    full_name: &str,
    value: &Bound<'py, PyAny>,
    budget: &ConversionBudget,
) -> PyResult<Bound<'py, PyMessage>> {
    if let Ok(m) = value.cast::<PyMessage>() {
        if m.borrow().desc.full_name != full_name {
            return Err(not_initializing(py, full_name, value));
        }
        return duplicate_with(py, m, budget);
    }
    if value.cast::<PyDict>().is_ok() || value.hasattr("keys").unwrap_or(false) {
        let child = new_instance(py, full_name)?;
        child
            .borrow_mut()
            .init_from_mapping_with(py, value, budget)?;
        return Ok(child);
    }
    if let Ok((name, bytes)) = name_and_bytes(value) {
        if name == full_name {
            let child = new_instance(py, full_name)?;
            codec::merge_in(py, &child, &bytes, budget)?;
            return Ok(child);
        }
    }
    Err(not_initializing(py, full_name, value))
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
            return duplicate_with(py, m, budget);
        }
    }
    let is_mapping = value.cast::<PyDict>().is_ok() || value.hasattr("keys").unwrap_or(false);
    let is_of_type = runtime_message.is_none()
        && value
            .getattr("DESCRIPTOR")
            .and_then(|d| d.getattr("full_name"))
            .and_then(|n| n.extract::<String>())
            .is_ok_and(|n| n == full_name);
    if !is_mapping && !is_of_type && assigns_values(py, full_name) {
        let child = new_instance(py, full_name)?;
        child.call_method1("_internal_assign", (value,))?;
        return Ok(child);
    }
    message_from_value(py, full_name, value, budget)
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

/// A copy of `message`, of its class, whose messages and containers are copies too.
pub(crate) fn duplicate<'py>(
    py: Python<'py>,
    message: &Bound<'py, PyMessage>,
) -> PyResult<Bound<'py, PyMessage>> {
    duplicate_with(py, message, &ConversionBudget::full())
}

/// A slot of a message, read for its copy: its scalars copied, and the messages that it holds,
/// which the copy copies once the message is no longer borrowed.
enum Shallow {
    Copied(Slot),
    Message(Py<PyMessage>),
    Composite(MessageClass, Vec<Py<PyMessage>>),
    Map(MapEntries),
}

/// A copy of `message`, as [`duplicate`] makes it, a step of `budget` for each message copied:
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
        Ok(x.unknown == y.unknown)
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
        return Ok(x == y);
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
    // A scalar that one sets is equal to the other's default, of a field without presence.
    Ok(match (value(a), value(b)) {
        (Some(v), Some(w)) => v == w,
        (Some(v), None) | (None, Some(v)) => {
            !field.has_presence() && *v == Scalar::default_of(field.kind)
        }
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
    /// each value of a repeated field and each item of a map, of every level, so that a constructor
    /// of many values (`Literal.Array(elements=[...])`, of a million messages) lets another thread
    /// take the GIL, and a signal interrupt it, as the copies of the runtime do.
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
                if !value.hasattr("items")? {
                    return Err(not_iterable());
                }
                let container = self.map(py, index, None)?;
                MapContainer::update_from(py, container.bind(py), &value, budget)?;
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
            merge_into_with(py, slf, other, budget)
        } else {
            let (name, bytes) = name_and_bytes(other_msg)
                .map_err(|_| not_same_class("MergeFrom", slf.as_any(), other_msg))?;
            if name != full_name {
                return Err(not_same_class("MergeFrom", slf.as_any(), other_msg));
            }
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
                    _ => Scalar::default_of(field.kind).to_py(py),
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
    #[new]
    #[classmethod]
    #[pyo3(signature = (*args, **kwargs))]
    fn py_new(
        cls: &Bound<'_, PyType>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let desc = message_type_of(cls)?;
        if !args.is_empty() {
            return Err(PyTypeError::new_err("No positional arguments allowed"));
        }
        let mut message = PyMessage::empty(desc);
        if let Some(kwargs) = kwargs {
            message.init_from_mapping(cls.py(), kwargs.as_any())?;
        }
        Ok(message)
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
        match slf.get_type().as_any().getattr(name) {
            Ok(value) => Ok(value.unbind()),
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
            let (name, bytes) = name_and_bytes(other_msg)
                .map_err(|_| not_same_class("CopyFrom", slf.as_any(), other_msg))?;
            if name != full_name {
                return Err(not_same_class("CopyFrom", slf.as_any(), other_msg));
            }
            mark_present(slf)?;
            slf.borrow_mut()
                .replace_slots(py, Slots::default(), Vec::new())?;
            codec::merge(py, slf, &bytes)
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
    fn __contains__(slf: &Bound<'_, Self>, name: &Bound<'_, PyAny>) -> PyResult<bool> {
        Self::has_field(slf, name)
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

    #[pyo3(name = "ListFields")]
    fn list_fields(slf: &Bound<'_, Self>) -> PyResult<Vec<(Py<PyAny>, Py<PyAny>)>> {
        let py = slf.py();
        let me = slf.borrow();
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
                out.push((field_descriptor_object(py, &me.desc, index)?, value));
            }
        }
        Ok(out)
    }

    /// The bytes of the message, of its map entries in the order of their keys (as upb's
    /// `deterministic=True` gives them, so either way here).
    #[pyo3(name = "SerializeToString", signature = (*args, **kwargs))]
    fn serialize_to_string<'py>(
        slf: &Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        check_serialize_arguments(args, kwargs)?;
        let py = slf.py();
        let bytes = codec::encode(py, &slf.borrow())?;
        Ok(PyBytes::new(py, &bytes))
    }

    #[pyo3(name = "SerializePartialToString", signature = (*args, **kwargs))]
    fn serialize_partial_to_string<'py>(
        slf: &Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        Self::serialize_to_string(slf, args, kwargs)
    }

    #[pyo3(name = "ByteSize")]
    fn byte_size(slf: &Bound<'_, Self>) -> PyResult<usize> {
        codec::encoded_len(slf.py(), &slf.borrow())
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

    #[classmethod]
    #[pyo3(name = "FromString")]
    fn from_string<'py>(
        cls: &Bound<'py, PyType>,
        s: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let message = cls.call0()?;
        let bytes = bytes_of(s)?;
        codec::merge(cls.py(), message.cast::<PyMessage>()?, &bytes)?;
        Ok(message)
    }

    #[pyo3(name = "IsInitialized", signature = (errors=None))]
    fn is_initialized(&self, errors: Option<&Bound<'_, PyAny>>) -> bool {
        let _ = errors;
        true
    }

    #[pyo3(name = "FindInitializationErrors")]
    fn find_initialization_errors(&self) -> Vec<String> {
        Vec::new()
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
        match op {
            CompareOp::Eq | CompareOp::Ne => {
                let Ok(other) = other.cast::<PyMessage>() else {
                    return Ok(py.NotImplemented());
                };
                let equal = content_eq(py, slf, other)?;
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

    /// Pickle as protobuf's messages pickle: the class, and the state of the bytes of the
    /// message, which `__setstate__` parses. A nested class pickles by its qualified name, which
    /// the generated modules of protobuf have too, so either runtime loads the pickle of the other.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let py = slf.py();
        Ok((slf.get_type(), PyTuple::empty(py), Self::__getstate__(slf)?))
    }

    fn __getstate__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDict>> {
        let py = slf.py();
        let state = PyDict::new(py);
        state.set_item(
            "serialized",
            PyBytes::new(py, &codec::encode(py, &slf.borrow())?),
        )?;
        Ok(state)
    }

    fn __setstate__(slf: &Bound<'_, Self>, state: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        let mut serialized = state.get_item("serialized")?;
        // As protobuf's: a str, of a pickle of Python 2, is latin-1.
        if serialized.cast::<PyBytes>().is_err() {
            serialized = serialized.call_method1("encode", ("latin1",))?;
        }
        let bytes = bytes_of(&serialized)?;
        slf.borrow_mut().clear_all(py);
        codec::merge(py, slf, &bytes)
    }

    /// The names of the attributes of the message, of its fields too, as upb's `dir()` lists.
    fn __dir__(slf: &Bound<'_, Self>) -> PyResult<Vec<String>> {
        let py = slf.py();
        let object = py.import("builtins")?.getattr("object")?;
        let mut names: Vec<String> = object.call_method1("__dir__", (slf,))?.extract()?;
        // As upb's: not `Extensions`, which a message that declares no extension ranges (as none
        // of Spark Connect does) has not, though its class lists it.
        names.retain(|name| name != "Extensions");
        names.extend(slf.borrow().desc.fields.iter().map(|f| f.name.clone()));
        names.sort();
        names.dedup();
        Ok(names)
    }

    #[pyo3(name = "UnknownFields")]
    fn unknown_fields(&self) -> PyResult<()> {
        Err(PyNotImplementedError::new_err("unknown field accessor"))
    }

    fn __deepcopy__<'py>(
        slf: &Bound<'py, Self>,
        memo: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyMessage>> {
        let _ = memo;
        duplicate(slf.py(), slf)
    }

    fn __copy__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyMessage>> {
        duplicate(slf.py(), slf)
    }
}
