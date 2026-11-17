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
use pyo3::types::{
    PyByteArray, PyBytes, PyDict, PyMemoryView, PyString, PyTuple, PyType, PyWeakrefMethods,
    PyWeakrefReference,
};
use pyo3::IntoPyObjectExt;

use super::containers::{MapContainer, MapValue, RepeatedComposite, RepeatedScalar};
use super::descriptors::{field_descriptor_object, PyDescriptor};
use super::pool::{pool, FieldDesc, Kind, MessageDesc};
use super::scalar::Scalar;
use super::{codec, grow, text};

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
    /// The set fields, by index.
    pub(crate) slots: Vec<Option<Slot>>,
    /// The unset children that `__getattr__` gave, by the index of their field.
    lazy: Vec<Option<Py<PyMessage>>>,
    /// The parent of an unset child.
    parent: Option<Parent>,
    /// The encoded fields that the message type has no field for, as the message read them.
    pub(crate) unknown: Vec<u8>,
}

fn classes() -> &'static Mutex<HashMap<String, Py<PyType>>> {
    static CLASSES: OnceLock<Mutex<HashMap<String, Py<PyType>>>> = OnceLock::new();
    CLASSES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Set the class of each message, by full name, that the runtime makes the instances of.
#[pyfunction]
pub(crate) fn _proto_register_classes(classes_by_name: &Bound<'_, PyDict>) -> PyResult<()> {
    let mut registered = classes().lock().unwrap_or_else(|e| e.into_inner());
    for (name, class) in classes_by_name.iter() {
        registered.insert(
            name.extract::<String>()?,
            class.cast_into::<PyType>()?.unbind(),
        );
    }
    Ok(())
}

/// The class of the message type `full_name` that `_proto_register_classes` set, if any.
pub(crate) fn registered_class(py: Python<'_>, full_name: &str) -> Option<Py<PyType>> {
    classes()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(full_name)
        .map(|c| c.clone_ref(py))
}

/// A new, empty message of the message type `full_name`, of its registered class.
pub(crate) fn new_instance<'py>(
    py: Python<'py>,
    full_name: &str,
) -> PyResult<Bound<'py, PyMessage>> {
    if let Some(class) = registered_class(py, full_name) {
        return Ok(class.bind(py).call0()?.cast_into::<PyMessage>()?);
    }
    let desc = pool()
        .message(full_name)
        .ok_or_else(|| PyTypeError::new_err(format!("no message type named {full_name}")))?
        .clone();
    Bound::new(py, PyMessage::empty(desc))
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

/// The name of a field (or a oneof) that `HasField`, `ClearField` and `WhichOneof` take, with
/// upb's error of an argument that is not a str.
fn field_name_of(name: &Bound<'_, PyAny>) -> PyResult<String> {
    match name.cast::<PyString>() {
        Ok(name) => Ok(name.to_cow()?.into_owned()),
        Err(_) => Err(PyValueError::new_err(format!(
            "Expected a field name, but got non-string argument {}.",
            name.repr()?
        ))),
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
/// runtime or of protobuf's, or a message whose fields are the items of a mapping.
pub(crate) fn message_from_value<'py>(
    py: Python<'py>,
    full_name: &str,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyMessage>> {
    if let Ok(m) = value.cast::<PyMessage>() {
        if m.borrow().desc.full_name != full_name {
            return Err(not_initializing(py, full_name, value));
        }
        return duplicate(py, m);
    }
    if value.cast::<PyDict>().is_ok() || value.hasattr("keys").unwrap_or(false) {
        let child = new_instance(py, full_name)?;
        child.borrow_mut().init_from_mapping(py, value)?;
        return Ok(child);
    }
    if let Ok((name, bytes)) = name_and_bytes(value) {
        if name == full_name {
            let child = new_instance(py, full_name)?;
            codec::merge(py, &child, &bytes)?;
            return Ok(child);
        }
    }
    Err(not_initializing(py, full_name, value))
}

/// A new message of the type `full_name` of `value`, as a message field of a constructor takes
/// it: as [`message_from_value`] does, and, as upb's, through the `_internal_assign` of a new
/// message of a well-known type whose class assigns values (a `Timestamp`, of a `datetime`), of a
/// value that is neither a mapping nor a message of the type.
fn message_for_field<'py>(
    py: Python<'py>,
    full_name: &str,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyMessage>> {
    let is_mapping = value.cast::<PyDict>().is_ok() || value.hasattr("keys").unwrap_or(false);
    let is_of_type = match value.cast::<PyMessage>() {
        Ok(m) => m.borrow().desc.full_name == full_name,
        Err(_) => value
            .getattr("DESCRIPTOR")
            .and_then(|d| d.getattr("full_name"))
            .and_then(|n| n.extract::<String>())
            .is_ok_and(|n| n == full_name),
    };
    if !is_mapping && !is_of_type && assigns_values(py, full_name) {
        let child = new_instance(py, full_name)?;
        child.call_method1("_internal_assign", (value,))?;
        return Ok(child);
    }
    message_from_value(py, full_name, value)
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
    grow(|| {
        let (full_name, slots, unknown) = {
            let m = message.borrow();
            (
                m.desc.full_name.clone(),
                m.copy_slots(py)?,
                m.unknown.clone(),
            )
        };
        let copy = new_instance(py, &full_name)?;
        {
            let mut c = copy.borrow_mut();
            c.slots = slots;
            c.unknown = unknown;
        }
        Ok(copy)
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
            let still_unset = p.lazy[parent.field]
                .as_ref()
                .is_some_and(|c| c.bind(py).is(&current));
            if !still_unset {
                return Ok(());
            }
            p.lazy[parent.field] = None;
            p.clear_oneof_siblings(py, parent.field);
            p.slots[parent.field] = Some(Slot::Message(current.clone().unbind()));
        }
        current = owner;
    }
}

/// Merge `source` into `target`, as protobuf's `MergeFrom` does: a set scalar replaces the value,
/// a message merges into the target's, and repeated fields and maps add their values.
pub(crate) fn merge_into(
    py: Python<'_>,
    target: &Bound<'_, PyMessage>,
    source: &Bound<'_, PyMessage>,
) -> PyResult<()> {
    if target.is(source) {
        let copy = duplicate(py, source)?;
        return merge_into(py, target, &copy);
    }
    enum Merge {
        Scalar(Scalar),
        Message(Py<PyMessage>),
        Repeated(Vec<Scalar>),
        Composite(Vec<Py<PyMessage>>),
        Map(Py<MapContainer>),
    }
    grow(|| {
        let desc = source.borrow().desc.clone();
        for index in 0..desc.fields.len() {
            let merge = {
                let s = source.borrow();
                match &s.slots[index] {
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
                    Some(Slot::Map(c)) => Merge::Map(c.clone_ref(py)),
                }
            };
            let owner = Some(weak(target)?);
            match merge {
                Merge::Scalar(v) => target.borrow_mut().set_scalar(py, index, v),
                Merge::Message(child) => {
                    let into = target.borrow_mut().child_for_merge(py, index)?;
                    merge_into(py, &into, child.bind(py))?;
                }
                Merge::Repeated(items) => {
                    let container = target.borrow_mut().repeated_scalar(py, index, owner)?;
                    container.bind(py).borrow_mut().items.extend(items);
                }
                Merge::Composite(items) => {
                    let container = target.borrow_mut().composite(py, index, owner)?;
                    for item in items {
                        let copy = duplicate(py, item.bind(py))?;
                        container.bind(py).borrow_mut().items.push(copy.unbind());
                    }
                }
                Merge::Map(map) => {
                    let container = target.borrow_mut().map(py, index, owner)?;
                    MapContainer::merge(py, container.bind(py), map.bind(py))?;
                }
            }
        }
        let unknown = source.borrow().unknown.clone();
        target.borrow_mut().unknown.extend(unknown);
        Ok(())
    })
}

/// Whether `a` and `b` hold the same fields, as protobuf's `==` compares two messages: of the
/// same type, with the same set fields of the same values.
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
        for (index, field) in x.desc.fields.iter().enumerate() {
            if !slot_eq(py, field, &x.slots[index], &y.slots[index])? {
                return Ok(false);
            }
        }
        Ok(x.unknown == y.unknown)
    })
}

fn slot_eq(
    py: Python<'_>,
    field: &FieldDesc,
    a: &Option<Slot>,
    b: &Option<Slot>,
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
        let items = |slot: &Option<Slot>| match slot {
            Some(Slot::Composite(c)) => c
                .bind(py)
                .borrow()
                .items
                .iter()
                .map(|m| m.clone_ref(py))
                .collect(),
            _ => Vec::new(),
        };
        let (x, y): (Vec<Py<PyMessage>>, Vec<Py<PyMessage>>) = (items(a), items(b));
        if x.len() != y.len() {
            return Ok(false);
        }
        for (m, n) in x.iter().zip(&y) {
            if !content_eq(py, m.bind(py), n.bind(py))? {
                return Ok(false);
            }
        }
        return Ok(true);
    }
    if field.repeated {
        let items = |slot: &Option<Slot>| match slot {
            Some(Slot::Repeated(c)) => c.bind(py).borrow().items.clone(),
            _ => Vec::new(),
        };
        return Ok(items(a) == items(b));
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
    let value = |slot: &Option<Slot>| match slot {
        Some(Slot::Scalar(v)) => v.clone(),
        _ => Scalar::default_of(field.kind),
    };
    if field.has_presence() && a.is_some() != b.is_some() {
        return Ok(false);
    }
    Ok(value(a) == value(b))
}

fn map_of(slot: &Option<Slot>) -> Option<&Py<MapContainer>> {
    match slot {
        Some(Slot::Map(m)) => Some(m),
        _ => None,
    }
}

impl PyMessage {
    pub(crate) fn empty(desc: Arc<MessageDesc>) -> Self {
        let n = desc.fields.len();
        PyMessage {
            desc,
            slots: (0..n).map(|_| None).collect(),
            lazy: (0..n).map(|_| None).collect(),
            parent: None,
            unknown: Vec::new(),
        }
    }

    /// Detach the unset child of the field `index` that `__getattr__` gave, so that setting its
    /// fields no longer sets the field.
    fn sever(&mut self, py: Python<'_>, index: usize) {
        if let Some(child) = self.lazy[index].take() {
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
        let desc = self.desc.clone();
        for &member in &desc.oneofs[oneof].fields {
            if member != index {
                self.slots[member] = None;
            }
        }
    }

    pub(crate) fn set_scalar(&mut self, py: Python<'_>, index: usize, value: Scalar) {
        self.clear_oneof_siblings(py, index);
        self.slots[index] = Some(Slot::Scalar(value));
    }

    pub(crate) fn set_message(&mut self, py: Python<'_>, index: usize, child: Py<PyMessage>) {
        self.clear_oneof_siblings(py, index);
        self.sever(py, index);
        self.slots[index] = Some(Slot::Message(child));
    }

    fn clear_index(&mut self, py: Python<'_>, index: usize) {
        self.slots[index] = None;
        self.sever(py, index);
    }

    fn clear_all(&mut self, py: Python<'_>) {
        for index in 0..self.slots.len() {
            self.clear_index(py, index);
        }
        self.unknown.clear();
    }

    /// Replace the fields with `slots` and the unknown fields with `unknown`, as upb's `CopyFrom`
    /// copies a message: an unset child that the message gave out becomes the child of its field
    /// when the copy sets that field, and stays unset (and the field's) when it does not.
    fn replace_slots(
        &mut self,
        py: Python<'_>,
        slots: Vec<Option<Slot>>,
        unknown: Vec<u8>,
    ) -> PyResult<()> {
        for (index, slot) in slots.into_iter().enumerate() {
            self.slots[index] = match (slot, self.lazy[index].take()) {
                (Some(Slot::Message(copied)), Some(stub)) => {
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
                    Some(Slot::Message(stub))
                }
                (slot, stub) => {
                    self.lazy[index] = stub;
                    slot
                }
            };
        }
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
        if let Some(Slot::Message(c)) = &self.slots[index] {
            return Ok(c.bind(py).clone());
        }
        // As upb's: an unset child that the message gave out becomes the child that the merge
        // sets, so that the object it gave out holds the merged fields.
        let child = match self.lazy[index].take() {
            Some(stub) => {
                let stub = stub.into_bound(py);
                stub.try_borrow_mut()?.parent = None;
                stub
            }
            None => new_instance(py, &self.desc.fields[index].type_name)?,
        };
        self.clear_oneof_siblings(py, index);
        self.slots[index] = Some(Slot::Message(child.clone().unbind()));
        Ok(child)
    }

    /// The container of the repeated scalar field `index`, made if absent.
    pub(crate) fn repeated_scalar(
        &mut self,
        py: Python<'_>,
        index: usize,
        owner: Option<Py<PyWeakrefReference>>,
    ) -> PyResult<Py<RepeatedScalar>> {
        if let Some(Slot::Repeated(c)) = &self.slots[index] {
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
        self.slots[index] = Some(Slot::Repeated(container.clone_ref(py)));
        Ok(container)
    }

    /// The container of the repeated message field `index`, made if absent.
    pub(crate) fn composite(
        &mut self,
        py: Python<'_>,
        index: usize,
        owner: Option<Py<PyWeakrefReference>>,
    ) -> PyResult<Py<RepeatedComposite>> {
        if let Some(Slot::Composite(c)) = &self.slots[index] {
            return Ok(c.clone_ref(py));
        }
        let field = &self.desc.fields[index];
        let container = Py::new(
            py,
            RepeatedComposite::new(field.type_name.clone(), Vec::new(), owner),
        )?;
        self.slots[index] = Some(Slot::Composite(container.clone_ref(py)));
        Ok(container)
    }

    /// The map of the map field `index`, made if absent.
    pub(crate) fn map(
        &mut self,
        py: Python<'_>,
        index: usize,
        owner: Option<Py<PyWeakrefReference>>,
    ) -> PyResult<Py<MapContainer>> {
        if let Some(Slot::Map(c)) = &self.slots[index] {
            return Ok(c.clone_ref(py));
        }
        let field = &self.desc.fields[index];
        let map = field
            .map
            .as_ref()
            .ok_or_else(|| PyTypeError::new_err(format!("{} is not a map", field.full_name)))?;
        let container = Py::new(
            py,
            MapContainer::new(field.name.clone(), map.clone(), owner),
        )?;
        self.slots[index] = Some(Slot::Map(container.clone_ref(py)));
        Ok(container)
    }

    fn copy_slots(&self, py: Python<'_>) -> PyResult<Vec<Option<Slot>>> {
        self.slots
            .iter()
            .map(|slot| match slot {
                None => Ok(None),
                Some(slot) => copy_slot(py, slot).map(Some),
            })
            .collect()
    }

    /// Set the fields of the items of `mapping`, as the constructor and a mapping for a message
    /// field do. None leaves a field unset. A mapping of a message field sets the fields of its
    /// message through here too, one level of a recursion as deep as the mappings nest, on a stack
    /// with room for it.
    pub(crate) fn init_from_mapping(
        &mut self,
        py: Python<'_>,
        mapping: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        grow(|| self.set_fields_of(py, mapping))
    }

    fn set_fields_of(&mut self, py: Python<'_>, mapping: &Bound<'_, PyAny>) -> PyResult<()> {
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
                MapContainer::update_from(py, container.bind(py), &value)?;
            } else if field.repeated && field.kind == Kind::Message {
                let mut items = Vec::new();
                for element in value.try_iter().map_err(|_| not_iterable())? {
                    items.push(message_for_field(py, &field.type_name, &element?)?.unbind());
                }
                let container = self.composite(py, index, None)?;
                container.bind(py).borrow_mut().items.extend(items);
            } else if field.repeated {
                let mut items = Vec::new();
                for element in value.try_iter().map_err(|_| not_iterable())? {
                    items.push(Scalar::from_py(
                        field.kind,
                        &field.full_name,
                        &field.type_name,
                        &element?,
                    )?);
                }
                let container = self.repeated_scalar(py, index, None)?;
                container.bind(py).borrow_mut().items.extend(items);
            } else if field.kind == Kind::Message {
                let child = message_for_field(py, &field.type_name, &value)?;
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
            for slot in self.slots.iter().flatten() {
                match slot {
                    Slot::Message(child) => child.bind(py).borrow_mut().discard_unknown(py, depth),
                    Slot::Composite(c) => {
                        for child in &c.bind(py).borrow().items {
                            child.bind(py).borrow_mut().discard_unknown(py, depth);
                        }
                    }
                    Slot::Map(c) => {
                        for (_, value) in &c.bind(py).borrow().entries {
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

    fn field_index(&self, name: &str) -> PyResult<usize> {
        self.desc.by_name.get(name).copied().ok_or_else(|| {
            PyValueError::new_err(format!(
                "Protocol message {} has no \"{name}\" field.",
                self.desc.name
            ))
        })
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

fn copy_slot(py: Python<'_>, slot: &Slot) -> PyResult<Slot> {
    Ok(match slot {
        Slot::Scalar(v) => Slot::Scalar(v.clone()),
        Slot::Message(c) => Slot::Message(duplicate(py, c.bind(py))?.unbind()),
        Slot::Repeated(c) => Slot::Repeated(Py::new(py, c.bind(py).borrow().copy())?),
        Slot::Composite(c) => {
            let c = c.bind(py).borrow();
            let items = c
                .items
                .iter()
                .map(|m| duplicate(py, m.bind(py)).map(Bound::unbind))
                .collect::<PyResult<Vec<_>>>()?;
            Slot::Composite(Py::new(
                py,
                RepeatedComposite::new(c.message_type.clone(), items, None),
            )?)
        }
        Slot::Map(c) => Slot::Map(Py::new(py, c.bind(py).borrow().copy(py)?)?),
    })
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
        let desc = class_descriptor(cls)?;
        if !args.is_empty() {
            return Err(PyTypeError::new_err("No positional arguments allowed"));
        }
        let mut message = PyMessage::empty(desc);
        if let Some(kwargs) = kwargs {
            message.init_from_mapping(cls.py(), kwargs.as_any())?;
        }
        Ok(message)
    }

    fn __getattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let desc = slf.borrow().desc.clone();
        let Some(&index) = desc.by_name.get(name) else {
            // As upb's: the `AttributeError` of the `Extensions` of a message that declares no
            // extension ranges has no message.
            if name == "Extensions" {
                return Err(PyAttributeError::new_err(()));
            }
            return Err(PyAttributeError::new_err(name.to_string()));
        };
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
        if field.kind == Kind::Message {
            {
                let me = slf.borrow();
                if let Some(Slot::Message(c)) = &me.slots[index] {
                    return Ok(c.clone_ref(py).into_any());
                }
                if let Some(c) = &me.lazy[index] {
                    return Ok(c.clone_ref(py).into_any());
                }
            }
            let child = new_instance(py, &field.type_name)?;
            child.borrow_mut().parent = Some(Parent {
                message: weak(slf)?,
                field: index,
            });
            slf.borrow_mut().lazy[index] = Some(child.clone().unbind());
            return Ok(child.into_any().unbind());
        }
        let me = slf.borrow();
        match &me.slots[index] {
            Some(Slot::Scalar(v)) => v.to_py(py),
            _ => Scalar::default_of(field.kind).to_py(py),
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
                let child = Self::__getattr__(slf, name)?;
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
            let other = other.borrow();
            if other.desc.full_name != full_name {
                return Err(not_same_class("CopyFrom", slf.as_any(), other_msg));
            }
            // An unset child (of no fields) copies as upb's: the message is cleared.
            let source = (other.parent.is_none())
                .then(|| Ok::<_, PyErr>((other.copy_slots(py)?, other.unknown.clone())));
            drop(other);
            mark_present(slf)?;
            let mut me = slf.borrow_mut();
            match source {
                Some(copied) => {
                    let (slots, unknown) = copied?;
                    me.replace_slots(py, slots, unknown)?;
                }
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
            let empty = (0..slf.borrow().slots.len()).map(|_| None).collect();
            slf.borrow_mut().replace_slots(py, empty, Vec::new())?;
            codec::merge(py, slf, &bytes)
        }
    }

    #[pyo3(name = "MergeFrom")]
    fn merge_from(slf: &Bound<'_, Self>, other_msg: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        let full_name = slf.borrow().desc.full_name.clone();
        if let Ok(other) = other_msg.cast::<PyMessage>() {
            if other.borrow().desc.full_name != full_name {
                return Err(not_same_class("MergeFrom", slf.as_any(), other_msg));
            }
            mark_present(slf)?;
            merge_into(py, slf, other)
        } else {
            let (name, bytes) = name_and_bytes(other_msg)
                .map_err(|_| not_same_class("MergeFrom", slf.as_any(), other_msg))?;
            if name != full_name {
                return Err(not_same_class("MergeFrom", slf.as_any(), other_msg));
            }
            mark_present(slf)?;
            codec::merge(py, slf, &bytes)
        }
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
        let field_name = field_name_of(field_name)?;
        let index = {
            let me = slf.borrow();
            match me.desc.oneof_by_name.get(field_name.as_str()) {
                Some(&oneof) => {
                    let set = me.desc.oneofs[oneof]
                        .fields
                        .iter()
                        .find(|&&i| me.slots[i].is_some());
                    match set {
                        Some(&index) => index,
                        None => return Ok(()),
                    }
                }
                None => me.field_index(&field_name)?,
            }
        };
        slf.borrow_mut().clear_index(py, index);
        Ok(())
    }

    #[pyo3(name = "HasField")]
    fn has_field(&self, field_name: &Bound<'_, PyAny>) -> PyResult<bool> {
        let field_name = &field_name_of(field_name)?;
        if let Some(&oneof) = self.desc.oneof_by_name.get(field_name.as_str()) {
            return Ok(self.desc.oneofs[oneof]
                .fields
                .iter()
                .any(|&i| self.slots[i].is_some()));
        }
        let index = self.field_index(field_name)?;
        let field = &self.desc.fields[index];
        if !field.has_presence() {
            return Err(PyValueError::new_err(format!(
                "Field {} does not have presence.",
                field.full_name
            )));
        }
        Ok(self.slots[index].is_some())
    }

    /// `name in message`, as upb's: whether the field `name` is set, as `HasField` tells it.
    fn __contains__(&self, name: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.has_field(name)
    }

    #[pyo3(name = "WhichOneof")]
    fn which_oneof(&self, oneof_group: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
        let oneof_group = &field_name_of(oneof_group)?;
        let Some(&oneof) = self.desc.oneof_by_name.get(oneof_group.as_str()) else {
            if self.desc.by_name.contains_key(oneof_group.as_str()) {
                return Err(PyValueError::new_err(format!(
                    "Expected a oneof name, but got field name {oneof_group}."
                )));
            }
            return Err(PyValueError::new_err(format!(
                "Protocol message {} has no \"{oneof_group}\" field.",
                self.desc.name
            )));
        };
        Ok(self.desc.oneofs[oneof]
            .fields
            .iter()
            .find(|&&i| self.slots[i].is_some())
            .map(|&i| self.desc.fields[i].name.clone()))
    }

    #[pyo3(name = "ListFields")]
    fn list_fields(slf: &Bound<'_, Self>) -> PyResult<Vec<(Py<PyAny>, Py<PyAny>)>> {
        let py = slf.py();
        let me = slf.borrow();
        let mut out = Vec::new();
        for &index in &me.desc.order {
            let field = &me.desc.fields[index];
            let value: Option<Py<PyAny>> = match &me.slots[index] {
                None => None,
                Some(Slot::Scalar(v)) => (field.has_presence() || !v.is_default())
                    .then(|| v.to_py(py))
                    .transpose()?,
                Some(Slot::Message(c)) => Some(c.clone_ref(py).into_any()),
                Some(Slot::Repeated(c)) => {
                    (!c.bind(py).borrow().items.is_empty()).then(|| c.clone_ref(py).into_any())
                }
                Some(Slot::Composite(c)) => {
                    (!c.bind(py).borrow().items.is_empty()).then(|| c.clone_ref(py).into_any())
                }
                Some(Slot::Map(c)) => {
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
        Ok(codec::encode(slf.py(), &slf.borrow())?.len())
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
    /// an `AttributeError` of no message (see `__getattr__`).
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
