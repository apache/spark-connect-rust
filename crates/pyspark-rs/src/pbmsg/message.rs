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
use std::hash::BuildHasherDefault;
use std::sync::{Arc, Mutex, OnceLock};

use pyo3::basic::CompareOp;
use pyo3::exceptions::{
    PyAttributeError, PyKeyError, PyNotImplementedError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::pyclass_init::PyClassInitializer;
use pyo3::types::{
    PyByteArray, PyBytes, PyDict, PyList, PyMemoryView, PyString, PyTuple, PyType,
    PyWeakrefMethods, PyWeakrefReference,
};
use pyo3::IntoPyObjectExt;

use super::arena::NRef;
use super::containers::{
    Entries, MapContainer, MapEntries, MapValue, RepeatedComposite, RepeatedScalar,
};
use super::descriptors::{field_descriptor_object, PyDescriptor};
use super::native::{self, CopyError, Native, NativeMessages, NativeValue, Value};
use super::pool::{pool, FieldDesc, KeptClass, Kind, MapDesc, MessageDesc};
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
    /// A copy of the value of a message, repeated or map field, held natively, of which the
    /// message makes the object that the field gives the first time that Python reads it (see
    /// [`native`] and [`PyMessage::materialize`]). The slots of a native message hold scalars and
    /// native values only.
    Native(Value),
}

impl Slot {
    /// Release the slot, its object as a `Bound`, which knows that the thread is attached (as a
    /// `Py` reads a thread local to know it).
    pub(crate) fn release(self, py: Python<'_>) {
        match self {
            Slot::Message(m) => drop(m.into_bound(py)),
            Slot::Repeated(c) => drop(c.into_bound(py)),
            Slot::Composite(c) => drop(c.into_bound(py)),
            Slot::Map(m) => drop(m.into_bound(py)),
            Slot::Scalar(_) | Slot::Native(_) => {}
        }
    }

    /// The slot of the same object, of an object (of a message, container or map).
    pub(crate) fn clone_slot(&self, py: Python<'_>) -> Slot {
        match self {
            Slot::Message(m) => Slot::Message(m.clone_ref(py)),
            Slot::Repeated(c) => Slot::Repeated(c.clone_ref(py)),
            Slot::Composite(c) => Slot::Composite(c.clone_ref(py)),
            Slot::Map(m) => Slot::Map(m.clone_ref(py)),
            Slot::Scalar(v) => Slot::Scalar(v.clone()),
            Slot::Native(v) => Slot::Native(v.clone()),
        }
    }
}

/// The fields of a message: its type, its set fields and the encoded fields that its type has no
/// field for, as the message read them. An object of a message holds them, and so does a native
/// message (see [`Native`]).
pub struct Fields {
    pub(crate) desc: &'static MessageDesc,
    /// The set fields.
    pub(crate) slots: Slots,
    /// The encoded fields that the message type has no field for, as the message read them.
    pub(crate) unknown: Vec<u8>,
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

    pub(crate) fn get_mut(&mut self, index: usize) -> Option<&mut Slot> {
        match self {
            Slots::Empty => None,
            Slots::One((i, slot)) => (*i == index).then_some(slot),
            Slots::Many(entries) => entries
                .iter_mut()
                .find(|(i, _)| *i == index)
                .map(|(_, slot)| slot),
        }
    }

    /// The set fields, by the index of their field, to change their values.
    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = (usize, &mut Slot)> {
        let entries: &mut [(usize, Slot)] = match self {
            Slots::Empty => &mut [],
            Slots::One(entry) => std::slice::from_mut(entry),
            Slots::Many(entries) => entries,
        };
        entries.iter_mut().map(|(i, slot)| (*i, slot))
    }

    /// The slots of the values of `slot` of each, of the same fields.
    pub(crate) fn map(&self, mut slot: impl FnMut(&Slot) -> Slot) -> Slots {
        match self {
            Slots::Empty => Slots::Empty,
            Slots::One((i, s)) => Slots::One((*i, slot(s))),
            Slots::Many(entries) => {
                Slots::Many(entries.iter().map(|(i, s)| (*i, slot(s))).collect())
            }
        }
    }

    /// Set the slot of the field `index` of `desc`, and give back the slot that it replaces.
    #[inline]
    pub(crate) fn set(&mut self, desc: &MessageDesc, index: usize, slot: Slot) -> Option<Slot> {
        // The first field of a message, as a parse most often sets it: the slot is written in
        // its place where `set` is inlined, of no copy of it.
        if let Slots::Empty = self {
            *self = Slots::One((index, slot));
            return None;
        }
        self.set_more(desc, index, slot)
    }

    /// [`Slots::set`] of slots that are not empty.
    fn set_more(&mut self, desc: &MessageDesc, index: usize, slot: Slot) -> Option<Slot> {
        // Of a field of a number after those of the set fields, as a parse most often sets its
        // fields, appended, of no search: no set field is of its number.
        if let Slots::Many(entries) = self {
            if let Some(&(last, _)) = entries.last() {
                if desc.fields[last].number < desc.fields[index].number {
                    entries.push((index, slot));
                    return None;
                }
            }
        }
        if let Some(old) = self.get_mut(index) {
            return Some(std::mem::replace(old, slot));
        }
        if let Slots::Empty = self {
            *self = Slots::One((index, slot));
            return None;
        }
        // Of room for four slots from the second, as a message of more than one field most often
        // sets a few, of no growth of the vector of one.
        let mut entries = match std::mem::take(self) {
            Slots::One(entry) => {
                let mut entries = Vec::with_capacity(4);
                entries.push(entry);
                entries
            }
            slots => slots.into_vec(),
        };
        let number = desc.fields[index].number;
        let at = entries
            .iter()
            .rposition(|(i, _)| desc.fields[*i].number < number)
            .map_or(0, |at| at + 1);
        entries.insert(at, (index, slot));
        *self = Slots::Many(entries);
        None
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

/// An empty container of the repeated or map field `field`, which holds its message `owner`
/// weakly, if it is given: of the kind and class that the container of the field is made of.
fn new_container(
    py: Python<'_>,
    field: &FieldDesc,
    owner: Option<Py<PyWeakrefReference>>,
) -> PyResult<Slot> {
    if let Some(map) = &field.map {
        let container = MapContainer::new(field.name.clone(), map.clone(), owner);
        return Ok(Slot::Map(container.into_object(py)?));
    }
    if field.kind == Kind::Message {
        let class = MessageClass::of(py, &field.type_name)?;
        let container = RepeatedComposite::new(class, Vec::new(), owner);
        return Ok(Slot::Composite(Py::new(py, container)?));
    }
    let container = RepeatedScalar::new(
        field.kind,
        field.full_name.clone(),
        field.type_name.clone(),
        Vec::new(),
        owner,
    );
    Ok(Slot::Repeated(Py::new(py, container)?))
}

#[pyclass(
    subclass,
    weakref,
    module = "pyspark.sql.connect.proto",
    name = "Message"
)]
pub struct PyMessage {
    /// The type, the set fields and the unknown fields of the message, which the message gives as
    /// its own (`message.slots`, of [`Deref`](std::ops::Deref)).
    pub(crate) fields: Fields,
    /// The unset children that `__getattr__` gave, each with the index of its field.
    lazy: Vec<(usize, Py<PyMessage>)>,
    /// The parent of an unset child, held weakly, and the index of its field (`parent_field`),
    /// which [`PyMessage::take_parent`] gives as a [`Parent`]: apart, so that the two and `clean`
    /// take the room of a `Parent`, and the object of a message that of 144 bytes.
    parent: Option<Py<PyWeakrefReference>>,
    parent_field: u32,
    /// The [`clean_epoch`] at which the message was made, of a message that held no native value
    /// at any level below it then (0 of none): a message made empty, which a read made of no
    /// native value (one of a value made whole, see `native::Whole`), as most of a read of a parse
    /// are, or that Python code made. A native value set in the message unmarks it, and a value
    /// set below it otherwise begins another epoch (see [`clean_epoch`]): so a message of the
    /// epoch now holds none of a slab of a parse, and [`PyMessage::unpin`] passes it by.
    clean: u32,
}

impl std::ops::Deref for PyMessage {
    type Target = Fields;

    fn deref(&self) -> &Fields {
        &self.fields
    }
}

impl std::ops::DerefMut for PyMessage {
    fn deref_mut(&mut self) -> &mut Fields {
        &mut self.fields
    }
}

/// A message released releases the objects of its fields as [`unpin_held`] does: a message, a
/// container or a map of a field that Python holds still keeps no slab of a parse alive, as a
/// message of a container released does (see `containers::release_held`): of a read of the mark
/// of the message only, inlined, of a message marked (see `PyMessage::clean`), as most of a read
/// are. (A `Drop` of its own makes the drop of a message call the drop of its fields apart, rather
/// than inline it: about 4 ns of each message released, 3% to 4% of the release of a parse of
/// 400,000 messages read whole.)
impl Drop for PyMessage {
    #[inline(always)]
    fn drop(&mut self) {
        // A message marked of the epoch now holds no native value of a slab below it, as most of
        // the messages of a read do, and so none that an object of its fields holds.
        if !self.is_clean(clean_epoch()) {
            unpin_fields_held(&self.fields.slots);
        }
    }
}

/// [`unpin_held`] of the objects of `slots`, the fields of a message released (see its `Drop`).
#[inline(never)]
fn unpin_fields_held(slots: &Slots) {
    for (_, slot) in slots.iter() {
        if let Some(object) = object_of(slot) {
            // SAFETY: of a live object, which the slot holds; a message holds objects only as an
            // object (its fields are set of the methods of the object), whose deallocation, which
            // drops it, is of the thread attached.
            if unsafe { pyo3::ffi::Py_REFCNT(object) } > 1 {
                debug_assert!(is_attached());
                unpin_below(unsafe { Python::assume_attached() }, slot);
            }
        }
    }
}

#[cfg(not(windows))]
extern "C" {
    /// Whether this thread holds the GIL: of the C API of CPython, not of its limited API, of
    /// which the extension is built; the interpreter that loads the extension has it (as
    /// `gc.rs` reads it). Not on Windows, where an extension of the limited API links against the
    /// DLL of that API only.
    fn PyGILState_Check() -> std::os::raw::c_int;
}

/// Whether the thread is attached to the interpreter, of CPython's check (see [`if_attached`]).
#[cfg(not(windows))]
fn is_attached() -> bool {
    // SAFETY: `PyGILState_Check` may be called of any thread at any time.
    unsafe { PyGILState_Check() != 0 }
}

/// Whether the thread is attached to the interpreter: of no check on Windows (see
/// [`if_attached`]).
#[cfg(windows)]
fn is_attached() -> bool {
    true
}

/// Run `release` of the token of the thread, if the thread is attached to the interpreter: of the
/// drop of a value that holds objects, which, of the deallocation of an object (whose thread is
/// attached), releases them as `Bound`s, of no read of PyO3's thread local, and of a value dropped
/// by a thread that is not attached leaves them as they are, whose release PyO3 defers until a
/// thread attaches. Of CPython's check (`PyGILState_Check`), which takes no lock, where
/// `Python::try_attach` locks PyO3's pool of the references to release later at each call (of
/// about 50 ns of each message released); on Windows, which has no such check of the limited API,
/// of `Python::try_attach`, which attaches the thread if it can. (CPython's check says yes of any
/// thread once a subinterpreter is made, which no thread that drops such a value runs.)
pub(crate) fn if_attached(release: impl FnOnce(Python<'_>)) {
    #[cfg(not(windows))]
    {
        // SAFETY: `PyGILState_Check` may be called of any thread at any time, and the token is of
        // a thread that is attached.
        if unsafe { PyGILState_Check() } != 0 {
            release(unsafe { Python::assume_attached() });
        }
    }
    #[cfg(windows)]
    {
        let _ = Python::try_attach(release);
    }
}

/// A class that `_proto_register_classes` set, with its message type and, of a class whose
/// `DESCRIPTOR` is `protobuf`'s (as the classes of the runtime have where `protobuf` imports, as
/// upb's do), `protobuf`'s descriptors of its fields, in the order of the fields of the message
/// type, which `ListFields` gives: read of the `DESCRIPTOR` once `ListFields` needs them (see
/// `registered_fields`), not as the class registers, as many classes register at once.
struct Registered {
    class: Py<PyType>,
    desc: &'static MessageDesc,
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
    /// The type of each class, by the address of the class, of a hash of a multiplication, which a
    /// parse looks up for each `FromString`, rather than SipHash.
    by_type: HashMap<usize, &'static MessageDesc, BuildHasherDefault<AddressHasher>>,
    /// The classes that each full name had before a registration gave it another (or none, of a
    /// name that had none), each with the [`generation`] that the registration began, in their
    /// order: the class of a name at a generation, which [`MessageClass::at`] gives.
    replaced: HashMap<String, Vec<Replaced>>,
    /// The records of the classes that each message type kept (see `MessageDesc::class`), by the
    /// address of the type: one of each class (and of none) that the type had at a read, which a
    /// later read of the type of the same class keeps again (see [`keep_class`]), so that the
    /// records are as many as the classes that a type had, not as the registrations.
    kept: HashMap<usize, Vec<&'static KeptClass>, BuildHasherDefault<AddressHasher>>,
}

/// The hasher of the addresses of [`Registry::by_type`] and [`Registry::kept`]: a multiplication
/// by an odd constant of the address without its low 4 bits, which the alignment of objects makes
/// 0, so that the low bits of the product, which pick the bucket of an address, differ as the
/// addresses do.
#[derive(Default)]
struct AddressHasher(u64);

impl std::hash::Hasher for AddressHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = (self.0.rotate_left(8) ^ u64::from(byte)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
    }

    fn write_usize(&mut self, address: usize) {
        self.0 = ((address >> 4) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

/// A class that a registration replaced (none, of a name that had none), with the [`generation`]
/// that the registration began.
type Replaced = (u64, Option<Py<PyType>>);

fn classes() -> &'static Mutex<Registry> {
    static CLASSES: OnceLock<Mutex<Registry>> = OnceLock::new();
    CLASSES.get_or_init(|| Mutex::new(Registry::default()))
}

/// The objects and the native messages of a message that outlives its container, up to which
/// [`PyMessage::unpin`] moves its native values out of their slabs.
const UNPIN_STEPS: usize = 1024;

/// The epoch of the marks of the messages that hold no native value below them (see
/// `PyMessage::clean`), from 1: one more at the end of each operation that sets native values
/// below a message other than in the message itself, which a mark of the message would not see,
/// so that it unmarks every message made before. Those are a merge into a message
/// (`merge_native`, and a parse into a message that holds fields, see `codec::merge_with`) and a
/// copy that shares the native values of a message, which the message that takes the copy then
/// holds (see `native::copy_parts`). A parse into an empty message begins none: the native
/// values that it sets are those of a parse of its own, all below the message, whose slabs no
/// message apart from it holds, which a mark of a message above it may pass by.
static CLEAN_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// The epoch of the marks of the messages now (see [`CLEAN_EPOCH`]).
#[inline]
pub(crate) fn clean_epoch() -> u64 {
    CLEAN_EPOCH.load(std::sync::atomic::Ordering::Acquire)
}

/// Begin the next epoch of the marks of the messages (see [`CLEAN_EPOCH`]), once native values
/// are set below a message.
pub(crate) fn unmark_all() {
    CLEAN_EPOCH.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
}

/// The mark of a message made at the epoch `epoch` (see `PyMessage::clean`), of 32 bits: past the
/// last epoch of 32 bits, that of no epoch, so that a mark is never of an epoch after its own.
fn mark_of(epoch: u64) -> u32 {
    u32::try_from(epoch).unwrap_or(u32::MAX)
}

/// The walks of [`PyMessage::unpin`] and the objects that they visited, which
/// `_proto_unpin_counts` of the tests gives.
static UNPIN_COUNTS: [std::sync::atomic::AtomicUsize; 2] = [
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
];

/// The walks of [`PyMessage::unpin`] so far and the objects that they visited, of the tests.
#[pyfunction]
pub(crate) fn _proto_unpin_counts() -> (usize, usize) {
    use std::sync::atomic::Ordering::Relaxed;
    (UNPIN_COUNTS[0].load(Relaxed), UNPIN_COUNTS[1].load(Relaxed))
}

/// A walk of [`PyMessage::unpin`]: the steps that it has left of [`UNPIN_STEPS`], the epoch of the
/// marks of the messages that it passes by (see `PyMessage::clean`), the objects that it visited,
/// and the messages left to visit, of one loop, of no recursion.
struct Unpin {
    steps: usize,
    epoch: u64,
    visited: usize,
    left: Vec<Py<PyMessage>>,
}

impl Unpin {
    fn new() -> Self {
        Unpin {
            steps: UNPIN_STEPS,
            epoch: clean_epoch(),
            visited: 0,
            left: Vec::new(),
        }
    }

    /// Add the objects below `slot` to the messages left to visit: the message of a message
    /// field, the messages of a container or the messages of the values of a map.
    fn add_below(&mut self, py: Python<'_>, slot: &Slot) {
        match slot {
            Slot::Message(child) => self.left.push(child.clone_ref(py)),
            Slot::Composite(c) => {
                if let Ok(c) = c.bind(py).try_borrow() {
                    self.left.extend(c.items.iter().map(|m| m.clone_ref(py)));
                }
            }
            Slot::Map(c) => {
                if let Ok(c) = c.bind(py).try_borrow() {
                    for (_, value) in c.entries.iter() {
                        if let MapValue::Message(m) = value {
                            self.left.push(m.clone_ref(py));
                        }
                    }
                }
            }
            Slot::Scalar(_) | Slot::Repeated(_) | Slot::Native(_) => {}
        }
    }

    /// Visit the messages left, and those below them (see [`PyMessage::unpin`]): a message marked
    /// of the epoch of the walk, which holds no native value below it, of no step.
    fn run(&mut self, py: Python<'_>) {
        while let Some(message) = self.left.pop() {
            // Released as a `Bound`, which knows that the thread is attached, with no read of a
            // thread local.
            let message = message.into_bound(py);
            if self.steps == 0 {
                break;
            }
            let Ok(mut m) = message.try_borrow_mut() else {
                continue;
            };
            if m.is_clean(self.epoch) {
                continue;
            }
            self.steps -= 1;
            self.visited += 1;
            for (_, slot) in m.slots.iter_mut() {
                if let Slot::Native(value) = slot {
                    let count = value.count() as usize;
                    if count > self.steps {
                        self.steps = 0;
                        break;
                    }
                    self.steps -= count;
                    value.unpin();
                } else {
                    self.add_below(py, slot);
                }
            }
        }
        for message in self.left.drain(..) {
            drop(message.into_bound(py));
        }
        use std::sync::atomic::Ordering::Relaxed;
        UNPIN_COUNTS[0].fetch_add(1, Relaxed);
        UNPIN_COUNTS[1].fetch_add(self.visited, Relaxed);
    }
}

/// Move the native values below the object of `slot` (a message, or the messages of a container
/// or of a map), which a message clears, replaces or releases, out of the slabs of their parse, if
/// Python holds the object still (see [`PyMessage::unpin`]): so that a child kept of a message
/// released, or of a field cleared (`ClearField`, `Clear`, `CopyFrom`, a member of a oneof that
/// another replaces), keeps no slab of the parse alive, as a message kept of a container does
/// (see `containers::release_held`).
#[inline]
pub(crate) fn unpin_held(py: Python<'_>, slot: &Slot) {
    if let Some(object) = object_of(slot) {
        // SAFETY: of a live object, which `slot` holds.
        if unsafe { pyo3::ffi::Py_REFCNT(object) } > 1 {
            unpin_below(py, slot);
        }
    }
}

/// [`unpin_held`] of the object of `slot`, which Python holds.
#[cold]
#[inline(never)]
fn unpin_below(py: Python<'_>, slot: &Slot) {
    let mut walk = Unpin::new();
    walk.add_below(py, slot);
    walk.run(py);
}

/// The object of `slot`, of a message, a container of messages or a map (whose messages
/// [`unpin_held`] walks).
#[inline]
fn object_of(slot: &Slot) -> Option<*mut pyo3::ffi::PyObject> {
    match slot {
        Slot::Message(m) => Some(m.as_ptr()),
        Slot::Composite(c) => Some(c.as_ptr()),
        Slot::Map(c) => Some(c.as_ptr()),
        Slot::Scalar(_) | Slot::Repeated(_) | Slot::Native(_) => None,
    }
}

/// Whether a slot holds a native value of messages (of a slab of a parse, it may be), which a
/// message that holds it is not marked of (see `PyMessage::clean`).
fn holds_native_messages(slot: &Slot) -> bool {
    matches!(
        slot,
        Slot::Native(Value::Message(_) | Value::Messages(_) | Value::Map(_))
    )
}

/// The generation of the classes of the messages: one more for each `_proto_register_classes`
/// that gives a full name another class (or a class, of one that had none). A native copy of a
/// message keeps the generation that it was made at, so that the objects that it makes when Python
/// reads it are of the classes that the copy made them of before (see [`MessageClass::at`]).
static GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The generation of the classes of the messages now (see [`GENERATION`]).
pub(crate) fn generation() -> u64 {
    GENERATION.load(std::sync::atomic::Ordering::Acquire)
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
            .ok_or_else(|| PyKeyError::new_err(name.clone()))?;
        found.push((name, class.unbind(), desc));
    }
    let mut registered = classes().lock().unwrap_or_else(|e| e.into_inner());
    // The generation that the registration begins if it changes the class of a name.
    let next = generation() + 1;
    let mut changed = false;
    for (name, class, desc) in found {
        registered.by_type.insert(class.as_ptr() as usize, desc);
        let previous = registered
            .by_name
            .get(&name)
            .map(|r| r.class.clone_ref(classes_by_name.py()));
        if previous.as_ref().is_none_or(|p| !p.is(&class)) {
            changed = true;
            // Before the generation begins, so that a read of the class that the type keeps of
            // the generation finds it changed (see `kept_class`).
            desc.class_changed
                .store(next, std::sync::atomic::Ordering::Release);
            registered
                .replaced
                .entry(name.clone())
                .or_default()
                .push((next, previous));
        }
        registered.by_name.insert(
            name,
            Registered {
                class,
                desc,
                fields: Arc::default(),
            },
        );
    }
    if changed {
        GENERATION.store(next, std::sync::atomic::Ordering::Release);
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
        (r.class.clone_ref(py), r.desc, r.fields.clone())
    };
    let fields = protobuf_fields_of(class.bind(py), desc)?;
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
    desc: &'static MessageDesc,
}

impl MessageClass {
    pub(crate) fn clone_ref(&self, py: Python<'_>) -> MessageClass {
        MessageClass {
            class: self.class.as_ref().map(|c| c.clone_ref(py)),
            desc: self.desc,
        }
    }

    /// Whether `self` and `other` make the messages of one class (of one message type, of no class).
    pub(crate) fn is_same(&self, other: &MessageClass) -> bool {
        match (&self.class, &other.class) {
            (Some(a), Some(b)) => a.as_ptr() == b.as_ptr(),
            (None, None) => std::ptr::eq(self.desc, other.desc),
            _ => false,
        }
    }

    /// The maker of the messages of the type `desc` of no class, of the tests.
    #[cfg(test)]
    pub(crate) fn of_no_class(desc: &'static MessageDesc) -> MessageClass {
        MessageClass { class: None, desc }
    }

    pub(crate) fn of(py: Python<'_>, full_name: &str) -> PyResult<MessageClass> {
        let registry = classes().lock().unwrap_or_else(|e| e.into_inner());
        MessageClass::of_locked(py, full_name, &registry)
    }

    /// [`MessageClass::of`] of `registry`, held locked.
    fn of_locked(py: Python<'_>, full_name: &str, registry: &Registry) -> PyResult<MessageClass> {
        if let Some(r) = registry.by_name.get(full_name) {
            return Ok(MessageClass {
                class: Some(r.class.clone_ref(py)),
                desc: r.desc,
            });
        }
        let desc = pool()
            .message(full_name)
            .ok_or_else(|| PyTypeError::new_err(format!("no message type named {full_name}")))?;
        Ok(MessageClass { class: None, desc })
    }

    /// The maker of the messages of the type `desc`, as [`MessageClass::of`] gives it: of the
    /// class that `desc` keeps of the [`generation`] now (see `MessageDesc::class`), which the
    /// registry gives once a generation, rather than of the lookup of the name of the type in the
    /// registry, under its lock, that each message of a parse would take.
    pub(crate) fn now(py: Python<'_>, desc: &'static MessageDesc) -> PyResult<MessageClass> {
        if let Some(class) = kept_class(desc, generation()) {
            return Ok(MessageClass {
                class: class.map(|c| c.clone_ref(py)),
                desc,
            });
        }
        // A type that is not the pool's type of its name, which no message has, is made as `of`
        // makes it, kept by no type.
        if !is_pool_type(desc) {
            return MessageClass::of(py, &desc.full_name);
        }
        // The class and the generation of one read of the registry, which a registration changes
        // under its lock (and registers the pool's type of the name with the class).
        let mut registry = classes().lock().unwrap_or_else(|e| e.into_inner());
        Ok(MessageClass {
            class: class_now(py, desc, &mut registry),
            desc,
        })
    }

    /// The maker of the messages of the type `desc` of the [`generation`] `at`: as
    /// [`MessageClass::now`] gives it now, of the class registered of the type then, which a later
    /// registration may have replaced since (see `Registry::replaced`).
    pub(crate) fn at(
        py: Python<'_>,
        desc: &'static MessageDesc,
        at: u64,
    ) -> PyResult<MessageClass> {
        if let Some(class) = kept_class(desc, at) {
            return Ok(MessageClass {
                class: class.map(|c| c.clone_ref(py)),
                desc,
            });
        }
        // The generation is read under the lock of the registry, which a registration holds
        // while it changes the classes and begins the next generation, so that the class read is
        // that of the generation `at`, also of threads that run at once (of no GIL).
        let mut registry = classes().lock().unwrap_or_else(|e| e.into_inner());
        if at == generation() {
            // As `now` makes it.
            if !is_pool_type(desc) {
                return MessageClass::of_locked(py, &desc.full_name, &registry);
            }
            return Ok(MessageClass {
                class: class_now(py, desc, &mut registry),
                desc,
            });
        }
        let replaced = registry.replaced.get(&desc.full_name).and_then(|changes| {
            changes
                .iter()
                .find(|(since, _)| *since > at)
                .map(|(_, previous)| previous.as_ref().map(|c| c.clone_ref(py)))
        });
        let class = match replaced {
            Some(class) => class,
            None => registry
                .by_name
                .get(&desc.full_name)
                .map(|r| r.class.clone_ref(py)),
        };
        let desc = match registry.by_name.get(&desc.full_name) {
            Some(r) if class.is_some() => r.desc,
            _ => desc,
        };
        Ok(MessageClass { class, desc })
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
        new_message_of(py, self.desc, self.class.as_ref())
    }
}

/// The class of the message type `desc` of the [`generation`] `at` (none, of no class), if it is
/// the one that `desc` keeps (see `MessageDesc::class`): of no registration that gave the type
/// another class since the earlier of the generation that it was kept of and `at`.
fn kept_class(desc: &MessageDesc, at: u64) -> Option<Option<&'static Py<PyType>>> {
    use std::sync::atomic::Ordering::Acquire;
    // SAFETY: a record of a class that a type kept is never released (see `MessageDesc::class`).
    let kept = unsafe { desc.class.load(Acquire).as_ref() }?;
    // The generation of the record is read before the change of the class. The record holds the
    // class of the type of each generation that it was kept of, the last of which, read here, a
    // read of the registry kept under its lock after the registrations up to it (see
    // `keep_class`), whose changes the change read next sees: so of a change not after that
    // generation, nor after `at`, the class of the type is the same from the change on, at both.
    let kept_at = kept.at.load(Acquire);
    let changed = desc.class_changed.load(Acquire);
    (changed <= kept_at && changed <= at).then_some(kept.class.as_ref())
}

/// Whether `desc` is the pool's type of its name, which the messages of the runtime are of.
fn is_pool_type(desc: &'static MessageDesc) -> bool {
    pool()
        .message(&desc.full_name)
        .is_some_and(|d| std::ptr::eq(d, desc))
}

/// The class registered now of `desc`, a type of the pool, of `registry` (held locked), which the
/// type keeps of the generation now (see [`keep_class`]).
fn class_now(
    py: Python<'_>,
    desc: &'static MessageDesc,
    registry: &mut Registry,
) -> Option<Py<PyType>> {
    let class = registry
        .by_name
        .get(&desc.full_name)
        .map(|r| r.class.clone_ref(py));
    keep_class(py, registry, desc, generation(), class.as_ref());
    class
}

/// Keep `class` as the class of the type `desc` of the generation `at`, the generation now (see
/// `MessageDesc::class`), unless the type keeps one of that generation already: a type keeps the
/// class of its first read, and of the first read after a registration that gave it another
/// class, so that the messages of the type are made of no lookup of the registry. Of `registry`,
/// held locked, which keeps a record of each class of each type (see `Registry::kept`): a class
/// that the type had before is kept again of its record, of the later generation, so that the
/// records, which are never released, are as many as the classes of the type, not as the
/// registrations that changed it.
fn keep_class(
    py: Python<'_>,
    registry: &mut Registry,
    desc: &'static MessageDesc,
    at: u64,
    class: Option<&Py<PyType>>,
) {
    use std::sync::atomic::Ordering;
    let current = desc.class.load(Ordering::Acquire);
    // SAFETY: a record of a class that a type kept is never released.
    if unsafe { current.as_ref() }.is_some_and(|kept| kept.at.load(Ordering::Acquire) >= at) {
        return;
    }
    let records = registry
        .kept
        .entry(std::ptr::from_ref(desc) as usize)
        .or_default();
    let of_class = |kept: &KeptClass| match (&kept.class, class) {
        (Some(a), Some(b)) => a.as_ptr() == b.as_ptr(),
        (None, None) => true,
        _ => false,
    };
    let record = match records.iter().find(|kept| of_class(kept)) {
        Some(&kept) => {
            // Of the later generation before the type keeps it again, which a read of the record
            // (see `kept_class`) sees then.
            kept.at.fetch_max(at, Ordering::AcqRel);
            kept
        }
        None => {
            let kept: &'static KeptClass = Box::leak(Box::new(KeptClass {
                at: std::sync::atomic::AtomicU64::new(at),
                class: class.map(|c| c.clone_ref(py)),
            }));
            records.push(kept);
            kept
        }
    };
    // Of the lock of the registry, which every keep of a class holds.
    desc.class
        .store(std::ptr::from_ref(record).cast_mut(), Ordering::Release);
}

/// The records of the classes that the message type `full_name` kept (see `Registry::kept`), of
/// the tests.
#[pyfunction]
pub(crate) fn _proto_kept_classes(full_name: &str) -> PyResult<usize> {
    let desc = pool()
        .message(full_name)
        .ok_or_else(|| PyKeyError::new_err(full_name.to_owned()))?;
    let registry = classes().lock().unwrap_or_else(|e| e.into_inner());
    Ok(registry
        .kept
        .get(&(std::ptr::from_ref(desc) as usize))
        .map_or(0, Vec::len))
}

/// A new, empty message of the type `desc` of the [`generation`] `at`, as
/// [`MessageClass::at`] makes it, of the class that the type keeps (see `MessageDesc::class`)
/// when that is the class of the generation, of no lookup of the registry.
pub(crate) fn new_message_at<'py>(
    py: Python<'py>,
    desc: &'static MessageDesc,
    at: u64,
) -> PyResult<Bound<'py, PyMessage>> {
    if let Some(class) = kept_class(desc, at) {
        return new_message_of(py, desc, class);
    }
    MessageClass::at(py, desc, at)?.new_message(py)
}

/// A new, empty message of the type `desc`, of `class` (see [`MessageClass::new_message`]), or of
/// the runtime's class of messages when it has none.
fn new_message_of<'py>(
    py: Python<'py>,
    desc: &'static MessageDesc,
    class: Option<&Py<PyType>>,
) -> PyResult<Bound<'py, PyMessage>> {
    let message = PyMessage::empty(desc);
    let Some(class) = class else {
        return Bound::new(py, message);
    };
    // SAFETY: `class` is a subclass of `PyMessage` (see `_proto_register_classes`), of which
    // PyO3's `tp_new_impl`, the `tp_new` of a `#[new]`, makes an object of an initializer.
    // The object is untracked once, before any Python code sees it; its deallocation
    // untracks it only if it is tracked.
    unsafe {
        let ptr = pyo3::impl_::pymethods::tp_new_impl::<PyClassInitializer<PyMessage>, PyMessage>(
            py,
            message.into(),
            class.bind(py).as_type_ptr(),
        )?;
        if pyo3::ffi::PyObject_GC_IsTracked(ptr) != 0 {
            pyo3::ffi::PyObject_GC_UnTrack(ptr.cast());
        }
        Ok(Bound::from_owned_ptr(py, ptr).cast_into_unchecked())
    }
}

/// A new, empty message of `class`, as its `FromString` makes one (tracked by the garbage
/// collector, as of a call of its class), if `class` is a class of a message type of the runtime;
/// else none.
pub(crate) fn new_message_of_class<'py>(
    class: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyMessage>>> {
    let Ok(class) = class.cast::<PyType>() else {
        return Ok(None);
    };
    if !class.is_subclass_of::<PyMessage>()? {
        return Ok(None);
    }
    let maker = MessageClass {
        class: Some(class.clone().unbind()),
        desc: message_type_of(class)?,
    };
    let message = maker.new_message(class.py())?;
    track(&message);
    Ok(Some(message))
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
fn message_type_of(cls: &Bound<'_, PyType>) -> PyResult<&'static MessageDesc> {
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
fn class_descriptor(cls: &Bound<'_, PyType>) -> PyResult<&'static MessageDesc> {
    let abstract_base =
        || PyTypeError::new_err("Message is the abstract base of the message classes");
    let descriptor = cls.getattr("DESCRIPTOR").map_err(|_| abstract_base())?;
    if let Ok(descriptor) = descriptor.cast::<PyDescriptor>() {
        return Ok(descriptor.get().desc);
    }
    let full_name = descriptor
        .getattr("full_name")
        .and_then(|name| name.extract::<String>())
        .map_err(|_| abstract_base())?;
    pool().message(&full_name).ok_or_else(|| {
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
/// (`Expression.Literal.Map` has one), which made it a mapping without `items`. Its copy of a
/// message of protobuf's of the type is of its bytes of any depth when `unlimited`, as upb's `==`
/// compares messages of any depth, else of upb's limit of a parse, as upb merges a message into a
/// field.
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
        PyMessage::init_fields(&child, value, budget)?;
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
/// of the runtime takes it (see [`message_from_value_of`]), of its bytes, each message that it
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

/// What a value given to a message field (or to a repeated message field) of the message type
/// `full_name` of a constructor is, of the checks that the runtime made of it, in their order (see
/// [`Given::of`]).
enum Given<'py> {
    /// A message of the runtime of the type.
    Message(Bound<'py, PyMessage>),
    /// A dict of a `Struct` whose class assigns values: of its `_internal_assign`, as upb's.
    StructDict,
    /// A dict, or an object of `keys` (a mapping), whose items are the fields of the message.
    Mapping,
    /// A message of protobuf's of the type, of its bytes.
    Protobufs(Vec<u8>),
    /// A value that the class of the well-known type assigns, as upb's `_internal_assign` of it
    /// (a Timestamp of a `datetime`).
    Assigned,
    /// Anything else: upb's `TypeError` (see [`not_initializing`]).
    Other,
}

impl<'py> Given<'py> {
    /// What `value` is to a message field of the type `full_name`: a message of the runtime of the
    /// type, at once, without the lookup of a `keys` of it, which raised and dropped an
    /// `AttributeError` for each message of a list of many; as upb's, a dict of a `Struct` of its
    /// `_internal_assign`, and a value that is neither a mapping nor a message of the type of the
    /// `_internal_assign` of a well-known type whose class assigns values (a Timestamp of a
    /// `datetime`); else a dict, a message of protobuf's of the type (taken before a mapping but a
    /// dict, as it can have a field named `keys`, as `Expression.Literal.Map` has, which made it
    /// a mapping without `items`), or another mapping.
    fn of(py: Python<'py>, full_name: &str, value: &Bound<'py, PyAny>) -> Given<'py> {
        let runtime_message = value.cast::<PyMessage>().ok();
        if let Some(m) = runtime_message {
            if m.borrow().desc.full_name == full_name {
                return Given::Message(m.clone());
            }
        }
        let is_dict = value.cast::<PyDict>().is_ok();
        if is_dict && full_name == STRUCT && assigns_values(py, full_name) {
            return Given::StructDict;
        }
        let is_mapping = is_dict || value.hasattr("keys").unwrap_or(false);
        let is_of_type = runtime_message.is_none() && is_protobufs_message_of(value, full_name);
        if !is_mapping && !is_of_type && assigns_values(py, full_name) {
            return Given::Assigned;
        }
        if runtime_message.is_some() {
            return Given::Other;
        }
        if is_dict {
            return Given::Mapping;
        }
        if is_protobufs_message_of(value, full_name) {
            if let Ok((_, bytes)) = name_and_bytes(value) {
                return Given::Protobufs(bytes);
            }
        }
        if value.hasattr("keys").unwrap_or(false) {
            return Given::Mapping;
        }
        Given::Other
    }
}

/// The full names of the well-known types of which upb's messages compare with, and find a
/// value in, a dict and a list: `Struct` and `ListValue`.
const STRUCT: &str = "google.protobuf.Struct";
const LIST_VALUE: &str = "google.protobuf.ListValue";

/// What a merge of a message reads, taken before the merge changes anything, as upb's `MergeFrom`
/// serializes its argument first: a native copy of the message, or, of a message of more levels
/// below it than upb's limit of a parse ([`codec::depth_limit`]), its bytes, which the merge
/// parses, of upb's `DecodeError` and of what that parse keeps.
pub(crate) enum Snapshot {
    Native(Native),
    Bytes(Vec<u8>),
}

impl Snapshot {
    /// The snapshot of `message`, a step of `budget` for each message that it copies.
    pub(crate) fn of(
        py: Python<'_>,
        message: &Bound<'_, PyMessage>,
        budget: &ConversionBudget,
    ) -> PyResult<Snapshot> {
        let at = native::At::top(Some(codec::depth_limit()));
        match native::copy_message(py, message, at, generation(), budget) {
            Ok(copy) => Ok(Snapshot::Native(copy)),
            Err(CopyError::TooDeep) => Ok(Snapshot::Bytes(codec::encode(py, &message.borrow())?)),
            Err(CopyError::Python(err)) => Err(err),
            Err(CopyError::Holds) => unreachable!("a snapshot watches no message"),
        }
    }

    /// Merge the snapshot into `target`, as upb's `MergeFrom` parses the bytes of its argument.
    pub(crate) fn merge_into(
        self,
        py: Python<'_>,
        target: &Bound<'_, PyMessage>,
        budget: &ConversionBudget,
    ) -> PyResult<()> {
        match self {
            Snapshot::Native(source) => merge_native(py, target, NRef::new(source), budget),
            Snapshot::Bytes(bytes) => codec::merge_in(py, target, &bytes, budget),
        }
    }
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

/// A copy of `message`, of the class of its type, whose fields are native copies (see
/// [`native::copy_slots`]), a step of `budget` for each message copied: each message is read
/// under a short borrow, and the messages that it holds are copied after it, so that the
/// checkpoints of the copy of a message of many messages, whose Python code may use the message,
/// find no borrow of it held.
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
        let (slots, unknown) = native::copy_slots_now(py, message, budget)?;
        let copy = match class {
            Some(class) => class.new_message(py)?,
            None => {
                let desc = message.borrow().desc;
                new_instance_of(py, desc)?
            }
        };
        copy.borrow_mut().take_parts(slots, unknown);
        Ok(copy)
    })
}

/// A copy of the fields and the unknown fields of `message` (see [`native::copy_slots`]).
fn copy_fields_of(
    py: Python<'_>,
    message: &Bound<'_, PyMessage>,
    budget: &ConversionBudget,
) -> PyResult<(Slots, Vec<u8>)> {
    native::copy_slots_now(py, message, budget)
}

/// Whether Python holds `child`, an unset child that a message gave out, as upb tells it of the
/// weak map of its unset children: upb's unset children and their containers hold their parents
/// strongly, so Python holds the child if it holds it, an unset child of it of any level, or a
/// container of one (`a.crosstab.input`, `a.crosstab.input.project.expressions`). The runtime's
/// hold their parents weakly, and each message holds its unset children and containers: so the
/// child of none of those has no reference but its parent's, nor has any of them. It runs no
/// Python code; a message that Rust code borrows meanwhile is taken as held.
fn held_by_python(py: Python<'_>, child: &Py<PyMessage>) -> bool {
    // SAFETY: each object is live, of the reference of the message that holds it.
    let held = |object: *mut pyo3::ffi::PyObject| unsafe { pyo3::ffi::Py_REFCNT(object) } > 1;
    if held(child.as_ptr()) {
        return true;
    }
    // The unset children below the child, each looked at before its reference here is taken.
    let mut pending = vec![child.clone_ref(py)];
    while let Some(stub) = pending.pop() {
        let stub = stub.bind(py);
        let Ok(stub) = stub.try_borrow() else {
            return true;
        };
        for (_, slot) in stub.slots.iter() {
            let container = match slot {
                Slot::Repeated(c) => c.as_ptr(),
                Slot::Composite(c) => c.as_ptr(),
                Slot::Map(c) => c.as_ptr(),
                _ => continue,
            };
            if held(container) {
                return true;
            }
        }
        for (_, grandchild) in &stub.lazy {
            if held(grandchild.as_ptr()) {
                return true;
            }
            pending.push(grandchild.clone_ref(py));
        }
    }
    false
}

/// Set `message` in its parent if it is an unset child, and the parent in its own, as setting a
/// field of an unset child does.
pub(crate) fn mark_present(message: &Bound<'_, PyMessage>) -> PyResult<()> {
    let py = message.py();
    let mut current = message.clone();
    loop {
        let parent = current.borrow_mut().take_parent();
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

/// Merge `source`, a native message (see [`Snapshot`]), into `target`, as protobuf's `MergeFrom`
/// does: a set scalar replaces the value, a message merges into the target's, and repeated fields
/// and maps add their values. A field that the target does not hold, of no unset child that it gave
/// out, takes the value of the source as it is, as a merge into a new message makes it, so a merge
/// into an empty message (`append`, a constructor) makes no object. The values that no other value
/// holds (of a snapshot) are moved, not copied. Each message merged or copied is a step of
/// `budget`, with no borrow of the target held but while a field is set; the source is immutable,
/// so what the target becomes meanwhile changes nothing of what is merged.
fn merge_native(
    py: Python<'_>,
    target: &Bound<'_, PyMessage>,
    source: NRef,
    budget: &ConversionBudget,
) -> PyResult<()> {
    let merged = merge_native_fields(py, target, source, budget);
    // The native values that the merge set below the target, of a merge that failed too, which
    // the marks of the messages above them do not see (see `CLEAN_EPOCH`).
    unmark_all();
    merged
}

/// [`merge_native`] of the fields of `source`.
fn merge_native_fields(
    py: Python<'_>,
    target: &Bound<'_, PyMessage>,
    source: NRef,
    budget: &ConversionBudget,
) -> PyResult<()> {
    grow(|| {
        budget.step(py)?;
        let fields = native::fields_at(py, source, generation());
        for (index, slot) in fields.slots.into_vec() {
            match slot {
                Slot::Scalar(v) => {
                    if !fields.desc.fields[index].has_presence() && v.is_default() {
                        continue;
                    }
                    target.borrow_mut().set_scalar(py, index, v);
                }
                Slot::Native(value) => merge_value(py, target, index, value, budget)?,
                _ => unreachable!("a native message holds no object"),
            }
        }
        if !fields.unknown.is_empty() {
            target.borrow_mut().unknown.extend(fields.unknown);
        }
        Ok(())
    })
}

/// Merge `value`, the native value of the field `index` of a source, into the field of `target`
/// (see [`merge_native`]).
fn merge_value(
    py: Python<'_>,
    target: &Bound<'_, PyMessage>,
    index: usize,
    value: Value,
    budget: &ConversionBudget,
) -> PyResult<()> {
    let Some(value) = share_into_absent(py, target, index, value)? else {
        return Ok(());
    };
    match value {
        Value::Message(child) => {
            let into = PyMessage::child_for_merge_of(target, index)?;
            merge_native(py, &into, child, budget)
        }
        Value::Scalars(items) => {
            let items = Arc::try_unwrap(items).unwrap_or_else(|items| (*items).clone());
            let container = PyMessage::repeated_scalar_of(target, index)?;
            container.bind(py).borrow_mut().items.extend(items);
            Ok(())
        }
        Value::Messages(items) => {
            let container = PyMessage::composite_of(target, index)?;
            let class = container.bind(py).borrow().class.clone_ref(py);
            let items = Arc::try_unwrap(items).map_or_else(
                |items| items.items.clone(),
                |mut m| std::mem::take(&mut m.items),
            );
            for item in items {
                let copy = native::duplicate_owned(py, item, Some(&class), budget)?;
                container.bind(py).borrow_mut().items.push(copy.unbind());
            }
            Ok(())
        }
        Value::Map(map) => {
            let container = PyMessage::map_of_field(target, index)?;
            MapContainer::merge_entries(py, container.bind(py), MapEntries::of_native(&map), budget)
        }
    }
}

/// Set `value` in the field `index` of `target` as it is, if the target does not hold the field
/// nor gave out an unset child of it, and the value is of the classes registered now (see
/// [`Native::generation`]): what a merge of the value into the empty field makes, of the class of
/// the container of the field of a repeated message field. The value back, if it is not set.
fn share_into_absent(
    py: Python<'_>,
    target: &Bound<'_, PyMessage>,
    index: usize,
    value: Value,
) -> PyResult<Option<Value>> {
    let now = generation();
    let class = match &value {
        Value::Message(m) if m.generation != now => return Ok(Some(value)),
        Value::Messages(m) if m.generation != now => return Ok(Some(value)),
        Value::Map(m) if m.generation != now => return Ok(Some(value)),
        Value::Messages(_) => {
            let desc = target.borrow().desc;
            Some(MessageClass::of(py, &desc.fields[index].type_name)?)
        }
        _ => None,
    };
    let mut t = target.borrow_mut();
    if t.slots.contains(index) || t.lazy_child(index).is_some() {
        return Ok(Some(value));
    }
    let value = match (value, class) {
        (Value::Messages(m), Some(class)) if !class.is_same(&m.class) => Value::Messages(Arc::new(
            NativeMessages::new(class, m.items.to_vec(), m.generation),
        )),
        (value, _) => value,
    };
    if matches!(value, Value::Message(_)) {
        t.set_child(py, index, Slot::Native(value));
    } else {
        t.set_slot(index, Slot::Native(value));
    }
    Ok(None)
}

/// Whether `a` and `b` hold the same fields, as protobuf's `==` compares two messages: of the
/// same type, with the same set fields of the same values. It reads the set fields of the two
/// only, in the order of their numbers, as a field that neither sets is equal, and compares the
/// values where they are, objects or native: it runs no Python code, so the messages and
/// containers that it reads stay borrowed meanwhile. It holds the GIL until it ends, as upb's `==`
/// does.
pub(crate) fn content_eq(
    py: Python<'_>,
    a: &Bound<'_, PyMessage>,
    b: &Bound<'_, PyMessage>,
) -> PyResult<bool> {
    if a.is(b) {
        return Ok(true);
    }
    fields_eq(py, &a.borrow(), &b.borrow())
}

/// Whether the fields `x` and `y` are equal, as [`content_eq`] compares two messages.
fn fields_eq(py: Python<'_>, x: &Fields, y: &Fields) -> PyResult<bool> {
    grow(|| {
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

/// A message that a field holds: an object, or a native message.
#[derive(Clone, Copy)]
pub(crate) enum MessageRef<'a> {
    Object(&'a Py<PyMessage>),
    Native(&'a Native),
}

impl MessageRef<'_> {
    /// `read` of the fields of the message: of an object, under its borrow.
    pub(crate) fn read<R>(self, py: Python<'_>, read: impl FnOnce(&Fields) -> R) -> R {
        match self {
            MessageRef::Object(m) => read(&m.bind(py).borrow()),
            MessageRef::Native(m) => read(&m.fields),
        }
    }
}

/// The message that the slot of a message field holds, if any.
pub(crate) fn message_of(slot: Option<&Slot>) -> Option<MessageRef<'_>> {
    match slot {
        Some(Slot::Message(m)) => Some(MessageRef::Object(m)),
        Some(Slot::Native(Value::Message(m))) => Some(MessageRef::Native(m)),
        _ => None,
    }
}

/// `read` of the messages that the slot of a repeated message field holds, in their order: of a
/// container of Python, under its borrow.
pub(crate) fn with_messages<R>(
    py: Python<'_>,
    slot: Option<&Slot>,
    read: impl FnOnce(&[MessageRef<'_>]) -> R,
) -> R {
    match slot {
        Some(Slot::Composite(c)) => {
            let c = c.bind(py).borrow();
            let items: Vec<MessageRef<'_>> = c
                .items
                .iter()
                .map(MessageRef::Object)
                .chain(c.pending_natives().iter().map(|m| MessageRef::Native(m)))
                .collect();
            read(&items)
        }
        Some(Slot::Native(Value::Messages(m))) => {
            let items: Vec<MessageRef<'_>> =
                m.items.iter().map(|m| MessageRef::Native(m)).collect();
            read(&items)
        }
        _ => read(&[]),
    }
}

/// `read` of the scalars that the slot of a repeated scalar field holds, in their order: of a
/// container of Python, under its borrow.
pub(crate) fn with_scalars<R>(
    py: Python<'_>,
    slot: Option<&Slot>,
    read: impl FnOnce(&[Scalar]) -> R,
) -> R {
    match slot {
        Some(Slot::Repeated(c)) => read(&c.bind(py).borrow().items),
        Some(Slot::Native(Value::Scalars(items))) => read(items),
        _ => read(&[]),
    }
}

/// The value of an entry of a map: a scalar or a message.
#[derive(Clone, Copy)]
pub(crate) enum EntryRef<'a> {
    Scalar(&'a Scalar),
    Message(MessageRef<'a>),
}

impl<'a> From<&'a MapValue> for EntryRef<'a> {
    fn from(value: &'a MapValue) -> Self {
        match value {
            MapValue::Scalar(v) => EntryRef::Scalar(v),
            MapValue::Message(m) => EntryRef::Message(MessageRef::Object(m)),
        }
    }
}

impl<'a> From<&'a NativeValue> for EntryRef<'a> {
    fn from(value: &'a NativeValue) -> Self {
        match value {
            NativeValue::Scalar(v) => EntryRef::Scalar(v),
            NativeValue::Message(m) => EntryRef::Message(MessageRef::Native(m)),
        }
    }
}

/// `read` of the map description and the entries of the slot of a map field, in the order of
/// their insertion, if it holds a map: of a map of Python, under its borrow.
pub(crate) fn with_entries<R>(
    py: Python<'_>,
    slot: Option<&Slot>,
    read: impl FnOnce(Option<(&MapDesc, Vec<(&Scalar, EntryRef<'_>)>)>) -> R,
) -> R {
    match slot {
        Some(Slot::Map(m)) => {
            let m = m.bind(py).borrow();
            let entries = m
                .entries
                .iter()
                .map(|(k, v)| (k, EntryRef::from(v)))
                .collect();
            read(Some((&m.map, entries)))
        }
        Some(Slot::Native(Value::Map(m))) => {
            let entries = m
                .entries
                .iter()
                .map(|(k, v)| (k, EntryRef::from(v)))
                .collect();
            read(Some((&m.map, entries)))
        }
        _ => read(None),
    }
}

/// Whether the messages `a` and `b` are equal, as [`content_eq`] compares them; two native
/// messages that are one are equal at once.
fn message_eq(py: Python<'_>, a: MessageRef<'_>, b: MessageRef<'_>) -> PyResult<bool> {
    match (a, b) {
        (MessageRef::Object(x), MessageRef::Object(y)) => content_eq(py, x.bind(py), y.bind(py)),
        (MessageRef::Native(x), MessageRef::Native(y)) if std::ptr::eq(x, y) => Ok(true),
        (a, b) => a.read(py, |x| b.read(py, |y| fields_eq(py, x, y))),
    }
}

/// Whether the values `a` and `b` of entries of maps are equal, as upb's `==` of two messages
/// compares the values of their maps: a float or a double of its bits.
fn entry_eq(py: Python<'_>, a: EntryRef<'_>, b: EntryRef<'_>) -> PyResult<bool> {
    match (a, b) {
        (EntryRef::Scalar(v), EntryRef::Scalar(w)) => Ok(v.eq_as_upb(w)),
        (EntryRef::Message(m), EntryRef::Message(n)) => message_eq(py, m, n),
        _ => Ok(false),
    }
}

/// Whether `x` and `y` have the same keys, of equal values (see [`entry_eq`]).
fn entries_eq<V, W>(py: Python<'_>, x: &Entries<V>, y: &Entries<W>) -> PyResult<bool>
where
    for<'a> EntryRef<'a>: From<&'a V> + From<&'a W>,
{
    if x.len() != y.len() {
        return Ok(false);
    }
    for (key, value) in x.iter() {
        let Some(other) = y.get(key) else {
            return Ok(false);
        };
        if !entry_eq(py, EntryRef::from(value), EntryRef::from(other))? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Whether the maps of the slots `a` and `b` of a map field are equal, as upb's `==` of two
/// messages compares them: a map that a message does not hold is an empty one.
fn maps_eq(py: Python<'_>, a: Option<&Slot>, b: Option<&Slot>) -> PyResult<bool> {
    let is_empty = |slot: &Slot| match slot {
        Slot::Map(m) => m.bind(py).borrow().entries.is_empty(),
        Slot::Native(value) => value.is_empty(),
        _ => true,
    };
    match (a, b) {
        (None, None) => Ok(true),
        (Some(m), None) | (None, Some(m)) => Ok(is_empty(m)),
        (Some(Slot::Map(x)), Some(Slot::Map(y))) => MapContainer::equal(py, x.bind(py), y.bind(py)),
        (Some(Slot::Map(x)), Some(Slot::Native(Value::Map(y)))) => {
            entries_eq(py, &x.bind(py).borrow().entries, &y.entries)
        }
        (Some(Slot::Native(Value::Map(x))), Some(Slot::Map(y))) => {
            entries_eq(py, &x.entries, &y.bind(py).borrow().entries)
        }
        (Some(Slot::Native(Value::Map(x))), Some(Slot::Native(Value::Map(y)))) => {
            if Arc::ptr_eq(x, y) {
                return Ok(true);
            }
            entries_eq(py, &x.entries, &y.entries)
        }
        _ => Ok(false),
    }
}

fn slot_eq(
    py: Python<'_>,
    field: &FieldDesc,
    a: Option<&Slot>,
    b: Option<&Slot>,
) -> PyResult<bool> {
    if field.map.is_some() {
        return maps_eq(py, a, b);
    }
    if field.repeated && field.kind == Kind::Message {
        return with_messages(py, a, |x| {
            with_messages(py, b, |y| {
                if x.len() != y.len() {
                    return Ok(false);
                }
                for (m, n) in x.iter().zip(y) {
                    if !message_eq(py, *m, *n)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            })
        });
    }
    if field.repeated {
        return Ok(with_scalars(py, a, |x| {
            with_scalars(py, b, |y| {
                x.len() == y.len() && x.iter().zip(y).all(|(v, w)| v.eq_as_upb(w))
            })
        }));
    }
    if field.kind == Kind::Message {
        return match (message_of(a), message_of(b)) {
            (None, None) => Ok(true),
            (Some(x), Some(y)) => message_eq(py, x, y),
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

impl PyMessage {
    /// The maker of the messages of the class of `slf` (see [`MessageClass`]), of which its copies
    /// are, as upb's.
    fn maker_of(slf: &Bound<'_, Self>) -> MessageClass {
        MessageClass {
            class: Some(slf.get_type().unbind()),
            desc: slf.borrow().desc,
        }
    }

    /// Whether the message holds no field and no unknown field (of the unset children that it
    /// gave out too): what a parse of its bytes then sets, it sets of a native parse (see
    /// `codec::merge_with`).
    pub(crate) fn is_empty(&self) -> bool {
        self.slots.len() == 0 && self.unknown.is_empty()
    }

    /// Set the fields of `parsed`, a native parse of bytes into an empty message of the type of
    /// `message` (see `codec::merge_with`), in `message`, as the parse into its object sets them,
    /// also of a parse that failed, of what it read: an unset child that the message gave out of a
    /// field that the parse set takes the parsed message, of every level (see
    /// [`PyMessage::adopt`]), as upb's parse makes its unset children the messages of the fields
    /// that it sets (`SyncSubobjs`). A message that Python code changed meanwhile (a checkpoint of
    /// the parse lets another thread take the GIL) merges them, as `MergeFrom` merges a message.
    pub(crate) fn take_parsed(
        message: &Bound<'_, Self>,
        mut parsed: Native,
        budget: Option<&ConversionBudget>,
    ) -> PyResult<()> {
        let py = message.py();
        let mut m = message.try_borrow_mut()?;
        if m.is_empty() {
            let slots = std::mem::take(&mut parsed.fields.slots);
            let unknown = std::mem::take(&mut parsed.fields.unknown);
            return m.replace_slots(py, slots, unknown);
        }
        drop(m);
        let full = ConversionBudget::full();
        merge_native(py, message, NRef::new(parsed), budget.unwrap_or(&full))
    }

    /// Move the native values of `message`, and of the messages below it (of its objects and of
    /// those of its containers too), out of the slabs of their parse (see `Value::unpin`): of a
    /// message that outlives the message, the container or the map that held it (see
    /// [`unpin_held`]), so that the native values of a part of a parse that it holds, unread, keep
    /// alive no slab of the messages released with the container, where a slab holds up to 64
    /// messages. Of up to [`UNPIN_STEPS`] objects and native messages, as a message that holds
    /// more holds most of the nodes of its slabs itself; a message that another borrows meanwhile
    /// is left as it is, and so is one that holds no native value below it (see
    /// `PyMessage::clean`), of no walk of its objects: as most of the objects of a read of a parse
    /// are, so that a message read whole is released of no walk of it.
    pub(crate) fn unpin(message: &Bound<'_, Self>) {
        let mut walk = Unpin::new();
        walk.left.push(message.clone().unbind());
        walk.run(message.py());
    }

    /// The native messages that `message` holds, of all its levels, of its objects and of its
    /// containers too, that are nodes of a slab of a parse (see `PyMessage::unpin`).
    pub(crate) fn slab_messages(message: &Bound<'_, Self>) -> usize {
        let py = message.py();
        let m = message.borrow();
        grow(|| {
            m.slots
                .iter()
                .map(|(_, slot)| match slot {
                    Slot::Native(value) => native::slab_messages(value),
                    Slot::Message(child) => Self::slab_messages(child.bind(py)),
                    Slot::Composite(c) => {
                        let c = c.bind(py).borrow();
                        let made: usize = c
                            .items
                            .iter()
                            .map(|item| Self::slab_messages(item.bind(py)))
                            .sum();
                        let pending: usize = c
                            .pending_natives()
                            .iter()
                            .map(|n| native::slab_messages(&Value::Message(n.clone())))
                            .sum();
                        made + pending
                    }
                    Slot::Map(c) => c
                        .bind(py)
                        .borrow()
                        .entries
                        .iter()
                        .map(|(_, value)| match value {
                            MapValue::Message(m) => Self::slab_messages(m.bind(py)),
                            MapValue::Scalar(_) => 0,
                        })
                        .sum(),
                    Slot::Scalar(_) | Slot::Repeated(_) => 0,
                })
                .sum()
        })
    }

    pub(crate) fn empty(desc: &'static MessageDesc) -> Self {
        PyMessage {
            fields: Fields {
                desc,
                slots: Slots::default(),
                unknown: Vec::new(),
            },
            lazy: Vec::new(),
            parent: None,
            parent_field: 0,
            clean: mark_of(clean_epoch()),
        }
    }

    /// Whether the message holds no native value of a slab at any level below it, as of its mark
    /// of the epoch `epoch` (see `PyMessage::clean`).
    #[inline]
    fn is_clean(&self, epoch: u64) -> bool {
        self.clean != 0 && u64::from(self.clean) == epoch
    }

    /// Unmark the message, once it holds the native value of `slot` (see `PyMessage::clean`).
    fn unmark_of(&mut self, slot: &Slot) {
        if holds_native_messages(slot) {
            self.clean = 0;
        }
    }

    /// Unmark the message, once it holds the native values of `slots` (see `PyMessage::clean`).
    fn unmark_of_slots(&mut self, slots: &Slots) {
        if slots.iter().any(|(_, slot)| holds_native_messages(slot)) {
            self.clean = 0;
        }
    }

    /// The parent of an unset child, which it no longer holds (see `PyMessage::parent`).
    fn take_parent(&mut self) -> Option<Parent> {
        let message = self.parent.take()?;
        Some(Parent {
            message,
            field: self.parent_field as usize,
        })
    }

    /// Take the set fields and the unknown fields of `fields`, the copy of a message of the type
    /// of this one (a new message), as its own.
    pub(crate) fn take_fields(&mut self, fields: Fields) {
        debug_assert_eq!(self.desc.full_name, fields.desc.full_name);
        self.take_parts(fields.slots, fields.unknown);
    }

    /// Take the set fields `slots` and the unknown fields `unknown` of a copy of a message of the
    /// type of this one (a new message) as its own.
    pub(crate) fn take_parts(&mut self, slots: Slots, unknown: Vec<u8>) {
        self.unmark_of_slots(&slots);
        self.fields.slots = slots;
        self.fields.unknown = unknown;
    }

    /// Set the slot of the field `index`, giving back the one that it replaces.
    pub(crate) fn set_slot(&mut self, index: usize, slot: Slot) -> Option<Slot> {
        self.unmark_of(&slot);
        let Fields { desc, slots, .. } = &mut self.fields;
        slots.set(desc, index, slot)
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
    /// The object of a member that Python holds keeps no slab of a parse (see [`unpin_held`]).
    pub(crate) fn clear_oneof_siblings(&mut self, py: Python<'_>, index: usize) {
        let Some(oneof) = self.desc.fields[index].oneof else {
            return;
        };
        // Of the set fields, which are few, not of the members, which may be many.
        let Fields { desc, slots, .. } = &mut self.fields;
        let cleared = |member: usize| member != index && desc.fields[member].oneof == Some(oneof);
        for (member, slot) in slots.iter() {
            if cleared(member) {
                unpin_held(py, slot);
            }
        }
        slots.retain(|member| !cleared(member));
    }

    pub(crate) fn set_scalar(&mut self, py: Python<'_>, index: usize, value: Scalar) {
        self.clear_oneof_siblings(py, index);
        self.set_slot(index, Slot::Scalar(value));
    }

    /// Set the message field `index` to `child`, an object or a native message: the other members
    /// of its oneof cleared, and an unset child of the field that the message gave out detached.
    fn set_child(&mut self, py: Python<'_>, index: usize, child: Slot) {
        self.clear_oneof_siblings(py, index);
        self.sever(py, index);
        self.set_slot(index, child);
    }

    /// Clear the field `index`: the object of its value that Python holds keeps no slab of a parse
    /// (see [`unpin_held`]).
    fn clear_index(&mut self, py: Python<'_>, index: usize) {
        if let Some(slot) = self.slots.remove(index) {
            unpin_held(py, &slot);
        }
        self.sever(py, index);
    }

    /// Clear the fields: the objects of their values that Python holds keep no slab of a parse
    /// (see [`unpin_held`]).
    fn clear_all(&mut self, py: Python<'_>) {
        for (_, slot) in self.slots.iter() {
            unpin_held(py, slot);
        }
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
    /// when the copy sets that field (see [`PyMessage::adopt`]), and stays unset (and the field's)
    /// when it does not.
    fn replace_slots(&mut self, py: Python<'_>, slots: Slots, unknown: Vec<u8>) -> PyResult<()> {
        // The objects of the fields replaced that Python holds keep no slab of a parse (see
        // `unpin_held`).
        for (_, slot) in self.fields.slots.iter() {
            unpin_held(py, slot);
        }
        self.unmark_of_slots(&slots);
        if self.lazy.is_empty() {
            self.fields.slots = slots;
            self.fields.unknown = unknown;
            return Ok(());
        }
        let mut replaced = Vec::with_capacity(slots.len());
        for (index, slot) in slots.into_vec() {
            let slot = match (slot, self.take_lazy_child(index)) {
                (Slot::Message(copied), Some(stub)) => {
                    let (fields, unknown) = {
                        let mut c = copied.bind(py).borrow_mut();
                        (std::mem::take(&mut c.slots), std::mem::take(&mut c.unknown))
                    };
                    Self::adopt(py, stub.bind(py), fields, unknown)?;
                    Slot::Message(stub)
                }
                (Slot::Native(Value::Message(copied)), Some(stub)) => {
                    // The stub holds the fields of the copy, as it would of the object of it.
                    let fields = copied.fields.clone_native_at(py, copied.generation);
                    Self::adopt(py, stub.bind(py), fields.slots, fields.unknown)?;
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

    /// Give `stub`, an unset child of a field that a copy sets, the fields `slots` and `unknown`
    /// of the copy's child, as upb's `Reify` of the stub does: the stub is a message of its own
    /// from then on, and its own unset children of the fields that the copy's child sets take
    /// theirs in turn (upb's `SyncSubobjs`), so that the objects that Python holds of any level
    /// are those of the copy, and a change of one keeps the fields that the copy set.
    fn adopt(
        py: Python<'_>,
        stub: &Bound<'_, PyMessage>,
        slots: Slots,
        unknown: Vec<u8>,
    ) -> PyResult<()> {
        grow(|| {
            let lazy = {
                let mut child = stub.try_borrow_mut()?;
                child.parent = None;
                child.unmark_of_slots(&slots);
                child.fields.slots = slots;
                child.fields.unknown = unknown;
                std::mem::take(&mut child.lazy)
            };
            for (index, grandchild) in lazy {
                enum Set {
                    Native(NRef),
                    Object(Py<PyMessage>),
                }
                let set = match stub.borrow().slots.get(index) {
                    Some(Slot::Native(Value::Message(m))) => Some(Set::Native(m.clone())),
                    Some(Slot::Message(m)) => Some(Set::Object(m.clone_ref(py))),
                    _ => None,
                };
                let (slots, unknown) = match set {
                    Some(Set::Native(m)) => {
                        let fields = m.fields.clone_native_at(py, m.generation);
                        (fields.slots, fields.unknown)
                    }
                    Some(Set::Object(m)) => {
                        let mut m = m.bind(py).try_borrow_mut()?;
                        (
                            std::mem::take(&mut m.fields.slots),
                            std::mem::take(&mut m.fields.unknown),
                        )
                    }
                    None => {
                        stub.borrow_mut().lazy.push((index, grandchild));
                        continue;
                    }
                };
                Self::adopt(py, grandchild.bind(py), slots, unknown)?;
                // The value that the object replaces is dropped once the stub is no longer
                // borrowed.
                let _replaced = stub.borrow_mut().set_slot(index, Slot::Message(grandchild));
            }
            Ok(())
        })
    }

    /// The child of the singular message field `index` to merge into: the set one, else a new
    /// one, set now.
    pub(crate) fn child_for_merge<'py>(
        &mut self,
        py: Python<'py>,
        index: usize,
    ) -> PyResult<Bound<'py, PyMessage>> {
        self.materialize_in_place(py, index)?;
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

    /// The child of the singular message field `index` of `message` to merge into (see
    /// [`PyMessage::child_for_merge`]), of the object of its native value, if it holds one.
    /// A new child is made with no borrow of the message held, as [`PyMessage::container_of`]
    /// makes a container, and set only if the field has no child by then.
    pub(crate) fn child_for_merge_of<'py>(
        message: &Bound<'py, PyMessage>,
        index: usize,
    ) -> PyResult<Bound<'py, PyMessage>> {
        let py = message.py();
        loop {
            Self::materialize(message, index)?;
            let type_name = {
                let m = message.try_borrow()?;
                match m.slots.get(index) {
                    Some(Slot::Message(c)) => return Ok(c.bind(py).clone()),
                    _ if m.lazy_child(index).is_some() => None,
                    _ => Some(m.desc.fields[index].type_name.clone()),
                }
            };
            let made = type_name
                .map(|type_name| new_instance(py, &type_name))
                .transpose()?;
            let mut m = message.try_borrow_mut()?;
            if matches!(m.slots.get(index), Some(Slot::Native(_) | Slot::Message(_))) {
                // Set meanwhile: read again; the child made is dropped once the message is no
                // longer borrowed.
                drop(m);
                continue;
            }
            // As upb's: an unset child that the message gave out becomes the child that the merge
            // sets, so that the object it gave out holds the merged fields.
            let child = match (m.take_lazy_child(index), made) {
                (Some(stub), _) => {
                    let stub = stub.into_bound(py);
                    stub.try_borrow_mut()?.parent = None;
                    stub
                }
                (None, Some(made)) => made,
                (None, None) => {
                    drop(m);
                    continue;
                }
            };
            m.clear_oneof_siblings(py, index);
            m.set_slot(index, Slot::Message(child.clone().unbind()));
            return Ok(child);
        }
    }

    /// Make the object of the native value of the field `index` of `message`, if it holds one,
    /// and set it in the field, which gives that object from then on (see [`native::made_of`]).
    /// The object is made with no borrow of the message held, as making it may run Python code (a
    /// collection of the garbage collector, a checkpoint of its budget, which another thread may
    /// take the GIL at), and set only if the field still holds the value then: else the field is
    /// read again. A message that Rust code borrows meanwhile (the code of a collection that an
    /// allocation of such code started, or another thread that it gave the GIL to) is Python's
    /// `RuntimeError` ("Already borrowed"), never a panic.
    pub(crate) fn materialize(message: &Bound<'_, PyMessage>, index: usize) -> PyResult<()> {
        let py = message.py();
        loop {
            let (value, desc) = {
                let m = message.try_borrow()?;
                match m.slots.get(index) {
                    Some(Slot::Native(value)) => (value.clone(), m.desc),
                    _ => return Ok(()),
                }
            };
            let budget = ConversionBudget::full();
            let made = native::made_of(py, &desc.fields[index], &value, Some(&budget), true)?;
            let taken = {
                let mut m = message.try_borrow_mut()?;
                match m.slots.get_mut(index) {
                    Some(slot) if matches!(&*slot, Slot::Native(held) if held.same(&value)) => {
                        let slot = std::mem::replace(slot, made.slot.clone_slot(py));
                        Some(slot)
                    }
                    _ => None,
                }
            };
            // The objects take the fields of the value once the message is no longer borrowed
            // (moved, of the messages that no other value holds, of this reference released
            // first, else copies of one level, see `Made::fill`), running no Python code
            // meanwhile; the objects that the field no longer takes are dropped.
            if let Some(Slot::Native(taken)) = taken {
                drop(value);
                made.fill(py, taken);
                return Ok(());
            }
        }
    }

    /// Make the objects of the native values of the fields of `message`, as
    /// [`PyMessage::materialize`] makes each, until it holds none.
    fn materialize_all(message: &Bound<'_, PyMessage>) -> PyResult<()> {
        loop {
            let native: Vec<usize> = message
                .borrow()
                .slots
                .iter()
                .filter(|(_, slot)| matches!(slot, Slot::Native(_)))
                .map(|(index, _)| index)
                .collect();
            if native.is_empty() {
                return Ok(());
            }
            for index in native {
                Self::materialize(message, index)?;
            }
        }
    }

    /// [`PyMessage::materialize`] of the field `index` of a message that is borrowed already, as
    /// its parse and its constructor borrow it: the object is made under that borrow, of no
    /// checkpoints.
    fn materialize_in_place(&mut self, py: Python<'_>, index: usize) -> PyResult<()> {
        let Some(Slot::Native(value)) = self.slots.get(index) else {
            return Ok(());
        };
        let value = value.clone();
        let made = native::made_of(py, &self.desc.fields[index], &value, None, false)?;
        let object = made.slot.clone_slot(py);
        if let Some(Slot::Native(taken)) = self.set_slot(index, object) {
            drop(value);
            made.fill(py, taken);
        }
        Ok(())
    }

    /// The container of the repeated or map field `index` of `message` (its slot), made if
    /// absent, which holds `message` weakly when `owned`: of the object of the native value of the
    /// field, if it holds one (see [`PyMessage::materialize`]). The container, and the weak
    /// reference of its owner, are made with no borrow of the message held, as their allocations
    /// may run the Python code of a collection of the garbage collector, which may read the
    /// message or give the GIL to a thread that does; the field takes the container only if it
    /// holds none by then. A message that Rust code borrows meanwhile is Python's
    /// `RuntimeError`, never a panic.
    fn container_of(message: &Bound<'_, PyMessage>, index: usize, owned: bool) -> PyResult<Slot> {
        let py = message.py();
        let held = |slot: Option<&Slot>| match slot {
            Some(slot @ (Slot::Repeated(_) | Slot::Composite(_) | Slot::Map(_))) => {
                Some(slot.clone_slot(py))
            }
            _ => None,
        };
        loop {
            Self::materialize(message, index)?;
            let desc = {
                let m = message.try_borrow()?;
                if let Some(slot) = held(m.slots.get(index)) {
                    return Ok(slot);
                }
                m.desc
            };
            let owner = if owned { Some(weak(message)?) } else { None };
            let made = new_container(py, &desc.fields[index], owner)?;
            let mut m = message.try_borrow_mut()?;
            if let Some(slot) = held(m.slots.get(index)) {
                // Made meanwhile; this one is dropped once the message is no longer borrowed.
                drop(m);
                drop(made);
                return Ok(slot);
            }
            if matches!(m.slots.get(index), Some(Slot::Native(_))) {
                // A native value set meanwhile, whose object the field gives.
                continue;
            }
            m.set_slot(index, made.clone_slot(py));
            return Ok(made);
        }
    }

    /// The container of the repeated scalar field `index` of `message`, made if absent, which
    /// holds `message` weakly (see [`PyMessage::container_of`]).
    pub(crate) fn repeated_scalar_of(
        message: &Bound<'_, PyMessage>,
        index: usize,
    ) -> PyResult<Py<RepeatedScalar>> {
        match Self::container_of(message, index, true)? {
            Slot::Repeated(c) => Ok(c),
            _ => Err(PyTypeError::new_err("not a repeated scalar field")),
        }
    }

    /// The container of the repeated message field `index` of `message`, made if absent (see
    /// [`PyMessage::container_of`]).
    pub(crate) fn composite_of(
        message: &Bound<'_, PyMessage>,
        index: usize,
    ) -> PyResult<Py<RepeatedComposite>> {
        match Self::container_of(message, index, true)? {
            Slot::Composite(c) => {
                // The messages appended follow all those of the container.
                RepeatedComposite::make_all(c.bind(message.py()))?;
                Ok(c)
            }
            _ => Err(PyTypeError::new_err("not a repeated message field")),
        }
    }

    /// The map of the map field `index` of `message`, made if absent (see
    /// [`PyMessage::container_of`]).
    pub(crate) fn map_of_field(
        message: &Bound<'_, PyMessage>,
        index: usize,
    ) -> PyResult<Py<MapContainer>> {
        match Self::container_of(message, index, true)? {
            Slot::Map(c) => Ok(c),
            _ => Err(PyTypeError::new_err("not a map field")),
        }
    }

    /// Set the fields of the items of `mapping` in `message`, in their order, as upb's
    /// `InitAttributes` sets those of the keyword arguments of `__init__` and `add`, and of a
    /// mapping of a message field: a name of no field is upb's `ValueError`, also of a value of
    /// None, which leaves the field as it is. A scalar is set. A message of a message field merges
    /// into the message that the field has, as `MergeFrom` (of a snapshot of the value first), and a
    /// mapping sets the fields of that message, set in its parent first; a field that the message
    /// does not hold (of no unset child that it gave out) takes the copy of a message as it is. The
    /// values of a repeated message field are appended in turn, each to the messages of those
    /// before it (see [`PyMessage::init_messages`]); those of a repeated scalar field as the
    /// container's `extend` takes them; a map takes its items as [`MapContainer::update_from`]
    /// does. A mapping of a message field is one level of a recursion as deep as the mappings
    /// nest, on a stack with room for it.
    ///
    /// No borrow of `message` is held while Python code runs (the conversion of a value, an
    /// iteration, a checkpoint), so that code that reaches the message (of the garbage collector's
    /// `get_objects`, or as a value of its own fields) finds it as upb's is then, of the fields
    /// before set, and a value that holds the message is read as upb reads it.
    ///
    /// One interrupt budget counts a step for each message that the fields copy or build, and for
    /// each value of a repeated field and each key of a map of messages, of every level, so that a
    /// constructor of many values (`Literal.Array(elements=[...])`, of a million messages) lets
    /// another thread take the GIL, and a signal interrupt it, as the copies of the runtime do. A
    /// map of scalars takes its value in Python, as upb's does (see [`MapContainer::update_from`]).
    pub(crate) fn init_fields(
        message: &Bound<'_, Self>,
        mapping: &Bound<'_, PyAny>,
        budget: &ConversionBudget,
    ) -> PyResult<()> {
        let py = message.py();
        grow(|| {
            budget.step(py)?;
            let desc = message.borrow().desc;
            let items = mapping.call_method0(pyo3::intern!(py, "items"))?;
            for item in items.try_iter()? {
                let (key, value): (String, Bound<'_, PyAny>) = item?.extract()?;
                let Some(&index) = desc.by_name.get(&key) else {
                    return Err(PyValueError::new_err(format!(
                        "Protocol message {} has no \"{key}\" field.",
                        desc.name
                    )));
                };
                if value.is_none() {
                    continue;
                }
                let field = &desc.fields[index];
                if field.map.is_some() {
                    let Slot::Map(container) = Self::container_of(message, index, false)? else {
                        unreachable!("the container of a map field")
                    };
                    let container = container.into_bound(py);
                    MapContainer::update_from(py, &container, &value, &field.full_name, budget)?;
                } else if field.repeated && field.kind == Kind::Message {
                    Self::init_messages(message, index, field, &value, budget)?;
                } else if field.repeated {
                    let Slot::Repeated(container) = Self::container_of(message, index, false)?
                    else {
                        unreachable!("the container of a repeated scalar field")
                    };
                    RepeatedScalar::extend_with(container.bind(py), &value, Some(budget))?;
                } else if field.kind == Kind::Message {
                    Self::init_message(message, index, field, &value, budget)?;
                } else {
                    let scalar =
                        Scalar::from_py(field.kind, &field.full_name, &field.type_name, &value)?;
                    message.borrow_mut().set_scalar(py, index, scalar);
                }
            }
            Ok(())
        })
    }

    /// Set `value` in the message field `index` of `message` (see [`PyMessage::init_fields`]), as
    /// upb's `InitMessageAttribute`: the message that the field has (or an unset child of it)
    /// merges a message, of a snapshot of it taken first, takes the fields of a mapping, set in
    /// its parent first, or of the `_internal_assign` of a well-known type whose class assigns
    /// values.
    fn init_message(
        message: &Bound<'_, Self>,
        index: usize,
        field: &FieldDesc,
        value: &Bound<'_, PyAny>,
        budget: &ConversionBudget,
    ) -> PyResult<()> {
        let py = message.py();
        match Given::of(py, &field.type_name, value) {
            Given::Message(m) => match Snapshot::of(py, &m, budget)? {
                Snapshot::Native(copy) => {
                    let copy = Value::Message(NRef::new(copy));
                    let Some(Value::Message(copy)) = share_into_absent(py, message, index, copy)?
                    else {
                        return Ok(());
                    };
                    let into = Self::child_for_merge_of(message, index)?;
                    merge_native(py, &into, copy, budget)
                }
                bytes => {
                    let into = Self::child_for_merge_of(message, index)?;
                    bytes.merge_into(py, &into, budget)
                }
            },
            Given::Mapping => {
                let into = Self::child_for_merge_of(message, index)?;
                Self::init_fields(&into, value, budget)
            }
            Given::StructDict => {
                let child = Self::field_value(message, index)?.into_bound(py);
                Self::assign_struct(child.cast::<PyMessage>()?, value, budget)
            }
            Given::Protobufs(bytes) => {
                let into = Self::child_for_merge_of(message, index)?;
                codec::merge_in(py, &into, &bytes, budget)
            }
            Given::Assigned => {
                let child = Self::field_value(message, index)?;
                child.bind(py).call_method1("_internal_assign", (value,))?;
                Ok(())
            }
            Given::Other => Err(not_initializing(py, &field.type_name, value)),
        }
    }

    /// Append the values of `value` to the repeated message field `index` of `message` (see
    /// [`PyMessage::init_fields`]), each in turn, as upb's constructor does: a new message
    /// appended, then a message merged into it or the fields of a mapping set in it (removed if
    /// they fail, as upb's `add` removes it), so the values before a failure stay, and a value that
    /// holds the message is read with the messages before it and the new one in the field. The
    /// copies of the messages of the runtime of a list or a tuple, whose iteration runs no Python
    /// code, are made at once, natively, and the field takes them when a value is not one, at a
    /// failure and at the end (as its native value, if it holds none yet): a value that holds the
    /// message (the copy watches for it) is taken in turn.
    fn init_messages(
        message: &Bound<'_, Self>,
        index: usize,
        field: &FieldDesc,
        value: &Bound<'_, PyAny>,
        budget: &ConversionBudget,
    ) -> PyResult<()> {
        let py = message.py();
        let values = value.try_iter().map_err(|_| {
            PyTypeError::new_err(format!(
                "Argument for field {} is not iterable",
                field.full_name
            ))
        })?;
        let fixed =
            value.is_exact_instance_of::<PyList>() || value.is_exact_instance_of::<PyTuple>();
        let mut copies: Vec<NRef> = Vec::new();
        let appended = (|| -> PyResult<()> {
            for element in values {
                let element = element?;
                budget.step(py)?;
                if fixed {
                    if let Ok(m) = element.cast::<PyMessage>() {
                        if m.borrow().desc.full_name == field.type_name {
                            let at = native::At::top(Some(codec::depth_limit())).watching(message);
                            match native::copy_message(py, m, at, generation(), budget) {
                                Ok(copy) => {
                                    copies.push(NRef::new(copy));
                                    continue;
                                }
                                Err(CopyError::Python(err)) => return Err(err),
                                // In turn: of its bytes, or with the messages before it.
                                Err(CopyError::TooDeep | CopyError::Holds) => {}
                            }
                        }
                    }
                }
                Self::append_copies(message, index, field, &mut copies)?;
                Self::append_value(message, index, field, &element, budget)?;
            }
            Ok(())
        })();
        let taken = Self::append_copies(message, index, field, &mut copies);
        appended.and(taken)
    }

    /// Append `copies`, native copies of messages of the repeated message field `index` of
    /// `message`, to the field, taking them: a field that the message does not hold takes them as
    /// its native value (of the class of the field's container, which the objects of them are
    /// made of), when they are of the classes registered now, else the field's container takes
    /// their objects, made with no borrow held.
    fn append_copies(
        message: &Bound<'_, Self>,
        index: usize,
        field: &FieldDesc,
        copies: &mut Vec<NRef>,
    ) -> PyResult<()> {
        if copies.is_empty() {
            return Ok(());
        }
        let py = message.py();
        let copies = std::mem::take(copies);
        let now = generation();
        if copies.iter().all(|m| m.generation == now) {
            let class = MessageClass::of(py, &field.type_name)?;
            let mut m = message.borrow_mut();
            if !m.slots.contains(index) {
                let value = Value::Messages(Arc::new(NativeMessages::new(class, copies, now)));
                m.set_slot(index, Slot::Native(value));
                return Ok(());
            }
        }
        let mut objects = Vec::with_capacity(copies.len());
        for copy in &copies {
            objects.push(native::message_object(py, copy)?.unbind());
        }
        let Slot::Composite(container) = Self::container_of(message, index, false)? else {
            unreachable!("the container of a repeated message field")
        };
        RepeatedComposite::make_all(container.bind(py))?;
        container.bind(py).borrow_mut().items.extend(objects);
        Ok(())
    }

    /// Append `element` to the repeated message field `index` of `message` as upb's constructor
    /// appends a value (see [`PyMessage::init_messages`]): a new message appended to the field's
    /// container, then the value merged into it (left empty, of the error of a value of another
    /// type), or the fields of a mapping set in it, which a failure removes, as upb's `add`
    /// removes the last message.
    fn append_value(
        message: &Bound<'_, Self>,
        index: usize,
        field: &FieldDesc,
        element: &Bound<'_, PyAny>,
        budget: &ConversionBudget,
    ) -> PyResult<()> {
        let py = message.py();
        let Slot::Composite(container) = Self::container_of(message, index, false)? else {
            unreachable!("the container of a repeated message field")
        };
        let container = container.into_bound(py);
        RepeatedComposite::make_all(&container)?;
        let class = container.borrow().class.clone_ref(py);
        let item = class.new_message(py)?;
        container.borrow_mut().items.push(item.clone().unbind());
        let remove_on_failure = |result: PyResult<()>| {
            if result.is_err() {
                // The message removed is dropped once the container is no longer borrowed.
                let _removed = container.borrow_mut().items.pop();
            }
            result
        };
        match Given::of(py, &field.type_name, element) {
            Given::Message(m) => Self::merge_from_with(&item, m.as_any(), budget),
            Given::Mapping => remove_on_failure(Self::init_fields(&item, element, budget)),
            Given::StructDict => remove_on_failure(Self::assign_struct(&item, element, budget)),
            Given::Protobufs(bytes) => codec::merge_in(py, &item, &bytes, budget),
            Given::Assigned => item.call_method1("_internal_assign", (element,)).map(drop),
            Given::Other => Err(not_initializing(py, &field.type_name, element)),
        }
    }

    /// Set the fields of the dict `value` in `message`, a `Struct` of a class that assigns values,
    /// as upb's constructor does: of its `_internal_assign`, else, of a dict of the key `fields`
    /// only, which that refuses, of the fields of the dict, the message cleared first.
    fn assign_struct(
        message: &Bound<'_, Self>,
        value: &Bound<'_, PyAny>,
        budget: &ConversionBudget,
    ) -> PyResult<()> {
        match message.call_method1("_internal_assign", (value,)) {
            Ok(_) => Ok(()),
            Err(err) => {
                let dict = value.cast::<PyDict>()?;
                if !(dict.len() == 1 && dict.contains("fields")?) {
                    return Err(err);
                }
                Self::clear(message)?;
                Self::init_fields(message, value, budget)
            }
        }
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
            for (_, slot) in self.slots.iter_mut() {
                match slot {
                    Slot::Message(child) => child.bind(py).borrow_mut().discard_unknown(py, depth),
                    Slot::Composite(c) => {
                        let mut c = c.bind(py).borrow_mut();
                        for child in &c.items {
                            child.bind(py).borrow_mut().discard_unknown(py, depth);
                        }
                        for child in c.pending_natives_mut(py) {
                            native::unique_native(child, |m| m.clone_native(py))
                                .discard_unknown(py, depth);
                        }
                    }
                    Slot::Map(c) => {
                        for (_, value) in c.bind(py).borrow().entries.iter() {
                            if let MapValue::Message(child) = value {
                                child.bind(py).borrow_mut().discard_unknown(py, depth);
                            }
                        }
                    }
                    // Of a clone of what the value shares with other messages.
                    Slot::Native(value) => value.discard_unknown(py, depth),
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
            // As upb's, which serializes the message first and parses its bytes into this one:
            // the merge reads a snapshot of it, taken before this message is set in its parent,
            // so a message below it (an unset child of it too) merges what it was, and one of
            // more than 100 levels fails, of what the parse of its bytes keeps.
            let snapshot = Snapshot::of(py, other, budget)?;
            mark_present(slf)?;
            snapshot.merge_into(py, slf, budget)
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
        // The native value of the field, if it holds one, becomes an object, which it gives.
        Self::materialize(slf, index)?;
        let desc = slf.try_borrow()?.desc;
        let field = &desc.fields[index];
        if field.map.is_some() || field.repeated {
            return Ok(match Self::container_of(slf, index, true)? {
                Slot::Repeated(c) => c.into_any(),
                Slot::Composite(c) => c.into_any(),
                Slot::Map(c) => c.into_any(),
                _ => unreachable!("the container of a repeated or map field"),
            });
        }
        // A message field: its message, or an unset child of it.
        let given = |me: &PyMessage| match me.slots.get(index) {
            Some(Slot::Message(c)) => Some(c.clone_ref(py).into_any()),
            _ => me.lazy_child(index).map(|c| c.clone_ref(py).into_any()),
        };
        let found = given(&*slf.try_borrow()?);
        if let Some(child) = found {
            return Ok(child);
        }
        // Made with no borrow of the message held (see `container_of`), and given out only if the
        // field has none by then.
        let parent = weak(slf)?;
        let child = new_instance(py, &field.type_name)?;
        {
            let mut c = child.try_borrow_mut()?;
            c.parent = Some(parent);
            c.parent_field = u32::try_from(index).expect("the index of a field of a message");
        }
        let mut me = slf.try_borrow_mut()?;
        if let Some(given) = given(&me) {
            drop(me);
            return Ok(given);
        }
        if matches!(me.slots.get(index), Some(Slot::Native(_))) {
            drop(me);
            return Self::field_value(slf, index);
        }
        me.lazy.push((index, child.clone().unbind()));
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
fn missing_fields(py: Python<'_>, message: &Fields) -> Vec<String> {
    let mut missing = Vec::new();
    if may_lack_required(message.desc) {
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

fn collect_missing(py: Python<'_>, message: &Fields, prefix: &str, out: &mut Vec<String>) {
    grow(|| {
        for &index in &message.desc.required {
            if !message.slots.contains(index) {
                out.push(format!("{prefix}{}", message.desc.fields[index].name));
            }
        }
        for (index, slot) in message.slots.iter() {
            let field = &message.desc.fields[index];
            let name = &field.name;
            if field.map.is_some() {
                with_entries(py, Some(slot), |entries| {
                    for (key, value) in entries.map(|(_, entries)| entries).unwrap_or_default() {
                        if let EntryRef::Message(child) = value {
                            let key = text::map_key_text(key);
                            let prefix = format!("{prefix}{name}[{key}].");
                            child.read(py, |child| collect_missing(py, child, &prefix, out));
                        }
                    }
                });
            } else if field.repeated && field.kind == Kind::Message {
                with_messages(py, Some(slot), |items| {
                    for (at, child) in items.iter().enumerate() {
                        let prefix = format!("{prefix}{name}[{at}].");
                        child.read(py, |child| collect_missing(py, child, &prefix, out));
                    }
                });
            } else if let Some(child) = message_of(Some(slot)) {
                let prefix = format!("{prefix}{name}.");
                child.read(py, |child| collect_missing(py, child, &prefix, out));
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

    /// As upb's: set the fields of the keyword arguments, over those that the message has, in
    /// their order (see [`PyMessage::init_fields`]): so `message.__init__(**fields)` merges them,
    /// as a constructor sets them of a new message. Of keyword arguments (even of none, of
    /// `**{}`), an unset child is set in its parent first, as upb's `EnsureReified`.
    #[pyo3(signature = (*args, **kwargs))]
    fn __init__(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        if !args.is_empty() {
            return Err(PyTypeError::new_err("No positional arguments allowed"));
        }
        let Some(kwargs) = kwargs else {
            return Ok(());
        };
        mark_present(slf)?;
        if kwargs.is_empty() {
            return Ok(());
        }
        Self::init_fields(slf, kwargs.as_any(), &ConversionBudget::full())
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
        let desc = slf.borrow().desc;
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
        let mut me = slf.borrow_mut();
        // As upb's, which sets an unset child of the field that Python holds in the message first
        // (its `EnsureReified`), then clears the field: so the member of its oneof that the
        // message holds is cleared too. (upb holds its unset children weakly, so one that Python
        // no longer holds is not; see `held_by_python`.)
        if me
            .lazy_child(index)
            .is_some_and(|child| held_by_python(py, child))
        {
            me.clear_oneof_siblings(py, index);
        }
        me.clear_index(py, index);
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
        // The native values of the fields become the objects that the fields give, and the fields
        // are listed under one borrow, which no field changes under.
        let me = loop {
            Self::materialize_all(slf)?;
            let me = slf.borrow();
            if !me
                .slots
                .iter()
                .any(|(_, slot)| matches!(slot, Slot::Native(_)))
            {
                break me;
            }
        };
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
                    (c.bind(py).borrow().len() > 0).then(|| c.clone_ref(py).into_any())
                }
                Slot::Map(c) => {
                    (!c.bind(py).borrow().entries.is_empty()).then(|| c.clone_ref(py).into_any())
                }
                Slot::Native(_) => None,
            };
            if let Some(value) = value {
                let descriptor = match &protobuf_fields {
                    Some(fields) => fields[index].clone_ref(py),
                    None => field_descriptor_object(py, me.desc, index)?,
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
            if !may_lack_required(m.desc) {
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
            if !may_lack_required(m.desc) {
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
        let budget = ConversionBudget::full();
        budget.step(py)?;
        let at = native::At::top(Some(codec::depth_limit()));
        match native::copy_slots(py, slf, at, generation(), &budget) {
            Ok((slots, unknown)) => {
                // Tracked by the garbage collector, as upb's copy of a call of its class is.
                let copied = maker.new_message(py)?;
                copied.borrow_mut().take_parts(slots, unknown);
                track(&copied);
                Ok(copied)
            }
            Err(CopyError::TooDeep) => {
                let copied = maker.new_message(py)?;
                track(&copied);
                let bytes = codec::encode(py, &slf.borrow())?;
                codec::merge(py, &copied, &bytes)?;
                Ok(copied)
            }
            Err(CopyError::Python(err)) => Err(err),
            Err(CopyError::Holds) => unreachable!("a copy that watches no message"),
        }
    }

    /// The pickle of the message, as protobuf's `Message.__reduce__` gives it, of which upb's
    /// messages pickle: the class and the state of the bytes of the message, which `__setstate__`
    /// parses, and of a nested type, where protobuf imports, protobuf's
    /// `_InternalConstructMessage` of the full name of the type (see `internal_construct` of the
    /// module).
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let state = Self::__getstate__(slf)?;
        let desc = slf.borrow().desc;
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

#[cfg(test)]
mod tests {
    use std::hash::Hasher;

    use super::AddressHasher;

    #[test]
    fn the_hashes_of_addresses_of_objects_differ_in_their_low_bits() {
        // Addresses of objects, 16 bytes apart, as the allocator aligns them: the low bits of
        // their hashes, which pick their buckets, take all their values, not a sixteenth of them.
        let hash = |address: usize| {
            let mut hasher = AddressHasher::default();
            hasher.write_usize(address);
            hasher.finish()
        };
        let low: std::collections::HashSet<u64> = (0..256)
            .map(|i| hash(0x7f3a_1c00_0000 + i * 16) & 0xff)
            .collect();
        assert!(low.len() > 128, "{}", low.len());
        assert!((0..256).any(|i| hash(0x7f3a_1c00_0000 + i * 16) & 0xf != 0));
    }
}
