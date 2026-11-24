//! The native values of the fields of the messages: a copy of the value of a message, repeated or
//! map field, held in Rust, of no object of Python, which a message holds in the slot of its field
//! ([`Slot::Native`]) until Python reads the field, when the message makes the object of the value,
//! of one level (see [`made_of`] and `PyMessage::materialize`): a message of its fields, whose
//! own values stay native until they are read, a container of its messages, a map of its entries.
//!
//! A copy of a message ([`copy_message`], [`copy_slots`]) is native: the constructor of a message
//! given a message of the runtime (as `proto.MergeAction(condition=c)`), `CopyFrom`, `copy.copy`
//! and `copy.deepcopy` copy the message without an object of Python for each message that it holds,
//! which the copies of the runtime made before, as an object of Python is a message of the runtime.
//! `MergeFrom` copies its argument so before it merges it (`message::Snapshot`), as upb serializes
//! it first, so that a merge into a message that the argument holds reads what it was.
//! A native value is never changed (but by [`Value::discard_unknown`], which clones what it shares
//! first), so a copy of a message that holds native values shares them.
//!
//! What Python reads of a native value is what it read of the objects of the copy before: the
//! object that it makes of a value is of the class that the copy would have made then (see
//! [`Native::generation`]), held by no message as a copy's objects were, and the field gives that
//! object from then on, so its identity is kept.

use std::sync::Arc;

use pyo3::prelude::*;

use super::containers::{
    Entries, MapContainer, MapEntries, MapValue, RepeatedComposite, RepeatedScalar,
};
use super::grow;
use super::message::{generation, Fields, MessageClass, PyMessage, Slot, Slots};
use super::pool::{FieldDesc, MapDesc, MessageDesc};
use super::scalar::Scalar;
use crate::interrupt::ConversionBudget;

/// A native value of a field: of a message field, a repeated scalar field, a repeated message
/// field or a map field.
#[derive(Clone)]
pub(crate) enum Value {
    Message(Arc<Native>),
    Scalars(Arc<Vec<Scalar>>),
    Messages(Arc<NativeMessages>),
    Map(Arc<NativeMap>),
}

/// A native message: its fields, whose slots hold scalars and native values only, never an object
/// of Python.
pub(crate) struct Native {
    pub(crate) fields: Fields,
    /// The levels of messages below the message, as upb's parse counts them: a message of a field
    /// or of a repeated field, a map entry and the message of the value of an entry each a level.
    pub(crate) depth: u32,
    /// The generation of the classes of the messages (see `message::generation`) of the copy that
    /// made the message, which the objects of it and of the messages below it are made of: of the
    /// class of their type registered then, as the copy made them before.
    pub(crate) generation: u64,
}

/// The native value of a repeated message field: its messages, of the class of the messages of
/// its container, which the objects of its messages are made of.
pub(crate) struct NativeMessages {
    pub(crate) class: MessageClass,
    pub(crate) items: Vec<Arc<Native>>,
    /// The levels of messages that the field adds below its message: one more than the deepest of
    /// its messages, none of no messages.
    pub(crate) depth: u32,
    pub(crate) generation: u64,
}

/// The native value of a map field: its entries, in the order of their insertion.
pub(crate) struct NativeMap {
    pub(crate) map: Arc<MapDesc>,
    pub(crate) entries: Entries<NativeValue>,
    /// The levels of messages that the map adds below its message: an entry is one, and the
    /// message of its value another.
    pub(crate) depth: u32,
    pub(crate) generation: u64,
}

/// The value of an entry of a native map.
#[derive(Clone)]
pub(crate) enum NativeValue {
    Scalar(Scalar),
    Message(Arc<Native>),
}

/// Why a copy stopped: an error of Python (of a checkpoint of its budget), a message of more
/// levels below the message copied than the copy takes, which the caller then copies of its bytes,
/// as upb copies it, of upb's `DecodeError`, or the message that the copy watches for (see
/// [`At::watching`]), which the copied message is or holds.
pub(crate) enum CopyError {
    Python(PyErr),
    TooDeep,
    Holds,
}

impl From<PyErr> for CopyError {
    fn from(err: PyErr) -> Self {
        CopyError::Python(err)
    }
}

/// The levels of messages that a copy may still take below a message: of any depth (`None`), as
/// `CopyFrom` and `copy.deepcopy` copy, or of upb's limit of a parse, as a constructor and
/// `copy.copy` copy (see `codec::depth_limit`).
pub(crate) type Levels = Option<u32>;

/// The levels that a message one level below a message of `levels` may take, or [`CopyError::TooDeep`]
/// if it may take none.
fn below(levels: Levels) -> Result<Levels, CopyError> {
    match levels {
        None => Ok(None),
        Some(0) => Err(CopyError::TooDeep),
        Some(levels) => Ok(Some(levels - 1)),
    }
}

/// Where a copy is in the message that it copies: the levels that it may still take below the
/// message that it is at ([`Levels`]), and the level of that message below the message copied, of
/// which every 16th level runs on a stack with room for 16 levels (see [`grow`]), as reading the
/// room left of the stack takes longer than a level of a copy of a small message.
#[derive(Clone, Copy)]
pub(crate) struct At {
    pub(crate) levels: Levels,
    level: u32,
    /// The object of the message that the copy stops at (see [`At::watching`]), if any.
    watch: Option<*mut pyo3::ffi::PyObject>,
}

impl At {
    /// The message copied, of the levels `levels` below it.
    pub(crate) fn top(levels: Levels) -> At {
        At {
            levels,
            level: 0,
            watch: None,
        }
    }

    /// The copy as it is, but that it stops with [`CopyError::Holds`] at `message`, if the message
    /// copied is it or holds it, at any level: a value that the field of a message is given, which
    /// holds the message, is copied once the field holds what it was given before, as upb copies
    /// each value of a repeated field after the one before it is appended (see
    /// `PyMessage::init_fields`).
    pub(crate) fn watching(self, message: &Bound<'_, PyMessage>) -> At {
        At {
            watch: Some(message.as_ptr()),
            ..self
        }
    }

    /// A message one level below, or [`CopyError::TooDeep`] if the copy may take none.
    pub(crate) fn below(self) -> Result<At, CopyError> {
        Ok(At {
            levels: below(self.levels)?,
            level: self.level + 1,
            watch: self.watch,
        })
    }

    /// `copy` of the message (or the map entry) that the copy is at, on a stack with room for the
    /// levels of the messages below it, up to the next level that grows it.
    pub(crate) fn grown<T>(self, copy: impl FnOnce() -> T) -> T {
        if self.level.is_multiple_of(16) {
            grow(copy)
        } else {
            copy()
        }
    }
}

impl Native {
    /// The native message of `fields`, made at `generation`, of the depth of its values.
    pub(crate) fn new(fields: Fields, generation: u64) -> Native {
        let depth = fields
            .slots
            .iter()
            .map(|(_, slot)| match slot {
                Slot::Native(value) => value.depth(),
                Slot::Scalar(_) => 0,
                _ => unreachable!("a native message holds no object"),
            })
            .max()
            .unwrap_or(0);
        Native {
            fields,
            depth,
            generation,
        }
    }

    /// The message as a message of a copy made at `generation`: itself, if it was made then, else
    /// a clone of it of that generation (which a class registered since makes of another class).
    pub(crate) fn at(self: &Arc<Native>, py: Python<'_>, generation: u64) -> Arc<Native> {
        if self.generation == generation {
            return self.clone();
        }
        grow(|| {
            Arc::new(Native {
                fields: self.fields.clone_native_at(py, generation),
                depth: self.depth,
                generation,
            })
        })
    }
}

impl NativeMessages {
    pub(crate) fn new(class: MessageClass, items: Vec<Arc<Native>>, generation: u64) -> Self {
        let depth = items
            .iter()
            .map(|item| item.depth.saturating_add(1))
            .max()
            .unwrap_or(0);
        NativeMessages {
            class,
            items,
            depth,
            generation,
        }
    }
}

impl NativeMap {
    pub(crate) fn new(map: Arc<MapDesc>, entries: Entries<NativeValue>, generation: u64) -> Self {
        let depth = entries
            .iter()
            .map(|(_, value)| match value {
                NativeValue::Scalar(_) => 1,
                NativeValue::Message(m) => m.depth.saturating_add(2),
            })
            .max()
            .unwrap_or(0);
        NativeMap {
            map,
            entries,
            depth,
            generation,
        }
    }
}

impl Value {
    /// The levels of messages that the value adds below the message of its field, as upb's parse
    /// counts them (see [`Native::depth`]).
    pub(crate) fn depth(&self) -> u32 {
        match self {
            Value::Message(m) => m.depth.saturating_add(1),
            Value::Scalars(_) => 0,
            Value::Messages(m) => m.depth,
            Value::Map(m) => m.depth,
        }
    }

    /// Whether the value is of no values: an empty repeated or map field, which `ListFields` does
    /// not give.
    pub(crate) fn is_empty(&self) -> bool {
        match self {
            Value::Message(_) => false,
            Value::Scalars(items) => items.is_empty(),
            Value::Messages(m) => m.items.is_empty(),
            Value::Map(m) => m.entries.is_empty(),
        }
    }

    /// Whether `self` and `other` are the same value (not equal values).
    pub(crate) fn same(&self, other: &Value) -> bool {
        match (self, other) {
            (Value::Message(a), Value::Message(b)) => Arc::ptr_eq(a, b),
            (Value::Scalars(a), Value::Scalars(b)) => Arc::ptr_eq(a, b),
            (Value::Messages(a), Value::Messages(b)) => Arc::ptr_eq(a, b),
            (Value::Map(a), Value::Map(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }

    /// The value as a value of a copy made at `generation`: itself, shared, if it was made then,
    /// else a clone of it of that generation.
    pub(crate) fn at(&self, py: Python<'_>, generation: u64) -> Value {
        match self {
            Value::Message(m) => Value::Message(m.at(py, generation)),
            Value::Scalars(_) => self.clone(),
            Value::Messages(m) if m.generation != generation => grow(|| {
                Value::Messages(Arc::new(NativeMessages {
                    class: m.class.clone_ref(py),
                    items: m.items.iter().map(|item| item.at(py, generation)).collect(),
                    depth: m.depth,
                    generation,
                }))
            }),
            Value::Map(m) if m.generation != generation => grow(|| {
                let entries = m.entries.map(|value| match value {
                    NativeValue::Scalar(v) => NativeValue::Scalar(v.clone()),
                    NativeValue::Message(m) => NativeValue::Message(m.at(py, generation)),
                });
                Value::Map(Arc::new(NativeMap {
                    map: m.map.clone(),
                    entries,
                    depth: m.depth,
                    generation,
                }))
            }),
            Value::Messages(_) | Value::Map(_) => self.clone(),
        }
    }

    /// Discard the unknown fields of the messages of the value, as `discard_unknown` of a message
    /// does at `depth` (see `PyMessage::discard_unknown`), of a clone of what the value shares.
    pub(crate) fn discard_unknown(&mut self, py: Python<'_>, depth: u32) {
        match self {
            Value::Message(m) => unique(m, |m| m.clone_native(py)).discard_unknown(py, depth),
            Value::Scalars(_) => {}
            Value::Messages(m) => {
                let m = unique(m, |m| NativeMessages {
                    class: m.class.clone_ref(py),
                    items: m.items.clone(),
                    depth: m.depth,
                    generation: m.generation,
                });
                for item in &mut m.items {
                    unique(item, |m| m.clone_native(py)).discard_unknown(py, depth);
                }
            }
            Value::Map(m) => {
                let m = unique(m, |m| NativeMap {
                    map: m.map.clone(),
                    entries: m.entries.map(NativeValue::clone),
                    depth: m.depth,
                    generation: m.generation,
                });
                for value in m.entries.values_mut() {
                    if let NativeValue::Message(child) = value {
                        unique(child, |m| m.clone_native(py)).discard_unknown(py, depth);
                    }
                }
            }
        }
    }
}

/// The value of `arc`, of its own: of a clone of it (`clone`) first, if another holds it too.
fn unique<T>(arc: &mut Arc<T>, clone: impl FnOnce(&T) -> T) -> &mut T {
    if Arc::get_mut(arc).is_none() {
        *arc = Arc::new(clone(arc));
    }
    Arc::get_mut(arc).expect("a value of its own")
}

impl Native {
    /// A clone of the message, of one level: its values are shared.
    fn clone_native(&self, py: Python<'_>) -> Native {
        Native {
            fields: self.fields.clone_native_at(py, self.generation),
            depth: self.depth,
            generation: self.generation,
        }
    }

    /// Discard the unknown fields of the message and of its messages, as upb's
    /// `_upb_Message_DiscardUnknown` does at `depth` (see `PyMessage::discard_unknown`).
    fn discard_unknown(&mut self, py: Python<'_>, depth: u32) {
        let depth = depth.saturating_sub(1);
        if depth == 0 {
            return;
        }
        grow(|| {
            self.fields.unknown.clear();
            for (_, slot) in self.fields.slots.iter_mut() {
                if let Slot::Native(value) = slot {
                    value.discard_unknown(py, depth);
                }
            }
        })
    }
}

impl Fields {
    /// A clone of the fields of a native message, of one level, as fields of a copy made at
    /// `generation`: the scalars cloned and the values shared (or cloned at `generation`).
    pub(crate) fn clone_native_at(&self, py: Python<'_>, generation: u64) -> Fields {
        Fields {
            desc: self.desc.clone(),
            slots: self.slots.map(|slot| match slot {
                Slot::Scalar(v) => Slot::Scalar(v.clone()),
                Slot::Native(value) => Slot::Native(value.at(py, generation)),
                _ => unreachable!("a native message holds no object"),
            }),
            unknown: self.unknown.clone(),
        }
    }
}

/// A native message is dropped a level at a time: the values that it alone holds are taken onto a
/// stack, so that a message of many levels is not dropped through a recursion of as many levels,
/// which CPython's trashcan bounds for the objects of the messages, and nothing would bound here.
impl Drop for Native {
    fn drop(&mut self) {
        // A message of few levels below it drops as it is, of as many levels of recursion.
        if self.depth < SHALLOW {
            return;
        }
        let mut values = Vec::new();
        take_own_values(&mut self.fields.slots, &mut values);
        while let Some(value) = values.pop() {
            if value.depth() < SHALLOW {
                continue;
            }
            match value {
                Value::Message(m) => {
                    if let Ok(mut m) = Arc::try_unwrap(m) {
                        take_own_values(&mut m.fields.slots, &mut values);
                    }
                }
                Value::Messages(m) => {
                    if let Ok(m) = Arc::try_unwrap(m) {
                        values.extend(m.items.into_iter().map(Value::Message));
                    }
                }
                Value::Map(m) => {
                    if let Ok(m) = Arc::try_unwrap(m) {
                        values.extend(m.entries.into_entries().filter_map(
                            |(_, value)| match value {
                                NativeValue::Message(m) => Some(Value::Message(m)),
                                NativeValue::Scalar(_) => None,
                            },
                        ));
                    }
                }
                Value::Scalars(_) => {}
            }
        }
    }
}

/// The levels below a native message (see [`Native::depth`]) from which its drop takes apart the
/// values that it alone holds, rather than dropping them through a recursion of as many levels.
const SHALLOW: u32 = 64;

/// Take the native values of messages of `slots` that nothing else holds onto `values` (those that
/// others hold are only released as the slots drop).
fn take_own_values(slots: &mut Slots, values: &mut Vec<Value>) {
    for (_, slot) in slots.iter_mut() {
        let own = match slot {
            Slot::Native(Value::Message(m)) => Arc::strong_count(m) == 1,
            Slot::Native(Value::Messages(m)) => Arc::strong_count(m) == 1,
            Slot::Native(Value::Map(m)) => Arc::strong_count(m) == 1,
            _ => false,
        };
        if own {
            if let Slot::Native(value) = std::mem::replace(slot, HELD) {
                values.push(value);
            }
        }
    }
}

/// A native copy of `message`, of the levels `levels` that may still be below it, made at
/// `generation`, a step of `budget` for each message that it copies, as a copy of the runtime
/// steps (see [`copy_parts`]).
pub(crate) fn copy_message(
    py: Python<'_>,
    message: &Bound<'_, PyMessage>,
    at: At,
    generation: u64,
    budget: &ConversionBudget,
) -> Result<Native, CopyError> {
    if at.watch == Some(message.as_ptr()) {
        return Err(CopyError::Holds);
    }
    at.grown(|| {
        budget.step(py)?;
        let (slots, unknown, desc) = copy_parts(py, message, at, generation, budget, true)?;
        let desc = match desc {
            Some(desc) => desc,
            None => message.borrow().desc.clone(),
        };
        let fields = Fields {
            desc,
            slots,
            unknown,
        };
        Ok(Native::new(fields, generation))
    })
}

/// The fields of a copy of a message, without its type: its slots and its unknown fields.
pub(crate) type Parts = (Slots, Vec<u8>);

/// The fields of a copy of a message, with its type when it was asked for (see [`copy_parts`]).
type PartsOfType = (Slots, Vec<u8>, Option<Arc<MessageDesc>>);

/// A native copy of the set fields and the unknown fields of `message` (see [`copy_parts`]), for an
/// object of the message's type, which has the type already.
pub(crate) fn copy_slots(
    py: Python<'_>,
    message: &Bound<'_, PyMessage>,
    at: At,
    generation: u64,
    budget: &ConversionBudget,
) -> Result<Parts, CopyError> {
    let (slots, unknown, _) = copy_parts(py, message, at, generation, budget, false)?;
    Ok((slots, unknown))
}

/// A native copy of the set fields and the unknown fields of `message`, of its type too when
/// `with_desc` (of a native message, which an object does not need), of the levels of `at` that
/// may still be below it, as upb's parse counts them: [`CopyError::TooDeep`] at the first
/// message past them. Its scalars, its repeated scalars and the native values that it holds
/// (shared) are read under a short borrow; the messages, the containers of messages and the maps
/// that it holds are copied after it, in their places, so that the checkpoints of the copy of a
/// message of many messages, whose Python code may use the message, find no borrow of it held.
fn copy_parts(
    py: Python<'_>,
    message: &Bound<'_, PyMessage>,
    at: At,
    generation: u64,
    budget: &ConversionBudget,
    with_desc: bool,
) -> Result<PartsOfType, CopyError> {
    let levels = at.levels;
    enum Held {
        Message(Py<PyMessage>),
        Composite(MessageClass, Vec<Py<PyMessage>>),
        Map(MapEntries),
    }
    // The messages, containers of messages and maps that the message holds, copied after its
    // borrow, each with the index of its field: the first apart, as most hold one or none.
    let mut first: Option<(usize, Held)> = None;
    let mut rest: Vec<(usize, Held)> = Vec::new();
    let (mut slots, desc, unknown) = {
        let m = message.borrow();
        // A message of scalars only, as most of the messages of a large message are, is copied
        // of its scalars at once.
        if m.slots
            .iter()
            .all(|(_, slot)| matches!(slot, Slot::Scalar(_)))
        {
            let slots = m.slots.map(|slot| match slot {
                Slot::Scalar(v) => Slot::Scalar(v.clone()),
                _ => HELD,
            });
            return Ok((slots, m.unknown.clone(), with_desc.then(|| m.desc.clone())));
        }
        // A slot for each of the message's, in their order, written in its place: of a held
        // value, `HELD`, until its copy is made.
        let mut copy_into = |index: usize, slot: &Slot, out: &mut Slot| -> Result<(), CopyError> {
            let held = match slot {
                Slot::Scalar(v) => {
                    *out = Slot::Scalar(v.clone());
                    return Ok(());
                }
                Slot::Native(value) => {
                    if levels.is_some_and(|levels| value.depth() > levels) {
                        return Err(CopyError::TooDeep);
                    }
                    *out = Slot::Native(value.at(py, generation));
                    return Ok(());
                }
                Slot::Repeated(c) => {
                    let items = Arc::new(c.bind(py).borrow().items.clone());
                    *out = Slot::Native(Value::Scalars(items));
                    return Ok(());
                }
                Slot::Message(c) => Held::Message(c.clone_ref(py)),
                Slot::Composite(c) => {
                    let c = c.bind(py).borrow();
                    let items = c.items.iter().map(|m| m.clone_ref(py)).collect();
                    Held::Composite(c.class.clone_ref(py), items)
                }
                Slot::Map(c) => Held::Map(c.bind(py).borrow().entries_for_copy(py)),
            };
            match first {
                None => first = Some((index, held)),
                Some(_) => rest.push((index, held)),
            }
            Ok(())
        };
        let slots = match &m.slots {
            Slots::Empty => Slots::Empty,
            Slots::One((index, slot)) => {
                let mut out = HELD;
                copy_into(*index, slot, &mut out)?;
                Slots::One((*index, out))
            }
            Slots::Many(entries) => {
                let mut copies: Vec<(usize, Slot)> =
                    entries.iter().map(|(index, _)| (*index, HELD)).collect();
                for ((index, slot), (_, out)) in entries.iter().zip(copies.iter_mut()) {
                    copy_into(*index, slot, out)?;
                }
                Slots::Many(copies)
            }
        };
        (slots, with_desc.then(|| m.desc.clone()), m.unknown.clone())
    };
    for (index, held) in first.into_iter().chain(rest) {
        let value = match held {
            Held::Message(child) => Value::Message(Arc::new(copy_message(
                py,
                child.bind(py),
                at.below()?,
                generation,
                budget,
            )?)),
            Held::Composite(class, items) => {
                let mut copies = Vec::with_capacity(items.len());
                if !items.is_empty() {
                    let at = at.below()?;
                    for item in &items {
                        let copy = copy_message(py, item.bind(py), at, generation, budget)?;
                        copies.push(Arc::new(copy));
                    }
                }
                Value::Messages(Arc::new(NativeMessages::new(class, copies, generation)))
            }
            Held::Map(entries) => {
                Value::Map(Arc::new(entries.native_copy(py, at, generation, budget)?))
            }
        };
        if let Some(slot) = slots.get_mut(index) {
            *slot = Slot::Native(value);
        }
    }
    Ok((slots, unknown, desc))
}

/// The slot that a copy of a message holds in the place of a message, a container of messages or a
/// map of the message until its copy is made, and that a native message holds in the place of a
/// value that its drop takes.
const HELD: Slot = Slot::Scalar(Scalar::Bool(false));

/// A copy of the set fields and the unknown fields of `message` of any depth, made now (see
/// [`copy_parts`]).
pub(crate) fn copy_slots_now(
    py: Python<'_>,
    message: &Bound<'_, PyMessage>,
    budget: &ConversionBudget,
) -> PyResult<Parts> {
    match copy_slots(py, message, At::top(None), generation(), budget) {
        Ok(parts) => Ok(parts),
        Err(CopyError::Python(err)) => Err(err),
        Err(CopyError::TooDeep | CopyError::Holds) => {
            unreachable!("a copy of any depth, watching no message, stops for neither")
        }
    }
}

/// An object of the native message `message`, as a copy made now of an object of it is made (see
/// `message::copy_as`): of `class` when it is given, else of the class registered now of its type,
/// of its fields of now, a step of `budget`.
pub(crate) fn duplicate_native<'py>(
    py: Python<'py>,
    message: &Native,
    class: Option<&MessageClass>,
    budget: &ConversionBudget,
) -> PyResult<Bound<'py, PyMessage>> {
    budget.step(py)?;
    let fields = message.fields.clone_native_at(py, generation());
    let copy = match class {
        Some(class) => class.new_message(py)?,
        None => MessageClass::of(py, &fields.desc.full_name)?.new_message(py)?,
    };
    copy.borrow_mut().take_fields(fields);
    Ok(copy)
}

/// An object of the native message `message`, as [`duplicate_native`] makes it, of its own fields,
/// moved where no other value holds the message, as a merge takes the messages of the snapshot
/// that it made (see `message::Snapshot`).
pub(crate) fn duplicate_owned<'py>(
    py: Python<'py>,
    message: Arc<Native>,
    class: Option<&MessageClass>,
    budget: &ConversionBudget,
) -> PyResult<Bound<'py, PyMessage>> {
    budget.step(py)?;
    let fields = fields_at(py, message, generation());
    let copy = match class {
        Some(class) => class.new_message(py)?,
        None => MessageClass::of(py, &fields.desc.full_name)?.new_message(py)?,
    };
    copy.borrow_mut().take_fields(fields);
    Ok(copy)
}

/// The fields of the native message `message`, as fields of a copy made at `generation`: its own,
/// moved, if no other value holds it and it was made then, else a copy of one level of them, at
/// `generation` (see [`Fields::clone_native_at`]).
pub(crate) fn fields_at(py: Python<'_>, message: Arc<Native>, generation: u64) -> Fields {
    if message.generation == generation {
        return own_fields(py, message);
    }
    message.fields.clone_native_at(py, generation)
}

/// Set the fields of `message`, a new object of the type of `native`, of one level of `native`.
fn fill(message: &Bound<'_, PyMessage>, native: &Native, py: Python<'_>) {
    message
        .borrow_mut()
        .take_fields(native.fields.clone_native_at(py, native.generation));
}

/// The object of a native value of a field, as the slot of the field holds it, made of one level
/// of the value but empty (see [`made_of`]): its messages take the fields of the value
/// ([`Made::fill`]) once the field holds the object in the place of the value, so that the fields
/// that no other message holds are moved rather than copied.
pub(crate) struct Made {
    /// The slot of the object.
    pub(crate) slot: Slot,
    /// The messages of a container or a map, in the order of the messages of the value (the
    /// message of a message field is that of the slot).
    messages: Vec<Py<PyMessage>>,
    /// The container of a repeated scalar field.
    scalars: Option<Py<RepeatedScalar>>,
}

/// The object of the native value `value` of `field`, of one level, empty (see [`Made`]): a
/// message, a container of messages (of the class of its container), a container of scalars, or
/// a map of the messages of its entries, held by no message, as the copies of the runtime were. A
/// step of `budget`, when it is given, for each message of a container or a map, so that a field of
/// many values can be interrupted.
pub(crate) fn made_of(
    py: Python<'_>,
    field: &FieldDesc,
    value: &Value,
    budget: Option<&ConversionBudget>,
) -> PyResult<Made> {
    let step = |py| budget.map_or(Ok(()), |budget| budget.step(py));
    let mut messages = Vec::new();
    let mut scalars = None;
    let slot = match value {
        Value::Message(m) => {
            let object = MessageClass::at(py, &m.fields.desc, m.generation)?.new_message(py)?;
            Slot::Message(object.unbind())
        }
        Value::Scalars(_) => {
            let container = Py::new(
                py,
                RepeatedScalar::new(
                    field.kind,
                    field.full_name.clone(),
                    field.type_name.clone(),
                    Vec::new(),
                    None,
                ),
            )?;
            scalars = Some(container.clone_ref(py));
            Slot::Repeated(container)
        }
        Value::Messages(m) => {
            messages.reserve(m.items.len());
            for _ in &m.items {
                step(py)?;
                messages.push(m.class.new_message(py)?.unbind());
            }
            let items = messages.iter().map(|m| m.clone_ref(py)).collect();
            let container = RepeatedComposite::new(m.class.clone_ref(py), items, None);
            Slot::Composite(Py::new(py, container)?)
        }
        Value::Map(m) => {
            let entries = m.entries.try_map(|value| -> PyResult<MapValue> {
                Ok(match value {
                    NativeValue::Scalar(v) => MapValue::Scalar(v.clone()),
                    NativeValue::Message(m) => {
                        step(py)?;
                        let class = MessageClass::at(py, &m.fields.desc, m.generation)?;
                        let object = class.new_message(py)?.unbind();
                        messages.push(object.clone_ref(py));
                        MapValue::Message(object)
                    }
                })
            })?;
            let map = MapContainer::of_entries(field.name.clone(), m.map.clone(), entries);
            Slot::Map(map.into_object(py)?)
        }
    };
    Ok(Made {
        slot,
        messages,
        scalars,
    })
}

impl Made {
    /// Give the objects the fields of `value`, the value that they were made of (see
    /// [`made_of`]), which the field no longer holds: moved where no other message holds them,
    /// else copied, of one level. It runs no Python code.
    pub(crate) fn fill(self, py: Python<'_>, value: Value) {
        let mut objects = self.messages.into_iter();
        let fill_next = |message: Arc<Native>| {
            if let Some(object) = objects.next() {
                object
                    .bind(py)
                    .borrow_mut()
                    .take_fields(own_fields(py, message));
            }
        };
        match value {
            Value::Message(m) => {
                if let Slot::Message(object) = &self.slot {
                    let fields = own_fields(py, m);
                    object.bind(py).borrow_mut().take_fields(fields);
                }
            }
            Value::Scalars(items) => {
                if let Some(container) = self.scalars {
                    let items = Arc::try_unwrap(items).unwrap_or_else(|items| (*items).clone());
                    container.bind(py).borrow_mut().items = items;
                }
            }
            Value::Messages(m) => match Arc::try_unwrap(m) {
                Ok(m) => m.items.into_iter().for_each(fill_next),
                Err(m) => m.items.iter().cloned().for_each(fill_next),
            },
            Value::Map(m) => {
                let messages = |value: &NativeValue| match value {
                    NativeValue::Message(m) => Some(m.clone()),
                    NativeValue::Scalar(_) => None,
                };
                let held: Vec<Arc<Native>> =
                    m.entries.iter().filter_map(|(_, v)| messages(v)).collect();
                // The map's own reference to each message is released first, so that a message
                // of a map that no other holds is moved.
                drop(m);
                held.into_iter().for_each(fill_next);
            }
        }
    }
}

/// The fields of the native message `message`: its own, moved, if no other holds it, else a copy
/// of one level of them.
fn own_fields(py: Python<'_>, message: Arc<Native>) -> Fields {
    match Arc::try_unwrap(message) {
        Ok(mut message) => Fields {
            desc: message.fields.desc.clone(),
            slots: std::mem::take(&mut message.fields.slots),
            unknown: std::mem::take(&mut message.fields.unknown),
        },
        Err(message) => message.fields.clone_native_at(py, message.generation),
    }
}

/// The object of the native message `message`, of one level: of the class registered of its type
/// when its copy was made (see [`Native::generation`]), of its fields.
pub(crate) fn message_object<'py>(
    py: Python<'py>,
    message: &Native,
) -> PyResult<Bound<'py, PyMessage>> {
    let object = MessageClass::at(py, &message.fields.desc, message.generation)?.new_message(py)?;
    fill(&object, message, py);
    Ok(object)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbmsg::pool::pool;

    /// A native message of the fields `fields` of `index` of the type `full_name`, holding `child`.
    fn holding(full_name: &str, field: &str, child: Native) -> Native {
        let desc = pool()
            .message(full_name)
            .expect("a type of the pool")
            .clone();
        let index = desc.by_name[field];
        let mut fields = Fields {
            desc: desc.clone(),
            slots: Slots::default(),
            unknown: Vec::new(),
        };
        let value = Slot::Native(Value::Message(Arc::new(child)));
        fields.slots.set(&desc, index, value);
        Native::new(fields, 0)
    }

    /// A native message of a million levels drops a level at a time, where a recursion of as many
    /// levels would overflow the stack of the thread of the test, and knows its levels as upb's
    /// parse counts them.
    #[test]
    fn deep_native_messages_drop_a_level_at_a_time() {
        let relation = pool()
            .message("spark.connect.Relation")
            .expect("Relation")
            .clone();
        let mut message = Native::new(
            Fields {
                desc: relation,
                slots: Slots::default(),
                unknown: Vec::new(),
            },
            0,
        );
        for _ in 0..500_000 {
            let limit = holding("spark.connect.Limit", "input", message);
            message = holding("spark.connect.Relation", "limit", limit);
        }
        assert_eq!(message.depth, 1_000_000);
        assert_eq!(Value::Message(Arc::new(message)).depth(), 1_000_001);
    }
}
