//! The native values of the fields of the messages: a copy of the value of a message, repeated or
//! map field, held in Rust, of no object of Python, which a message holds in the slot of its field
//! ([`Slot::Native`]) until Python reads the field, when the message makes the object of the value
//! (see [`made_of`] and `PyMessage::materialize`): of all its levels at once, of a value of few
//! messages ([`Whole`]), else of one level, a message of its fields, whose own values stay native
//! until they are read, a container of its messages, a map of its entries. The container of a
//! repeated message field of a larger value holds its messages natively and makes their objects as
//! they are read (`RepeatedComposite::of_native`).
//!
//! A copy of a message ([`copy_message`], [`copy_slots`]) is native: the constructor of a message
//! given a message of the runtime (as `proto.MergeAction(condition=c)`), `CopyFrom`, `copy.copy`
//! and `copy.deepcopy` copy the message without an object of Python for each message that it holds,
//! which the copies of the runtime made before, as an object of Python is a message of the runtime.
//! `MergeFrom` copies its argument so before it merges it (`message::Snapshot`), as upb serializes
//! it first, so that a merge into a message that the argument holds reads what it was. So is a
//! parse into a message that holds no field (`codec::NativeParse`), of `FromString`,
//! `ParseFromString` and the drop-in's own parses (`Column.to_plan`).
//! A native value is never changed (but by [`Value::discard_unknown`], which clones what it shares
//! first), so a copy of a message that holds native values shares them. A native message is of a
//! count of its references ([`NRef`], see `arena`), of a slab of the messages of its parse, or of
//! an allocation of its own, of a copy.
//!
//! What Python reads of a native value is what it read of the objects of the copy before: the
//! object that it makes of a value is of the class that the copy would have made then (see
//! [`Native::generation`]), held by no message as a copy's objects were, and the field gives that
//! object from then on, so its identity is kept.

use std::sync::Arc;

use pyo3::prelude::*;

use super::arena::{NRef, Releases};
use super::containers::{
    Entries, MapContainer, MapEntries, MapValue, RepeatedComposite, RepeatedScalar,
};
use super::grow;
use super::message::{generation, new_message_at, Fields, MessageClass, PyMessage, Slot, Slots};
use super::pool::{FieldDesc, MapDesc, MessageDesc};
use super::scalar::Scalar;
use crate::interrupt::ConversionBudget;

/// A native value of a field: of a message field, a repeated scalar field, a repeated message
/// field or a map field.
#[derive(Clone)]
pub(crate) enum Value {
    Message(NRef),
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
    /// The messages of the message, itself and those below it, of which Python's read of the
    /// message makes the objects at once, if they are few (see [`WHOLE`]).
    pub(crate) count: u32,
    /// The generation of the classes of the messages (see `message::generation`) of the copy that
    /// made the message, which the objects of it and of the messages below it are made of: of the
    /// class of their type registered then, as the copy made them before.
    pub(crate) generation: u64,
}

/// The native value of a repeated message field: its messages, of the class of the messages of
/// its container, which the objects of its messages are made of.
pub(crate) struct NativeMessages {
    pub(crate) class: MessageClass,
    pub(crate) items: Items,
    /// The levels of messages that the field adds below its message: one more than the deepest of
    /// its messages, none of no messages.
    pub(crate) depth: u32,
    /// The messages of the field and those below them (see [`Native::count`]).
    pub(crate) count: u32,
    pub(crate) generation: u64,
}

/// The native value of a map field: its entries, in the order of their insertion.
pub(crate) struct NativeMap {
    pub(crate) map: Arc<MapDesc>,
    pub(crate) entries: Entries<NativeValue>,
    /// The levels of messages that the map adds below its message: an entry is one, and the
    /// message of its value another.
    pub(crate) depth: u32,
    /// The messages of the values of the map and those below them (see [`Native::count`]).
    pub(crate) count: u32,
    pub(crate) generation: u64,
}

/// The value of an entry of a native map.
#[derive(Clone)]
pub(crate) enum NativeValue {
    Scalar(Scalar),
    Message(NRef),
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
        let count = count_of(&fields.slots);
        Native {
            fields,
            depth,
            count,
            generation,
        }
    }

    /// An empty native message of the type `desc`, made at `generation`, which a native parse
    /// fills (see `codec::NativeParse`).
    pub(crate) fn empty(desc: &'static MessageDesc, generation: u64) -> Native {
        Native {
            fields: Fields {
                desc,
                slots: Slots::default(),
                unknown: Vec::new(),
            },
            depth: 0,
            count: 1,
            generation,
        }
    }

    /// Count the levels below the message and its messages again, of its values, once a parse has
    /// changed them.
    pub(crate) fn recount(&mut self) {
        self.count = count_of(&self.fields.slots);
        self.depth = self
            .fields
            .slots
            .iter()
            .map(|(_, slot)| match slot {
                Slot::Native(value) => value.depth(),
                _ => 0,
            })
            .max()
            .unwrap_or(0);
    }

    /// The message as a message of a copy made at `generation`: itself, if it was made then, else
    /// a clone of it of that generation (which a class registered since makes of another class).
    pub(crate) fn at(this: &NRef, py: Python<'_>, generation: u64) -> NRef {
        if this.generation == generation {
            return this.clone();
        }
        grow(|| {
            NRef::new(Native {
                fields: this.fields.clone_native_at(py, generation),
                depth: this.depth,
                count: this.count,
                generation,
            })
        })
    }
}

/// The messages of a native message of the slots `slots`, itself and those of its values.
fn count_of(slots: &Slots) -> u32 {
    slots.iter().fold(1, |count, (_, slot)| match slot {
        Slot::Native(value) => count.saturating_add(value.count()),
        _ => count,
    })
}

/// The messages of the native value of a repeated message field: up to two in place, more of a
/// vector, as most repeated message fields of a plan hold one or two (the arguments of a
/// function), of no allocation of their own then. Of the slice of the messages, in their order.
#[derive(Clone, Default)]
pub(crate) enum Items {
    #[default]
    Empty,
    One([NRef; 1]),
    Two([NRef; 2]),
    Many(Vec<NRef>),
}

impl Items {
    pub(crate) fn push(&mut self, item: NRef) {
        *self = match std::mem::take(self) {
            Items::Empty => Items::One([item]),
            Items::One([a]) => Items::Two([a, item]),
            Items::Two([a, b]) => {
                let mut items = Vec::with_capacity(4);
                items.extend([a, b, item]);
                Items::Many(items)
            }
            Items::Many(mut items) => {
                items.push(item);
                Items::Many(items)
            }
        };
    }
}

impl From<Vec<NRef>> for Items {
    fn from(items: Vec<NRef>) -> Items {
        match items.len() {
            0 => Items::Empty,
            1 | 2 => {
                let mut items = items.into_iter();
                let mut out = Items::Empty;
                for item in items.by_ref() {
                    out.push(item);
                }
                out
            }
            _ => Items::Many(items),
        }
    }
}

impl std::ops::Deref for Items {
    type Target = [NRef];

    fn deref(&self) -> &[NRef] {
        match self {
            Items::Empty => &[],
            Items::One(items) => items,
            Items::Two(items) => items,
            Items::Many(items) => items,
        }
    }
}

impl std::ops::DerefMut for Items {
    fn deref_mut(&mut self) -> &mut [NRef] {
        match self {
            Items::Empty => &mut [],
            Items::One(items) => items,
            Items::Two(items) => items,
            Items::Many(items) => items,
        }
    }
}

/// The messages of [`Items`], by value, in their order.
pub(crate) enum ItemsIntoIter {
    Small(std::iter::Chain<std::array::IntoIter<NRef, 1>, std::option::IntoIter<NRef>>),
    Many(std::vec::IntoIter<NRef>),
}

impl Iterator for ItemsIntoIter {
    type Item = NRef;

    fn next(&mut self) -> Option<NRef> {
        match self {
            ItemsIntoIter::Small(items) => items.next(),
            ItemsIntoIter::Many(items) => items.next(),
        }
    }
}

impl IntoIterator for Items {
    type Item = NRef;
    type IntoIter = ItemsIntoIter;

    fn into_iter(self) -> ItemsIntoIter {
        let small = |first: [NRef; 1], second: Option<NRef>| {
            ItemsIntoIter::Small(first.into_iter().chain(second))
        };
        match self {
            Items::Empty => ItemsIntoIter::Many(Vec::new().into_iter()),
            Items::One(first) => small(first, None),
            Items::Two([a, b]) => small([a], Some(b)),
            Items::Many(items) => ItemsIntoIter::Many(items.into_iter()),
        }
    }
}

impl<'a> IntoIterator for &'a Items {
    type Item = &'a NRef;
    type IntoIter = std::slice::Iter<'a, NRef>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> IntoIterator for &'a mut Items {
    type Item = &'a mut NRef;
    type IntoIter = std::slice::IterMut<'a, NRef>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl FromIterator<NRef> for Items {
    fn from_iter<I: IntoIterator<Item = NRef>>(iter: I) -> Items {
        let mut items = Items::Empty;
        for item in iter {
            items.push(item);
        }
        items
    }
}

impl Items {
    /// The messages, of a vector (of no allocation of the messages of a vector).
    pub(crate) fn into_vec(self) -> Vec<NRef> {
        match self {
            Items::Many(items) => items,
            items => items.into_iter().collect(),
        }
    }
}

impl NativeMessages {
    pub(crate) fn new(class: MessageClass, items: Vec<NRef>, generation: u64) -> Self {
        let depth = items
            .iter()
            .map(|item| item.depth.saturating_add(1))
            .max()
            .unwrap_or(0);
        let count = items
            .iter()
            .fold(0u32, |count, item| count.saturating_add(item.count));
        NativeMessages {
            class,
            items: Items::from(items),
            depth,
            count,
            generation,
        }
    }
}

/// The messages of a list of messages of a parse of at most one level below them released one after
/// another, of the releases of the list (`NRef::release_with`): the nodes of a slab that the list
/// releases in a row (of the messages of a flat array, which a parse makes in a row) are released
/// of one atomic operation for each slab, rather than of one for each message. The messages of a
/// short list, of messages of more levels, whose nodes are apart in their slabs (between those of
/// the messages below each), and of a copy, which are allocations of their own, are released as
/// they are.
impl Drop for NativeMessages {
    #[inline]
    fn drop(&mut self) {
        if self.items.len() >= BATCHED && self.depth <= 2 && NRef::in_slab(&self.items[0]) {
            release_batched(std::mem::take(&mut self.items));
        }
    }
}

/// Release the messages `items` of a list, of the releases of the list (see the drop of
/// `NativeMessages`).
#[inline(never)]
fn release_batched(items: Items) {
    release_list(items);
}

/// Release the messages `items`, one after another, of one release of the nodes of each slab that
/// they release in a row (`NRef::release_with`).
pub(crate) fn release_list(items: impl IntoIterator<Item = NRef>) {
    let mut releases = Releases::default();
    for item in items {
        NRef::release_with(item, &mut releases);
    }
}

/// The messages of a list from which its drop releases their nodes of a slab at once (see the
/// drop of `NativeMessages`).
const BATCHED: usize = 8;

impl NativeMessages {
    /// Append `item`, a message that a native parse read, counting its levels.
    pub(crate) fn push(&mut self, item: NRef) {
        self.depth = self.depth.max(item.depth.saturating_add(1));
        self.count = self.count.saturating_add(item.count);
        self.items.push(item);
    }
}

/// The levels of messages that an entry of `value` adds below the message of its map, as
/// `NativeMap::depth` counts them.
fn entry_depth(value: &NativeValue) -> u32 {
    match value {
        NativeValue::Scalar(_) => 1,
        NativeValue::Message(m) => m.depth.saturating_add(2),
    }
}

impl NativeMap {
    /// Set `value` of `key`, of an entry that a native parse read, as a map sets it, counting the
    /// levels of the map again of the entry that it replaces, if any.
    pub(crate) fn insert(&mut self, key: Scalar, value: NativeValue) {
        let depth = entry_depth(&value);
        let count = entry_count(&value);
        if self.entries.insert(key, value).is_some() {
            self.depth = self
                .entries
                .iter()
                .map(|(_, v)| entry_depth(v))
                .max()
                .unwrap_or(0);
            self.count = entries_count(&self.entries);
        } else {
            self.depth = self.depth.max(depth);
            self.count = self.count.saturating_add(count);
        }
    }

    pub(crate) fn new(map: Arc<MapDesc>, entries: Entries<NativeValue>, generation: u64) -> Self {
        let depth = entries
            .iter()
            .map(|(_, value)| match value {
                NativeValue::Scalar(_) => 1,
                NativeValue::Message(m) => m.depth.saturating_add(2),
            })
            .max()
            .unwrap_or(0);
        let count = entries_count(&entries);
        NativeMap {
            map,
            entries,
            depth,
            count,
            generation,
        }
    }
}

/// The messages of the value of an entry of a native map and those below it.
fn entry_count(value: &NativeValue) -> u32 {
    match value {
        NativeValue::Scalar(_) => 0,
        NativeValue::Message(m) => m.count,
    }
}

/// The messages of the values of the entries `entries` and those below them.
fn entries_count(entries: &Entries<NativeValue>) -> u32 {
    entries.iter().fold(0u32, |count, (_, value)| {
        count.saturating_add(entry_count(value))
    })
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

    /// The messages of the value and those below them (see [`Native::count`]).
    pub(crate) fn count(&self) -> u32 {
        match self {
            Value::Message(m) => m.count,
            Value::Scalars(_) => 0,
            Value::Messages(m) => m.count,
            Value::Map(m) => m.count,
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
            (Value::Message(a), Value::Message(b)) => NRef::ptr_eq(a, b),
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
            Value::Message(m) => Value::Message(Native::at(m, py, generation)),
            Value::Scalars(_) => self.clone(),
            Value::Messages(m) if m.generation != generation => grow(|| {
                Value::Messages(Arc::new(NativeMessages {
                    class: m.class.clone_ref(py),
                    items: m
                        .items
                        .iter()
                        .map(|item| Native::at(item, py, generation))
                        .collect(),
                    depth: m.depth,
                    count: m.count,
                    generation,
                }))
            }),
            Value::Map(m) if m.generation != generation => grow(|| {
                let entries = m.entries.map(|value| match value {
                    NativeValue::Scalar(v) => NativeValue::Scalar(v.clone()),
                    NativeValue::Message(m) => NativeValue::Message(Native::at(m, py, generation)),
                });
                Value::Map(Arc::new(NativeMap {
                    map: m.map.clone(),
                    entries,
                    depth: m.depth,
                    count: m.count,
                    generation,
                }))
            }),
            Value::Messages(_) | Value::Map(_) => self.clone(),
        }
    }

    /// Move the messages of the value, of a value of up to [`UNPIN`] messages, out of the slabs of
    /// their parse (see `arena`), each of an allocation of its own, where no other value holds
    /// them: of a message that outlives the container that held it (see `PyMessage::unpin`), so
    /// that it keeps alive no slab of the messages released with the container, as a slab is
    /// released with the last of its messages. A larger value holds most of the nodes of its slabs
    /// itself, and is left as it is.
    pub(crate) fn unpin(&mut self) {
        if self.count() <= UNPIN {
            unpin_value(self);
        }
    }

    /// Discard the unknown fields of the messages of the value, as `discard_unknown` of a message
    /// does at `depth` (see `PyMessage::discard_unknown`), of a clone of what the value shares.
    pub(crate) fn discard_unknown(&mut self, py: Python<'_>, depth: u32) {
        match self {
            Value::Message(m) => {
                unique_native(m, |m| m.clone_native(py)).discard_unknown(py, depth)
            }
            Value::Scalars(_) => {}
            Value::Messages(m) => {
                let m = unique(m, |m| NativeMessages {
                    class: m.class.clone_ref(py),
                    items: m.items.clone(),
                    depth: m.depth,
                    count: m.count,
                    generation: m.generation,
                });
                for item in &mut m.items {
                    unique_native(item, |m| m.clone_native(py)).discard_unknown(py, depth);
                }
            }
            Value::Map(m) => {
                let m = unique(m, |m| NativeMap {
                    map: m.map.clone(),
                    entries: m.entries.map(NativeValue::clone),
                    depth: m.depth,
                    count: m.count,
                    generation: m.generation,
                });
                for value in m.entries.values_mut() {
                    if let NativeValue::Message(child) = value {
                        unique_native(child, |m| m.clone_native(py)).discard_unknown(py, depth);
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

/// The messages of a native value up to which [`Value::unpin`] moves them out of their slabs.
const UNPIN: u32 = 256;

/// [`Value::unpin`] of `value`, of any size, a level at a time.
fn unpin_value(value: &mut Value) {
    match value {
        Value::Message(m) => unpin_native(m),
        Value::Messages(m) => {
            if let Some(m) = Arc::get_mut(m) {
                m.items.iter_mut().for_each(unpin_native);
            }
        }
        Value::Map(m) => {
            if let Some(m) = Arc::get_mut(m) {
                for value in m.entries.values_mut() {
                    if let NativeValue::Message(m) = value {
                        unpin_native(m);
                    }
                }
            }
        }
        Value::Scalars(_) => {}
    }
}

/// The native messages of `value`, of all its levels, that are nodes of a slab of a parse (of
/// `_proto_slab_messages` of the tests).
pub(crate) fn slab_messages(value: &Value) -> usize {
    fn of(message: &NRef) -> usize {
        let below: usize = message
            .fields
            .slots
            .iter()
            .map(|(_, slot)| match slot {
                Slot::Native(value) => slab_messages(value),
                _ => 0,
            })
            .sum();
        below + usize::from(NRef::in_slab(message))
    }
    grow(|| match value {
        Value::Message(m) => of(m),
        Value::Messages(m) => m.items.iter().map(of).sum(),
        Value::Map(m) => m
            .entries
            .iter()
            .map(|(_, value)| match value {
                NativeValue::Message(m) => of(m),
                NativeValue::Scalar(_) => 0,
            })
            .sum(),
        Value::Scalars(_) => 0,
    })
}

/// Move the native message `message` and the messages below it out of their slabs, of those that
/// no other value holds (see [`Value::unpin`]).
pub(crate) fn unpin_native(message: &mut NRef) {
    grow(|| {
        if let Some(native) = NRef::get_mut(message) {
            for (_, slot) in native.fields.slots.iter_mut() {
                if let Slot::Native(value) = slot {
                    unpin_value(value);
                }
            }
        }
        NRef::rebox(message);
    })
}

/// The messages of the native value `value` of a repeated message field, of its own: of a clone of
/// one level of it first, if another holds it too.
pub(crate) fn unique_messages<'v>(
    py: Python<'_>,
    value: &'v mut Arc<NativeMessages>,
) -> &'v mut NativeMessages {
    unique(value, |m| NativeMessages {
        class: m.class.clone_ref(py),
        items: m.items.clone(),
        depth: m.depth,
        count: m.count,
        generation: m.generation,
    })
}

/// The native message of `message`, of its own: of a clone of it (`clone`) first, if another holds
/// it too.
pub(crate) fn unique_native(
    message: &mut NRef,
    clone: impl FnOnce(&Native) -> Native,
) -> &mut Native {
    if NRef::get_mut(message).is_none() {
        *message = NRef::new(clone(message));
    }
    NRef::get_mut(message).expect("a message of its own")
}

impl Native {
    /// A clone of the message, of one level: its values are shared.
    pub(crate) fn clone_native(&self, py: Python<'_>) -> Native {
        Native {
            fields: self.fields.clone_native_at(py, self.generation),
            depth: self.depth,
            count: self.count,
            generation: self.generation,
        }
    }

    /// Discard the unknown fields of the message and of its messages, as upb's
    /// `_upb_Message_DiscardUnknown` does at `depth` (see `PyMessage::discard_unknown`).
    pub(crate) fn discard_unknown(&mut self, py: Python<'_>, depth: u32) {
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
            desc: self.desc,
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
                    if let Ok(mut m) = NRef::try_unwrap(m) {
                        take_own_values(&mut m.fields.slots, &mut values);
                    }
                }
                Value::Messages(m) => {
                    if let Ok(mut m) = Arc::try_unwrap(m) {
                        values.extend(std::mem::take(&mut m.items).into_iter().map(Value::Message));
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
            Slot::Native(Value::Message(m)) => NRef::strong_count(m) == 1,
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
            None => message.borrow().desc,
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
type PartsOfType = (Slots, Vec<u8>, Option<&'static MessageDesc>);

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
        /// The class, the objects and the messages held natively that follow them.
        Composite(MessageClass, Vec<Py<PyMessage>>, Vec<NRef>),
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
            return Ok((slots, m.unknown.clone(), with_desc.then(|| m.desc)));
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
                    let pending = c.pending_natives();
                    let depth = pending.iter().map(|m| m.depth.saturating_add(1)).max();
                    if levels
                        .zip(depth)
                        .is_some_and(|(levels, depth)| depth > levels)
                    {
                        return Err(CopyError::TooDeep);
                    }
                    let pending = pending
                        .iter()
                        .map(|m| Native::at(m, py, generation))
                        .collect();
                    Held::Composite(c.class.clone_ref(py), items, pending)
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
        (slots, with_desc.then(|| m.desc), m.unknown.clone())
    };
    for (index, held) in first.into_iter().chain(rest) {
        let value = match held {
            Held::Message(child) => Value::Message(NRef::new(copy_message(
                py,
                child.bind(py),
                at.below()?,
                generation,
                budget,
            )?)),
            Held::Composite(class, items, pending) => {
                let mut copies = Vec::with_capacity(items.len() + pending.len());
                if !items.is_empty() {
                    let at = at.below()?;
                    for item in &items {
                        let copy = copy_message(py, item.bind(py), at, generation, budget)?;
                        copies.push(NRef::new(copy));
                    }
                }
                copies.extend(pending);
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
    message: NRef,
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
pub(crate) fn fields_at(py: Python<'_>, message: NRef, generation: u64) -> Fields {
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
    /// The objects below the object of the slot, of a value made whole (see [`WHOLE`]), which take
    /// the fields of the value of all its levels.
    whole: Option<Whole>,
    /// Whether the slot is a container that makes the objects of the messages of the value as they
    /// are read (see `RepeatedComposite::of_native`).
    lazy: bool,
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
    whole: bool,
) -> PyResult<Made> {
    let step = |py| budget.map_or(Ok(()), |budget| budget.step(py));
    if whole && value.count() <= whole_limit() {
        let mut objects = Whole {
            messages: Vec::with_capacity(value.count() as usize),
            ..Whole::default()
        };
        let slot = objects.make(py, field, value, budget, 0)?;
        return Ok(Made {
            slot,
            messages: Vec::new(),
            scalars: None,
            whole: Some(objects),
            lazy: false,
        });
    }
    let mut messages = Vec::new();
    let mut scalars = None;
    let slot = match value {
        Value::Message(m) => {
            let object = new_message_at(py, m.fields.desc, m.generation)?;
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
        Value::Messages(m) if whole => {
            // Of a read of Python: the container makes the objects of the messages as they are
            // read (see `RepeatedComposite::of_native`).
            let container = RepeatedComposite::of_native(m.class.clone_ref(py), m.clone());
            return Ok(Made {
                slot: Slot::Composite(Py::new(py, container)?),
                messages: Vec::new(),
                scalars: None,
                whole: None,
                lazy: true,
            });
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
                        let object = new_message_at(py, m.fields.desc, m.generation)?.unbind();
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
        whole: None,
        lazy: false,
    })
}

/// The messages of a native value of a field (see [`Value::count`]) up to which the object of the
/// value that Python reads is made whole, the objects of all its messages made at once
/// ([`Whole`]), rather than of one level: the read of each level of a native value reads and sets
/// the fields of its message, which the objects of a whole value take once, of all its levels.
/// The objects of a larger value are made of one level, so that a read of a part of it makes the
/// objects of that part; the tests read values of every size of each way (see [`whole_limit`]).
const WHOLE: u32 = 256;

/// The limit of [`WHOLE`] that a read of a native value takes: [`WHOLE`], or that of
/// `_proto_whole_limit` of the tests.
static WHOLE_LIMIT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(WHOLE);

fn whole_limit() -> u32 {
    WHOLE_LIMIT.load(std::sync::atomic::Ordering::Relaxed)
}

/// Set the messages of a native value up to which a read of Python makes the objects of all its
/// levels at once (see [`WHOLE`]), of none, as of a value of more than 256 messages, at 0, and give
/// the limit that it replaces: of the tests, which read the values of their messages each way
/// (`None` gives the runtime's own limit back).
#[pyfunction]
#[pyo3(signature = (limit=None))]
pub(crate) fn _proto_whole_limit(limit: Option<u32>) -> u32 {
    WHOLE_LIMIT.swap(limit.unwrap_or(WHOLE), std::sync::atomic::Ordering::Relaxed)
}

/// The native messages that `message` holds, of all its levels, that are nodes of a slab of a
/// parse (see `PyMessage::unpin`): of the tests, which check that a message that outlives the
/// container that held it keeps no slab alive.
#[pyfunction]
pub(crate) fn _proto_slab_messages(message: &Bound<'_, PyMessage>) -> usize {
    PyMessage::slab_messages(message)
}

/// The objects of a native value made whole (see [`WHOLE`]), empty, in the order of a walk of the
/// value that visits a message before the values of its fields, and a container of messages or a
/// map after its messages: the messages, the containers of repeated scalar fields, those of
/// repeated message fields and the maps, which hold the objects of their messages. They take the
/// fields of the value of all its levels once the field holds the object of the value
/// ([`Made::fill`]), moved where no other value holds them, as the objects of one level do.
#[derive(Default)]
struct Whole {
    messages: Vec<Py<PyMessage>>,
    scalars: Vec<Py<RepeatedScalar>>,
    composites: Vec<Py<RepeatedComposite>>,
    maps: Vec<Py<MapContainer>>,
    /// The releases of the nodes of the messages whose fields the objects take.
    releases: Releases,
}

impl Whole {
    /// The object of the native value `value` of `field`, empty, with the objects below it, which
    /// `self` keeps, as a slot of a field holds it.
    fn make(
        &mut self,
        py: Python<'_>,
        field: &FieldDesc,
        value: &Value,
        budget: Option<&ConversionBudget>,
        level: u32,
    ) -> PyResult<Slot> {
        Ok(match value {
            Value::Message(m) => Slot::Message(self.make_message(py, m, None, budget, level)?),
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
                self.scalars.push(container.clone_ref(py));
                Slot::Repeated(container)
            }
            Value::Messages(m) => {
                let mut items = Vec::with_capacity(m.items.len());
                for item in &m.items {
                    items.push(self.make_message(py, item, Some(&m.class), budget, level)?);
                }
                let container = RepeatedComposite::new(m.class.clone_ref(py), items, None);
                let container = Py::new(py, container)?;
                self.composites.push(container.clone_ref(py));
                Slot::Composite(container)
            }
            Value::Map(m) => {
                let entries = m.entries.try_map(|value| -> PyResult<MapValue> {
                    Ok(match value {
                        NativeValue::Scalar(v) => MapValue::Scalar(v.clone()),
                        NativeValue::Message(m) => {
                            MapValue::Message(self.make_message(py, m, None, budget, level)?)
                        }
                    })
                })?;
                let map = MapContainer::of_entries(field.name.clone(), m.map.clone(), entries)
                    .into_object(py)?;
                self.maps.push(map.clone_ref(py));
                Slot::Map(map)
            }
        })
    }

    /// The object of the native message `message`, empty, of the class of its type of the
    /// generation of its copy (of `class`, of a message of a repeated field, as its container makes
    /// it), with the objects of the native values of its fields, which `self` keeps. A step of
    /// `budget`, when it is given, for each message. The message is at the level `level` below
    /// the value, of which every 16th level runs on a stack with room for 16 levels (see
    /// [`grow`]).
    fn make_message(
        &mut self,
        py: Python<'_>,
        message: &Native,
        class: Option<&MessageClass>,
        budget: Option<&ConversionBudget>,
        level: u32,
    ) -> PyResult<Py<PyMessage>> {
        if level > 0 && level.is_multiple_of(16) {
            return grow(|| self.make_message_here(py, message, class, budget, level));
        }
        self.make_message_here(py, message, class, budget, level)
    }

    /// [`Whole::make_message`] on the stack that it runs on.
    fn make_message_here(
        &mut self,
        py: Python<'_>,
        message: &Native,
        class: Option<&MessageClass>,
        budget: Option<&ConversionBudget>,
        level: u32,
    ) -> PyResult<Py<PyMessage>> {
        if let Some(budget) = budget {
            budget.step(py)?;
        }
        let object = match class {
            Some(class) => class.new_message(py)?,
            None => new_message_at(py, message.fields.desc, message.generation)?,
        };
        self.messages.push(object.clone().unbind());
        for (index, slot) in message.fields.slots.iter() {
            if let Slot::Native(value) = slot {
                let field = &message.fields.desc.fields[index];
                self.make(py, field, value, budget, level + 1)?.release(py);
            }
        }
        Ok(object.unbind())
    }

    /// Give the objects the fields of `value`, the value that they were made of, in the order
    /// that [`Whole::make`] made them, moved where no other value holds them (see [`own_parts`]):
    /// the object of the value, as its field's slot holds it.
    fn fill(&mut self, py: Python<'_>, value: Value, level: u32) -> Slot {
        match value {
            Value::Message(m) => Slot::Message(self.fill_message(py, m, level)),
            Value::Scalars(items) => {
                let container = take(&mut self.scalars);
                let items = Arc::try_unwrap(items).unwrap_or_else(|items| (*items).clone());
                container.bind(py).borrow_mut().items = items;
                Slot::Repeated(container)
            }
            Value::Messages(m) => {
                match Arc::try_unwrap(m) {
                    Ok(mut m) => std::mem::take(&mut m.items).into_iter().for_each(|item| {
                        drop(self.fill_message(py, item, level).into_bound(py));
                    }),
                    Err(m) => m.items.iter().cloned().for_each(|item| {
                        drop(self.fill_message(py, item, level).into_bound(py));
                    }),
                }
                Slot::Composite(take(&mut self.composites))
            }
            Value::Map(m) => {
                for message in map_messages(m) {
                    drop(self.fill_message(py, message, level).into_bound(py));
                }
                Slot::Map(take(&mut self.maps))
            }
        }
    }

    /// Give the object of the native message `message` its fields, of the objects of the native
    /// values of its fields: the object. The message is at the level `level` below the value (see
    /// [`Whole::make_message`]).
    fn fill_message(&mut self, py: Python<'_>, message: NRef, level: u32) -> Py<PyMessage> {
        if level > 0 && level.is_multiple_of(16) {
            return grow(|| self.fill_message_here(py, message, level));
        }
        self.fill_message_here(py, message, level)
    }

    /// [`Whole::fill_message`] on the stack that it runs on.
    fn fill_message_here(&mut self, py: Python<'_>, message: NRef, level: u32) -> Py<PyMessage> {
        let object = take(&mut self.messages);
        let (mut slots, unknown) = own_parts(py, message, &mut self.releases);
        for (_, slot) in slots.iter_mut() {
            if matches!(slot, Slot::Native(_)) {
                if let Slot::Native(value) = std::mem::replace(slot, HELD) {
                    *slot = self.fill(py, value, level + 1);
                }
            }
        }
        object.bind(py).borrow_mut().take_parts(slots, unknown);
        object
    }
}

/// The first of the objects `objects` that [`Whole::make`] made, which [`Whole::fill`] takes in
/// the order that they were made in (`objects` is reversed once, see [`Made::fill`]).
fn take<T>(objects: &mut Vec<Py<T>>) -> Py<T> {
    objects
        .pop()
        .expect("an object of each native value of a whole value")
}

impl Made {
    /// Give the objects the fields of `value`, the value that they were made of (see
    /// [`made_of`]), which the field no longer holds: moved where no other value holds them
    /// (see [`own_parts`]), else copied, of one level. It runs no Python code.
    pub(crate) fn fill(self, py: Python<'_>, value: Value) {
        // The references of the objects are released as `Bound`'s, which know that the thread is
        // attached, with no read of a thread local.
        let Made {
            slot,
            messages,
            scalars,
            whole,
            lazy,
        } = self;
        if let Some(mut objects) = whole {
            // The objects in the order that they were made, of which `take` takes the last.
            objects.messages.reverse();
            objects.scalars.reverse();
            objects.composites.reverse();
            objects.maps.reverse();
            objects.fill(py, value, 0).release(py);
            debug_assert!(objects.messages.is_empty() && objects.composites.is_empty());
            slot.release(py);
            return;
        }
        let mut objects = messages.into_iter().map(|m| m.into_bound(py));
        let mut releases = Releases::default();
        let mut fill = |object: &Bound<'_, PyMessage>, message: NRef| {
            let (slots, unknown) = own_parts(py, message, &mut releases);
            object.borrow_mut().take_parts(slots, unknown);
        };
        let fill_next = |message: NRef| {
            if let Some(object) = objects.next() {
                fill(&object, message);
            }
        };
        match value {
            Value::Message(m) => match slot {
                Slot::Message(object) => fill(&object.into_bound(py), m),
                slot => slot.release(py),
            },
            Value::Scalars(items) => {
                if let Some(container) = scalars {
                    let items = Arc::try_unwrap(items).unwrap_or_else(|items| (*items).clone());
                    container.into_bound(py).borrow_mut().items = items;
                }
                slot.release(py);
            }
            Value::Messages(m) => {
                match (slot, lazy) {
                    // A container of the messages held natively (see `made_of`), of the value as
                    // its own once the field's reference is released.
                    (Slot::Composite(container), true) => {
                        drop(m);
                        container.into_bound(py).borrow_mut().own_pending();
                    }
                    (slot, _) => {
                        match Arc::try_unwrap(m) {
                            Ok(mut m) => {
                                std::mem::take(&mut m.items).into_iter().for_each(fill_next)
                            }
                            Err(m) => m.items.iter().cloned().for_each(fill_next),
                        }
                        slot.release(py);
                    }
                }
            }
            Value::Map(m) => {
                map_messages(m).into_iter().for_each(fill_next);
                slot.release(py);
            }
        }
    }
}

/// The native messages of the values of the map `map`, in the order of its entries, of the map's
/// own references released first, so that a message of a map that no other holds is moved.
fn map_messages(map: Arc<NativeMap>) -> Vec<NRef> {
    match Arc::try_unwrap(map) {
        Ok(map) => map
            .entries
            .into_entries()
            .filter_map(|(_, value)| match value {
                NativeValue::Message(m) => Some(m),
                NativeValue::Scalar(_) => None,
            })
            .collect(),
        Err(map) => map
            .entries
            .iter()
            .filter_map(|(_, value)| match value {
                NativeValue::Message(m) => Some(m.clone()),
                NativeValue::Scalar(_) => None,
            })
            .collect(),
    }
}

/// The set fields and the unknown fields of the native message `message` for its object: its own,
/// moved, if no other holds it (its node released of `releases`), else a copy of one level of
/// them. The messages of a parse are of slabs (see `arena`), so that the values moved leave no
/// small hole behind them.
fn own_parts(py: Python<'_>, message: NRef, releases: &mut Releases) -> (Slots, Vec<u8>) {
    match NRef::into_parts_releasing(message, releases) {
        Ok(parts) => parts,
        Err(message) => {
            let fields = message.fields.clone_native_at(py, message.generation);
            (fields.slots, fields.unknown)
        }
    }
}

/// The fields of the native message `message`: its own, moved, if no other holds it, else a copy
/// of one level of them.
fn own_fields(py: Python<'_>, message: NRef) -> Fields {
    match NRef::try_unwrap(message) {
        Ok(mut message) => Fields {
            desc: message.fields.desc,
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
    let object = new_message_at(py, message.fields.desc, message.generation)?;
    fill(&object, message, py);
    Ok(object)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbmsg::pool::pool;

    /// A native message of the fields `fields` of `index` of the type `full_name`, holding `child`.
    fn holding(full_name: &str, field: &str, child: Native) -> Native {
        let desc = pool().message(full_name).expect("a type of the pool");
        let index = desc.by_name[field];
        let mut fields = Fields {
            desc,
            slots: Slots::default(),
            unknown: Vec::new(),
        };
        let value = Slot::Native(Value::Message(NRef::new(child)));
        fields.slots.set(desc, index, value);
        Native::new(fields, 0)
    }

    /// The messages of a native repeated message field are held in place up to two, of a vector
    /// from the third, in their order, of a push, a collection and a vector alike, and give
    /// themselves back of each way of reading them.
    #[test]
    fn the_messages_of_a_repeated_field_keep_their_order_of_each_size() {
        let literal = pool()
            .message("spark.connect.Expression.Literal")
            .expect("Literal");
        let native = |i: u8| {
            NRef::new(Native::new(
                Fields {
                    desc: literal,
                    slots: Slots::default(),
                    unknown: vec![i],
                },
                0,
            ))
        };
        let tags = |items: &Items| {
            items
                .iter()
                .map(|m| m.fields.unknown[0])
                .collect::<Vec<_>>()
        };
        for count in 0..6u8 {
            let mut pushed = Items::default();
            (0..count).for_each(|i| pushed.push(native(i)));
            let collected: Items = (0..count).map(native).collect();
            let of_vec = Items::from((0..count).map(native).collect::<Vec<_>>());
            let expected: Vec<u8> = (0..count).collect();
            for items in [&pushed, &collected, &of_vec] {
                assert_eq!(tags(items), expected);
                assert_eq!(items.len(), count as usize);
                assert_eq!(matches!(items, Items::Many(_)), count > 2);
            }
            assert_eq!(tags(&pushed.clone()), expected);
            let by_value: Vec<u8> = collected.into_iter().map(|m| m.fields.unknown[0]).collect();
            assert_eq!(by_value, expected);
            let as_vec: Vec<u8> = of_vec
                .into_vec()
                .iter()
                .map(|m| m.fields.unknown[0])
                .collect();
            assert_eq!(as_vec, expected);
            for item in &mut pushed {
                NRef::get_mut(item).expect("its own").fields.unknown[0] += 10;
            }
            assert_eq!(tags(&pushed), (10..10 + count).collect::<Vec<_>>());
        }
    }

    /// A native message of a million levels drops a level at a time, where a recursion of as many
    /// levels would overflow the stack of the thread of the test, and knows its levels as upb's
    /// parse counts them.
    #[test]
    fn deep_native_messages_drop_a_level_at_a_time() {
        let relation = pool().message("spark.connect.Relation").expect("Relation");
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
        assert_eq!(Value::Message(NRef::new(message)).depth(), 1_000_001);
    }
}
