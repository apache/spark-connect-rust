//! The protobuf wire format of the messages of the runtime: [`encode`] writes the set fields in
//! the order of their numbers, packs repeated numeric fields, writes a map entry's key and value,
//! and ends with the unknown fields that the message read; [`merge`] reads every wire type, takes
//! a repeated numeric field packed or not, merges a message that a field repeats into the one it
//! holds, and keeps the fields that the message type has no field for, and the map entries that
//! have such fields.

use pyo3::prelude::*;

use super::containers::{MapValue, RepeatedComposite, RepeatedScalar};
use super::message::{new_instance, MessageClass, PyMessage, Slot};
use super::pool::{FieldDesc, Kind, MapDesc, MessageDesc};
use super::scalar::Scalar;
use super::text::upb_wire_key_order;
use super::{decode_error_of, grow, DecodeError};
use crate::interrupt::ConversionBudget;

fn write_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn write_tag(out: &mut Vec<u8>, number: u32, wire_type: u8) {
    write_varint(out, ((number as u64) << 3) | wire_type as u64);
}

fn write_len_delimited(out: &mut Vec<u8>, bytes: &[u8]) {
    write_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

/// Write `value`, a value of a field of `kind`, without its tag.
fn write_value(out: &mut Vec<u8>, kind: Kind, value: &Scalar) {
    match (kind, value) {
        (Kind::Double, Scalar::F64(v)) => out.extend_from_slice(&v.to_le_bytes()),
        (Kind::Float, Scalar::F32(v)) => out.extend_from_slice(&v.to_le_bytes()),
        (Kind::Int64, Scalar::I64(v)) => write_varint(out, *v as u64),
        (Kind::SInt64, Scalar::I64(v)) => write_varint(out, ((v << 1) ^ (v >> 63)) as u64),
        (Kind::SFixed64, Scalar::I64(v)) => out.extend_from_slice(&v.to_le_bytes()),
        (Kind::UInt64, Scalar::U64(v)) => write_varint(out, *v),
        (Kind::Fixed64, Scalar::U64(v)) => out.extend_from_slice(&v.to_le_bytes()),
        (Kind::Int32, Scalar::I32(v)) => write_varint(out, *v as i64 as u64),
        (Kind::SInt32, Scalar::I32(v)) => write_varint(out, ((v << 1) ^ (v >> 31)) as u32 as u64),
        (Kind::SFixed32, Scalar::I32(v)) => out.extend_from_slice(&v.to_le_bytes()),
        (Kind::UInt32, Scalar::U32(v)) => write_varint(out, *v as u64),
        (Kind::Fixed32, Scalar::U32(v)) => out.extend_from_slice(&v.to_le_bytes()),
        (Kind::Bool, Scalar::Bool(v)) => write_varint(out, *v as u64),
        (Kind::String, Scalar::Str(v)) => write_len_delimited(out, v.as_bytes()),
        (Kind::Bytes, Scalar::Bytes(v)) => write_len_delimited(out, v),
        (Kind::Enum, Scalar::Enum(v)) => write_varint(out, *v as i64 as u64),
        // A value always has the variant of the kind of its field (see `Scalar::from_py`).
        _ => {}
    }
}

/// The length of the varint of `value`.
fn varint_len(value: u64) -> usize {
    (64 - (value | 1).leading_zeros() as usize).div_ceil(7)
}

fn tag_len(number: u32, wire_type: u8) -> usize {
    varint_len(((number as u64) << 3) | wire_type as u64)
}

/// The length of `value`, a value of a field of `kind`, as [`write_value`] writes it.
fn value_len(kind: Kind, value: &Scalar) -> usize {
    match (kind, value) {
        (Kind::Double | Kind::Fixed64 | Kind::SFixed64, _) => 8,
        (Kind::Float | Kind::Fixed32 | Kind::SFixed32, _) => 4,
        (Kind::Int64, Scalar::I64(v)) => varint_len(*v as u64),
        (Kind::SInt64, Scalar::I64(v)) => varint_len(((v << 1) ^ (v >> 63)) as u64),
        (Kind::UInt64, Scalar::U64(v)) => varint_len(*v),
        (Kind::Int32, Scalar::I32(v)) => varint_len(*v as i64 as u64),
        (Kind::SInt32, Scalar::I32(v)) => varint_len(((v << 1) ^ (v >> 31)) as u32 as u64),
        (Kind::UInt32, Scalar::U32(v)) => varint_len(*v as u64),
        (Kind::Bool, _) => 1,
        (Kind::String, Scalar::Str(v)) => varint_len(v.len() as u64) + v.len(),
        (Kind::Bytes, Scalar::Bytes(v)) => varint_len(v.len() as u64) + v.len(),
        (Kind::Enum, Scalar::Enum(v)) => varint_len(*v as i64 as u64),
        // A value always has the variant of the kind of its field (see `Scalar::from_py`).
        _ => 0,
    }
}

/// The bytes of `message`, as upb writes a message once into one buffer: the sizes of the
/// message and of the messages that it holds first ([`measure`]), then the bytes, each message
/// after its tag and size.
pub(crate) fn encode(py: Python<'_>, message: &PyMessage) -> PyResult<Vec<u8>> {
    let mut sizes = Vec::new();
    let len = measure(py, message, &mut sizes)?;
    let mut out = Vec::with_capacity(len);
    write(py, message, &sizes, &mut 0, &mut out)?;
    Ok(out)
}

/// The length of the bytes of `message`, as [`encode`] writes them.
pub(crate) fn encoded_len(py: Python<'_>, message: &PyMessage) -> PyResult<usize> {
    measure(py, message, &mut Vec::new())
}

/// The entries of `map`, in upb's order of the keys of its deterministic serialization.
fn sorted_entries(map: &super::containers::MapContainer) -> Vec<&(Scalar, MapValue)> {
    let mut entries: Vec<_> = map.entries.iter().collect();
    entries.sort_by(|a, b| upb_wire_key_order(&a.0, &b.0));
    entries
}

/// The length of the bytes of `message`, with the lengths of the messages and the packed fields
/// that it holds (and of the entries of its maps), each pushed onto `sizes` in the order in which
/// [`write`] writes them.
fn measure(py: Python<'_>, message: &PyMessage, sizes: &mut Vec<usize>) -> PyResult<usize> {
    grow(|| {
        let mut total = 0;
        for (index, slot) in message.slots.iter() {
            let field = &message.desc.fields[index];
            match slot {
                Slot::Scalar(value) => {
                    if field.has_presence() || !value.is_default() {
                        total += tag_len(field.number, field.kind.wire_type())
                            + value_len(field.kind, value);
                    }
                }
                Slot::Message(child) => {
                    total += measure_field(py, field.number, &child.bind(py).borrow(), sizes)?;
                }
                Slot::Repeated(container) => {
                    let container = container.bind(py).borrow();
                    if container.items.is_empty() {
                        continue;
                    }
                    let values: usize = container
                        .items
                        .iter()
                        .map(|value| value_len(field.kind, value))
                        .sum();
                    if field.packed {
                        sizes.push(values);
                        total += tag_len(field.number, 2) + varint_len(values as u64) + values;
                    } else {
                        total += container.items.len()
                            * tag_len(field.number, field.kind.wire_type())
                            + values;
                    }
                }
                Slot::Composite(container) => {
                    for child in &container.bind(py).borrow().items {
                        total += measure_field(py, field.number, &child.bind(py).borrow(), sizes)?;
                    }
                }
                Slot::Map(map) => {
                    let map = map.bind(py).borrow();
                    let desc = &map.map;
                    for (key, value) in sorted_entries(&map) {
                        let at = sizes.len();
                        sizes.push(0);
                        let mut entry = tag_len(1, desc.key.wire_type()) + value_len(desc.key, key);
                        entry += match value {
                            MapValue::Scalar(value) => {
                                tag_len(2, desc.value.wire_type()) + value_len(desc.value, value)
                            }
                            MapValue::Message(child) => {
                                measure_field(py, 2, &child.bind(py).borrow(), sizes)?
                            }
                        };
                        sizes[at] = entry;
                        total += tag_len(field.number, 2) + varint_len(entry as u64) + entry;
                    }
                }
            }
        }
        Ok(total + message.unknown.len())
    })
}

/// The length of `child` as the field `number` of its message, its tag and size included, with
/// its size pushed onto `sizes` before those of its own messages.
fn measure_field(
    py: Python<'_>,
    number: u32,
    child: &PyMessage,
    sizes: &mut Vec<usize>,
) -> PyResult<usize> {
    let at = sizes.len();
    sizes.push(0);
    let len = measure(py, child, sizes)?;
    sizes[at] = len;
    Ok(tag_len(number, 2) + varint_len(len as u64) + len)
}

/// Write the bytes of `message`, of the sizes that [`measure`] pushed, from `next`.
fn write(
    py: Python<'_>,
    message: &PyMessage,
    sizes: &[usize],
    next: &mut usize,
    out: &mut Vec<u8>,
) -> PyResult<()> {
    grow(|| {
        for (index, slot) in message.slots.iter() {
            let field = &message.desc.fields[index];
            match slot {
                Slot::Scalar(value) => {
                    if field.has_presence() || !value.is_default() {
                        write_tag(out, field.number, field.kind.wire_type());
                        write_value(out, field.kind, value);
                    }
                }
                Slot::Message(child) => {
                    write_field(py, field.number, &child.bind(py).borrow(), sizes, next, out)?
                }
                Slot::Repeated(container) => {
                    let container = container.bind(py).borrow();
                    if container.items.is_empty() {
                        continue;
                    }
                    if field.packed {
                        write_tag(out, field.number, 2);
                        write_varint(out, take_size(sizes, next) as u64);
                        for value in &container.items {
                            write_value(out, field.kind, value);
                        }
                    } else {
                        for value in &container.items {
                            write_tag(out, field.number, field.kind.wire_type());
                            write_value(out, field.kind, value);
                        }
                    }
                }
                Slot::Composite(container) => {
                    for child in &container.bind(py).borrow().items {
                        write_field(py, field.number, &child.bind(py).borrow(), sizes, next, out)?;
                    }
                }
                Slot::Map(map) => {
                    let map = map.bind(py).borrow();
                    let desc = &map.map;
                    for (key, value) in sorted_entries(&map) {
                        write_tag(out, field.number, 2);
                        write_varint(out, take_size(sizes, next) as u64);
                        write_tag(out, 1, desc.key.wire_type());
                        write_value(out, desc.key, key);
                        match value {
                            MapValue::Scalar(value) => {
                                write_tag(out, 2, desc.value.wire_type());
                                write_value(out, desc.value, value);
                            }
                            MapValue::Message(child) => {
                                write_field(py, 2, &child.bind(py).borrow(), sizes, next, out)?
                            }
                        }
                    }
                }
            }
        }
        out.extend_from_slice(&message.unknown);
        Ok(())
    })
}

fn take_size(sizes: &[usize], next: &mut usize) -> usize {
    let size = sizes[*next];
    *next += 1;
    size
}

/// Write `child` as the field `number` of its message: its tag, its size and its bytes.
fn write_field(
    py: Python<'_>,
    number: u32,
    child: &PyMessage,
    sizes: &[usize],
    next: &mut usize,
    out: &mut Vec<u8>,
) -> PyResult<()> {
    write_tag(out, number, 2);
    write_varint(out, take_size(sizes, next) as u64);
    write(py, child, sizes, next, out)
}

/// Write `child` as the field `number` of its message, measured alone.
fn encode_message_field(
    py: Python<'_>,
    number: u32,
    child: &Py<PyMessage>,
    out: &mut Vec<u8>,
) -> PyResult<()> {
    let child = child.bind(py).borrow();
    let mut sizes = Vec::new();
    measure_field(py, number, &child, &mut sizes)?;
    write_field(py, number, &child, &sizes, &mut 0, out)
}

/// The cause of a decode error, which [`merge`] gives in upb's error of the message type.
fn decode_error(what: &str) -> PyErr {
    DecodeError::new_err(what.to_string())
}

/// upb's reason of a decode error of `what`.
fn upb_reason(what: &str) -> &'static str {
    if what.contains("UTF-8") {
        "String field had bad UTF-8"
    } else {
        "Wire format was corrupt"
    }
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Reader { data, pos: 0 }
    }

    fn done(&self) -> bool {
        self.pos >= self.data.len()
    }

    /// The field number and the wire type of a tag, as upb reads a tag: a varint of at most five
    /// bytes whose value fits in 32 bits, so a field number is at most 2^29 - 1. A longer tag, or
    /// one of more bits, is corrupt (and not cut to the bits of a field number).
    fn tag(&mut self) -> PyResult<(u32, u8)> {
        let mut value = 0u64;
        for shift in (0..35).step_by(7) {
            let byte = *self
                .data
                .get(self.pos)
                .ok_or_else(|| decode_error("truncated tag"))?;
            self.pos += 1;
            value |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                let tag =
                    u32::try_from(value).map_err(|_| decode_error("tag of more than 32 bits"))?;
                return Ok((tag >> 3, (tag & 7) as u8));
            }
        }
        Err(decode_error("tag of more than five bytes"))
    }

    fn varint(&mut self) -> PyResult<u64> {
        let mut value = 0u64;
        for shift in (0..70).step_by(7) {
            let byte = *self
                .data
                .get(self.pos)
                .ok_or_else(|| decode_error("truncated varint"))?;
            self.pos += 1;
            if shift < 64 {
                value |= ((byte & 0x7f) as u64) << shift;
            }
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(decode_error("varint too long"))
    }

    fn take(&mut self, n: usize) -> PyResult<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&end| end <= self.data.len())
            .ok_or_else(|| decode_error("truncated message"))?;
        let bytes = &self.data[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }

    fn fixed32(&mut self) -> PyResult<[u8; 4]> {
        Ok(self.take(4)?.try_into().expect("4 bytes"))
    }

    fn fixed64(&mut self) -> PyResult<[u8; 8]> {
        Ok(self.take(8)?.try_into().expect("8 bytes"))
    }

    fn len_delimited(&mut self) -> PyResult<&'a [u8]> {
        let len = self.varint()?;
        let len = usize::try_from(len).map_err(|_| decode_error("length too large"))?;
        self.take(len)
    }

    /// A value of a field of `kind`, of its wire type.
    fn value(&mut self, kind: Kind) -> PyResult<Scalar> {
        Ok(match kind {
            Kind::Double => Scalar::F64(f64::from_le_bytes(self.fixed64()?)),
            Kind::Float => Scalar::F32(f32::from_le_bytes(self.fixed32()?)),
            Kind::Int64 => Scalar::I64(self.varint()? as i64),
            Kind::SInt64 => {
                let v = self.varint()?;
                Scalar::I64(((v >> 1) as i64) ^ -((v & 1) as i64))
            }
            Kind::SFixed64 => Scalar::I64(i64::from_le_bytes(self.fixed64()?)),
            Kind::UInt64 => Scalar::U64(self.varint()?),
            Kind::Fixed64 => Scalar::U64(u64::from_le_bytes(self.fixed64()?)),
            Kind::Int32 => Scalar::I32(self.varint()? as i64 as i32),
            Kind::SInt32 => {
                let v = self.varint()? as u32;
                Scalar::I32(((v >> 1) as i32) ^ -((v & 1) as i32))
            }
            Kind::SFixed32 => Scalar::I32(i32::from_le_bytes(self.fixed32()?)),
            Kind::UInt32 => Scalar::U32(self.varint()? as u32),
            Kind::Fixed32 => Scalar::U32(u32::from_le_bytes(self.fixed32()?)),
            Kind::Bool => Scalar::Bool(self.varint()? != 0),
            Kind::String => Scalar::Str(
                String::from_utf8(self.len_delimited()?.to_vec())
                    .map_err(|_| decode_error("invalid UTF-8 in a string field"))?,
            ),
            Kind::Bytes => Scalar::Bytes(self.len_delimited()?.to_vec()),
            Kind::Enum => Scalar::Enum(self.varint()? as i64 as i32),
            Kind::Message => return Err(decode_error("a message is not a scalar")),
        })
    }

    /// Skip the value of a field of `wire_type`, a group up to its end. As upb skips a group, the
    /// tags of its fields are read as tags are, but a field number of 0 among them is not corrupt.
    fn skip(&mut self, wire_type: u8, number: u32) -> PyResult<()> {
        match wire_type {
            0 => {
                self.varint()?;
            }
            1 => {
                self.take(8)?;
            }
            2 => {
                self.len_delimited()?;
            }
            5 => {
                self.take(4)?;
            }
            3 => loop {
                let (inner, inner_type) = self.tag()?;
                if inner_type == 4 {
                    if inner != number {
                        return Err(decode_error("mismatched end of group"));
                    }
                    break;
                }
                grow(|| self.skip(inner_type, inner))?;
            },
            _ => return Err(decode_error("invalid wire type")),
        }
        Ok(())
    }
}

/// The value of a field of the unknown fields of a message, as upb's comparison of unknown fields
/// reads it: a varint of its value, whatever its encoding, a length-delimited field of its bytes,
/// and a group of its fields.
#[derive(PartialEq)]
enum UnknownValue<'a> {
    Varint(u64),
    Fixed64(u64),
    Fixed32(u32),
    Delimited(&'a [u8]),
    Group(Vec<(u32, UnknownValue<'a>)>),
}

/// The fields of the unknown fields that `reader` reads, up to its end, or up to the end of the
/// group `group`: each of its tag and value, stably sorted by their tags, as upb sorts them to
/// compare them (`upb_UnknownFields_Build`), the fields of a group too.
fn unknown_fields_of<'a>(
    reader: &mut Reader<'a>,
    group: Option<u32>,
) -> PyResult<Vec<(u32, UnknownValue<'a>)>> {
    let mut fields = Vec::new();
    loop {
        if reader.done() {
            if group.is_some() {
                return Err(decode_error("unterminated group"));
            }
            break;
        }
        let (number, wire_type) = reader.tag()?;
        let value = match wire_type {
            0 => UnknownValue::Varint(reader.varint()?),
            1 => UnknownValue::Fixed64(u64::from_le_bytes(reader.fixed64()?)),
            2 => UnknownValue::Delimited(reader.len_delimited()?),
            3 => UnknownValue::Group(grow(|| unknown_fields_of(reader, Some(number)))?),
            4 if group == Some(number) => break,
            5 => UnknownValue::Fixed32(u32::from_le_bytes(reader.fixed32()?)),
            _ => return Err(decode_error("invalid unknown field")),
        };
        fields.push(((number << 3) | u32::from(wire_type), value));
    }
    fields.sort_by_key(|&(tag, _)| tag);
    Ok(fields)
}

/// Whether the unknown fields `a` and `b` of two messages are equal as upb's `==` compares them
/// (`upb_Message_UnknownFieldsAreEqual`): of the same bytes, or of the same fields, each of its
/// number, wire type and value, in the order of their tags, so that the order of fields of other
/// numbers or wire types, and the encoding of a varint, do not count. The fields of the same tag
/// keep their order. A length-delimited field compares its bytes as they are.
pub(crate) fn unknown_fields_eq(a: &[u8], b: &[u8]) -> bool {
    if a == b {
        return true;
    }
    match (
        unknown_fields_of(&mut Reader::new(a), None),
        unknown_fields_of(&mut Reader::new(b), None),
    ) {
        (Ok(x), Ok(y)) => x == y,
        _ => false,
    }
}

/// Merge the encoded message `data` into `message`, as protobuf's `MergeFromString` does. A
/// failure is upb's `DecodeError` of the message type, also of a failure in a message that it
/// holds: "Error parsing message with type 'spark.connect.Plan': Wire format was corrupt", of the
/// runtime's class of it, which is protobuf's `google.protobuf.message.DecodeError` too where
/// protobuf imports (see [`super::decode_error_of`]).
pub(crate) fn merge(py: Python<'_>, message: &Bound<'_, PyMessage>, data: &[u8]) -> PyResult<()> {
    merge_with(py, message, data, Some(&ConversionBudget::full()))
}

/// Merge `data` into `message` as [`merge`] does, a step of `budget` for each message and each map
/// entry: the budget of the operation that the merge is a part of, such as the constructor or the
/// `extend` of many of protobuf's messages.
pub(crate) fn merge_in(
    py: Python<'_>,
    message: &Bound<'_, PyMessage>,
    data: &[u8],
    budget: &ConversionBudget,
) -> PyResult<()> {
    merge_with(py, message, data, Some(budget))
}

/// Merge `data` into `message` as [`merge`] does, without checkpoints: for a merge of the text
/// of a message, which holds borrows of the messages that it prints.
pub(crate) fn merge_unchecked(
    py: Python<'_>,
    message: &Bound<'_, PyMessage>,
    data: &[u8],
) -> PyResult<()> {
    merge_with(py, message, data, None)
}

fn merge_with(
    py: Python<'_>,
    message: &Bound<'_, PyMessage>,
    data: &[u8],
    budget: Option<&ConversionBudget>,
) -> PyResult<()> {
    merge_message(py, message, data, budget).map_err(|err| {
        if !err.is_instance_of::<DecodeError>(py) {
            return err;
        }
        let what = err.value(py).to_string();
        let full_name = message.borrow().desc.full_name.clone();
        decode_error_of(
            py,
            format!(
                "Error parsing message with type '{full_name}': {}",
                upb_reason(&what)
            ),
        )
    })
}

/// Merge `data` into `message`, one level of a merge, on a stack with room for it, with the
/// cause of a failure. Each message merged, and each map entry, is a step of `budget`, whose
/// checkpoints find no borrow held: the merge borrows a message only to set one of its fields.
fn merge_message(
    py: Python<'_>,
    message: &Bound<'_, PyMessage>,
    data: &[u8],
    budget: Option<&ConversionBudget>,
) -> PyResult<()> {
    grow(|| merge_inner(py, message, data, budget))
}

/// The repeated fields that the merge of a message has set values of, whose next values (which
/// the wire format gives in a row) take the same container, and their elements the same class.
#[derive(Default)]
struct Repeated {
    messages: Option<(usize, Py<RepeatedComposite>, MessageClass)>,
    scalars: Option<(usize, Py<RepeatedScalar>)>,
}

impl Repeated {
    fn messages(
        &mut self,
        py: Python<'_>,
        message: &Bound<'_, PyMessage>,
        field: &FieldDesc,
    ) -> PyResult<(Py<RepeatedComposite>, &MessageClass)> {
        if !matches!(&self.messages, Some((index, _, _)) if *index == field.index) {
            let container = PyMessage::composite_of(message, field.index)?;
            let class = container.bind(py).borrow().class.clone_ref(py);
            self.messages = Some((field.index, container, class));
        }
        let (_, container, class) = self.messages.as_ref().expect("set above");
        Ok((container.clone_ref(py), class))
    }

    fn scalars(
        &mut self,
        py: Python<'_>,
        message: &Bound<'_, PyMessage>,
        field: &FieldDesc,
    ) -> PyResult<Py<RepeatedScalar>> {
        if !matches!(&self.scalars, Some((index, _)) if *index == field.index) {
            let container = PyMessage::repeated_scalar_of(message, field.index)?;
            self.scalars = Some((field.index, container));
        }
        let (_, container) = self.scalars.as_ref().expect("set above");
        Ok(container.clone_ref(py))
    }
}

/// The field of the number `number` of `desc`, of a binary search of the fields in the order of
/// their numbers, which takes less time than the hash of the number in `by_number`, as the parse
/// of a message looks up each of its fields.
fn field_of_number(desc: &MessageDesc, number: u32) -> Option<&FieldDesc> {
    desc.order
        .binary_search_by_key(&number, |&i| desc.fields[i].number)
        .ok()
        .map(|at| &desc.fields[desc.order[at]])
}

fn merge_inner(
    py: Python<'_>,
    message: &Bound<'_, PyMessage>,
    data: &[u8],
    budget: Option<&ConversionBudget>,
) -> PyResult<()> {
    if let Some(budget) = budget {
        budget.step(py)?;
    }
    let desc = message.borrow().desc.clone();
    let mut repeated = Repeated::default();
    let mut reader = Reader::new(data);
    while !reader.done() {
        let start = reader.pos;
        let (number, wire_type) = reader.tag()?;
        if number == 0 {
            return Err(decode_error("invalid field number 0"));
        }
        if wire_type == 4 {
            return Err(decode_error("unexpected end of group"));
        }
        let known = match field_of_number(&desc, number) {
            Some(field) => merge_field(
                py,
                message,
                field,
                wire_type,
                &mut reader,
                &mut repeated,
                budget,
            )?,
            None => false,
        };
        if !known {
            reader.skip(wire_type, number)?;
            message
                .borrow_mut()
                .unknown
                .extend_from_slice(&data[start..reader.pos]);
        }
    }
    Ok(())
}

/// Merge the value of `field` that `reader` is at, of `wire_type`, into `message`, and return
/// whether the wire type is that of the field: a value of another wire type is an unknown field,
/// as protobuf reads it. As upb's, the message of a message field is set (or appended) before its
/// bytes are read, so a failure in them leaves it set, of the fields read before the failure.
fn merge_field(
    py: Python<'_>,
    message: &Bound<'_, PyMessage>,
    field: &FieldDesc,
    wire_type: u8,
    reader: &mut Reader<'_>,
    repeated: &mut Repeated,
    budget: Option<&ConversionBudget>,
) -> PyResult<bool> {
    let index = field.index;
    if let Some(map) = &field.map {
        if wire_type != 2 {
            return Ok(false);
        }
        let entry = reader.len_delimited()?;
        merge_map_entry(py, message, field, map, entry, budget)?;
        return Ok(true);
    }
    if field.kind == Kind::Message {
        if wire_type != 2 {
            return Ok(false);
        }
        let child = if field.repeated {
            let (container, class) = repeated.messages(py, message, field)?;
            let child = class.new_message(py)?;
            container
                .bind(py)
                .borrow_mut()
                .items
                .push(child.clone().unbind());
            child
        } else {
            message.borrow_mut().child_for_merge(py, index)?
        };
        let bytes = reader.len_delimited()?;
        merge_message(py, &child, bytes, budget)?;
        return Ok(true);
    }
    if field.repeated {
        if wire_type == 2 && field.kind.packable() {
            let payload = reader.len_delimited()?;
            // Values of a fixed width are read at once, of a payload of a whole number of them
            // (else corrupt, as upb's, which then sets none); varints in turn.
            let width = match field.kind.wire_type() {
                1 => 8,
                5 => 4,
                _ => 0,
            };
            if width > 0 {
                if payload.len() % width != 0 {
                    return Err(decode_error("truncated packed field"));
                }
                let container = repeated.scalars(py, message, field)?;
                let mut c = container.bind(py).borrow_mut();
                c.items.reserve(payload.len() / width);
                let mut packed = Reader::new(payload);
                while !packed.done() {
                    let value = packed.value(field.kind)?;
                    c.items.push(value);
                }
                return Ok(true);
            }
            let mut packed = Reader::new(payload);
            let mut values = Vec::new();
            while !packed.done() {
                values.push(packed.value(field.kind)?);
            }
            let container = repeated.scalars(py, message, field)?;
            container.bind(py).borrow_mut().items.extend(values);
        } else if wire_type == field.kind.wire_type() {
            let value = reader.value(field.kind)?;
            let container = repeated.scalars(py, message, field)?;
            container.bind(py).borrow_mut().items.push(value);
        } else {
            return Ok(false);
        }
        return Ok(true);
    }
    if wire_type != field.kind.wire_type() {
        return Ok(false);
    }
    let value = reader.value(field.kind)?;
    message.borrow_mut().set_scalar(py, index, value);
    Ok(true)
}

/// Merge the map entry `entry` of the map field `field` into `message`, as upb decodes one: as a
/// message of the entry type, whose key and value are set in the map. An entry with fields that
/// the entry type has no field for, or with a key or value of another wire type (which upb reads
/// as such fields), is not set: the message keeps it in its unknown fields, as upb encodes the
/// entry message again, of its key and value (each written as a field of a message is, so a
/// scalar of the default is not) and then those fields.
fn merge_map_entry(
    py: Python<'_>,
    message: &Bound<'_, PyMessage>,
    field: &FieldDesc,
    map: &MapDesc,
    entry: &[u8],
    budget: Option<&ConversionBudget>,
) -> PyResult<()> {
    if let Some(budget) = budget {
        budget.step(py)?;
    }
    let mut key = Scalar::default_of(map.key);
    let mut scalar = None;
    let mut child: Option<Bound<'_, PyMessage>> = None;
    let mut unknown = Vec::new();
    let mut reader = Reader::new(entry);
    while !reader.done() {
        let start = reader.pos;
        let (number, wire_type) = reader.tag()?;
        match number {
            0 => return Err(decode_error("invalid field number 0")),
            1 if wire_type == map.key.wire_type() => key = reader.value(map.key)?,
            2 if map.value == Kind::Message && wire_type == 2 => {
                let bytes = reader.len_delimited()?;
                if child.is_none() {
                    child = Some(new_instance(py, &map.value_type)?);
                }
                merge_message(py, child.as_ref().expect("made above"), bytes, budget)?;
            }
            2 if map.value != Kind::Message && wire_type == map.value.wire_type() => {
                scalar = Some(reader.value(map.value)?);
            }
            _ => {
                reader.skip(wire_type, number)?;
                unknown.extend_from_slice(&entry[start..reader.pos]);
            }
        }
    }
    if !unknown.is_empty() {
        let mut bytes = Vec::new();
        if !key.is_default() {
            write_tag(&mut bytes, 1, map.key.wire_type());
            write_value(&mut bytes, map.key, &key);
        }
        match (&child, scalar) {
            (Some(child), _) => encode_message_field(py, 2, child.as_unbound(), &mut bytes)?,
            (None, Some(value)) if !value.is_default() => {
                write_tag(&mut bytes, 2, map.value.wire_type());
                write_value(&mut bytes, map.value, &value);
            }
            _ => {}
        }
        bytes.extend_from_slice(&unknown);
        let mut m = message.borrow_mut();
        write_tag(&mut m.unknown, field.number, 2);
        write_len_delimited(&mut m.unknown, &bytes);
        return Ok(());
    }
    let value = if map.value == Kind::Message {
        let child = match child {
            Some(child) => child,
            None => new_instance(py, &map.value_type)?,
        };
        MapValue::Message(child.unbind())
    } else {
        MapValue::Scalar(scalar.unwrap_or_else(|| Scalar::default_of(map.value)))
    };
    let container = PyMessage::map_of_field(message, field.index)?;
    // The value that the entry replaces is dropped once the map is no longer borrowed.
    let _replaced = container.bind(py).borrow_mut().insert(key, value);
    Ok(())
}
