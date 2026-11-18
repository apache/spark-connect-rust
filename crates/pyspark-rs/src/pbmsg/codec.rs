//! The protobuf wire format of the messages of the runtime: [`encode`] writes the set fields in
//! the order of their numbers, packs repeated numeric fields, writes a map entry's key and value,
//! and ends with the unknown fields that the message read; [`merge`] reads every wire type, takes
//! a repeated numeric field packed or not, merges a message that a field repeats into the one it
//! holds, and keeps the fields that the message type has no field for, and the map entries that
//! have such fields.

use pyo3::prelude::*;

use super::containers::MapValue;
use super::message::{new_instance, weak, PyMessage, Slot};
use super::pool::{FieldDesc, Kind, MapDesc};
use super::scalar::Scalar;
use super::text::upb_wire_key_order;
use super::{grow, DecodeError};

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

/// The bytes of `message`.
pub(crate) fn encode(py: Python<'_>, message: &PyMessage) -> PyResult<Vec<u8>> {
    let mut out = Vec::new();
    encode_into(py, message, &mut out)?;
    Ok(out)
}

fn encode_message_field(
    py: Python<'_>,
    number: u32,
    message: &Py<PyMessage>,
    out: &mut Vec<u8>,
) -> PyResult<()> {
    let mut buffer = Vec::new();
    encode_into(py, &message.bind(py).borrow(), &mut buffer)?;
    write_tag(out, number, 2);
    write_len_delimited(out, &buffer);
    Ok(())
}

fn encode_into(py: Python<'_>, message: &PyMessage, out: &mut Vec<u8>) -> PyResult<()> {
    grow(|| {
        for &index in &message.desc.order {
            let field = &message.desc.fields[index];
            match &message.slots[index] {
                None => {}
                Some(Slot::Scalar(value)) => {
                    if field.has_presence() || !value.is_default() {
                        write_tag(out, field.number, field.kind.wire_type());
                        write_value(out, field.kind, value);
                    }
                }
                Some(Slot::Message(child)) => encode_message_field(py, field.number, child, out)?,
                Some(Slot::Repeated(container)) => {
                    let container = container.bind(py).borrow();
                    if container.items.is_empty() {
                        continue;
                    }
                    if field.packed {
                        let mut payload = Vec::new();
                        for value in &container.items {
                            write_value(&mut payload, field.kind, value);
                        }
                        write_tag(out, field.number, 2);
                        write_len_delimited(out, &payload);
                    } else {
                        for value in &container.items {
                            write_tag(out, field.number, field.kind.wire_type());
                            write_value(out, field.kind, value);
                        }
                    }
                }
                Some(Slot::Composite(container)) => {
                    for child in &container.bind(py).borrow().items {
                        encode_message_field(py, field.number, child, out)?;
                    }
                }
                Some(Slot::Map(map)) => {
                    let map = map.bind(py).borrow();
                    let desc = &map.map;
                    // In upb's order of the keys of its deterministic serialization.
                    let mut entries: Vec<_> = map.entries.iter().collect();
                    entries.sort_by(|a, b| upb_wire_key_order(&a.0, &b.0));
                    for (key, value) in entries {
                        let mut entry = Vec::new();
                        write_tag(&mut entry, 1, desc.key.wire_type());
                        write_value(&mut entry, desc.key, key);
                        match value {
                            MapValue::Scalar(value) => {
                                write_tag(&mut entry, 2, desc.value.wire_type());
                                write_value(&mut entry, desc.value, value);
                            }
                            MapValue::Message(child) => {
                                encode_message_field(py, 2, child, &mut entry)?
                            }
                        }
                        write_tag(out, field.number, 2);
                        write_len_delimited(out, &entry);
                    }
                }
            }
        }
        out.extend_from_slice(&message.unknown);
        Ok(())
    })
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

/// Merge the encoded message `data` into `message`, as protobuf's `MergeFromString` does. A
/// failure is upb's `DecodeError` of the message type, also of a failure in a message that it
/// holds: "Error parsing message with type 'spark.connect.Plan': Wire format was corrupt".
pub(crate) fn merge(py: Python<'_>, message: &Bound<'_, PyMessage>, data: &[u8]) -> PyResult<()> {
    merge_message(py, message, data).map_err(|err| {
        if !err.is_instance_of::<DecodeError>(py) {
            return err;
        }
        let what = err.value(py).to_string();
        let full_name = message.borrow().desc.full_name.clone();
        DecodeError::new_err(format!(
            "Error parsing message with type '{full_name}': {}",
            upb_reason(&what)
        ))
    })
}

/// Merge `data` into `message`, one level of a merge, on a stack with room for it, with the
/// cause of a failure.
fn merge_message(py: Python<'_>, message: &Bound<'_, PyMessage>, data: &[u8]) -> PyResult<()> {
    grow(|| merge_inner(py, message, data))
}

fn merge_inner(py: Python<'_>, message: &Bound<'_, PyMessage>, data: &[u8]) -> PyResult<()> {
    let desc = message.borrow().desc.clone();
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
        let field = desc.by_number.get(&number).map(|&i| &desc.fields[i]);
        let known = match field {
            Some(field) => merge_field(py, message, field, wire_type, &mut reader)?,
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
/// as protobuf reads it.
fn merge_field(
    py: Python<'_>,
    message: &Bound<'_, PyMessage>,
    field: &FieldDesc,
    wire_type: u8,
    reader: &mut Reader<'_>,
) -> PyResult<bool> {
    let index = field.index;
    if let Some(map) = &field.map {
        if wire_type != 2 {
            return Ok(false);
        }
        let entry = reader.len_delimited()?;
        merge_map_entry(py, message, field, map, entry)?;
        return Ok(true);
    }
    if field.kind == Kind::Message {
        if wire_type != 2 {
            return Ok(false);
        }
        let bytes = reader.len_delimited()?;
        if field.repeated {
            let child = new_instance(py, &field.type_name)?;
            merge_message(py, &child, bytes)?;
            let container = message
                .borrow_mut()
                .composite(py, index, Some(weak(message)?))?;
            container.bind(py).borrow_mut().items.push(child.unbind());
        } else {
            let child = message.borrow_mut().child_for_merge(py, index)?;
            merge_message(py, &child, bytes)?;
        }
        return Ok(true);
    }
    if field.repeated {
        let mut values = Vec::new();
        if wire_type == 2 && field.kind.packable() {
            let mut packed = Reader::new(reader.len_delimited()?);
            while !packed.done() {
                values.push(packed.value(field.kind)?);
            }
        } else if wire_type == field.kind.wire_type() {
            values.push(reader.value(field.kind)?);
        } else {
            return Ok(false);
        }
        let container = message
            .borrow_mut()
            .repeated_scalar(py, index, Some(weak(message)?))?;
        container.bind(py).borrow_mut().items.extend(values);
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
) -> PyResult<()> {
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
                merge_message(py, child.as_ref().expect("made above"), bytes)?;
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
    let container = message
        .borrow_mut()
        .map(py, field.index, Some(weak(message)?))?;
    // The value that the entry replaces is dropped once the map is no longer borrowed.
    let _replaced = container.bind(py).borrow_mut().insert(key, value);
    Ok(())
}
