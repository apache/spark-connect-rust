//! The protobuf wire format of the messages of the runtime: [`encode`] writes the set fields in
//! the order of their numbers, packs repeated numeric fields, writes a map entry's key and value,
//! and ends with the unknown fields that the message read; [`merge`] reads every wire type, takes
//! a repeated numeric field packed or not, merges a message that a field repeats into the one it
//! holds, and keeps the fields that the message type has no field for, and the map entries that
//! have such fields.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

use pyo3::prelude::*;

use super::containers::{Entries, MapValue, RepeatedComposite, RepeatedScalar};
use super::message::{new_instance, EntryRef, Fields, MessageClass, PyMessage, Slot};
use super::native::Value;
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
        (Kind::String | Kind::Bytes, Scalar::Bytes(v)) => write_len_delimited(out, v),
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
        (Kind::String | Kind::Bytes, Scalar::Bytes(v)) => varint_len(v.len() as u64) + v.len(),
        (Kind::Enum, Scalar::Enum(v)) => varint_len(*v as i64 as u64),
        // A value always has the variant of the kind of its field (see `Scalar::from_py`).
        _ => 0,
    }
}

/// The bytes of `message`, as upb writes a message once into one buffer: the sizes of the
/// message and of the messages that it holds first ([`measure`]), then the bytes, each message
/// after its tag and size. The values that the message holds natively are written as their
/// objects would be.
pub(crate) fn encode(py: Python<'_>, message: &Fields) -> PyResult<Vec<u8>> {
    let mut sizes = Vec::new();
    let len = measure(py, message, &mut sizes)?;
    let mut out = Vec::with_capacity(len);
    write(py, message, &sizes, &mut 0, &mut out)?;
    Ok(out)
}

/// The length of the bytes of `message`, as [`encode`] writes them.
pub(crate) fn encoded_len(py: Python<'_>, message: &Fields) -> PyResult<usize> {
    measure(py, message, &mut Vec::new())
}

/// The entries of `entries`, in upb's order of the keys of its deterministic serialization.
fn sorted_entries<V>(entries: &Entries<V>) -> Vec<&(Scalar, V)> {
    let mut entries: Vec<_> = entries.iter().collect();
    entries.sort_by(|a, b| upb_wire_key_order(&a.0, &b.0));
    entries
}

/// The length of the values `items` of the repeated scalar field `field`, with the length of their
/// payload pushed onto `sizes` if they are packed.
fn measure_scalars(field: &FieldDesc, items: &[Scalar], sizes: &mut Vec<usize>) -> usize {
    if items.is_empty() {
        return 0;
    }
    let values: usize = items.iter().map(|value| value_len(field.kind, value)).sum();
    if field.packed {
        sizes.push(values);
        tag_len(field.number, 2) + varint_len(values as u64) + values
    } else {
        items.len() * tag_len(field.number, field.kind.wire_type()) + values
    }
}

/// The length of the entries `entries` of the map field `field` of `map`, with the length of each
/// entry, and of the messages that it holds, pushed onto `sizes`.
fn measure_map<V>(
    py: Python<'_>,
    field: &FieldDesc,
    map: &MapDesc,
    entries: &Entries<V>,
    sizes: &mut Vec<usize>,
) -> PyResult<usize>
where
    for<'a> EntryRef<'a>: From<&'a V>,
{
    let mut total = 0;
    for (key, value) in sorted_entries(entries) {
        let at = sizes.len();
        sizes.push(0);
        let mut entry = tag_len(1, map.key.wire_type()) + value_len(map.key, key);
        entry += match EntryRef::from(value) {
            EntryRef::Scalar(value) => {
                tag_len(2, map.value.wire_type()) + value_len(map.value, value)
            }
            EntryRef::Message(child) => {
                child.read(py, |child| measure_field(py, 2, child, sizes))?
            }
        };
        sizes[at] = entry;
        total += tag_len(field.number, 2) + varint_len(entry as u64) + entry;
    }
    Ok(total)
}

/// The length of the bytes of `message`, with the lengths of the messages and the packed fields
/// that it holds (and of the entries of its maps), each pushed onto `sizes` in the order in which
/// [`write`] writes them.
fn measure(py: Python<'_>, message: &Fields, sizes: &mut Vec<usize>) -> PyResult<usize> {
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
                    total += measure_child(py, field, &child.bind(py).borrow(), sizes)?;
                }
                Slot::Native(Value::Message(child)) => {
                    total += measure_child(py, field, &child.fields, sizes)?;
                }
                Slot::Repeated(container) => {
                    total += measure_scalars(field, &container.bind(py).borrow().items, sizes);
                }
                Slot::Native(Value::Scalars(items)) => {
                    total += measure_scalars(field, items, sizes);
                }
                Slot::Composite(container) => {
                    for child in &container.bind(py).borrow().items {
                        total += measure_child(py, field, &child.bind(py).borrow(), sizes)?;
                    }
                }
                Slot::Native(Value::Messages(items)) => {
                    for child in &items.items {
                        total += measure_child(py, field, &child.fields, sizes)?;
                    }
                }
                Slot::Map(map) => {
                    let map = map.bind(py).borrow();
                    total += measure_map(py, field, &map.map, &map.entries, sizes)?;
                }
                Slot::Native(Value::Map(map)) => {
                    total += measure_map(py, field, &map.map, &map.entries, sizes)?;
                }
            }
        }
        Ok(total + message.unknown.len())
    })
}

/// The length of `child` as a value of `field`, its tags (and size) included: of a group, between
/// its start and end tags, with no size; else as [`measure_field`] measures it.
fn measure_child(
    py: Python<'_>,
    field: &FieldDesc,
    child: &Fields,
    sizes: &mut Vec<usize>,
) -> PyResult<usize> {
    if field.delimited {
        let len = measure(py, child, sizes)?;
        return Ok(tag_len(field.number, 3) + len + tag_len(field.number, 4));
    }
    measure_field(py, field.number, child, sizes)
}

/// The length of `child` as the field `number` of its message, its tag and size included, with
/// its size pushed onto `sizes` before those of its own messages.
fn measure_field(
    py: Python<'_>,
    number: u32,
    child: &Fields,
    sizes: &mut Vec<usize>,
) -> PyResult<usize> {
    let at = sizes.len();
    sizes.push(0);
    let len = measure(py, child, sizes)?;
    sizes[at] = len;
    Ok(tag_len(number, 2) + varint_len(len as u64) + len)
}

/// Write the values `items` of the repeated scalar field `field`, of their payload's size of
/// `sizes` if they are packed.
fn write_scalars(
    field: &FieldDesc,
    items: &[Scalar],
    sizes: &[usize],
    next: &mut usize,
    out: &mut Vec<u8>,
) {
    if items.is_empty() {
        return;
    }
    if field.packed {
        write_tag(out, field.number, 2);
        write_varint(out, take_size(sizes, next) as u64);
        for value in items {
            write_value(out, field.kind, value);
        }
    } else {
        for value in items {
            write_tag(out, field.number, field.kind.wire_type());
            write_value(out, field.kind, value);
        }
    }
}

/// Write the entries `entries` of the map field `field` of `map`, of the sizes of `sizes`.
fn write_map<V>(
    py: Python<'_>,
    field: &FieldDesc,
    map: &MapDesc,
    entries: &Entries<V>,
    sizes: &[usize],
    next: &mut usize,
    out: &mut Vec<u8>,
) -> PyResult<()>
where
    for<'a> EntryRef<'a>: From<&'a V>,
{
    for (key, value) in sorted_entries(entries) {
        write_tag(out, field.number, 2);
        write_varint(out, take_size(sizes, next) as u64);
        write_tag(out, 1, map.key.wire_type());
        write_value(out, map.key, key);
        match EntryRef::from(value) {
            EntryRef::Scalar(value) => {
                write_tag(out, 2, map.value.wire_type());
                write_value(out, map.value, value);
            }
            EntryRef::Message(child) => {
                child.read(py, |child| write_field(py, 2, child, sizes, next, out))?
            }
        }
    }
    Ok(())
}

/// Write the bytes of `message`, of the sizes that [`measure`] pushed, from `next`.
fn write(
    py: Python<'_>,
    message: &Fields,
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
                    write_child(py, field, &child.bind(py).borrow(), sizes, next, out)?
                }
                Slot::Native(Value::Message(child)) => {
                    write_child(py, field, &child.fields, sizes, next, out)?
                }
                Slot::Repeated(container) => {
                    write_scalars(field, &container.bind(py).borrow().items, sizes, next, out)
                }
                Slot::Native(Value::Scalars(items)) => {
                    write_scalars(field, items, sizes, next, out)
                }
                Slot::Composite(container) => {
                    for child in &container.bind(py).borrow().items {
                        write_child(py, field, &child.bind(py).borrow(), sizes, next, out)?;
                    }
                }
                Slot::Native(Value::Messages(items)) => {
                    for child in &items.items {
                        write_child(py, field, &child.fields, sizes, next, out)?;
                    }
                }
                Slot::Map(map) => {
                    let map = map.bind(py).borrow();
                    write_map(py, field, &map.map, &map.entries, sizes, next, out)?;
                }
                Slot::Native(Value::Map(map)) => {
                    write_map(py, field, &map.map, &map.entries, sizes, next, out)?;
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

/// Write `child` as a value of `field`: of a group, its start tag, its bytes and its end tag;
/// else as [`write_field`] writes it.
fn write_child(
    py: Python<'_>,
    field: &FieldDesc,
    child: &Fields,
    sizes: &[usize],
    next: &mut usize,
    out: &mut Vec<u8>,
) -> PyResult<()> {
    if field.delimited {
        write_tag(out, field.number, 3);
        write(py, child, sizes, next, out)?;
        write_tag(out, field.number, 4);
        return Ok(());
    }
    write_field(py, field.number, child, sizes, next, out)
}

/// Write `child` as the field `number` of its message: its tag, its size and its bytes.
fn write_field(
    py: Python<'_>,
    number: u32,
    child: &Fields,
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

/// The cause of the error of a parse of a message nested deeper than its limit.
const TOO_DEEP: &str = "Exceeded upb_DecodeOptions_MaxDepth";

/// upb's reason of a decode error of `what`.
fn upb_reason(what: &str) -> &'static str {
    if what.contains("UTF-8") {
        "String field had bad UTF-8"
    } else if what == TOO_DEEP {
        TOO_DEEP
    } else {
        "Wire format was corrupt"
    }
}

/// The bytes past the end of the data that upb reads of a packed field of a fixed width, as
/// zeros, once its input stream reads of its patch buffer, of these bytes of slop
/// (`kUpb_EpsCopyInputStream_SlopBytes`): when the data is no longer than them, or once the parse
/// read a tag of the last of them.
const PACKED_SLOP: usize = 16;

/// The limit of the nesting of the messages that a parse reads, as upb's of its parsers
/// (`kUpb_WireFormat_DefaultDepthLimit`): a message, a map entry or a group of more than 100
/// levels below the message that the parse reads fails it.
pub(crate) const DEPTH_LIMIT: u32 = 100;

/// The limit of the parses, of the levels of messages that a parse reads below the message that
/// it parses: [`DEPTH_LIMIT`], or the limit of the parsers of the `protobuf` of the process, which
/// `pyspark.sql.connect.proto._rust` sets (see [`set_depth_limit`]): 65535 levels once upb's
/// `SetAllowOversizeProtos(True)` is called, or that of the `decoder.SetRecursionLimit` of
/// protobuf's pure-Python implementation.
static LIMIT: AtomicU32 = AtomicU32::new(DEPTH_LIMIT);

/// A function of Python that gives the limit of the parsers of `protobuf` (see [`LIMIT`]), which a
/// parse calls when it reaches its limit, as the limit may have changed since it was set, of a
/// call of `protobuf` that the runtime does not see.
static DEPTH_PROBE: Mutex<Option<Py<PyAny>>> = Mutex::new(None);

/// The levels of messages that a parse reads below the message that it parses, of the limit of
/// the parsers of the process (see [`LIMIT`]).
pub(crate) fn depth_limit() -> u32 {
    LIMIT.load(Ordering::Relaxed)
}

/// Set the limit of the parses (see [`LIMIT`]). upb's parser recurses on the stack, so a message of
/// tens of thousands of levels can crash it; this runtime's parser grows its stack, and does not.
/// The responses of the server that the core decodes take it too, as grpc's deserializer of
/// PySpark's client parses them with upb (see `spark_connect_proto::ClientCodec`).
pub(crate) fn set_depth_limit(levels: u32) {
    LIMIT.store(levels, Ordering::Relaxed);
    spark_connect_proto::set_depth_limit(levels);
}

/// Set the function of Python that gives the limit of the parsers of `protobuf` (see
/// [`DEPTH_PROBE`]), or none.
pub(crate) fn set_depth_probe(probe: Option<Py<PyAny>>) {
    let previous = std::mem::replace(
        &mut *DEPTH_PROBE.lock().unwrap_or_else(|e| e.into_inner()),
        probe,
    );
    // Released once the function is no longer locked.
    drop(previous);
}

/// The limit of the parses once [`DEPTH_PROBE`] has told it again, which a parse that reached its
/// limit reads: of a probe that fails or gives no number, the limit as it is.
fn probed_depth_limit() -> u32 {
    Python::attach(|py| {
        let probe = DEPTH_PROBE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map(|probe| probe.clone_ref(py));
        if let Some(levels) = probe.and_then(|p| p.call0(py).ok()?.extract::<u32>(py).ok()) {
            set_depth_limit(levels);
        }
    });
    depth_limit()
}

/// The limit of the parsers of `protobuf` for a response of the server that would nest past the
/// limit of the core (see `spark_connect_proto::set_depth_probe`), as [`probed_depth_limit`] tells
/// it, once the runtime has installed, which a deep response installs: the runtime takes the limit
/// of upb when it installs. Only on a thread of Python, as that of a call of the client is: a
/// thread of the runtime has no thread state of Python, and does not wait for the GIL.
pub(crate) fn response_depth_probe() -> Option<u32> {
    // SAFETY: reads the thread state that the interpreter keeps for this thread, without the
    // GIL, which it does not need.
    if unsafe { pyo3::ffi::PyGILState_GetThisThreadState() }.is_null() {
        return None;
    }
    let installed = DEPTH_PROBE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .is_some();
    if !installed {
        // The runtime sets the limit and the probe when it installs; of an import that fails, the
        // limit stays.
        Python::attach(|py| py.import("pyspark.sql.connect.proto").map(drop).ok());
    }
    Some(probed_depth_limit())
}

/// The levels of messages that a parse may still read below the message that it reads, of the
/// limit that it started of ([`depth_limit`]), or of no limit, for the parses of the drop-in itself
/// of the messages that PySpark builds in Python, of any depth (see [`merge_unlimited`]).
#[derive(Clone, Copy)]
pub(crate) struct Depth {
    left: Option<u32>,
    limit: u32,
}

impl Depth {
    pub(crate) fn limited() -> Depth {
        let limit = depth_limit();
        Depth {
            left: Some(limit),
            limit,
        }
    }

    fn unlimited() -> Depth {
        Depth {
            left: None,
            limit: u32::MAX,
        }
    }

    /// The depth of a message, map entry or group one level below, or upb's error past the limit.
    /// A parse that reaches its limit asks `protobuf` for its limit again (see
    /// [`probed_depth_limit`]), which a call of upb's `SetAllowOversizeProtos(True)` of a reference
    /// to it taken before the runtime installed raised, and then reads on to it.
    fn below(self) -> PyResult<Depth> {
        match self.left {
            None => Ok(self),
            Some(0) => {
                let limit = probed_depth_limit();
                if limit > self.limit {
                    Ok(Depth {
                        left: Some(limit - self.limit - 1),
                        limit,
                    })
                } else {
                    Err(decode_error(TOO_DEEP))
                }
            }
            Some(levels) => Ok(Depth {
                left: Some(levels - 1),
                limit: self.limit,
            }),
        }
    }
}

/// A reader of the bytes of a message, from `pos` to `end`, of `data`, the bytes of the parse, as
/// upb's decoder reads them: a value, a tag or a length that runs past `end` is read of the bytes
/// that follow it in the data, and of zeros past the end of the data (as upb reads of its patch
/// buffer), and the parse fails once the value is set, when the message's next field is to be
/// read ([`Reader::check_end`]). A string or the bytes of a bytes field are read of the data past
/// `end` too, but not past the end of the data; a message, a map entry and a packed field must end
/// within `end`.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
    end: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Reader {
            data,
            pos: 0,
            end: data.len(),
        }
    }

    fn done(&self) -> bool {
        self.pos >= self.end
    }

    /// The failure of a read past the end of the message, which upb tells before it reads its
    /// next field, once it set the value that it read.
    fn check_end(&self) -> PyResult<()> {
        if self.pos > self.end {
            return Err(decode_error("a value past the end of its message"));
        }
        Ok(())
    }

    /// The byte at the position, or a zero past the end of the data.
    fn byte(&mut self) -> u8 {
        let byte = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        byte
    }

    /// The field number and the wire type of a tag, as upb reads a tag: a varint of at most five
    /// bytes whose value fits in 32 bits, so a field number is at most 2^29 - 1. A longer tag, or
    /// one of more bits, is corrupt (and not cut to the bits of a field number).
    fn tag(&mut self) -> PyResult<(u32, u8)> {
        let mut value = 0u64;
        for shift in (0..35).step_by(7) {
            let byte = self.byte();
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
            let byte = self.byte();
            if shift < 64 {
                value |= ((byte & 0x7f) as u64) << shift;
            }
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(decode_error("varint too long"))
    }

    fn fixed32(&mut self) -> [u8; 4] {
        std::array::from_fn(|_| self.byte())
    }

    fn fixed64(&mut self) -> [u8; 8] {
        std::array::from_fn(|_| self.byte())
    }

    /// The length of a length-delimited value, as upb reads one: a varint of at most five bytes
    /// (a longer one is corrupt, even of a small value) and of at most 2^31 - 1, which the read of
    /// the value checks.
    fn size(&mut self) -> PyResult<u64> {
        let mut len = 0u64;
        for shift in (0..35).step_by(7) {
            let byte = self.byte();
            len |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                if len > i32::MAX as u64 {
                    return Err(decode_error("length too large"));
                }
                return Ok(len);
            }
        }
        Err(decode_error("length of more than five bytes"))
    }

    /// The position `len` bytes past the position, if it is at most `bound`.
    fn end_of(&self, len: u64, bound: usize) -> Option<usize> {
        usize::try_from(len)
            .ok()
            .and_then(|len| self.pos.checked_add(len))
            .filter(|&end| end <= bound)
    }

    /// The `len` bytes at the position, of a string or of a bytes field, as upb reads them: past
    /// the end of the message too, but within the data.
    fn bytes(&mut self, len: u64) -> PyResult<&'a [u8]> {
        let end = self
            .end_of(len, self.data.len())
            .ok_or_else(|| decode_error("truncated string"))?;
        let bytes = &self.data[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }

    /// A reader of the `len` bytes at the position, of a message, a map entry or a packed field of
    /// varints, which must end within the message, as upb's limit of them (`PushLimit`). The
    /// position of this reader is then past them.
    fn sub(&mut self, len: u64) -> PyResult<Reader<'a>> {
        let end = self
            .end_of(len, self.end)
            .ok_or_else(|| decode_error("truncated message"))?;
        let sub = Reader {
            data: self.data,
            pos: self.pos,
            end,
        };
        self.pos = end;
        Ok(sub)
    }

    /// A value of a field of `kind`, of its wire type, but of a string or bytes field (see
    /// [`Reader::bytes`]).
    fn value(&mut self, kind: Kind) -> PyResult<Scalar> {
        Ok(match kind {
            Kind::Double => Scalar::F64(f64::from_le_bytes(self.fixed64())),
            Kind::Float => Scalar::F32(f32::from_le_bytes(self.fixed32())),
            Kind::Int64 => Scalar::I64(self.varint()? as i64),
            Kind::SInt64 => {
                let v = self.varint()?;
                Scalar::I64(((v >> 1) as i64) ^ -((v & 1) as i64))
            }
            Kind::SFixed64 => Scalar::I64(i64::from_le_bytes(self.fixed64())),
            Kind::UInt64 => Scalar::U64(self.varint()?),
            Kind::Fixed64 => Scalar::U64(u64::from_le_bytes(self.fixed64())),
            Kind::Int32 => Scalar::I32(self.varint()? as i64 as i32),
            Kind::SInt32 => {
                let v = self.varint()? as u32;
                Scalar::I32(((v >> 1) as i32) ^ -((v & 1) as i32))
            }
            Kind::SFixed32 => Scalar::I32(i32::from_le_bytes(self.fixed32())),
            Kind::UInt32 => Scalar::U32(self.varint()? as u32),
            Kind::Fixed32 => Scalar::U32(u32::from_le_bytes(self.fixed32())),
            Kind::Bool => Scalar::Bool(self.varint()? != 0),
            Kind::Enum => Scalar::Enum(self.varint()? as i64 as i32),
            Kind::String | Kind::Bytes => {
                let len = self.size()?;
                let bytes = self.bytes(len)?;
                string_of(kind, bytes, true)?
            }
            Kind::Message => return Err(decode_error("a message is not a scalar")),
        })
    }

    /// Skip the value of an unknown field of `wire_type`, as upb skips one: a varint, a fixed
    /// value or the bytes of a length (past the end of the message too, which the next read
    /// tells), or a group up to its end, a level of `depth`, whose fields it reads as tags are
    /// read, but for which a field number of 0 is not corrupt.
    fn skip(&mut self, wire_type: u8, number: u32, depth: Depth) -> PyResult<()> {
        match wire_type {
            0 => {
                self.varint()?;
            }
            1 => self.pos += 8,
            2 => {
                let len = self.size()?;
                self.pos = usize::try_from(len)
                    .ok()
                    .and_then(|len| self.pos.checked_add(len))
                    .unwrap_or(usize::MAX);
            }
            5 => self.pos += 4,
            3 => {
                // As upb's, which skips an unknown group of more levels than its limit as corrupt.
                let depth = depth
                    .below()
                    .map_err(|_| decode_error("unknown group of too many levels"))?;
                loop {
                    if self.done() {
                        self.check_end()?;
                        return Err(decode_error("unterminated group"));
                    }
                    let (inner, inner_type) = self.tag()?;
                    if inner_type == 4 {
                        if inner != number {
                            return Err(decode_error("mismatched end of group"));
                        }
                        break;
                    }
                    grow(|| self.skip(inner_type, inner, depth))?;
                    self.check_end()?;
                }
            }
            _ => return Err(decode_error("invalid wire type")),
        }
        Ok(())
    }
}

/// The value of a string or bytes field of `bytes`: of a string field, its UTF-8, which it must be
/// if `validate`; else, of bytes that are not UTF-8, the bytes, as upb gives the value of a string
/// field of proto2 (or of the `NONE` validation of editions) that is not UTF-8.
fn string_of(kind: Kind, bytes: &[u8], validate: bool) -> PyResult<Scalar> {
    if kind == Kind::Bytes {
        return Ok(Scalar::Bytes(bytes.to_vec()));
    }
    match std::str::from_utf8(bytes) {
        Ok(text) => Ok(Scalar::Str(text.to_string())),
        Err(_) if !validate => Ok(Scalar::Bytes(bytes.to_vec())),
        Err(_) => Err(decode_error("invalid UTF-8 in a string field")),
    }
}

/// A value of `field` that `reader` reads, of its kind, but of a string of its UTF-8 validation.
fn value_of_field(reader: &mut Reader<'_>, field: &FieldDesc) -> PyResult<Scalar> {
    if field.kind == Kind::String && !field.validate_utf8 {
        let len = reader.size()?;
        let bytes = reader.bytes(len)?;
        return string_of(Kind::String, bytes, false);
    }
    reader.value(field.kind)
}

/// Whether `value`, a value of an enum that a parse read, of the values `closed` of a closed enum
/// (of proto2, or of the `CLOSED` enum type of editions), is one that the enum takes: any of an
/// open enum (of no `closed`), a declared one of a closed enum, whose other values upb keeps in
/// the unknown fields.
fn enum_value_known(closed: Option<&[i32]>, value: &Scalar) -> bool {
    match (closed, value) {
        (Some(values), Scalar::Enum(number)) => values.binary_search(number).is_ok(),
        _ => true,
    }
}

/// Keep the field that `reader` read from `tag_start`, of a value of a closed enum that its enum
/// does not declare, in the unknown fields of `message`, as upb keeps it: of the bytes of its tag
/// and value as they are. A value read past the end of the message is not kept, and the parse
/// fails, as upb's does.
fn keep_unknown_enum(
    message: &Bound<'_, PyMessage>,
    reader: &Reader<'_>,
    tag_start: usize,
) -> PyResult<()> {
    reader.check_end()?;
    keep_unknown(message, &reader.data[tag_start..reader.pos]);
    Ok(())
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
            1 => UnknownValue::Fixed64(u64::from_le_bytes(reader.fixed64())),
            2 => {
                let len = reader.size()?;
                UnknownValue::Delimited(reader.sub(len)?.rest())
            }
            3 => UnknownValue::Group(grow(|| unknown_fields_of(reader, Some(number)))?),
            4 if group == Some(number) => break,
            5 => UnknownValue::Fixed32(u32::from_le_bytes(reader.fixed32())),
            _ => return Err(decode_error("invalid unknown field")),
        };
        reader.check_end()?;
        fields.push(((number << 3) | u32::from(wire_type), value));
    }
    fields.sort_by_key(|&(tag, _)| tag);
    Ok(fields)
}

impl<'a> Reader<'a> {
    /// The bytes of the reader from its position to its end.
    fn rest(&self) -> &'a [u8] {
        &self.data[self.pos.min(self.end)..self.end]
    }
}

/// The fields of the unknown fields `data` of a message, each the bytes of its tag and the bytes
/// of its value (of a length-delimited field, of its length too, and of a group, of its fields and
/// of its end tag), in their order, as protobuf's pure-Python messages hold them
/// (`_unknown_fields`), which its `==` reads.
pub(crate) fn unknown_field_pairs(data: &[u8]) -> PyResult<Vec<(&[u8], &[u8])>> {
    let mut reader = Reader::new(data);
    let mut pairs = Vec::new();
    while !reader.done() {
        let start = reader.pos;
        let (number, wire_type) = reader.tag()?;
        let value = reader.pos;
        reader.skip(wire_type, number, Depth::unlimited())?;
        reader.check_end()?;
        pairs.push((&data[start..value], &data[value..reader.pos]));
    }
    Ok(pairs)
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
/// runtime's class of it, which is protobuf's `google.protobuf.message.DecodeError` where
/// protobuf imports (see [`super::decode_error_of`]). As upb's, a message of more than 100 levels
/// below `message` fails it ([`DEPTH_LIMIT`]).
pub(crate) fn merge(py: Python<'_>, message: &Bound<'_, PyMessage>, data: &[u8]) -> PyResult<()> {
    merge_with(
        py,
        message,
        data,
        Some(&ConversionBudget::full()),
        Depth::limited(),
    )
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
    merge_with(py, message, data, Some(budget), Depth::limited())
}

/// Merge `data` into `message` as [`merge`] does, without checkpoints: for a merge of the text
/// of a message, which holds borrows of the messages that it prints.
pub(crate) fn merge_unchecked(
    py: Python<'_>,
    message: &Bound<'_, PyMessage>,
    data: &[u8],
) -> PyResult<()> {
    merge_with(py, message, data, None, Depth::limited())
}

/// Merge `data` into `message` as [`merge`] does, but of any depth: the drop-in's own parse of the
/// bytes of a message that PySpark builds in Python, as the relation of a DataFrame and the
/// expression of a Column, which it parses as upb never does (see `parse_deep` of
/// `pyspark.sql.connect._column_reduce`), and the copy of a message of protobuf's into one of the
/// runtime, which upb copies of any depth (`CopyFrom`, `==`).
pub(crate) fn merge_unlimited(
    py: Python<'_>,
    message: &Bound<'_, PyMessage>,
    data: &[u8],
) -> PyResult<()> {
    merge_unlimited_in(py, message, data, &ConversionBudget::full())
}

/// Merge `data` into `message` as [`merge_unlimited`] does, a step of `budget` for each message and
/// each map entry (see [`merge_in`]).
pub(crate) fn merge_unlimited_in(
    py: Python<'_>,
    message: &Bound<'_, PyMessage>,
    data: &[u8],
    budget: &ConversionBudget,
) -> PyResult<()> {
    merge_with(py, message, data, Some(budget), Depth::unlimited())
}

fn merge_with(
    py: Python<'_>,
    message: &Bound<'_, PyMessage>,
    data: &[u8],
    budget: Option<&ConversionBudget>,
    depth: Depth,
) -> PyResult<()> {
    let mut reader = Reader::new(data);
    merge_message(py, message, &mut reader, budget, depth).map_err(|err| {
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

/// Merge the message that `reader` reads into `message`, one level of a merge, on a stack with
/// room for it, with the cause of a failure. Each message merged, and each map entry, is a step of
/// `budget`, whose checkpoints find no borrow held: the merge borrows a message only to set one of
/// its fields.
fn merge_message(
    py: Python<'_>,
    message: &Bound<'_, PyMessage>,
    reader: &mut Reader<'_>,
    budget: Option<&ConversionBudget>,
    depth: Depth,
) -> PyResult<()> {
    grow(|| merge_inner(py, message, reader, budget, depth, None))
}

/// Merge the group that `reader` reads into `message`, up to its end tag of `number`, one level of
/// a merge, as [`merge_message`] merges a message: a group that the end of the message that holds
/// it ends first is corrupt.
fn merge_group(
    py: Python<'_>,
    message: &Bound<'_, PyMessage>,
    reader: &mut Reader<'_>,
    budget: Option<&ConversionBudget>,
    (depth, number): (Depth, u32),
) -> PyResult<()> {
    grow(|| merge_inner(py, message, reader, budget, depth, Some(number)))
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

/// Whether `field` reads a value of `wire_type`: a value of another wire type is an unknown field,
/// as protobuf reads it. A repeated numeric field takes its values packed or not.
fn reads_wire_type(field: &FieldDesc, wire_type: u8) -> bool {
    if field.map.is_some() {
        return wire_type == 2;
    }
    wire_type == field.wire_type() || (field.repeated && wire_type == 2 && field.kind.packable())
}

fn merge_inner(
    py: Python<'_>,
    message: &Bound<'_, PyMessage>,
    reader: &mut Reader<'_>,
    budget: Option<&ConversionBudget>,
    depth: Depth,
    group: Option<u32>,
) -> PyResult<()> {
    if let Some(budget) = budget {
        budget.step(py)?;
    }
    let desc = message.borrow().desc.clone();
    let mut repeated = Repeated::default();
    // As upb's: the run of the fields of numbers that the message type has no field of, which the
    // message read since the tag of its last field of a number of its type, which it keeps at the
    // next such tag, or at its end (also at an end of a group), so that a failure before either
    // drops them. A field of a number of its type but of another wire type is kept as it is read.
    let mut unknown_from: Option<usize> = None;
    while !reader.done() {
        let start = reader.pos;
        let (number, wire_type) = reader.tag()?;
        if wire_type == 4 {
            keep_unknown_run(message, reader, unknown_from, start);
            if group == Some(number) {
                return Ok(());
            }
            return Err(decode_error("unexpected end of group"));
        }
        let field = field_of_number(&desc, number);
        if field.is_some() {
            keep_unknown_run(message, reader, unknown_from.take(), start);
        } else if number == 0 && !(desc.fields.is_empty() && matches!(wire_type, 0..=3 | 5)) {
            // As upb's: a field of the number 0 is corrupt, but in a message type of no fields
            // (`google.protobuf.Empty`, `ResultComplete`), of which a field of the number 0 of a
            // varint, a fixed value, a length or a group is an unknown field.
            return Err(decode_error("invalid field number 0"));
        }
        match field {
            Some(field) if reads_wire_type(field, wire_type) => merge_field(
                py,
                message,
                field,
                (start, wire_type),
                reader,
                &mut repeated,
                budget,
                depth,
            )?,
            Some(_) => {
                reader.skip(wire_type, number, depth)?;
                reader.check_end()?;
                keep_unknown_run(message, reader, Some(start), reader.pos);
            }
            None => {
                reader.skip(wire_type, number, depth)?;
                unknown_from.get_or_insert(start);
            }
        }
        reader.check_end()?;
    }
    keep_unknown_run(message, reader, unknown_from, reader.end);
    if group.is_some() {
        return Err(decode_error("unterminated group"));
    }
    Ok(())
}

/// Keep the fields that `reader` read from `from` to `to` in the unknown fields of `message`.
fn keep_unknown_run(
    message: &Bound<'_, PyMessage>,
    reader: &Reader<'_>,
    from: Option<usize>,
    to: usize,
) {
    if let Some(from) = from {
        keep_unknown(message, &reader.data[from..to]);
    }
}

fn keep_unknown(message: &Bound<'_, PyMessage>, bytes: &[u8]) {
    message.borrow_mut().unknown.extend_from_slice(bytes);
}

/// Merge the value of `field` that `reader` is at, of `wire_type`, which the field reads, into
/// `message`. As upb's, the message of a message field is set (or appended) before its bytes are
/// read, so a failure in them leaves it set, of the fields read before the failure, and so is a
/// member of a oneof or an `optional` string or bytes field whose bytes fail, of the value that
/// [`failed_string_value`] gives.
#[allow(clippy::too_many_arguments)]
fn merge_field(
    py: Python<'_>,
    message: &Bound<'_, PyMessage>,
    field: &FieldDesc,
    (tag_start, wire_type): (usize, u8),
    reader: &mut Reader<'_>,
    repeated: &mut Repeated,
    budget: Option<&ConversionBudget>,
    depth: Depth,
) -> PyResult<()> {
    let index = field.index;
    if let Some(map) = &field.map {
        let len = reader.size()?;
        let mut entry = reader.sub(len)?;
        return merge_map_entry(py, message, field, map, &mut entry, budget, depth.below()?);
    }
    if field.kind == Kind::Message {
        let len = if field.delimited { 0 } else { reader.size()? };
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
        if field.delimited {
            return merge_group(py, &child, reader, budget, (depth.below()?, field.number));
        }
        let mut sub = reader.sub(len)?;
        return merge_message(py, &child, &mut sub, budget, depth.below()?);
    }
    if field.repeated {
        if wire_type == 2 && field.kind.packable() {
            let len = reader.size()?;
            // Values of a fixed width are read at once, of a payload of a whole number of them
            // (else corrupt, as upb's, which then sets none); varints in turn, each set once it
            // is read, as upb sets them, so that one cut short by the end of the payload is set.
            let width = match field.kind.wire_type() {
                1 => 8,
                5 => 4,
                _ => 0,
            };
            if width > 0 {
                if len % width != 0 {
                    return Err(decode_error("truncated packed field"));
                }
                // As upb copies them: of the bytes past the end of the message too, within the
                // data, or, of its patch buffer, of zeros past the end of the data, up to the
                // bytes of its slop.
                let patched = reader.data.len() <= PACKED_SLOP
                    || tag_start + PACKED_SLOP >= reader.data.len();
                let bound = reader.data.len() + if patched { PACKED_SLOP } else { 0 };
                let end = reader
                    .end_of(len, bound)
                    .ok_or_else(|| decode_error("truncated packed field"))?;
                let container = repeated.scalars(py, message, field)?;
                let mut c = container.bind(py).borrow_mut();
                c.items.reserve((end - reader.pos) / width as usize);
                while reader.pos < end {
                    let value = reader.value(field.kind)?;
                    c.items.push(value);
                }
                return Ok(());
            }
            let mut packed = reader.sub(len)?;
            let container = repeated.scalars(py, message, field)?;
            let closed = field.closed_values.as_deref();
            while !packed.done() {
                if let Some(values) = closed {
                    // As upb's: a value that the closed enum does not declare goes to the unknown
                    // fields as a varint field of the number of the field (of a tag of its own),
                    // of the 64 bits of the value as it is read, also one read past the end of the
                    // packed values, before the parse fails.
                    let raw = packed.varint()?;
                    let number = raw as i64 as i32;
                    if values.binary_search(&number).is_ok() {
                        container
                            .bind(py)
                            .borrow_mut()
                            .items
                            .push(Scalar::Enum(number));
                    } else {
                        let mut m = message.borrow_mut();
                        write_tag(&mut m.unknown, field.number, 0);
                        write_varint(&mut m.unknown, raw);
                    }
                    continue;
                }
                let value = packed.value(field.kind)?;
                container.bind(py).borrow_mut().items.push(value);
            }
            return packed.check_end();
        }
        let value = value_of_field(reader, field)?;
        if !enum_value_known(field.closed_values.as_deref(), &value) {
            return keep_unknown_enum(message, reader, tag_start);
        }
        let container = repeated.scalars(py, message, field)?;
        container.bind(py).borrow_mut().items.push(value);
        return Ok(());
    }
    if matches!(field.kind, Kind::String | Kind::Bytes) {
        let len = reader.size()?;
        let read = reader
            .bytes(len)
            .and_then(|bytes| string_of(field.kind, bytes, field.validate_utf8));
        return match read {
            Ok(value) => {
                message.borrow_mut().set_scalar(py, index, value);
                Ok(())
            }
            Err(err) => {
                if field.has_presence() {
                    let value = failed_string_value(message, field);
                    message.borrow_mut().set_scalar(py, index, value);
                }
                Err(err)
            }
        };
    }
    let value = reader.value(field.kind)?;
    if !enum_value_known(field.closed_values.as_deref(), &value) {
        return keep_unknown_enum(message, reader, tag_start);
    }
    message.borrow_mut().set_scalar(py, index, value);
    Ok(())
}

/// The value that a string or bytes field of presence `field` of `message` takes of a read of its
/// bytes that fails, which upb sets before it reads them, of the memory of the field: the value
/// that the field holds, if it is set; the value of the member of its oneof that holds a string or
/// bytes, which upb's oneof reads of the same memory (as a string, if it is UTF-8); else an empty
/// value (where upb reads the memory of another member as a string, which only an empty one is
/// safely, or the zeros of an unset field rather than its default).
fn failed_string_value(message: &Bound<'_, PyMessage>, field: &FieldDesc) -> Scalar {
    let m = message.borrow();
    let empty = Scalar::default_of(field.kind);
    let held_by = |member: usize| match m.slots.get(member) {
        Some(Slot::Scalar(Scalar::Str(text))) => Some(text.as_bytes().to_vec()),
        Some(Slot::Scalar(Scalar::Bytes(bytes))) => Some(bytes.clone()),
        _ => None,
    };
    let held = match field.oneof {
        Some(oneof) => m.desc.oneofs[oneof].fields.iter().find_map(|&i| held_by(i)),
        None => held_by(field.index),
    };
    match held {
        Some(bytes) => string_of(field.kind, &bytes, field.validate_utf8).unwrap_or(empty),
        None => empty,
    }
}

/// A key or a scalar value of `kind` of a map entry that `reader` reads, of a string of the UTF-8
/// validation `validate` of the field of the entry message (see [`string_of`]).
fn map_value(reader: &mut Reader<'_>, kind: Kind, validate: bool) -> PyResult<Scalar> {
    if kind == Kind::String && !validate {
        let len = reader.size()?;
        let bytes = reader.bytes(len)?;
        return string_of(Kind::String, bytes, false);
    }
    reader.value(kind)
}

/// Merge the map entry that `entry` reads, of the map field `field`, into `message`, as upb decodes
/// one: as a message of the entry type, a level of `depth`, whose key and value are set in the
/// map. An entry with fields that the entry type has no field for, or with a key or value of
/// another wire type (which upb reads as such fields), is not set: the message keeps it in its
/// unknown fields, as upb encodes the entry message again, of its key and value (each written as a
/// field of a message is, so a scalar of the default is not) and then those fields.
fn merge_map_entry(
    py: Python<'_>,
    message: &Bound<'_, PyMessage>,
    field: &FieldDesc,
    map: &MapDesc,
    reader: &mut Reader<'_>,
    budget: Option<&ConversionBudget>,
    depth: Depth,
) -> PyResult<()> {
    if let Some(budget) = budget {
        budget.step(py)?;
    }
    let mut key = Scalar::default_of(map.key);
    let mut key_present = false;
    let mut scalar = None;
    let mut child: Option<Bound<'_, PyMessage>> = None;
    let mut unknown = Vec::new();
    while !reader.done() {
        let start = reader.pos;
        let (number, wire_type) = reader.tag()?;
        match number {
            0 => return Err(decode_error("invalid field number 0")),
            1 if wire_type == map.key.wire_type() => {
                key = map_value(reader, map.key, map.key_validate_utf8)?;
                key_present = true;
            }
            2 if map.value == Kind::Message && wire_type == 2 => {
                let len = reader.size()?;
                if child.is_none() {
                    child = Some(new_instance(py, &map.value_type)?);
                }
                let mut sub = reader.sub(len)?;
                merge_message(
                    py,
                    child.as_ref().expect("made above"),
                    &mut sub,
                    budget,
                    depth.below()?,
                )?;
            }
            2 if map.value != Kind::Message && wire_type == map.value.wire_type() => {
                let value = map_value(reader, map.value, map.value_validate_utf8)?;
                if enum_value_known(map.value_closed.as_deref(), &value) {
                    scalar = Some(value);
                } else {
                    // As upb's: an entry of a value that a closed enum does not declare is kept
                    // in the unknown fields, of the value as an unknown field of the entry, of
                    // its bytes as they are.
                    reader.check_end()?;
                    unknown.extend_from_slice(&reader.data[start..reader.pos]);
                }
            }
            _ => {
                reader.skip(wire_type, number, depth)?;
                reader.check_end()?;
                unknown.extend_from_slice(&reader.data[start..reader.pos]);
            }
        }
        reader.check_end()?;
    }
    if !unknown.is_empty() {
        // As upb encodes the entry again: a key or a value of the default is written when it was
        // read, of an entry message whose fields have presence (of proto2, or of the `EXPLICIT`
        // presence of editions), and not of one of proto3.
        let mut bytes = Vec::new();
        if !key.is_default() || (key_present && map.key_presence) {
            write_tag(&mut bytes, 1, map.key.wire_type());
            write_value(&mut bytes, map.key, &key);
        }
        match (&child, scalar) {
            (Some(child), _) => encode_message_field(py, 2, child.as_unbound(), &mut bytes)?,
            (None, Some(value)) if !value.is_default() || map.value_presence => {
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
