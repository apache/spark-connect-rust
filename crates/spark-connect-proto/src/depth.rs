//! The nesting of the messages of an encoded message, found without decoding it: a scan of its
//! wire format with a stack of its own, which does not recurse, of a table of the fields of each
//! message type that hold messages (`MESSAGE_TYPES`, which `build.rs` writes of the descriptor set
//! of the protos), so that the client decodes a response only when its messages nest at most as
//! deep as the limit of the client ([`depth_limit`]).
//!
//! The scan counts the levels as upb's decoder and prost's do: a message, a map entry and a group
//! (also one of a field that the type does not know) are each a level below the message that
//! holds them, and the message that the scan reads is at level 0. A field that the type does not
//! know, but a group, is skipped, of no level, as upb does. It reads the bytes as prost does, so
//! that prost never nests deeper than the scan found: a message or a map entry ends within the
//! data, not within the message that holds it, and a map field takes an entry of any wire type.
//! Where prost fails to decode the bytes, the scan stops there, and prost reports its own error.

use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::OnceLock;

/// A message type of the descriptor set (see `build.rs`).
pub(crate) struct MessageType {
    /// The full name of the type, such as `spark.connect.DataType`.
    pub(crate) name: &'static str,
    /// The identifier of the struct of the type, which prost's errors name it by (`DataType`).
    pub(crate) ident: &'static str,
    /// The fields of the type that hold messages, by their numbers.
    pub(crate) fields: &'static [Field],
}

/// A field of a message type that holds messages.
pub(crate) struct Field {
    pub(crate) number: u32,
    pub(crate) kind: FieldKind,
    /// The index of the type of its messages (of its map entries, of a map) in `MESSAGE_TYPES`.
    pub(crate) message: u32,
    /// The identifier of the field in the struct, or of its oneof, which prost's errors name it by.
    pub(crate) ident: &'static str,
}

/// How a field holds messages.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum FieldKind {
    /// A message of a message type, length-delimited.
    Message,
    /// A map entry, a message of the entry type, which prost reads of any wire type.
    Map,
    /// A group, delimited by its start and end tags. The protos of Spark Connect, of proto3, have
    /// none, but the scan reads them as prost does.
    #[allow(dead_code)]
    Group,
}

include!(concat!(env!("OUT_DIR"), "/message_fields.rs"));

/// The index in `MESSAGE_TYPES` of the message type of the full name `name`, if the descriptor
/// set has it.
pub(crate) fn message_type_index(name: &str) -> Option<u32> {
    MESSAGE_TYPES
        .binary_search_by(|message| message.name.cmp(name))
        .ok()
        .map(|at| at as u32)
}

/// The index in `MESSAGE_TYPES` of the message type of the full name `name` if it is a response
/// of the service (`RESPONSE_TYPES`), whose messages the client decodes within the limit: not of a
/// request, which the generated server decodes with the same codec.
pub(crate) fn response_type_index(name: &str) -> Option<u32> {
    RESPONSE_TYPES
        .binary_search(&name)
        .ok()
        .and_then(|_| message_type_index(name))
}

/// The limit of the levels of messages of upb's and prost's decoders
/// (`kUpb_WireFormat_DefaultDepthLimit`, prost's `RECURSION_LIMIT`).
pub const DEFAULT_DEPTH_LIMIT: u32 = 100;

static LIMIT: AtomicU32 = AtomicU32::new(DEFAULT_DEPTH_LIMIT);

/// The function that a scan calls when a message would nest past the limit, which may give a
/// higher limit, that the limit then is (see [`set_depth_probe`]).
static PROBE: OnceLock<fn() -> Option<u32>> = OnceLock::new();

/// The levels of messages that a response of the client may nest below it, 100 unless
/// [`set_depth_limit`] set another: a deeper response fails to decode, with the status of prost's
/// error of its limit (see [`crate::ClientCodec`]). Not a stable API.
#[doc(hidden)]
pub fn depth_limit() -> u32 {
    LIMIT.load(Ordering::Relaxed)
}

/// Set the levels of messages that the responses of the clients of the process may nest below
/// them, of every client: the Python binding follows upb's limit of its process with it, 65535
/// levels once `SetAllowOversizeProtos(True)` is called. Not a stable API.
#[doc(hidden)]
pub fn set_depth_limit(levels: u32) {
    LIMIT.store(levels, Ordering::Relaxed);
}

/// Set the function that a scan calls when a response would nest past the limit, once in a
/// process: it gives the limit as it is then, when it can tell it, which the scan takes when it is
/// higher (the Python binding asks upb for its limit, which a call that it does not see may have
/// raised). Not a stable API.
#[doc(hidden)]
pub fn set_depth_probe(probe: fn() -> Option<u32>) {
    let _ = PROBE.set(probe);
}

/// The limit of a scan that reached `limit`, once the probe has told it again.
fn probed_limit(limit: u32) -> u32 {
    match PROBE.get().and_then(|probe| probe()) {
        Some(levels) if levels > limit => {
            set_depth_limit(levels);
            levels
        }
        _ => limit,
    }
}

/// The levels of the messages of the fields that their types have, which the decoded message keeps,
/// that a response may nest below it, whatever the limit: the client converts, clones and drops the
/// messages that it keeps by recursing once per level, which this bounds. It is past the limit of
/// prost's decoder and of upb's (100); only a limit of upb's `SetAllowOversizeProtos(True)`
/// (65535), which the Python binding follows, reaches it. A response past it fails with the
/// [`TooDeep`] of [`TooDeep::is_kept`], whose source in the status of the client is the error,
/// which the client raises as a `RecursionError`.
pub const MAX_KEPT_DEPTH: u32 = 1024;

/// The error of a message that nests past the limit, as prost reports its own limit, of the fields
/// that lead to the level past it, of the identifiers of the generated code, innermost first; or
/// of one whose kept messages nest past [`MAX_KEPT_DEPTH`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TooDeep {
    path: Vec<(&'static str, &'static str)>,
    kept: bool,
}

impl TooDeep {
    /// Whether the message nests its kept messages past [`MAX_KEPT_DEPTH`], within the limit.
    pub fn is_kept(&self) -> bool {
        self.kept
    }
}

impl fmt::Display for TooDeep {
    /// prost's text of its error, `failed to decode Protobuf message: DataType.kind: ...:
    /// recursion limit reached`, or the client's text of [`MAX_KEPT_DEPTH`], which it raises as a
    /// `RecursionError`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.kept {
            return write!(
                f,
                "maximum recursion depth exceeded while decoding a response; the messages of a \
                 response may be nested at most {MAX_KEPT_DEPTH} levels deep"
            );
        }
        f.write_str("failed to decode Protobuf message: ")?;
        for (message, field) in &self.path {
            write!(f, "{message}.{field}: ")?;
        }
        f.write_str("recursion limit reached")
    }
}

/// The [`TooDeep`] of [`TooDeep::is_kept`] of the status of a response that the client failed to
/// decode of it, if any. Not a stable API.
#[doc(hidden)]
pub fn kept_too_deep(status: &tonic::Status) -> Option<&TooDeep> {
    std::iter::successors(std::error::Error::source(status), |error| error.source())
        .find_map(|error| error.downcast_ref::<TooDeep>())
        .filter(|error| error.kept)
}

impl std::error::Error for TooDeep {}

/// What a scan found of a message within the limit: the levels below it of its deepest message,
/// of the messages that prost keeps (of the fields that the types know) and of all.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Depth {
    /// The level of the deepest message, map entry or group.
    pub levels: u32,
    /// The level of the deepest message or map entry of the fields that the types know, which the
    /// decoded message holds.
    pub kept: u32,
}

/// A level of the scan: a message (or map entry) of a type, or a group, of a type or of a field that
/// the type of the level above does not know.
#[derive(Clone, Copy)]
struct Frame {
    /// The index of the type of the level in the table, if it is known.
    message: Option<u32>,
    /// The fields of the type that hold messages, none of a group of no known type.
    fields: &'static [Field],
    /// The end of a message in the data, or `usize::MAX` of a group, which its end tag ends.
    end: usize,
    /// The field number of the end tag of a group, or 0 of a message (no field is of 0).
    group: u32,
    /// The field of the level above that leads to this one, if the type above knows it.
    via: Option<&'static Field>,
    /// Whether the decoded message keeps this level: of known fields, from the top down.
    kept: bool,
    /// Whether prost's errors name the fields of this level: those of a message and of a group of
    /// a known type, but not of a map entry (which has no struct), nor of an unknown group.
    named: bool,
}

/// The reader of the scan: a position in the data.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    /// A varint, as prost's `decode_varint` reads one, or none where prost fails: of more than ten
    /// bytes, of a tenth byte above 1, or cut short by the end of the data.
    fn varint(&mut self) -> Option<u64> {
        let first = *self.data.get(self.pos)?;
        if first < 0x80 {
            self.pos += 1;
            return Some(u64::from(first));
        }
        let mut value = 0u64;
        for at in 0..10 {
            let byte = *self.data.get(self.pos + at)?;
            if at == 9 && byte > 1 {
                return None;
            }
            value |= u64::from(byte & 0x7f) << (7 * at);
            if byte < 0x80 {
                self.pos += at + 1;
                return Some(value);
            }
        }
        None
    }

    /// The field number and wire type of a key, as prost's `decode_key` reads them, or none where
    /// prost fails: a key past 32 bits, a wire type of 6 or 7, or a field number of 0.
    fn key(&mut self) -> Option<(u32, u8)> {
        let key = u32::try_from(self.varint()?).ok()?;
        let wire_type = (key & 7) as u8;
        let number = key >> 3;
        (wire_type < 6 && number > 0).then_some((number, wire_type))
    }

    /// The end of a length-delimited value of the length that the reader reads, which must end
    /// within the data, as prost checks it.
    fn delimited(&mut self) -> Option<usize> {
        let len = usize::try_from(self.varint()?).ok()?;
        let end = self.pos.checked_add(len)?;
        (end <= self.data.len()).then_some(end)
    }

    /// Skip a value of a field of `wire_type` that is not a group, as prost's `skip_field` does,
    /// or none where prost fails.
    fn skip(&mut self, wire_type: u8) -> Option<()> {
        let end = match wire_type {
            0 => return self.varint().map(drop),
            1 => self.pos.checked_add(8)?,
            2 => self.delimited()?,
            5 => self.pos.checked_add(4)?,
            _ => return None,
        };
        (end <= self.data.len()).then(|| self.pos = end)
    }
}

/// Scan `data`, the bytes of a message of the type `message` (an index of `MESSAGE_TYPES`), for
/// the levels of its messages, within `limit` levels (see the module): its [`Depth`], or the
/// [`TooDeep`] of prost's error at the first level past the limit. Bytes that prost fails to
/// decode end the scan, with the depth it found.
pub(crate) fn scan(message: u32, data: &[u8], limit: u32) -> Result<Depth, TooDeep> {
    scan_in(MESSAGE_TYPES, message, data, limit)
}

/// [`scan`] of the message types `types`.
fn scan_in(types: &[MessageType], message: u32, data: &[u8], limit: u32) -> Result<Depth, TooDeep> {
    let mut limit = limit;
    let mut reader = Reader { data, pos: 0 };
    let mut depth = Depth::default();
    // The levels above the level that the scan reads (`top`), the first of the message itself.
    let mut frames: Vec<Frame> = Vec::new();
    let mut top = Frame {
        message: Some(message),
        fields: types[message as usize].fields,
        end: data.len(),
        group: 0,
        via: None,
        kept: true,
        named: true,
    };
    loop {
        if reader.pos >= top.end {
            // prost fails a message that a value read past its end overran.
            if reader.pos > top.end {
                return Ok(depth);
            }
            match frames.pop() {
                Some(parent) => top = parent,
                None => return Ok(depth),
            }
            continue;
        }
        let Some((number, wire_type)) = reader.key() else {
            return Ok(depth);
        };
        if top.group != 0 && wire_type == 4 {
            // The end of a group, which must be of its number.
            if number != top.group {
                return Ok(depth);
            }
            top = frames.pop().expect("a group is below a message");
            continue;
        }
        let field = top
            .fields
            .binary_search_by_key(&number, |field| field.number)
            .ok()
            .map(|at| &top.fields[at]);
        // The next level, if the field holds a message (of the type of the index, if known), which
        // ends at the position, or at the end tag of a group of the number; else the value is
        // skipped.
        let (message, end, group) = match field {
            // prost takes a map entry of any wire type, of the length that follows the key.
            Some(field) if field.kind == FieldKind::Map => match reader.delimited() {
                Some(end) => (Some(field.message), end, 0),
                None => return Ok(depth),
            },
            Some(field) if field.kind == FieldKind::Message => {
                // prost fails a message of another wire type.
                if wire_type != 2 {
                    return Ok(depth);
                }
                match reader.delimited() {
                    Some(end) => (Some(field.message), end, 0),
                    None => return Ok(depth),
                }
            }
            Some(field) => {
                if wire_type != 3 {
                    return Ok(depth);
                }
                (Some(field.message), usize::MAX, number)
            }
            // A field that the type does not know, or that holds no message.
            None if wire_type == 3 => (None, usize::MAX, number),
            None => {
                if reader.skip(wire_type).is_none() {
                    return Ok(depth);
                }
                continue;
            }
        };
        let at = frames.len() as u32 + 1;
        if at > limit {
            limit = probed_limit(limit);
            if at > limit {
                return Err(TooDeep {
                    path: path_to(types, &frames, &top, field),
                    kept: false,
                });
            }
        }
        let kept = top.kept && field.is_some();
        if kept && at > MAX_KEPT_DEPTH {
            return Err(TooDeep {
                path: Vec::new(),
                kept: true,
            });
        }
        depth.levels = depth.levels.max(at);
        if kept {
            depth.kept = depth.kept.max(at);
        }
        // prost names the fields of a message, but not of a map entry, and of a group of a known
        // type, but not of an unknown one.
        let named = match (group, field) {
            (0, field) => field.is_none_or(|field| field.kind != FieldKind::Map),
            _ => message.is_some(),
        };
        frames.push(top);
        top = Frame {
            message,
            fields: message.map_or(&[], |message| types[message as usize].fields),
            end,
            group,
            via: field,
            kept,
            named,
        };
    }
}

/// The fields of prost's error of the level past the limit below `top`, of `field` of it, of
/// `frames`, the levels above `top`: of each level whose fields prost names, the field that leads
/// to the next level, if the type of the level knows it, innermost first.
fn path_to(
    types: &[MessageType],
    frames: &[Frame],
    top: &Frame,
    field: Option<&'static Field>,
) -> Vec<(&'static str, &'static str)> {
    let mut path = Vec::new();
    let mut via = field;
    for level in std::iter::once(top).chain(frames.iter().rev()) {
        if let (true, Some(message), Some(field)) = (level.named, level.message, via) {
            path.push((types[message as usize].ident, field.ident));
        }
        via = level.via;
    }
    path
}

/// The [`Depth`] of `data`, the bytes of a message of the type `M`, within the limit of the
/// responses ([`depth_limit`]), or prost's error of the limit. A type that the descriptor set does
/// not have is of no depth. Not a stable API.
#[doc(hidden)]
pub fn response_depth<M: prost::Name>(data: &[u8]) -> Result<Depth, TooDeep> {
    response_depth_of(&M::full_name(), data)
}

/// [`response_depth`] of the type of the full name `name`, such as
/// `spark.connect.AnalyzePlanResponse`. Not a stable API.
#[doc(hidden)]
pub fn response_depth_of(name: &str, data: &[u8]) -> Result<Depth, TooDeep> {
    match message_type_index(name) {
        Some(message) => scan(message, data, depth_limit()),
        None => Ok(Depth::default()),
    }
}

/// The stack that prost's decode of a message takes for each of its levels: prost recurses once
/// per level (of a message, a map entry or a group, also of a field that the type does not know).
/// Measured on x86-64 Linux, a level of an `Expression` takes 777 bytes in a release build, of a
/// `Literal` 602 and of a `Relation` 456, and of a `Relation` 3.4 KiB in a debug build, which this
/// takes about twice of.
const DECODE_STACK_PER_LEVEL: usize = if cfg!(debug_assertions) {
    8 << 10
} else {
    2 << 10
};

/// The stack, beyond that of its levels, that a decode takes.
const DECODE_STACK_BASE: usize = 64 << 10;

/// Run `decode`, the decode of a message of `depth`, on a stack that holds its levels: the stack of
/// the thread when it has room for them, as it has of any response within prost's limit, else a
/// segment of its own.
pub(crate) fn on_stack_of<R>(depth: Depth, decode: impl FnOnce() -> R) -> R {
    let stack = DECODE_STACK_BASE + depth.levels as usize * DECODE_STACK_PER_LEVEL;
    stacker::maybe_grow(stack, stack, decode)
}

/// The message types of the details of a failed call that the client reads, of the
/// `grpc-status-details-bin` of its status: the `google.rpc.Status`, whose `details` (3) hold a
/// `google.protobuf.Any` each, of an `ErrorInfo`, whose `metadata` (3) is a map, or of a
/// `RetryInfo`, whose `retry_delay` (1) is a `google.protobuf.Duration`.
static STATUS_TYPES: &[MessageType] = &[
    MessageType {
        name: "google.protobuf.Any",
        ident: "Any",
        fields: &[],
    },
    MessageType {
        name: "google.protobuf.Duration",
        ident: "Duration",
        fields: &[],
    },
    MessageType {
        name: "google.rpc.ErrorInfo",
        ident: "ErrorInfo",
        fields: &[Field {
            number: 3,
            kind: FieldKind::Map,
            message: 3,
            ident: "metadata",
        }],
    },
    MessageType {
        name: "google.rpc.ErrorInfo.MetadataEntry",
        ident: "MetadataEntry",
        fields: &[],
    },
    MessageType {
        name: "google.rpc.RetryInfo",
        ident: "RetryInfo",
        fields: &[Field {
            number: 1,
            kind: FieldKind::Message,
            message: 1,
            ident: "retry_delay",
        }],
    },
    MessageType {
        name: "google.rpc.Status",
        ident: "Status",
        fields: &[Field {
            number: 3,
            kind: FieldKind::Message,
            message: 0,
            ident: "details",
        }],
    },
];

/// `M` of `data`, the bytes of a message of the details of a failed call of the full name `name`
/// (`google.rpc.Status`, `google.rpc.ErrorInfo` or `google.rpc.RetryInfo`, of the fields of
/// [`STATUS_TYPES`], which `M` must have), as prost decodes it, within the limit of the responses
/// and on a stack that holds its levels: none of a message that nests past the limit, as of one
/// that prost fails to decode, or of another name. Not a stable API.
#[doc(hidden)]
pub fn decode_status_detail<M: prost::Message + Default>(name: &str, data: &[u8]) -> Option<M> {
    let message = STATUS_TYPES
        .binary_search_by(|message| message.name.cmp(name))
        .ok()?;
    let depth = scan_in(STATUS_TYPES, message as u32, data, depth_limit()).ok()?;
    on_stack_of(depth, || M::decode(data)).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{expression, DataType, ExecutePlanResponse, MlParams, Relation};
    use prost::Message;

    fn varint(mut value: u64) -> Vec<u8> {
        let mut out = Vec::new();
        while value >= 0x80 {
            out.push((value as u8) | 0x80);
            value >>= 7;
        }
        out.push(value as u8);
        out
    }

    fn key(number: u32, wire_type: u8) -> Vec<u8> {
        varint((u64::from(number) << 3) | u64::from(wire_type))
    }

    fn delimited(number: u32, value: &[u8]) -> Vec<u8> {
        [key(number, 2), varint(value.len() as u64), value.to_vec()].concat()
    }

    /// `levels` messages, each the field of the number `numbers[level % numbers.len()]` of the one
    /// above, around `inner`, the bytes of the innermost: built without recursion, of the sizes of
    /// the levels from the inside out.
    fn nested(levels: usize, numbers: &[u32], inner: &[u8]) -> Vec<u8> {
        let number = |level: usize| numbers[level % numbers.len()];
        let mut sizes = vec![0; levels];
        let mut size = inner.len();
        for level in (0..levels).rev() {
            sizes[level] = size;
            size += key(number(level), 2).len() + varint(size as u64).len();
        }
        let mut out = Vec::with_capacity(size);
        for (level, &size) in sizes.iter().enumerate() {
            out.extend(key(number(level), 2));
            out.extend(varint(size as u64));
        }
        out.extend_from_slice(inner);
        out
    }

    /// Groups of the field `number`, nested `levels` deep.
    fn groups(levels: usize, number: u32) -> Vec<u8> {
        [key(number, 3).repeat(levels), key(number, 4).repeat(levels)].concat()
    }

    /// The bytes of a DataType of `arrays` nested arrays of `leaf`, the bytes of the innermost
    /// DataType: DataType.array (20) of DataType.Array.element_type (1), two levels an array.
    fn arrays_of(arrays: usize, leaf: &[u8]) -> Vec<u8> {
        nested(2 * arrays, &[20, 1], leaf)
    }

    /// A DataType of a long (DataType.long, 7, of an empty DataType.Long), of `arrays` nested
    /// arrays: its deepest message, the Long, is at level `2 * arrays + 1`.
    fn arrays(arrays: usize) -> Vec<u8> {
        arrays_of(arrays, &delimited(7, &[]))
    }

    /// An ExecutePlanResponse of a schema (ExecutePlanResponse.schema, 7) of `data_type`.
    fn response_of(data_type: &[u8]) -> Vec<u8> {
        delimited(7, data_type)
    }

    fn index_of<M: prost::Name>() -> u32 {
        message_type_index(&M::full_name()).expect("a type of the descriptor set")
    }

    fn scan_of<M: prost::Name>(data: &[u8], limit: u32) -> Result<Depth, TooDeep> {
        scan(index_of::<M>(), data, limit)
    }

    /// Whether prost of this build has its limit of 100 levels: it has none of the
    /// `no-recursion-limit` feature, which `cargo test` of the crates of the workspace that enable
    /// it (the Python binding, and the tests of `apache-spark-connect`) unifies.
    fn prost_has_limit() -> bool {
        DataType::decode(arrays(60).as_slice()).is_err()
    }

    #[test]
    fn the_boundary_is_100_levels_below_the_message() {
        for (root, bytes, levels) in [
            (index_of::<DataType>(), arrays(49), 99),
            (
                index_of::<ExecutePlanResponse>(),
                response_of(&arrays(49)),
                100,
            ),
            (index_of::<DataType>(), arrays(50), 101),
            (
                index_of::<ExecutePlanResponse>(),
                response_of(&arrays(50)),
                102,
            ),
        ] {
            let scanned = scan(root, &bytes, DEFAULT_DEPTH_LIMIT);
            if levels <= 100 {
                let depth = scanned.unwrap();
                assert_eq!((depth.levels, depth.kept), (levels, levels));
            } else {
                assert!(scanned.is_err(), "{levels} levels");
            }
            // Of a limit of as many levels, they are all within it.
            assert_eq!(scan(root, &bytes, levels).unwrap().levels, levels);
            assert!(scan(root, &bytes, levels - 1).is_err());
            if prost_has_limit() {
                let prost = match root == index_of::<DataType>() {
                    true => DataType::decode(bytes.as_slice()).map(drop),
                    false => ExecutePlanResponse::decode(bytes.as_slice()).map(drop),
                };
                assert_eq!(prost.is_ok(), levels <= 100, "{levels} levels");
            }
        }
    }

    #[test]
    fn the_error_past_the_limit_is_prosts_of_its_limit() {
        // The fields that lead to the level past the limit, innermost first, of the identifiers
        // of the generated code: the oneof `kind` of a DataType, the boxed `element_type` of an
        // array, the `schema` of the response.
        let bytes = response_of(&arrays(50));
        let error = scan_of::<ExecutePlanResponse>(&bytes, 100).unwrap_err();
        let expected = format!(
            "failed to decode Protobuf message: {}ExecutePlanResponse.schema: recursion limit \
             reached",
            "Array.element_type: DataType.kind: ".repeat(50)
        );
        assert_eq!(error.to_string(), expected);
        // A map field and its entry: MlParams.params -> entry -> Literal -> Literal.Array ->
        // Literal..., of no entry of the entry, which has no struct. The innermost Literal of 49
        // arrays is at level 100, and its Literal.Array of a 50th at 101.
        let literals = |arrays: usize| {
            let literal = nested(2 * arrays, &[22, 2], &[key(7, 0), varint(1)].concat());
            delimited(1, &[delimited(1, b"k"), delimited(2, &literal)].concat())
        };
        assert_eq!(scan_of::<MlParams>(&literals(49), 100).unwrap().levels, 100);
        let error = scan_of::<MlParams>(&literals(50), 100).unwrap_err();
        let expected = format!(
            "failed to decode Protobuf message: Literal.literal_type: {}MlParams.params: \
             recursion limit reached",
            "Array.elements: Literal.literal_type: ".repeat(49)
        );
        assert_eq!(error.to_string(), expected);
        // A map entry of strings, of a data source of a read of nested projects:
        // Relation.project (3) -> Project.input (1) -> ... Relation.read (2) -> Read.data_source
        // (2) -> DataSource.options (3), an entry of a level of its own.
        let options = |projects: usize| {
            let entry = delimited(3, &[delimited(1, b"k"), delimited(2, b"v")].concat());
            nested(2 * projects, &[3, 1], &delimited(2, &delimited(2, &entry)))
        };
        // The entry of 48 projects at level 99, of 49 at 101.
        assert_eq!(scan_of::<Relation>(&options(48), 100).unwrap().levels, 99);
        let error = scan_of::<Relation>(&options(49), 100).unwrap_err();
        let expected = format!(
            "failed to decode Protobuf message: DataSource.options: Read.read_type: \
             Relation.rel_type: {}recursion limit reached",
            "Project.input: Relation.rel_type: ".repeat(49)
        );
        assert_eq!(error.to_string(), expected);
        if prost_has_limit() {
            for (prost, ours) in [
                (
                    ExecutePlanResponse::decode(bytes.as_slice()).unwrap_err(),
                    scan_of::<ExecutePlanResponse>(&bytes, 100).unwrap_err(),
                ),
                (
                    MlParams::decode(literals(50).as_slice()).unwrap_err(),
                    scan_of::<MlParams>(&literals(50), 100).unwrap_err(),
                ),
                (
                    Relation::decode(options(49).as_slice()).unwrap_err(),
                    scan_of::<Relation>(&options(49), 100).unwrap_err(),
                ),
            ] {
                assert_eq!(prost.to_string(), ours.to_string());
            }
            assert!(MlParams::decode(literals(49).as_slice()).is_ok());
            assert!(Relation::decode(options(48).as_slice()).is_ok());
        }
    }

    #[test]
    fn repeated_and_packed_fields_are_no_levels_of_their_own() {
        // A struct of many fields is of the levels of one: Struct.fields (1) is repeated.
        // DataType.struct (21) -> Struct.fields (1) -> StructField.data_type (2) -> DataType of
        // an array of a long: the Long at level 6.
        let field = [delimited(1, b"a"), delimited(2, &arrays(1))].concat();
        let wide = delimited(21, &delimited(1, &field).repeat(1000));
        assert_eq!(scan_of::<DataType>(&wide, 100).unwrap().levels, 6);
        // The packed values of Literal.specialized_array (25) -> SpecializedArray.ints (2) ->
        // Ints.values (1) are skipped as bytes, and so are the same values unpacked.
        let values: Vec<u8> = (0..10_000u64).flat_map(varint).collect();
        let packed = delimited(25, &delimited(2, &delimited(1, &values)));
        assert_eq!(
            scan_of::<expression::Literal>(&packed, 100).unwrap().levels,
            2
        );
        let unpacked: Vec<u8> = (0..100u64)
            .flat_map(|value| [key(1, 0), varint(value)].concat())
            .collect();
        let unpacked = delimited(25, &delimited(2, &unpacked));
        assert_eq!(
            scan_of::<expression::Literal>(&unpacked, 100)
                .unwrap()
                .levels,
            2
        );
        if prost_has_limit() {
            assert!(DataType::decode(wide.as_slice()).is_ok());
            let literal = expression::Literal::decode(packed.as_slice()).unwrap();
            assert!(literal.literal_type.is_some());
            assert!(expression::Literal::decode(unpacked.as_slice()).is_ok());
        }
    }

    #[test]
    fn unknown_fields_are_of_no_level_but_their_groups_are() {
        // The fields that the type does not know, of every wire type but groups, are skipped, also
        // a length-delimited one that holds any number of groups.
        let unknown = [
            [key(900, 0), varint(u64::MAX)].concat(),
            [key(901, 1), vec![0; 8]].concat(),
            [key(905, 5), vec![0; 4]].concat(),
            delimited(904, &groups(10_000, 1)),
        ]
        .concat();
        let bytes = [response_of(&arrays(49)), unknown.clone()].concat();
        let depth = scan_of::<ExecutePlanResponse>(&bytes, 100).unwrap();
        assert_eq!((depth.levels, depth.kept), (100, 100));
        // Unknown groups are levels, of messages that the decoded message does not keep: 100
        // within the limit, and 101 past it, of no field that prost names.
        let depth = scan_of::<ExecutePlanResponse>(&groups(100, 950), 100).unwrap();
        assert_eq!((depth.levels, depth.kept), (100, 0));
        let error = scan_of::<ExecutePlanResponse>(&groups(101, 950), 100).unwrap_err();
        assert_eq!(
            error.to_string(),
            "failed to decode Protobuf message: recursion limit reached"
        );
        // Groups below a known message: prost's error names the known fields above them.
        let bytes = response_of(&groups(100, 800));
        let error = scan_of::<ExecutePlanResponse>(&bytes, 100).unwrap_err();
        assert_eq!(
            error.to_string(),
            "failed to decode Protobuf message: ExecutePlanResponse.schema: recursion limit reached"
        );
        // A group of a known field that holds no message (of a scalar, of another wire type) is
        // read as an unknown group: prost fails it, and the scan does not stop there.
        let bytes = [key(15, 3), key(15, 4)].concat();
        assert_eq!(
            scan_of::<ExecutePlanResponse>(&bytes, 100).unwrap().levels,
            1
        );
        // The unknown fields of a message at the limit are taken, as upb takes them; prost of its
        // limit fails them, as it checks its limit for each field that it skips.
        let leaf = delimited(7, &[key(15, 0), varint(1)].concat());
        let bytes = response_of(&arrays_of(49, &leaf));
        assert_eq!(
            scan_of::<ExecutePlanResponse>(&bytes, 100).unwrap().levels,
            100
        );
        if prost_has_limit() {
            for groups in [
                groups(100, 950),
                groups(101, 950),
                response_of(&groups(100, 800)),
            ] {
                let prost = ExecutePlanResponse::decode(groups.as_slice()).map(drop);
                let ours = scan_of::<ExecutePlanResponse>(&groups, 100).map(drop);
                assert_eq!(
                    prost.map_err(|error| error.to_string()),
                    ours.map_err(|error| error.to_string())
                );
            }
            assert!(ExecutePlanResponse::decode(
                [response_of(&arrays(49)), unknown].concat().as_slice()
            )
            .is_ok());
            assert!(ExecutePlanResponse::decode(bytes.as_slice()).is_err());
        }
    }

    /// A message type of fields of groups (1, of itself), of messages (2, of itself) and of a map
    /// (3) of entries whose values (2) are of the type.
    static NODES: &[MessageType] = &[
        MessageType {
            name: "t.Node",
            ident: "Node",
            fields: &[
                Field {
                    number: 1,
                    kind: FieldKind::Group,
                    message: 0,
                    ident: "group",
                },
                Field {
                    number: 2,
                    kind: FieldKind::Message,
                    message: 0,
                    ident: "child",
                },
                Field {
                    number: 3,
                    kind: FieldKind::Map,
                    message: 1,
                    ident: "children",
                },
            ],
        },
        MessageType {
            name: "t.Node.ChildrenEntry",
            ident: "ChildrenEntry",
            fields: &[Field {
                number: 2,
                kind: FieldKind::Message,
                message: 0,
                ident: "value",
            }],
        },
    ];

    #[test]
    fn groups_and_map_entries_of_a_known_type_are_levels() {
        assert_eq!(scan_in(NODES, 0, &groups(100, 1), 100).unwrap().levels, 100);
        let error = scan_in(NODES, 0, &groups(101, 1), 100).unwrap_err();
        assert_eq!(
            error.to_string(),
            format!(
                "failed to decode Protobuf message: {}recursion limit reached",
                "Node.group: ".repeat(101)
            )
        );
        // A message field of a group, and a group field of a message.
        let mixed = nested(99, &[2], &groups(1, 1));
        let depth = scan_in(NODES, 0, &mixed, 100).unwrap();
        assert_eq!((depth.levels, depth.kept), (100, 100));
        // A group ends at its own end tag: of another number, prost fails, and so the scan stops.
        let bytes = [key(1, 3), key(2, 4), groups(200, 1)].concat();
        assert_eq!(scan_in(NODES, 0, &bytes, 100).unwrap().levels, 1);
        // A group field of another wire type fails prost, which the scan leaves to it.
        let bytes = [delimited(1, &[]), groups(200, 1)].concat();
        assert_eq!(scan_in(NODES, 0, &bytes, 100).unwrap().levels, 0);
        // Map entries, each a level, of a Node each: 50 put the innermost Node at level 100,
        // and 51 their last entry at 101.
        let entries = |count: usize| {
            let mut bytes = Vec::new();
            for _ in 0..count {
                bytes = delimited(3, &[delimited(1, b"k"), delimited(2, &bytes)].concat());
            }
            bytes
        };
        assert_eq!(scan_in(NODES, 0, &entries(50), 100).unwrap().levels, 100);
        let error = scan_in(NODES, 0, &entries(51), 100).unwrap_err();
        assert_eq!(
            error.to_string(),
            format!(
                "failed to decode Protobuf message: {}recursion limit reached",
                "Node.children: ".repeat(51)
            )
        );
        // prost takes a map entry of any wire type, of the length that follows its key: of a
        // varint, the entry here of an empty Node.
        let bytes = [key(3, 0), varint(2), key(2, 2), varint(0)].concat();
        assert_eq!(scan_in(NODES, 0, &bytes, 100).unwrap().levels, 2);
    }

    #[test]
    fn bytes_that_prost_fails_to_decode_end_the_scan() {
        let deep = groups(1000, 900);
        for bytes in [
            // A length past the end of the data.
            [key(20, 2), varint(10)].concat(),
            // A varint cut short, one of eleven bytes, and one of a tenth byte above 1.
            vec![0x80],
            [vec![0x80; 10], vec![0x01]].concat(),
            [vec![0xff; 9], vec![0x02]].concat(),
            // A key of a wire type of 6 or 7, of the field number 0, and past 32 bits.
            key(1, 6),
            key(1, 7),
            key(0, 0),
            varint(1 << 35),
            // A message field of another wire type, also of a group, which upb would skip as an
            // unknown group of its 1000 levels.
            [key(20, 3), deep.clone()].concat(),
            [key(20, 0), varint(1), deep.clone()].concat(),
            [key(20, 5), vec![0; 4], deep.clone()].concat(),
            // An end group of no group.
            [key(20, 4), deep.clone()].concat(),
            [key(900, 4), deep.clone()].concat(),
            // A message whose last field runs past its end, before deep groups.
            [
                delimited(20, &[key(3, 0), vec![0x80]].concat()),
                vec![0x01],
                deep.clone(),
            ]
            .concat(),
            // A fixed value cut short.
            [key(900, 1), vec![0; 7]].concat(),
        ] {
            assert!(
                scan_of::<DataType>(&bytes, 100).is_ok(),
                "{bytes:?} is left to prost"
            );
            assert!(DataType::decode(bytes.as_slice()).is_err(), "{bytes:?}");
        }
        // A message that ends past the message that holds it is read to its end, as prost reads it
        // within the data before it fails the message that holds it, so its levels count: the
        // DataType.Array (20) of 2 bytes, of an element_type (1) of the 2000 bytes that follow it,
        // of 500 groups.
        let groups = groups(500, 900);
        let bytes = [
            key(20, 2),
            varint(1 + varint(groups.len() as u64).len() as u64),
            key(1, 2),
            varint(groups.len() as u64),
            groups,
        ]
        .concat();
        let error = scan_of::<DataType>(&bytes, 100).unwrap_err();
        assert_eq!(
            error.to_string(),
            "failed to decode Protobuf message: Array.element_type: DataType.kind: recursion \
             limit reached"
        );
        assert!(DataType::decode(bytes.as_slice()).is_err());
        if prost_has_limit() {
            let prost = DataType::decode(bytes.as_slice()).unwrap_err();
            assert_eq!(prost.to_string(), error.to_string());
        }
    }

    #[test]
    fn a_million_levels_do_not_overflow_the_stack() {
        // Of unknown groups and of known messages, within a limit of them all and past the
        // default one, on a thread of a small stack: the scan does not recurse.
        let unknown = groups(1_000_000, 900);
        let known = arrays(500_000);
        std::thread::Builder::new()
            .stack_size(128 << 10)
            .spawn(move || {
                let response = index_of::<ExecutePlanResponse>();
                let data_type = index_of::<DataType>();
                assert!(scan(response, &unknown, 100).is_err());
                let depth = scan(response, &unknown, u32::MAX).unwrap();
                assert_eq!((depth.levels, depth.kept), (1_000_000, 0));
                // The path of the levels 0 to 100, to the level past the limit.
                let error = scan(data_type, &known, 100).unwrap_err();
                assert_eq!((error.path.len(), error.is_kept()), (101, false));
                // Within any limit, kept messages may nest at most MAX_KEPT_DEPTH levels.
                let error = scan(data_type, &known, u32::MAX).unwrap_err();
                assert!(error.is_kept() && error.path.is_empty());
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn kept_messages_nest_at_most_max_kept_depth_levels_within_any_limit() {
        // The Long of 511 arrays of the schema of a response is at level 1024; of 512, the
        // DataType of the 512th array is at 1025.
        let at_bound = response_of(&arrays(511));
        let depth = scan_of::<ExecutePlanResponse>(&at_bound, 65535).unwrap();
        assert_eq!((depth.levels, depth.kept), (MAX_KEPT_DEPTH, MAX_KEPT_DEPTH));
        let past = response_of(&arrays(512));
        let error = scan_of::<ExecutePlanResponse>(&past, 65535).unwrap_err();
        assert!(error.is_kept());
        assert_eq!(
            error.to_string(),
            "maximum recursion depth exceeded while decoding a response; the messages of a \
             response may be nested at most 1024 levels deep"
        );
        // Unknown groups, which the decoded message does not keep, nest to the limit, and past it
        // are of the limit's error.
        let within = scan_of::<ExecutePlanResponse>(&groups(65535, 900), 65535).unwrap();
        assert_eq!((within.levels, within.kept), (65535, 0));
        let past_limit = scan_of::<ExecutePlanResponse>(&groups(65536, 900), 65535);
        assert!(!past_limit.unwrap_err().is_kept());
        // Within the default limit, the limit fails first.
        assert!(!scan_of::<ExecutePlanResponse>(&past, 100)
            .unwrap_err()
            .is_kept());
    }

    #[test]
    fn random_bytes_are_taken_wherever_prost_takes_them() {
        // DataTypes of a random spine of 85 to 115 levels, of arrays and structs, with unknown
        // groups and fields of every wire type beside it, half of them mutated
        // at random: the scan takes every one that prost takes within its limit, and prost fails
        // every one that the scan does not take, where prost has its limit. Of a fixed seed.
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut random = move |bound: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % bound
        };
        let check = prost_has_limit();
        let (mut taken, mut too_deep, mut same_error) = (0, 0, 0);
        for _ in 0..3000 {
            let target = 85 + random(31) as usize;
            // The bytes of a DataType, from the innermost out, and the levels below it.
            let mut bytes = delimited(7, &[]);
            let mut levels = 1;
            while levels < target {
                bytes = match random(10) {
                    0..=4 => {
                        levels += 2;
                        delimited(20, &delimited(1, &bytes))
                    }
                    5 | 6 => {
                        levels += 3;
                        let field = [delimited(1, b"a"), delimited(2, &bytes)].concat();
                        delimited(21, &delimited(1, &field))
                    }
                    7 | 8 => {
                        // Beside it, an unknown group of a few levels.
                        let group = groups(1 + random(4) as usize, 910);
                        [bytes, group].concat()
                    }
                    _ => {
                        let noise = match random(4) {
                            0 => [key(912, 0), varint(random(1 << 40))].concat(),
                            1 => [key(913, 1), vec![7; 8]].concat(),
                            2 => [key(914, 5), vec![7; 4]].concat(),
                            _ => delimited(915, b"noise"),
                        };
                        [noise, bytes].concat()
                    }
                };
            }
            if random(2) == 0 {
                let at = random(bytes.len() as u64) as usize;
                match random(3) {
                    0 => bytes[at] = random(256) as u8,
                    1 => bytes.truncate(at),
                    _ => bytes.insert(at, random(256) as u8),
                }
            }
            let ours = scan_of::<DataType>(&bytes, 100);
            match &ours {
                Ok(_) => taken += 1,
                Err(_) => too_deep += 1,
            }
            if check {
                let prost = DataType::decode(bytes.as_slice());
                if prost.is_ok() {
                    assert!(ours.is_ok(), "{bytes:?}");
                }
                if let Err(error) = &ours {
                    let prost = prost.expect_err("prost fails what the scan does not take");
                    same_error += usize::from(prost.to_string() == error.to_string());
                }
            }
        }
        eprintln!("taken {taken}, too deep {too_deep}, of prost's error {same_error}");
        assert!(taken > 500 && too_deep > 500, "{taken} {too_deep}");
        if check {
            // Most of the errors are prost's own, but where prost fails first, of a value that the
            // scan skips, as of a mutated string, or of an unknown field at the limit.
            assert!(same_error * 10 > too_deep * 8, "{same_error} of {too_deep}");
        }
    }

    #[test]
    fn a_probe_that_raises_the_limit_lets_the_scan_read_on() {
        thread_local! {
            static LIMIT_OF_PROBE: std::cell::Cell<Option<u32>> = const {
                std::cell::Cell::new(None)
            };
        }
        // The probe of the process, set once: of the threads of this test only.
        set_depth_probe(|| LIMIT_OF_PROBE.with(std::cell::Cell::get));
        let bytes = arrays(60);
        assert!(scan_of::<DataType>(&bytes, 100).is_err());
        // A probe of no higher limit leaves it.
        LIMIT_OF_PROBE.with(|limit| limit.set(Some(100)));
        assert!(scan_of::<DataType>(&bytes, 100).is_err());
        // A higher limit is the limit of the scan, and of the process.
        let before = depth_limit();
        LIMIT_OF_PROBE.with(|limit| limit.set(Some(200)));
        assert_eq!(scan_of::<DataType>(&bytes, 100).unwrap().levels, 121);
        assert_eq!(depth_limit(), 200);
        set_depth_limit(before);
        LIMIT_OF_PROBE.with(|limit| limit.set(None));
    }

    #[test]
    fn the_limit_is_of_the_responses_of_the_service() {
        // The responses of the service, which the client decodes within the limit, are types of
        // the table; the requests, which the generated server decodes with the same codec, are not
        // responses, and neither are the messages that they hold.
        assert!(!RESPONSE_TYPES.is_empty());
        for name in RESPONSE_TYPES {
            assert!(response_type_index(name).is_some(), "{name}");
            assert!(name.ends_with("Response"), "{name}");
        }
        for name in [
            "spark.connect.ExecutePlanRequest",
            "spark.connect.AnalyzePlanRequest",
            "spark.connect.Relation",
        ] {
            assert!(message_type_index(name).is_some(), "{name}");
            assert!(response_type_index(name).is_none(), "{name}");
        }
    }
}
