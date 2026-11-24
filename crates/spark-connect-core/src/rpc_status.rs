// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! The details of the status of a failed call, its `grpc-status-details-bin`, as PySpark's client
//! reads them, in the dialect of grpcio: the `google.rpc.Status` that grpc_status's
//! `rpc_status.from_call` parses (`status_pb2.Status.FromString`), and the `google.rpc.ErrorInfo`
//! and `google.rpc.RetryInfo` of its details, which `Any.Unpack` parses, each with upb's parser,
//! the protobuf of PySpark's client (see `crate::error::from_call_error`).
//!
//! upb reads a field of a number of the type but of another wire type as a field that the type
//! does not know, which it skips; a varint of at most ten bytes, of any tenth byte; a tag and a
//! length of at most five bytes, of a tag of at most 32 bits and a length below 2^31; and a string
//! field, of proto3 in these types, only of valid UTF-8, as it reads the field. A message and a
//! map entry are a level below the message that holds them, and so is a group of a field that the
//! type does not know, whose fields it skips, but for which a field of the number 0 is not
//! corrupt, as it is in a message. A group past the limit of upb's parser
//! (`spark_connect_proto::depth_limit`, 100, or 65535 once `SetAllowOversizeProtos(True)` is
//! called) is corrupt, as is a message past it ([`Corrupt::TooDeep`]), which no message of these
//! types reaches. A map entry of a field that the entry type does not know, or of a key or a
//! value of another wire type, is not set in the map. upb reads a value, a tag or a length that
//! runs past the end of a message within the details of the bytes that follow it, and of zeros past
//! the end of the details, and fails the parse once the value is read; a string, past the end of
//! its message too, but within the details, whose UTF-8 it checks first. The first failure of a
//! parse fails it, of protobuf's `DecodeError` of its [`Corrupt`].

use std::collections::HashMap;

/// Why upb's parser fails a message (`upb_DecodeStatus`), which the text of protobuf's
/// `DecodeError` of the parse gives ([`decode_error_text`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Corrupt {
    /// A wire format that is corrupt (`kUpb_DecodeStatus_Malformed`), also of a group past the
    /// limit of the levels.
    Malformed,
    /// A string field that is not UTF-8 (`kUpb_DecodeStatus_BadUtf8`).
    BadUtf8,
    /// A message past the limit of the levels (`kUpb_DecodeStatus_MaxDepthExceeded`), which no
    /// message of these types reaches within a limit of upb's: they hold messages one level deep.
    TooDeep,
}

impl Corrupt {
    /// upb's text of the failure (`upb_DecodeStatus_String`).
    fn reason(self) -> &'static str {
        match self {
            Corrupt::Malformed => "Wire format was corrupt",
            Corrupt::BadUtf8 => "String field had bad UTF-8",
            Corrupt::TooDeep => "Exceeded upb_DecodeOptions_MaxDepth",
        }
    }
}

/// The text of protobuf's `DecodeError` of a parse of a message of the type of the full name
/// `name` that fails of `corrupt`, as upb gives it: `Error parsing message with type
/// 'google.rpc.Status': Wire format was corrupt`.
pub(crate) fn decode_error_text(name: &str, corrupt: Corrupt) -> String {
    format!(
        "Error parsing message with type '{name}': {}",
        corrupt.reason()
    )
}

/// The full name of `google.rpc.Status`.
pub(crate) const STATUS: &str = "google.rpc.Status";
/// The full name of `google.rpc.ErrorInfo`.
pub(crate) const ERROR_INFO: &str = "google.rpc.ErrorInfo";
/// The full name of `google.rpc.RetryInfo`.
pub(crate) const RETRY_INFO: &str = "google.rpc.RetryInfo";

/// A `google.rpc.Status`, as upb reads it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Status {
    pub(crate) code: i32,
    pub(crate) message: String,
    pub(crate) details: Vec<Any>,
}

/// A `google.protobuf.Any` of the details of a [`Status`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Any {
    pub(crate) type_url: String,
    pub(crate) value: Vec<u8>,
}

impl Any {
    /// Whether the message is of the type of the full name `name`, as protobuf's `Any.Is` tells
    /// it: of a type URL of a `/`, whose part after the last `/` is the name.
    pub(crate) fn is(&self, name: &str) -> bool {
        self.type_url.contains('/') && self.type_url.rsplit('/').next() == Some(name)
    }
}

/// A `google.rpc.ErrorInfo`, as upb reads it: its metadata of the entries that upb sets in the map,
/// of the last value of a key.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ErrorInfo {
    pub(crate) reason: String,
    pub(crate) domain: String,
    pub(crate) metadata: HashMap<String, String>,
}

/// A `google.rpc.RetryInfo`, as upb reads it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct RetryInfo {
    pub(crate) retry_delay: Option<Duration>,
}

/// A `google.protobuf.Duration` of a [`RetryInfo`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Duration {
    pub(crate) seconds: i64,
    pub(crate) nanos: i32,
}

/// The details of the status of a failure as upb reads them, read once for the readers of the
/// failure: the `google.rpc.Status`, or why it fails, and its first `ErrorInfo` and its first
/// `RetryInfo` (of the details that `Any.Is` tells of the type), or why each fails, which are read
/// as they are first asked for.
pub(crate) struct Details {
    status: Result<Status, Corrupt>,
    error_info: std::sync::OnceLock<Option<Result<ErrorInfo, Corrupt>>>,
    retry_info: std::sync::OnceLock<Option<Result<RetryInfo, Corrupt>>>,
}

impl Details {
    /// The details of `data`, the `grpc-status-details-bin` of a status.
    pub(crate) fn of(data: &[u8]) -> Self {
        Details {
            status: status(data),
            error_info: std::sync::OnceLock::new(),
            retry_info: std::sync::OnceLock::new(),
        }
    }

    /// The `google.rpc.Status`, or why upb fails to read it.
    pub(crate) fn status(&self) -> Result<&Status, Corrupt> {
        self.status.as_ref().map_err(|corrupt| *corrupt)
    }

    /// The first `ErrorInfo` of the status, or why upb fails to read it; none of a status of none,
    /// or that upb fails to read.
    pub(crate) fn error_info(&self) -> Option<Result<&ErrorInfo, Corrupt>> {
        self.error_info
            .get_or_init(|| Some(error_info(&self.first(ERROR_INFO)?.value)))
            .as_ref()
            .map(|read| read.as_ref().map_err(|corrupt| *corrupt))
    }

    /// [`Details::error_info`], taken of the details.
    pub(crate) fn into_error_info(self) -> Option<Result<ErrorInfo, Corrupt>> {
        self.error_info();
        self.error_info.into_inner().flatten()
    }

    /// The first `RetryInfo` of the status, or why upb fails to read it; none of a status of none,
    /// or that upb fails to read.
    pub(crate) fn retry_info(&self) -> Option<Result<&RetryInfo, Corrupt>> {
        self.retry_info
            .get_or_init(|| Some(retry_info(&self.first(RETRY_INFO)?.value)))
            .as_ref()
            .map(|read| read.as_ref().map_err(|corrupt| *corrupt))
    }

    /// The first detail of the type of the full name `name` (`Any.Is`), of a status that upb reads.
    fn first(&self, name: &str) -> Option<&Any> {
        let status = self.status.as_ref().ok()?;
        status.details.iter().find(|detail| detail.is(name))
    }
}

/// The `google.rpc.Status` of `data`, as upb reads it, or why it fails.
pub(crate) fn status(data: &[u8]) -> Result<Status, Corrupt> {
    read_status(data, &mut Limit::new())
}

/// [`status`] within `limit`.
fn read_status(data: &[u8], limit: &mut Limit) -> Result<Status, Corrupt> {
    let mut reader = Reader::new(data);
    let mut status = Status::default();
    while let Some((number, wire_type)) = reader.field()? {
        match (number, wire_type) {
            (1, 0) => status.code = reader.varint()? as i32,
            (2, 2) => status.message = reader.string()?,
            (3, 2) => {
                let mut any = reader.message(1, limit)?;
                status.details.push(read_any(&mut any, limit)?);
            }
            _ => reader.skip(number, wire_type, 0, limit)?,
        }
        reader.check_end()?;
    }
    Ok(status)
}

/// The `google.protobuf.Any` of a detail of a status, of `reader`, at level 1.
fn read_any(reader: &mut Reader<'_>, limit: &mut Limit) -> Result<Any, Corrupt> {
    let mut any = Any::default();
    while let Some((number, wire_type)) = reader.field()? {
        match (number, wire_type) {
            (1, 2) => any.type_url = reader.string()?,
            (2, 2) => any.value = reader.bytes()?.to_vec(),
            _ => reader.skip(number, wire_type, 1, limit)?,
        }
        reader.check_end()?;
    }
    Ok(any)
}

/// The `google.rpc.ErrorInfo` of `data`, the value of a detail, as upb reads it, or why it fails.
pub(crate) fn error_info(data: &[u8]) -> Result<ErrorInfo, Corrupt> {
    let (mut reader, mut limit) = (Reader::new(data), Limit::new());
    let mut info = ErrorInfo::default();
    while let Some((number, wire_type)) = reader.field()? {
        match (number, wire_type) {
            (1, 2) => info.reason = reader.string()?,
            (2, 2) => info.domain = reader.string()?,
            (3, 2) => {
                let mut entry = reader.message(1, &mut limit)?;
                if let Some((key, value)) = read_entry(&mut entry, &mut limit)? {
                    info.metadata.insert(key, value);
                }
            }
            _ => reader.skip(number, wire_type, 0, &mut limit)?,
        }
        reader.check_end()?;
    }
    Ok(info)
}

/// The key and the value of an entry of the map of strings of an `ErrorInfo`, of `reader`, at
/// level 1, as upb reads one: none of an entry of a field that the entry type does not know, or
/// of a key or a value of another wire type, which upb keeps in the unknown fields of the message
/// rather than in the map.
fn read_entry(
    reader: &mut Reader<'_>,
    limit: &mut Limit,
) -> Result<Option<(String, String)>, Corrupt> {
    let (mut key, mut value, mut unknown) = (String::new(), String::new(), false);
    while let Some((number, wire_type)) = reader.field()? {
        match (number, wire_type) {
            (1, 2) => key = reader.string()?,
            (2, 2) => value = reader.string()?,
            _ => {
                reader.skip(number, wire_type, 1, limit)?;
                unknown = true;
            }
        }
        reader.check_end()?;
    }
    Ok((!unknown).then_some((key, value)))
}

/// The `google.rpc.RetryInfo` of `data`, the value of a detail, as upb reads it, or why it fails:
/// a `retry_delay` that comes again merges into the one before it.
pub(crate) fn retry_info(data: &[u8]) -> Result<RetryInfo, Corrupt> {
    let (mut reader, mut limit) = (Reader::new(data), Limit::new());
    let mut info = RetryInfo::default();
    while let Some((number, wire_type)) = reader.field()? {
        match (number, wire_type) {
            (1, 2) => {
                let mut delay = reader.message(1, &mut limit)?;
                let duration = info.retry_delay.get_or_insert_with(Duration::default);
                while let Some((number, wire_type)) = delay.field()? {
                    match (number, wire_type) {
                        (1, 0) => duration.seconds = delay.varint()? as i64,
                        (2, 0) => duration.nanos = delay.varint()? as i32,
                        _ => delay.skip(number, wire_type, 1, &mut limit)?,
                    }
                    delay.check_end()?;
                }
            }
            _ => reader.skip(number, wire_type, 0, &mut limit)?,
        }
        reader.check_end()?;
    }
    Ok(info)
}

/// The limit of the levels of a parse, of upb's parsers of the process (see
/// `spark_connect_proto::depth_limit`).
struct Limit(u32);

impl Limit {
    fn new() -> Self {
        Limit(spark_connect_proto::depth_limit())
    }

    /// Whether a level is within the limit, once the limit is told again of one past it, as a
    /// call of upb's `SetAllowOversizeProtos(True)` that the client did not see may have raised it
    /// (see `spark_connect_proto::set_depth_probe`).
    fn admits(&mut self, level: u32) -> bool {
        if level > self.0 {
            self.0 = spark_connect_proto::probed_depth_limit();
        }
        level <= self.0
    }
}

/// A reader of the bytes of a message, from `pos` to `end`, of `data`, the bytes of the parse, as
/// upb's decoder reads them (see the module): past `end`, of the bytes that follow it, and of zeros
/// past the end of the data, until the next field of the message is to be read
/// ([`Reader::check_end`]).
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

    /// The failure of a read past the end of the message, which upb tells once it read the value.
    fn check_end(&self) -> Result<(), Corrupt> {
        if self.pos > self.end {
            return Err(Corrupt::Malformed);
        }
        Ok(())
    }

    /// The byte at the position, or a zero past the end of the data.
    fn byte(&mut self) -> u8 {
        let byte = self.data.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        byte
    }

    /// The number and the wire type of the next field of the message, or none at its end. These
    /// types hold no groups of theirs, so the end of a group is corrupt, and so is a field of the
    /// number 0.
    fn field(&mut self) -> Result<Option<(u32, u8)>, Corrupt> {
        if self.done() {
            return Ok(None);
        }
        let (number, wire_type) = self.tag()?;
        if wire_type == 4 || number == 0 {
            return Err(Corrupt::Malformed);
        }
        Ok(Some((number, wire_type)))
    }

    /// The field number and the wire type of a tag, as upb reads one: a varint of at most five
    /// bytes, of a value of at most 32 bits, so that a field number is below 2^29.
    fn tag(&mut self) -> Result<(u32, u8), Corrupt> {
        let mut value = 0u64;
        for shift in (0..35).step_by(7) {
            let byte = self.byte();
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                let tag = u32::try_from(value).map_err(|_| Corrupt::Malformed)?;
                return Ok((tag >> 3, (tag & 7) as u8));
            }
        }
        Err(Corrupt::Malformed)
    }

    /// A varint, as upb reads one: of at most ten bytes, of the low 64 bits of its value.
    fn varint(&mut self) -> Result<u64, Corrupt> {
        let mut value = 0u64;
        for shift in (0..70).step_by(7) {
            let byte = self.byte();
            if shift < 64 {
                value |= u64::from(byte & 0x7f) << shift;
            }
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(Corrupt::Malformed)
    }

    /// The length of a length-delimited value, as upb reads one: a varint of at most five bytes,
    /// even of a small value, and of at most 2^31 - 1.
    fn size(&mut self) -> Result<usize, Corrupt> {
        let mut len = 0u64;
        for shift in (0..35).step_by(7) {
            let byte = self.byte();
            len |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                if len > i32::MAX as u64 {
                    return Err(Corrupt::Malformed);
                }
                return usize::try_from(len).map_err(|_| Corrupt::Malformed);
            }
        }
        Err(Corrupt::Malformed)
    }

    /// The bytes of a string or bytes field at the position, as upb reads them: past the end of
    /// the message too, but within the data.
    fn bytes(&mut self) -> Result<&'a [u8], Corrupt> {
        let len = self.size()?;
        let end = self
            .pos
            .checked_add(len)
            .filter(|&end| end <= self.data.len())
            .ok_or(Corrupt::Malformed)?;
        let bytes = &self.data[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }

    /// The value of a string field at the position, which must be UTF-8.
    fn string(&mut self) -> Result<String, Corrupt> {
        let bytes = self.bytes()?;
        std::str::from_utf8(bytes)
            .map(str::to_string)
            .map_err(|_| Corrupt::BadUtf8)
    }

    /// A reader of the message or the map entry at the position, at `level`, which must end within
    /// the message, as upb's limit of it (`PushLimit`). The position of this reader is then past
    /// it.
    fn message(&mut self, level: u32, limit: &mut Limit) -> Result<Reader<'a>, Corrupt> {
        let len = self.size()?;
        let end = self
            .pos
            .checked_add(len)
            .filter(|&end| end <= self.end)
            .ok_or(Corrupt::Malformed)?;
        if !limit.admits(level) {
            return Err(Corrupt::TooDeep);
        }
        let message = Reader {
            data: self.data,
            pos: self.pos,
            end,
        };
        self.pos = end;
        Ok(message)
    }

    /// Skip the value of a field of `number` and `wire_type` that the message at `level` does not
    /// read, as upb skips one: a varint, a fixed value, the bytes of a length, past the end of the
    /// message too, which the next read tells, or a group, a level below the message, up to its
    /// end tag, of its number, whose fields it skips of their tags, also of the groups in it, each
    /// a level below the one that holds it, without recursion.
    fn skip(
        &mut self,
        number: u32,
        wire_type: u8,
        level: u32,
        limit: &mut Limit,
    ) -> Result<(), Corrupt> {
        if wire_type != 3 {
            return self.skip_value(wire_type);
        }
        // The numbers of the groups that are open, the innermost last, and the one that opens.
        let mut open: Vec<u32> = Vec::new();
        let mut opening = Some(number);
        loop {
            if let Some(group) = opening.take() {
                let at = level.saturating_add(1 + open.len() as u32);
                if !limit.admits(at) {
                    return Err(Corrupt::Malformed);
                }
                open.push(group);
            }
            if self.done() {
                // A group that the data or the message ends in.
                return Err(Corrupt::Malformed);
            }
            let (inner, inner_type) = self.tag()?;
            match inner_type {
                3 => opening = Some(inner),
                4 if open.last() == Some(&inner) => {
                    open.pop();
                    if open.is_empty() {
                        return Ok(());
                    }
                }
                _ => self.skip_value(inner_type)?,
            }
            self.check_end()?;
        }
    }

    /// Skip a value of `wire_type` that is not a group: of 6 or 7, or of the end of a group that
    /// is not open, the data is corrupt.
    fn skip_value(&mut self, wire_type: u8) -> Result<(), Corrupt> {
        match wire_type {
            0 => {
                self.varint()?;
            }
            1 => self.pos += 8,
            2 => {
                let len = self.size()?;
                self.pos = self.pos.saturating_add(len);
            }
            5 => self.pos += 4,
            _ => return Err(Corrupt::Malformed),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// `levels` groups of the field `number` nested in each other.
    fn groups(levels: usize, number: u32) -> Vec<u8> {
        [key(number, 3).repeat(levels), key(number, 4).repeat(levels)].concat()
    }

    /// The bytes of a status of the code 13 and the message `boom`, then `extra`.
    fn boom(extra: &[u8]) -> Vec<u8> {
        [key(1, 0), varint(13), delimited(2, b"boom"), extra.to_vec()].concat()
    }

    fn of_boom(details: Vec<Any>) -> Status {
        Status {
            code: 13,
            message: "boom".to_string(),
            details,
        }
    }

    fn any(type_url: &str, value: &[u8]) -> Any {
        Any {
            type_url: type_url.to_string(),
            value: value.to_vec(),
        }
    }

    #[test]
    fn a_status_nests_its_groups_and_its_details_within_upb_s_limit() {
        // As upb's parser, of `kUpb_WireFormat_DefaultDepthLimit`: the status is at level 0, a
        // detail at 1, and a group a level below what holds it, so that 100 nested groups read
        // and 101 are corrupt, and in a detail, 99 and 100.
        assert_eq!(status(&boom(&groups(100, 5))), Ok(of_boom(Vec::new())));
        assert_eq!(status(&boom(&groups(101, 5))), Err(Corrupt::Malformed));
        assert_eq!(status(&boom(&groups(101, 1000))), Err(Corrupt::Malformed));
        let in_any = |levels| {
            boom(&delimited(
                3,
                &[delimited(1, b"x/y"), groups(levels, 5)].concat(),
            ))
        };
        assert_eq!(status(&in_any(99)), Ok(of_boom(vec![any("x/y", b"")])));
        assert_eq!(status(&in_any(100)), Err(Corrupt::Malformed));
        // Groups of fields in turn, and the field of the details as a group, which upb reads as a
        // field that the type does not know, a level of its own.
        let mut mixed = Vec::new();
        for level in 0..101 {
            mixed.extend(key(if level % 2 == 0 { 5 } else { 1000 }, 3));
        }
        for level in (0..101).rev() {
            mixed.extend(key(if level % 2 == 0 { 5 } else { 1000 }, 4));
        }
        assert_eq!(status(&boom(&mixed)), Err(Corrupt::Malformed));
        let as_group = |levels| boom(&[key(3, 3), groups(levels, 5), key(3, 4)].concat());
        assert_eq!(status(&as_group(99)), Ok(of_boom(Vec::new())));
        assert_eq!(status(&as_group(100)), Err(Corrupt::Malformed));
        // Within the limit of `SetAllowOversizeProtos(True)`, 65535, without recursion, on a thread
        // of a small stack, which a parser that recursed per level would overflow.
        let read = std::thread::Builder::new()
            .stack_size(64 << 10)
            .spawn(|| {
                let at = |levels| read_status(&boom(&groups(levels, 5)), &mut Limit(65535));
                (at(65535), at(65536))
            })
            .unwrap()
            .join()
            .unwrap();
        assert_eq!(read, (Ok(of_boom(Vec::new())), Err(Corrupt::Malformed)));
    }

    #[test]
    fn a_status_takes_a_known_field_of_another_wire_type_as_unknown() {
        // As upb, where prost fails: the code as a length, a fixed value or a group, the message
        // as a varint, the details as a varint, each skipped, and the last value of a field read.
        assert_eq!(
            status(&[delimited(1, b""), delimited(2, b"boom")].concat()),
            Ok(Status {
                message: "boom".to_string(),
                ..Status::default()
            })
        );
        for code in [
            [key(1, 5), vec![0; 4]].concat(),
            [key(1, 1), vec![0; 8]].concat(),
            [key(1, 3), key(1, 4)].concat(),
        ] {
            assert_eq!(
                status(&[code, delimited(2, b"boom")].concat())
                    .unwrap()
                    .code,
                0
            );
        }
        assert_eq!(
            status(&[delimited(1, b"zz"), boom(&[key(3, 0), varint(7)].concat())].concat()),
            Ok(of_boom(Vec::new()))
        );
        assert_eq!(
            status(&[key(1, 0), varint(13), key(2, 0), varint(5)].concat()),
            Ok(Status {
                code: 13,
                ..Status::default()
            })
        );
        let details = [
            delimited(3, &delimited(1, b"a/b")),
            delimited(
                3,
                &[
                    delimited(1, b"c/d"),
                    delimited(1, b"e/f"),
                    key(2, 0),
                    varint(1),
                ]
                .concat(),
            ),
        ]
        .concat();
        assert_eq!(
            status(&boom(&details)),
            Ok(of_boom(vec![any("a/b", b""), any("e/f", b"")]))
        );
    }

    #[test]
    fn a_status_reads_varints_tags_and_lengths_as_upb() {
        // A varint of ten bytes of any tenth byte, of the low bits of its value; a tag and a length
        // of at most five bytes; a tag of at most 32 bits.
        let ten = [
            key(1, 0),
            vec![0x8d],
            vec![0x80; 8],
            vec![0x7f],
            delimited(2, b"boom"),
        ]
        .concat();
        assert_eq!(status(&ten), Ok(of_boom(Vec::new())));
        let eleven = [
            key(1, 0),
            vec![0x8d],
            vec![0x80; 9],
            vec![0],
            delimited(2, b"boom"),
        ]
        .concat();
        assert_eq!(status(&eleven), Err(Corrupt::Malformed));
        let code = |tag: &[u8]| [tag.to_vec(), varint(13), delimited(2, b"boom")].concat();
        assert_eq!(
            status(&code(b"\x88\x80\x80\x80\x00")),
            Ok(of_boom(Vec::new()))
        );
        assert_eq!(
            status(&code(b"\x88\x80\x80\x80\x80\x00")),
            Err(Corrupt::Malformed)
        );
        assert_eq!(
            status(&code(b"\x88\x80\x80\x80\x7f")),
            Err(Corrupt::Malformed)
        );
        let message = |len: &[u8]| {
            [
                key(1, 0),
                varint(13),
                key(2, 2),
                len.to_vec(),
                b"boom".to_vec(),
            ]
            .concat()
        };
        assert_eq!(
            status(&message(b"\x84\x80\x80\x80\x00")),
            Ok(of_boom(Vec::new()))
        );
        assert_eq!(
            status(&message(b"\x84\x80\x80\x80\x80\x00")),
            Err(Corrupt::Malformed)
        );
        let unknown = [
            key(9, 2),
            b"\x84\x80\x80\x80\x80\x00boom".to_vec(),
            boom(b""),
        ]
        .concat();
        assert_eq!(status(&unknown), Err(Corrupt::Malformed));
        // A code of a negative int32 and of more than 32 bits, as upb casts it.
        let wide = [key(1, 0), varint((1 << 32) + 13), delimited(2, b"boom")].concat();
        assert_eq!(status(&wide).unwrap().code, 13);
        let negative = [key(1, 0), varint(u64::MAX), delimited(2, b"boom")].concat();
        assert_eq!(status(&negative).unwrap().code, -1);
    }

    #[test]
    fn a_status_fails_where_upb_s_wire_format_is_corrupt() {
        for corrupt in [
            b"abc".to_vec(),
            b"\x00".to_vec(),
            boom(b"\x00\x01"),
            boom(&key(5, 4)),
            boom(&key(1, 6)),
            boom(&key(5, 7)),
            boom(&[key(5, 3), key(6, 4)].concat()),
            boom(&key(5, 3).repeat(100)),
            [key(2, 2), varint(10), b"ab".to_vec()].concat(),
            [key(1, 0), vec![0x80]].concat(),
            [key(9, 5), vec![0; 3]].concat(),
            [key(9, 1), vec![0; 7]].concat(),
            [key(2, 2), varint(1 << 31), b"ab".to_vec()].concat(),
            boom(&delimited(3, &[delimited(1, b"x/y"), key(5, 4)].concat())),
            boom(&[key(3, 2), varint(1), key(9, 3), key(9, 4)].concat()),
            boom(&delimited(3, b"\x00\x01")),
        ] {
            assert_eq!(status(&corrupt), Err(Corrupt::Malformed), "{corrupt:?}");
        }
        // But a fixed value that ends the data, and what a group holds: a field of the number 0,
        // a string of any bytes, a varint of ten bytes, a tag of five bytes, an end tag of more.
        assert_eq!(
            status(&[key(9, 5), vec![0; 4]].concat()),
            Ok(Status::default())
        );
        for in_group in [
            b"\x00\x01".to_vec(),
            b"\x03\x04".to_vec(),
            delimited(2, b"\xff"),
            [key(7, 0), vec![0x80; 9], vec![0x7f]].concat(),
            b"\xb0\x80\x80\x80\x00\x01".to_vec(),
            [key(6, 1), vec![0; 8]].concat(),
        ] {
            let data = boom(&[key(5, 3), in_group.clone(), key(5, 4)].concat());
            assert_eq!(status(&data), Ok(of_boom(Vec::new())), "{in_group:?}");
        }
        assert_eq!(
            status(&boom(&[key(5, 3), b"\xac\x80\x00".to_vec()].concat())),
            Ok(of_boom(Vec::new()))
        );
        assert_eq!(
            status(&boom(
                &[key(5, 3), b"\xb0\x80\x80\x80\x70\x01".to_vec(), key(5, 4)].concat()
            )),
            Err(Corrupt::Malformed)
        );
    }

    #[test]
    fn a_status_s_strings_are_utf_8_as_upb_reads_them() {
        // Checked as the field is read, so that the first failure is that of the parse; also of a
        // string that runs past the end of its detail, before the end is told.
        let bad = [key(1, 0), varint(13), delimited(2, b"\xff\xfe")].concat();
        assert_eq!(status(&bad), Err(Corrupt::BadUtf8));
        assert_eq!(
            status(&[bad.clone(), groups(101, 5)].concat()),
            Err(Corrupt::BadUtf8)
        );
        assert_eq!(
            status(&[groups(101, 5), bad.clone()].concat()),
            Err(Corrupt::Malformed)
        );
        assert_eq!(
            status(&[delimited(2, b"\xff"), b"\xff".to_vec()].concat()),
            Err(Corrupt::BadUtf8)
        );
        assert_eq!(
            status(&[b"\x0b".to_vec(), delimited(2, b"\xff")].concat()),
            Err(Corrupt::Malformed)
        );
        assert_eq!(
            status(&[delimited(2, b"\xff"), delimited(2, b"ok")].concat()),
            Err(Corrupt::BadUtf8)
        );
        assert_eq!(
            status(&boom(&delimited(3, &delimited(1, b"\xff")))),
            Err(Corrupt::BadUtf8)
        );
        for text in [&b"\xed\xa0\x80"[..], b"\xc0\x80", b"\xf4\x90\x80\x80"] {
            assert_eq!(status(&delimited(2, text)), Err(Corrupt::BadUtf8));
        }
        assert_eq!(
            status(&delimited(2, "\u{1f600}\0".as_bytes()))
                .unwrap()
                .message,
            "\u{1f600}\0"
        );
        let past = |rest: &[u8]| {
            boom(&[key(3, 2), varint(2), key(1, 2), varint(5), rest.to_vec()].concat())
        };
        assert_eq!(
            status(&past(b"\xff\xfe\xfd\xfc\xfb")),
            Err(Corrupt::BadUtf8)
        );
        assert_eq!(status(&past(b"abcde")), Err(Corrupt::Malformed));
        assert_eq!(status(&past(b"\xff")), Err(Corrupt::Malformed));
        // The bytes of the value of a detail are not checked.
        assert_eq!(
            status(&boom(&delimited(3, &delimited(2, b"\xff")))),
            Ok(of_boom(vec![any("", b"\xff")]))
        );
    }

    #[test]
    fn an_any_is_of_a_type_as_protobuf_tells_it() {
        assert!(any("type.googleapis.com/google.rpc.ErrorInfo", b"").is(ERROR_INFO));
        assert!(any("x.com/a/google.rpc.ErrorInfo", b"").is(ERROR_INFO));
        assert!(any("/google.rpc.ErrorInfo", b"").is(ERROR_INFO));
        assert!(!any("google.rpc.ErrorInfo", b"").is(ERROR_INFO));
        assert!(!any("type.googleapis.com/google.rpc.ErrorInfo/", b"").is(ERROR_INFO));
        assert!(!any("type.googleapis.com/google.rpc.RetryInfo", b"").is(ERROR_INFO));
    }

    #[test]
    fn an_error_info_sets_the_entries_of_its_map_as_upb() {
        let entry = |key: &[u8], value: &[u8], extra: &[u8]| {
            delimited(
                3,
                &[delimited(1, key), delimited(2, value), extra.to_vec()].concat(),
            )
        };
        let info = error_info(
            &[
                delimited(1, b"REASON"),
                delimited(2, b"org.apache.spark"),
                entry(b"k", b"v1", b""),
                entry(b"k", b"v2", b""),
                entry(b"dropped", b"x", &[key(3, 0), varint(1)].concat()),
                entry(b"group", b"x", &groups(1, 4)),
                delimited(3, &[key(1, 0), varint(1), delimited(2, b"x")].concat()),
                delimited(3, &delimited(2, b"of no key")),
                delimited(
                    3,
                    &[delimited(1, b"a"), delimited(1, b"b"), delimited(2, b"v")].concat(),
                ),
                [key(3, 0), varint(1)].concat(),
                [key(3, 3), key(3, 4)].concat(),
            ]
            .concat(),
        )
        .unwrap();
        assert_eq!(
            (info.reason.as_str(), info.domain.as_str()),
            ("REASON", "org.apache.spark")
        );
        let mut metadata: Vec<_> = info.metadata.into_iter().collect();
        metadata.sort();
        let expected = [("", "of no key"), ("b", "v"), ("k", "v2")];
        assert_eq!(
            metadata,
            expected.map(|(k, v)| (k.to_string(), v.to_string()))
        );
        // Of UTF-8, of the limit of its levels, and of a wire format that is not corrupt.
        for (data, corrupt) in [
            (delimited(1, b"\xff"), Corrupt::BadUtf8),
            (entry(b"\xff", b"v", b""), Corrupt::BadUtf8),
            (entry(b"k", b"\xff", b""), Corrupt::BadUtf8),
            (groups(101, 5), Corrupt::Malformed),
            (entry(b"k", b"v", &groups(100, 5)), Corrupt::Malformed),
            (
                delimited(3, &[delimited(1, b"k"), key(1, 4)].concat()),
                Corrupt::Malformed,
            ),
            (
                delimited(3, &[delimited(1, b"k"), b"\x00\x01".to_vec()].concat()),
                Corrupt::Malformed,
            ),
            (
                [key(3, 2), b"\x80\x80\x80\x80\x80\x00".to_vec()].concat(),
                Corrupt::Malformed,
            ),
            (b"abc".to_vec(), Corrupt::Malformed),
        ] {
            assert_eq!(error_info(&data), Err(corrupt), "{data:?}");
        }
        assert_eq!(error_info(&groups(100, 5)), Ok(ErrorInfo::default()));
        assert!(error_info(&entry(b"k", b"v", &groups(99, 5)))
            .unwrap()
            .metadata
            .is_empty());
    }

    #[test]
    fn a_retry_info_merges_its_delays_as_upb() {
        let delay = |fields: &[u8]| delimited(1, fields);
        assert_eq!(
            retry_info(
                &[
                    delay(&[key(1, 0), varint(2)].concat()),
                    delay(&[key(2, 0), varint(5)].concat())
                ]
                .concat()
            ),
            Ok(RetryInfo {
                retry_delay: Some(Duration {
                    seconds: 2,
                    nanos: 5
                })
            })
        );
        assert_eq!(
            retry_info(&delay(
                &[
                    key(1, 0),
                    varint(u64::MAX - 4),
                    key(2, 0),
                    varint((1 << 32) + 5)
                ]
                .concat()
            )),
            Ok(RetryInfo {
                retry_delay: Some(Duration {
                    seconds: -5,
                    nanos: 5
                })
            })
        );
        assert_eq!(
            retry_info(&[key(1, 0), varint(5)].concat()),
            Ok(RetryInfo::default())
        );
        assert_eq!(
            retry_info(&delay(&groups(99, 5))),
            Ok(RetryInfo {
                retry_delay: Some(Duration::default())
            })
        );
        assert_eq!(retry_info(&delay(&groups(100, 5))), Err(Corrupt::Malformed));
        assert_eq!(retry_info(b"abc"), Err(Corrupt::Malformed));
    }

    #[test]
    fn the_details_read_each_message_once_as_they_are_asked_for() {
        // The first ErrorInfo and the first RetryInfo of `Any.Is`, of a status that upb reads.
        let detail = |type_url: &str, value: &[u8]| {
            delimited(
                3,
                &[delimited(1, type_url.as_bytes()), delimited(2, value)].concat(),
            )
        };
        let info = "type.googleapis.com/google.rpc.ErrorInfo";
        let retry = "x/google.rpc.RetryInfo";
        let data = boom(
            &[
                detail("google.rpc.ErrorInfo", &delimited(1, b"no slash")),
                detail(info, &delimited(1, b"first")),
                detail(info, &delimited(1, b"second")),
                detail(retry, b"abc"),
                detail(retry, b""),
            ]
            .concat(),
        );
        let details = Details::of(&data);
        assert_eq!(details.status().map(|status| status.details.len()), Ok(5));
        assert_eq!(
            details
                .error_info()
                .map(|info| info.map(|info| info.reason.clone())),
            Some(Ok("first".to_string()))
        );
        assert_eq!(details.retry_info(), Some(Err(Corrupt::Malformed)));
        let none = Details::of(&boom(b""));
        assert_eq!((none.error_info(), none.retry_info()), (None, None));
        let corrupt = Details::of(b"abc");
        assert_eq!(corrupt.status().err(), Some(Corrupt::Malformed));
        assert_eq!((corrupt.error_info(), corrupt.retry_info()), (None, None));
    }

    #[test]
    fn the_texts_of_the_failures_are_upb_s() {
        assert_eq!(
            decode_error_text(STATUS, Corrupt::Malformed),
            "Error parsing message with type 'google.rpc.Status': Wire format was corrupt"
        );
        assert_eq!(
            decode_error_text(ERROR_INFO, Corrupt::BadUtf8),
            "Error parsing message with type 'google.rpc.ErrorInfo': String field had bad UTF-8"
        );
        assert_eq!(
            decode_error_text(RETRY_INFO, Corrupt::TooDeep),
            "Error parsing message with type 'google.rpc.RetryInfo': Exceeded \
             upb_DecodeOptions_MaxDepth"
        );
    }
}
