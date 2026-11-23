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

//! The limits of the size of the metadata that a call receives, as gRPC's core enforces them on
//! the metadata of a response, its headers (the initial metadata, or the status of a response of
//! its trailers only) and its trailers.
//!
//! gRPC's core counts each entry as HPACK does (RFC 7541, section 4.1), of its name, its value and
//! 32 bytes, as they are on the wire. At its hard limit (`grpc.absolute_max_metadata_size`, 16 KiB
//! by default) the call fails, `RESOURCE_EXHAUSTED` of the text `received metadata size exceeds
//! hard limit (...)`, of the entry whose name or value alone, and 32 bytes, reaches the limit, or of
//! the size at the entry that reaches it and of the entries up to it; past its soft limit
//! (`grpc.max_metadata_size`, 8 KiB by default) the call fails at random, the more often the
//! larger the metadata is up to the hard limit, of `received metadata size exceeds soft limit
//! (...)`, of all the entries, which the call still has, as the details of an error that grpcio
//! gives the application. A binary value that gRPC's core does not decode as base64 (see
//! [`unbase64`]) fails the call `INTERNAL`, of `Error parsing '<name>' metadata: illegal base64
//! encoding`, as it decodes it as it reads it; of the `grpc-status-details-bin` of an error, the
//! call still has it, empty, as an empty one of base64 (see [`EmptyDetails`]). The entries of these
//! texts are those of gRPC's core's metadata, as its
//! encoder gives them: the headers that it knows first, in the order of its table, of their values
//! as it encodes them again, then the others, in the order that they came in, of their values as
//! they came but of the decoded bytes of a binary one (of a name ending in `-bin`).
//!
//! The channel ([`Checked`]) checks the metadata of each response so, of the soft limit too in the
//! dialect of grpcio, where a Rust client keeps the hard limit, which hyper kept before: hyper takes
//! header lists far larger, so that a large one fails its call at once, of this error, rather than
//! as a reset stream, or as a connection that hyper ends. The connection advertises the hard limit
//! as the size of a header list that the client takes ([`ADVERTISED_HEADER_LIST_SIZE`]), as
//! grpcio's channel does: a server that honours it, as grpc-java's, resets the stream of a call
//! whose metadata would reach it, as it does of grpcio's calls.
//!
//! The channel ends the calls of a client that are in flight when the client is closed, as
//! grpcio's `close` cancels the calls of its channel: of `CANCELLED`, `Channel closed!`, before the
//! response of a call begins, and `Stream removed (Channel closed!)` while it is read (see
//! [`crate::client::ChannelClosed`]).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use hyper::body::{Body, Frame, SizeHint};
use tonic::codegen::{http, BoxFuture, Bytes, Service, StdError};
use tonic::transport::Channel;

/// gRPC's soft limit of the size of the metadata that a call receives (`grpc.max_metadata_size`).
const SOFT_LIMIT: usize = 8 * 1024;

/// gRPC's hard limit of the size of the metadata that a call receives
/// (`grpc.absolute_max_metadata_size`).
const HARD_LIMIT: usize = 16 * 1024;

/// HPACK's overhead of an entry (RFC 7541, section 4.1), which gRPC's core counts in the size of
/// the metadata.
const ENTRY_OVERHEAD: usize = 32;

/// The size of a header list that hyper takes, far past the hard limit, so that the channel tells
/// a larger one by its size (see the module).
pub(crate) const MAX_HEADER_LIST_SIZE: u32 = 16 << 20;

/// The size of a header list that a connection of the client advertises in its settings
/// (`SETTINGS_MAX_HEADER_LIST_SIZE`), the hard limit, as grpcio's channel advertises it (see
/// [`crate::connector::Handover`]).
pub(crate) const ADVERTISED_HEADER_LIST_SIZE: u32 = HARD_LIMIT as u32;

/// The headers that gRPC's core knows of a response, in the order of its table of metadata
/// (`grpc_metadata_batch`), in which it lists them before the others.
const KNOWN_HEADERS: [&str; 17] = [
    ":status",
    "content-type",
    "te",
    "grpc-encoding",
    "grpc-accept-encoding",
    "grpc-status",
    "grpc-previous-rpc-attempts",
    "grpc-retry-pushback-ms",
    "user-agent",
    "grpc-message",
    "host",
    "endpoint-load-metrics-bin",
    "grpc-server-stats-bin",
    "grpc-trace-bin",
    "grpc-tags-bin",
    "lb-token",
    "x-envoy-peer-metadata",
];

/// The source of the client's status of metadata that gRPC's core refuses (see [`checked`]).
#[derive(Debug, Clone, Copy)]
pub(crate) struct MetadataRefused;

impl std::fmt::Display for MetadataRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the client refuses metadata that gRPC refuses")
    }
}

impl std::error::Error for MetadataRefused {}

/// The source of a status whose `grpc-status-details-bin` the call has, but empty, of which tonic
/// gives no details: an empty value of base64, or, of the client's own status of details that
/// gRPC's core does not decode ([`MetadataRefused`], its source then), the value that it gives the
/// call of them. grpcio's error of such a call has the details, empty, as PySpark's
/// `rpc_status.from_call` reads them (see [`crate::error::GrpcioError`]).
#[derive(Debug, Clone, Copy)]
pub(crate) struct EmptyDetails {
    /// The refusal of the details, of a status of the client's own.
    pub(crate) refused: Option<MetadataRefused>,
}

impl std::fmt::Display for EmptyDetails {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the details of the status are empty")
    }
}

impl std::error::Error for EmptyDetails {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.refused
            .as_ref()
            .map(|refused| refused as &(dyn std::error::Error + 'static))
    }
}

/// Whether the call of `status` has its `grpc-status-details-bin`, empty (see [`EmptyDetails`]).
pub(crate) fn has_empty_details(status: &tonic::Status) -> bool {
    std::iter::successors(std::error::Error::source(status), |error| error.source())
        .any(|error| error.is::<EmptyDetails>())
}

/// The name of the details of the status of an error.
const STATUS_DETAILS: &str = "grpc-status-details-bin";

/// Of the metadata `headers` of a response, its headers or its trailers, whose
/// `grpc-status-details-bin` gRPC's core decodes (see [`checked`] for one that it does not), the
/// value that tonic reads of the bytes that gRPC's core decodes, where tonic's own decoding would
/// differ, or panic, and of empty bytes, the status of an error of them, tonic's, of its details
/// empty but present (see [`EmptyDetails`]). None of metadata of no details, or of no error.
fn read_details(headers: &mut http::HeaderMap) -> Option<tonic::Status> {
    let details = unbase64(headers.get(STATUS_DETAILS)?.as_bytes())?;
    if !details.is_empty() {
        use base64::Engine;
        let canonical = base64::engine::general_purpose::STANDARD.encode(&details);
        if headers.get(STATUS_DETAILS).map(|value| value.as_bytes()) != Some(canonical.as_bytes()) {
            let canonical = http::HeaderValue::from_str(&canonical)
                .unwrap_or_else(|_| http::HeaderValue::from_static(""));
            headers.insert(STATUS_DETAILS, canonical);
        }
        return None;
    }
    headers.insert(STATUS_DETAILS, http::HeaderValue::from_static(""));
    let mut status = tonic::Status::from_header_map(headers)?;
    if status.code() == tonic::Code::Ok {
        return None;
    }
    status.set_source(std::sync::Arc::new(EmptyDetails { refused: None }));
    Some(status)
}

/// Whether `status` is the client's own of metadata that gRPC's core refuses (see [`checked`]).
pub(crate) fn is_refused(status: &tonic::Status) -> bool {
    std::iter::successors(std::error::Error::source(status), |error| error.source())
        .any(|error| error.is::<MetadataRefused>())
}

/// The status of the metadata of `status`, the code of a response, None of trailers, and
/// `headers` that gRPC's core refuses (see the module), of the soft limit too if `soft`, whose
/// rejection is of the draw `draw` in `[0, 1)`; None of metadata within the limits.
fn checked(
    status: Option<&str>,
    headers: &http::HeaderMap,
    soft: bool,
    draw: impl FnOnce() -> f64,
) -> Option<tonic::Status> {
    let entries = status
        .map(|status| (":status", status.as_bytes()))
        .into_iter()
        .chain(
            headers
                .iter()
                .map(|(name, value)| (name.as_str(), value.as_bytes())),
        );
    let mut listed: Vec<(&str, &[u8])> = Vec::new();
    let mut total = 0;
    for (name, value) in entries {
        if name.len() + ENTRY_OVERHEAD >= HARD_LIMIT {
            return Some(too_large(format!(
                "received metadata size exceeds hard limit (key length {} vs. {HARD_LIMIT})",
                name.len()
            )));
        }
        if value.len() + ENTRY_OVERHEAD >= HARD_LIMIT {
            return Some(too_large(format!(
                "received metadata size exceeds hard limit (value length {} vs. {HARD_LIMIT})",
                value.len()
            )));
        }
        // As gRPC's core decodes a binary value as it reads it; of the details of an error, the
        // call still has them, empty (see `EmptyDetails`).
        if name.ends_with("-bin") && unbase64(value).is_none() {
            let mut status = tonic::Status::internal(format!(
                "Error parsing '{name}' metadata: illegal base64 encoding"
            ));
            status.set_source(if name == STATUS_DETAILS {
                std::sync::Arc::new(EmptyDetails {
                    refused: Some(MetadataRefused),
                })
            } else {
                std::sync::Arc::new(MetadataRefused)
            });
            return Some(status);
        }
        total += name.len() + value.len() + ENTRY_OVERHEAD;
        listed.push((name, value));
        if total >= HARD_LIMIT {
            return Some(too_large(format!(
                "received metadata size exceeds hard limit ({total} vs. {HARD_LIMIT});  {}",
                listing(&listed)
            )));
        }
    }
    let rejected = || {
        let past = (total - SOFT_LIMIT) as f64 / (HARD_LIMIT - SOFT_LIMIT) as f64;
        draw() < past
    };
    if soft && total > SOFT_LIMIT && rejected() {
        let message = format!(
            "received metadata size exceeds soft limit ({total} vs. {SOFT_LIMIT});  {}",
            listing(&listed)
        );
        return Some(soft_rejected(message, headers));
    }
    None
}

/// The entries of `entries` as gRPC's core lists them (see the module), of each name and size.
fn listing(entries: &[(&str, &[u8])]) -> String {
    let rank = |name: &str| {
        KNOWN_HEADERS
            .iter()
            .position(|known| *known == name)
            .unwrap_or(KNOWN_HEADERS.len())
    };
    let mut ordered: Vec<&(&str, &[u8])> = entries.iter().collect();
    // Stable: the headers that it does not know keep the order that they came in.
    ordered.sort_by_key(|(name, _)| rank(name));
    ordered
        .iter()
        .map(|(name, value)| {
            let size = name.len() + encoded_length(name, value) + ENTRY_OVERHEAD;
            format!("{name}:{size}B")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// The length of the value `value` of the header `name` as gRPC's core encodes it again: of the
/// value that it parsed of a header that it knows, of the decoded bytes of a binary one, else as
/// it came.
fn encoded_length(name: &str, value: &[u8]) -> usize {
    let number = || {
        std::str::from_utf8(value)
            .ok()
            .and_then(|text| text.parse::<u64>().ok())
            .map(|number| number.to_string().len())
    };
    let length = match name {
        ":status" | "grpc-status" | "grpc-previous-rpc-attempts" | "grpc-retry-pushback-ms" => {
            number()
        }
        "content-type" => Some(if value.is_empty() {
            0
        } else if value == b"application/grpc"
            || value.starts_with(b"application/grpc;")
            || value.starts_with(b"application/grpc+")
        {
            "application/grpc".len()
        } else {
            "application/grpc+unknown".len()
        }),
        "grpc-accept-encoding" => Some(accepted_encodings(value).len()),
        _ if name.ends_with("-bin") => unbase64(value).map(|decoded| decoded.len()),
        _ => None,
    };
    length.unwrap_or(value.len())
}

/// The compression algorithms of a `grpc-accept-encoding` as gRPC's core encodes them again: of
/// those that it knows, and `identity`, always, in its order, separated by `, `.
fn accepted_encodings(value: &[u8]) -> String {
    let named: Vec<&[u8]> = value
        .split(|byte| *byte == b',')
        .map(|name| name.trim_ascii())
        .collect();
    ["identity", "deflate", "gzip"]
        .into_iter()
        .filter(|algorithm| *algorithm == "identity" || named.contains(&algorithm.as_bytes()))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The bytes of a binary value of base64, as gRPC's core decodes the metadata that it reads: of
/// the standard alphabet, of any `=` at its end dropped, of whole groups of four characters, and
/// of a last group of two or three characters whose unused bits are zero; None of any other
/// value, of a character out of the alphabet, an `=` within it, or a last group of one.
fn unbase64(value: &[u8]) -> Option<Vec<u8>> {
    let sextet = |byte: u8| -> Option<u32> {
        Some(u32::from(match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        }))
    };
    let end = value.len() - value.iter().rev().take_while(|byte| **byte == b'=').count();
    let value = &value[..end];
    let mut decoded = Vec::with_capacity(value.len() / 4 * 3 + 2);
    let (groups, rest) = value.as_chunks::<4>();
    for group in groups {
        let bits = group
            .iter()
            .try_fold(0u32, |bits, byte| Some(bits << 6 | sextet(*byte)?))?;
        decoded.extend_from_slice(&bits.to_be_bytes()[1..]);
    }
    match rest.len() {
        0 => {}
        1 => return None,
        length => {
            let bits = rest
                .iter()
                .try_fold(0u32, |bits, byte| Some(bits << 6 | sextet(*byte)?))?
                << (6 * (4 - length));
            // The bits past the bytes of the group are zero, as gRPC's core asks.
            let bytes = length - 1;
            if bits & (0xff_ffff >> (8 * bytes)) != 0 {
                return None;
            }
            decoded.extend_from_slice(&bits.to_be_bytes()[1..1 + bytes]);
        }
    }
    Some(decoded)
}

/// The status of metadata `headers` past the soft limit, of `message`: of the details of the
/// server's error and of its other entries, which gRPC's core still gives the call, so that
/// grpcio's error of it has them in its trailing metadata, as PySpark's `rpc_status.from_call`
/// reads them (see [`crate::error::from_call_error`]).
fn soft_rejected(message: String, headers: &http::HeaderMap) -> tonic::Status {
    let present = headers.get(STATUS_DETAILS);
    let details = present
        .and_then(|details| unbase64(details.as_bytes()))
        .unwrap_or_default();
    let empty = present.is_some() && details.is_empty();
    let mut metadata = headers.clone();
    for name in ["grpc-status", "grpc-message", STATUS_DETAILS] {
        metadata.remove(name);
    }
    let mut status = tonic::Status::with_details_and_metadata(
        tonic::Code::ResourceExhausted,
        message,
        details.into(),
        tonic::metadata::MetadataMap::from_headers(metadata),
    );
    if empty {
        status.set_source(std::sync::Arc::new(EmptyDetails {
            refused: Some(MetadataRefused),
        }));
    } else {
        status.set_source(std::sync::Arc::new(MetadataRefused));
    }
    status
}

/// `RESOURCE_EXHAUSTED` of `message`, tagged as the client's own.
fn too_large(message: String) -> tonic::Status {
    refused(tonic::Code::ResourceExhausted, message)
}

/// The status of `code` and `message`, tagged as the client's own.
fn refused(code: tonic::Code, message: String) -> tonic::Status {
    let mut status = tonic::Status::new(code, message);
    status.set_source(std::sync::Arc::new(MetadataRefused));
    status
}

/// The status of the metadata of a response that the client refuses (see [`checked`]), of the
/// soft limit in the dialect of grpcio.
fn refusal(status: Option<&str>, headers: &http::HeaderMap) -> Option<tonic::Status> {
    checked(
        status,
        headers,
        crate::connector::grpcio_dialect(),
        crate::retries::random_unit,
    )
}

/// The mark of a request whose metadata `MetadataSource` gives as it is sent, which [`Checked`]
/// reads then (see `SparkConnectClient::set_metadata_source`), and the metadata of the call
/// credentials, the `authorization` of the token, which gRPC sends after the metadata of the call.
#[derive(Clone)]
pub(crate) struct PendingMetadata {
    pub(crate) source: std::sync::Arc<dyn crate::channel::MetadataSource>,
    pub(crate) after: Vec<(
        tonic::metadata::AsciiMetadataKey,
        tonic::metadata::AsciiMetadataValue,
    )>,
}

/// The keys of the metadata of an application that gRPC's core does not send, as it sends its
/// own: of HTTP/2 and of gRPC (`te`, `content-type`, `content-length`, `grpc-timeout`), and those
/// that tonic drops too (`grpc-message`, `grpc-message-type`, `grpc-status`), as grpcio's calls of
/// them succeed (but `grpc-status`, of which a call of grpcio hangs).
const NOT_SENT: [&str; 7] = [
    "te",
    "content-type",
    "content-length",
    "grpc-timeout",
    "grpc-message",
    "grpc-message-type",
    "grpc-status",
];

/// The fields of a connection that HTTP/2 forbids in a request (RFC 9113, 8.2.2), which gRPC's core
/// sends of the metadata of an application, so that its server resets the stream as malformed
/// (`RST_STREAM` of `PROTOCOL_ERROR`), where hyper drops them.
const CONNECTION_FIELDS: [&str; 5] = [
    "connection",
    "keep-alive",
    "proxy-connection",
    "transfer-encoding",
    "upgrade",
];

/// Whether gRPC's core sends `key` of the metadata of an application (see [`NOT_SENT`]).
pub(crate) fn sent_of_an_application(key: &str) -> bool {
    !NOT_SENT.contains(&key)
}

/// The status of a call of a field of a connection in its headers (see [`CONNECTION_FIELDS`]), in
/// the dialect of grpcio (`grpcio`), as grpcio's call fails once its server resets its stream:
/// `INTERNAL`, `Stream removed (RST_STREAM (Received RST_STREAM with error code 1))`.
fn of_connection_fields(headers: &http::HeaderMap, grpcio: bool) -> Option<tonic::Status> {
    if !grpcio
        || !CONNECTION_FIELDS
            .iter()
            .any(|name| headers.contains_key(*name))
    {
        return None;
    }
    Some(tonic::Status::internal(
        "Stream removed (RST_STREAM (Received RST_STREAM with error code 1))",
    ))
}

/// Append `metadata`, the metadata of a call, of its order, to `headers`, and then `after`, as
/// gRPC sends the metadata of a call and then that of its call credentials. A key of several values
/// keeps them all, in their order.
pub(crate) fn append_metadata(
    headers: &mut http::HeaderMap,
    metadata: tonic::metadata::MetadataMap,
    after: &[(
        tonic::metadata::AsciiMetadataKey,
        tonic::metadata::AsciiMetadataValue,
    )],
) {
    let mut metadata = metadata;
    for (key, value) in after {
        metadata.append(key.clone(), value.clone());
    }
    // The iterator names the first value of each key only, and the values after it of None.
    let mut last = None;
    for (name, value) in metadata.into_headers() {
        if name.is_some() {
            last = name;
        }
        if let Some(name) = &last {
            headers.append(name.clone(), value);
        }
    }
}

/// The error of the source of the metadata of a call, as the source of the status that fails the
/// call, which is then the error of the call as it is, of no retry (see [`metadata_failure`]).
#[derive(Debug)]
pub(crate) struct MetadataFailed(pub(crate) crate::error::SparkError);

impl std::fmt::Display for MetadataFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl std::error::Error for MetadataFailed {}

/// The error of the source of the metadata of the call that `status` failed, if the source failed
/// it (see [`MetadataFailed`]).
pub(crate) fn metadata_failure(status: &tonic::Status) -> Option<crate::error::SparkError> {
    std::error::Error::source(status)?
        .downcast_ref::<MetadataFailed>()
        .map(|failed| failed.0.clone())
}

/// The channel of a client, whose responses of metadata past the limits of gRPC fail their calls
/// (see the module): their headers as they come, and their trailers as the body ends; whose
/// calls in flight end when the client is closed; whose request of [`PendingMetadata`] is sent
/// once its source gave its metadata, or fails of the error of the source, unsent; and whose
/// calls take the connection of the subchannel that the channel of the client picks (see
/// `crate::connector::ClientChannel`), once it resolved its name and connected.
#[derive(Clone)]
pub(crate) struct Checked {
    /// The channel of the client: its resolver and its `pick_first`, over the subchannels that it
    /// shares with the other channels of the process.
    pub(crate) channel: Arc<crate::connector::ClientChannel>,
    /// Whether the client of the calls closed its channel (see `SparkConnectClient::close`).
    pub(crate) closed: Arc<crate::client::ChannelClosed>,
    /// Whether the channel is of a target that gRPC refuses, whose status the transport makes
    /// itself, as gRPC's channel fails its calls of no response, of no limit of the metadata.
    pub(crate) refused: bool,
}

impl Service<http::Request<tonic::body::Body>> for Checked {
    type Response = http::Response<CheckedBody>;
    type Error = StdError;
    type Future = BoxFuture<Self::Response, Self::Error>;

    /// Ready at once: a call waits for its connection as it is made (see [`Checked::call`]).
    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, mut request: http::Request<tonic::body::Body>) -> Self::Future {
        let closed = Arc::clone(&self.closed);
        let refused = self.refused;
        let channel = Arc::clone(&self.channel);
        let pending = request.extensions_mut().remove::<PendingMetadata>();
        Box::pin(async move {
            if let Some(PendingMetadata { source, after }) = pending {
                let pairs = match source.metadata().await {
                    Ok(pairs) => pairs,
                    Err(error) => {
                        let mut status = tonic::Status::unknown(error.to_string());
                        status.set_source(std::sync::Arc::new(MetadataFailed(error)));
                        return Err(Box::new(status) as StdError);
                    }
                };
                match call_metadata(&pairs) {
                    Ok(metadata) => append_metadata(request.headers_mut(), metadata, &after),
                    Err(status) => return Err(Box::new(status) as StdError),
                }
            }
            if let Some(status) =
                of_connection_fields(request.headers(), crate::connector::grpcio_dialect())
            {
                return Err(Box::new(status) as StdError);
            }
            // The connection that the channel picks, which its name resolved first, of its own
            // resolution, and which the close of the client ends, as the call that waits for it.
            let picked = tokio::select! {
                biased;
                picked = channel.pick() => picked,
                () = closed.closed() => {
                    return Err(Box::new(tonic::Status::cancelled(crate::client::CLOSED_CHANNEL))
                        as StdError);
                }
            };
            let mut connection = match picked {
                Ok(connection) => connection,
                Err(unpicked) => {
                    let status = Box::new(unpicked.status(channel.grpcio())) as StdError;
                    return Err(unless_closed(status, &closed));
                }
            };
            std::future::poll_fn(|cx| connection.poll_ready(cx))
                .await
                .map_err(|error| unless_closed(StdError::from(error), &closed))?;
            checked_response(connection.call(request), closed, refused).await
        })
    }
}

/// The metadata of a call of `pairs`, the keys and the values of a source of the metadata (see
/// `crate::channel::MetadataSource`), as gRPC's core validates them: a key of `[0-9a-z_.-]`, a
/// value of printable ASCII and the space, or of any bytes, sent in base64, of a key of `-bin`; a
/// `user-agent`, which grpcio replaces with its own, and the keys that gRPC's core sets itself
/// (see [`NOT_SENT`]) are not sent. Of another, gRPC's `INTERNAL` (`Invalid metadata`), which
/// fails the call.
pub(crate) fn call_metadata(
    pairs: &[(Vec<u8>, Vec<u8>)],
) -> Result<tonic::metadata::MetadataMap, tonic::Status> {
    let invalid = || tonic::Status::internal("Invalid metadata");
    let mut metadata = tonic::metadata::MetadataMap::new();
    for (key, value) in pairs {
        let legal = |b: &u8| b.is_ascii_lowercase() || b.is_ascii_digit() || b"_.-".contains(b);
        if key.is_empty() || !key.iter().all(legal) {
            return Err(invalid());
        }
        if key.ends_with(b"-bin") {
            let key = tonic::metadata::BinaryMetadataKey::from_bytes(key).map_err(|_| invalid())?;
            metadata.append_bin(key, tonic::metadata::BinaryMetadataValue::from_bytes(value));
            continue;
        }
        if key == b"user-agent" || NOT_SENT.iter().any(|name| key == name.as_bytes()) {
            if !value.iter().all(|b| (0x20..=0x7e).contains(b)) {
                return Err(invalid());
            }
            continue;
        }
        if !value.iter().all(|b| (0x20..=0x7e).contains(b)) {
            return Err(invalid());
        }
        let key = tonic::metadata::AsciiMetadataKey::from_bytes(key).map_err(|_| invalid())?;
        let value = tonic::metadata::AsciiMetadataValue::try_from(value.as_slice())
            .map_err(|_| invalid())?;
        metadata.append(key, value);
    }
    Ok(metadata)
}

/// The response of `response`, the call of the channel of [`Checked`], whose headers past the
/// limits of gRPC, or of details that are not read, fail the call, and which ends as cancelled
/// when the client closes the channel before it begins, as grpcio's `close` cancels the calls in
/// flight on its channel.
async fn checked_response(
    response: <Channel as Service<http::Request<tonic::body::Body>>>::Future,
    closed: Arc<crate::client::ChannelClosed>,
    refused: bool,
) -> Result<http::Response<CheckedBody>, StdError> {
    let mut response = tokio::select! {
        biased;
        response = response => {
            response.map_err(|error| unless_closed(StdError::from(error), &closed))?
        }
        () = closed.closed() => {
            return Err(Box::new(tonic::Status::cancelled(crate::client::CLOSED_CHANNEL)) as StdError);
        }
    };
    if let Some(status) =
        refusal(Some(response.status().as_str()), response.headers()).filter(|_| !refused)
    {
        return Err(Box::new(status) as StdError);
    }
    if let Some(status) = read_details(response.headers_mut()) {
        return Err(Box::new(status) as StdError);
    }
    Ok(response.map(|body| CheckedBody::new(body, closed)))
}

/// `error`, of a call of [`Checked`] that failed before its response, unless the client of the
/// call closed the channel by then: then `CANCELLED` (`Channel closed!`), as grpcio's `close`
/// cancels the calls in flight on its channel, whatever ended the call, such as the connector of a
/// subchannel that the close let go of as the call waited for its attempt (see
/// `crate::connector`): the close marks the channel closed before it lets go of its subchannels.
fn unless_closed(error: StdError, closed: &crate::client::ChannelClosed) -> StdError {
    if closed.is_closed() {
        return Box::new(tonic::Status::cancelled(crate::client::CLOSED_CHANNEL));
    }
    error
}

/// The body of a response of [`Checked`], whose trailers past the limits fail its call, and which
/// ends as cancelled when the client of the call closes the channel while it is read.
pub(crate) struct CheckedBody {
    body: tonic::body::Body,
    /// Ends once the client of the call closes the channel (see
    /// [`crate::client::ChannelClosed::closed`]), None once the body ended so.
    closed: Option<Pin<Box<dyn Future<Output = ()> + Send>>>,
}

impl CheckedBody {
    fn new(body: tonic::body::Body, closed: Arc<crate::client::ChannelClosed>) -> Self {
        CheckedBody {
            body,
            closed: Some(Box::pin(async move { closed.closed().await })),
        }
    }
}

impl Body for CheckedBody {
    type Data = Bytes;
    type Error = StdError;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        let Some(closed) = this.closed.as_mut() else {
            // Cancelled: the call has nothing more.
            return Poll::Ready(None);
        };
        // A frame that has come is read first, so that a stream whose end came before the close
        // ends as the server ended it.
        let polled = Pin::new(&mut this.body).poll_frame(cx);
        // As grpcio's `close`, which cancels the calls in flight on its channel: a response that
        // is read as the client closes the channel ends then, as cancelled, of the stream of the
        // call removed.
        if polled.is_pending() && closed.as_mut().poll(cx).is_ready() {
            this.closed = None;
            return Poll::Ready(Some(Err(Box::new(tonic::Status::cancelled(
                crate::client::STREAM_OF_A_CLOSED_CHANNEL,
            )))));
        }
        match polled {
            Poll::Ready(Some(Ok(mut frame))) => {
                if let Some(status) = frame
                    .trailers_ref()
                    .and_then(|trailers| refusal(None, trailers))
                {
                    return Poll::Ready(Some(Err(Box::new(status))));
                }
                if let Some(status) = frame.trailers_mut().and_then(read_details) {
                    return Poll::Ready(Some(Err(Box::new(status))));
                }
                Poll::Ready(Some(Ok(frame)))
            }
            Poll::Ready(Some(Err(error))) => Poll::Ready(Some(Err(Box::new(error)))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }

    fn is_end_stream(&self) -> bool {
        self.closed.is_none() || self.body.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.body.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(entries: &[(&str, String)]) -> http::HeaderMap {
        let mut headers = http::HeaderMap::new();
        for (name, value) in entries {
            headers.append(
                http::HeaderName::from_bytes(name.as_bytes()).unwrap(),
                http::HeaderValue::from_str(value).unwrap(),
            );
        }
        headers
    }

    /// The headers of a response of its trailers only, as a server of gRPC's core sends them, and
    /// `more`.
    fn trailers_only(more: &[(&str, String)]) -> http::HeaderMap {
        let mut entries = vec![
            ("content-type", "application/grpc".to_string()),
            ("grpc-status", "13".to_string()),
            ("grpc-message", "failed".to_string()),
        ];
        entries.extend(more.iter().cloned());
        headers(&entries)
    }

    fn refused_with(
        status: Option<&str>,
        headers: &http::HeaderMap,
        soft: bool,
        draw: f64,
    ) -> Option<String> {
        checked(status, headers, soft, || draw).map(|status| {
            assert_eq!(status.code(), tonic::Code::ResourceExhausted);
            assert!(is_refused(&status));
            status.message().to_string()
        })
    }

    #[test]
    fn metadata_past_the_hard_limit_is_refused_as_grpc_s_core_refuses_it() {
        // Of grpcio 1.85's texts of the same metadata: an entry too large alone, of its value or
        // its name, and entries that pass the limit together, of the size at the entry that
        // passes it and of the entries up to it.
        let value = |length| "a".repeat(length);
        assert_eq!(
            refused_with(
                Some("200"),
                &trailers_only(&[("x-big", value(20000))]),
                true,
                0.9
            )
            .as_deref(),
            Some("received metadata size exceeds hard limit (value length 20000 vs. 16384)")
        );
        assert_eq!(
            refused_with(
                Some("200"),
                &trailers_only(&[("x-big", value(16352))]),
                true,
                0.9
            )
            .as_deref(),
            Some("received metadata size exceeds hard limit (value length 16352 vs. 16384)")
        );
        assert_eq!(
            refused_with(
                Some("200"),
                &trailers_only(&[("x-big", value(16320))]),
                true,
                0.9
            )
            .as_deref(),
            Some(
                "received metadata size exceeds hard limit (16554 vs. 16384);  :status:42B \
                 content-type:60B grpc-status:45B grpc-message:50B x-big:16357B"
            )
        );
        let key = "x".repeat(17000);
        assert_eq!(
            refused_with(
                None,
                &headers(&[(key.as_str(), "v".to_string())]),
                true,
                0.9
            )
            .as_deref(),
            Some("received metadata size exceeds hard limit (key length 17000 vs. 16384)")
        );
        let many: Vec<(String, String)> =
            (0..200).map(|i| (format!("x-k{i}"), value(100))).collect();
        let many: Vec<(&str, String)> = many
            .iter()
            .map(|(name, value)| (name.as_str(), value.clone()))
            .collect();
        let text = refused_with(Some("200"), &trailers_only(&many), false, 0.9).unwrap();
        assert!(
            text.starts_with(
                "received metadata size exceeds hard limit (16509 vs. 16384);  :status:42B \
                 content-type:60B grpc-status:45B grpc-message:50B x-k0:136B x-k1:136B"
            ),
            "{text}"
        );
        assert!(text.ends_with("x-k117:138B x-k118:138B"), "{text}");
        // Within it, of no error.
        assert_eq!(
            refused_with(
                Some("200"),
                &trailers_only(&[("x-big", value(8000))]),
                false,
                0.0
            ),
            None
        );
    }

    #[test]
    fn metadata_past_the_soft_limit_is_refused_at_random_in_the_dialect_of_grpcio() {
        // The more often the larger it is up to the hard limit, as gRPC's core refuses it, of the
        // draw: here 12234 bytes, of a chance of 4042 / 8192.
        let metadata = trailers_only(&[("x-big", "a".repeat(12000))]);
        let text = "received metadata size exceeds soft limit (12234 vs. 8192);  :status:42B \
                    content-type:60B grpc-status:45B grpc-message:50B x-big:12037B";
        assert_eq!(
            refused_with(Some("200"), &metadata, true, 0.49).as_deref(),
            Some(text)
        );
        assert_eq!(refused_with(Some("200"), &metadata, true, 0.5), None);
        // Of a Rust client, the hard limit only.
        assert_eq!(refused_with(Some("200"), &metadata, false, 0.0), None);
        // At the soft limit, never.
        let at = trailers_only(&[("x-big", "a".repeat(7958))]);
        assert_eq!(refused_with(Some("200"), &at, true, 0.0), None);
    }

    #[test]
    fn the_limits_are_reached_as_grpc_s_core_reaches_them() {
        // Of grpcio 1.85's texts at the boundaries: a value or a name that reaches the hard
        // limit with 32 bytes, whatever the other, and a size of the entries that reaches it.
        let value = |length| "a".repeat(length);
        let refused = |length| {
            refused_with(
                Some("200"),
                &trailers_only(&[("x-big", value(length))]),
                true,
                0.9,
            )
            .unwrap()
        };
        let listing = |total, size| {
            format!(
                "received metadata size exceeds hard limit ({total} vs. 16384);  :status:42B \
                 content-type:60B grpc-status:45B grpc-message:50B x-big:{size}B"
            )
        };
        for (length, total) in [(16347, 16581), (16348, 16582), (16351, 16585)] {
            assert_eq!(refused(length), listing(total, length + 37));
        }
        assert_eq!(
            refused(16352),
            "received metadata size exceeds hard limit (value length 16352 vs. 16384)"
        );
        // A size of exactly the limit.
        assert_eq!(refused(16150), listing(16384, 16187));
        assert_eq!(
            refused_with(
                Some("200"),
                &trailers_only(&[("x-big", value(16149))]),
                false,
                0.0
            ),
            None
        );
        // A name that reaches it, of a short value.
        let key = "k".repeat(16352);
        assert_eq!(
            refused_with(
                None,
                &headers(&[(key.as_str(), "v".to_string())]),
                true,
                0.9
            )
            .as_deref(),
            Some("received metadata size exceeds hard limit (key length 16352 vs. 16384)")
        );
        let key = "k".repeat(16351);
        let at_the_limit =
            format!("received metadata size exceeds hard limit (16384 vs. 16384);  {key}:16384B");
        assert_eq!(
            refused_with(
                None,
                &headers(&[(key.as_str(), "v".to_string())]),
                true,
                0.9
            ),
            Some(at_the_limit)
        );
    }

    #[test]
    fn the_entries_are_listed_as_grpc_s_core_encodes_them() {
        // Of grpcio 1.85's text of the same metadata: the headers that gRPC's core knows first,
        // in the order of its table, of their values as it encodes them again, such as the
        // algorithms of `grpc-accept-encoding`, then the others, in the order that they came,
        // of the decoded bytes of a binary one; the size is of the entries as they came.
        let entries = [
            ("x-a", "1"),
            ("x-b-bin", "AQIDBAU"),
            ("x-c", "zz"),
            ("x-envoy-peer-metadata", "envoy"),
            ("lb-token", "tok"),
            ("grpc-tags-bin", "dGFncw=="),
            ("grpc-trace-bin", "dHJhY2U="),
            ("grpc-server-stats-bin", "c3RhdHM="),
            ("endpoint-load-metrics-bin", "bG9hZA=="),
            ("host", "h"),
            ("grpc-message", "failed"),
            ("user-agent", "ua"),
            ("grpc-retry-pushback-ms", "100"),
            ("grpc-previous-rpc-attempts", "2"),
            ("grpc-status", "13"),
            ("grpc-accept-encoding", "gzip"),
            ("grpc-encoding", "identity"),
            ("te", "trailers"),
            ("content-type", "application/grpc"),
        ];
        let mut entries: Vec<(&str, String)> = entries
            .iter()
            .map(|(name, value)| (*name, value.to_string()))
            .collect();
        entries.push(("x-big", "a".repeat(16000)));
        assert_eq!(
            refused_with(Some("200"), &headers(&entries), true, 0.9).as_deref(),
            Some(
                "received metadata size exceeds hard limit (17035 vs. 16384);  :status:42B \
                 content-type:60B te:42B grpc-encoding:53B grpc-accept-encoding:66B \
                 grpc-status:45B grpc-previous-rpc-attempts:59B grpc-retry-pushback-ms:57B \
                 user-agent:44B grpc-message:50B host:37B endpoint-load-metrics-bin:61B \
                 grpc-server-stats-bin:58B grpc-trace-bin:51B grpc-tags-bin:49B lb-token:43B \
                 x-envoy-peer-metadata:58B x-a:36B x-b-bin:44B x-c:37B x-big:16037B"
            )
        );
        // Of their values as gRPC's core encodes them again.
        let encoded = |name: &str, value: &str| encoded_length(name, value.as_bytes());
        assert_eq!(encoded("content-type", "application/grpc+proto"), 16);
        assert_eq!(encoded("content-type", "application/grpc;x=1"), 16);
        assert_eq!(encoded("content-type", "text/plain"), 24);
        assert_eq!(encoded("content-type", ""), 0);
        assert_eq!(encoded("grpc-accept-encoding", "identity"), 8);
        assert_eq!(encoded("grpc-accept-encoding", "deflate,gzip"), 23);
        assert_eq!(encoded("grpc-accept-encoding", "gzip,deflate,identity"), 23);
        assert_eq!(encoded("grpc-accept-encoding", "br"), 8);
        assert_eq!(encoded("grpc-encoding", "gzip"), 4);
        assert_eq!(encoded("grpc-status", "013"), 2);
        assert_eq!(encoded("grpc-message", "%E2%82%AC"), 9);
        // A binary value of base64 with or without its padding, and one that is not base64.
        let padded =
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, [b'x'; 100]);
        assert_eq!(encoded("x-d-bin", &padded), 100);
        assert_eq!(encoded("x-d-bin", padded.trim_end_matches('=')), 100);
        assert_eq!(encoded("x-d-bin", "not base64!"), 11);
    }

    #[test]
    fn binary_values_are_decoded_as_grpc_s_core_decodes_them() {
        // As the base64 of gRPC core's reads of the metadata, which grpcio 1.85 reads so: of the
        // standard alphabet, of any `=` at its end dropped, and of a last group of two or three
        // characters whose unused bits are zero.
        let decoded = |value: &str| unbase64(value.as_bytes());
        for (value, bytes) in [
            ("", &b""[..]),
            ("=", b""),
            ("==", b""),
            ("YWJj", b"abc"),
            ("YWJj=", b"abc"),
            ("YWJj===", b"abc"),
            ("YWI=", b"ab"),
            ("YWI", b"ab"),
            ("abc", b"i\xb7"),
            ("YQ", b"a"),
            ("YQ=", b"a"),
            ("YQ==", b"a"),
            ("YWJjZA", b"abcd"),
            ("YWJjZGU", b"abcde"),
            ("CA0SBGJvb20", b"\x08\x0d\x12\x04boom"),
        ] {
            assert_eq!(decoded(value).as_deref(), Some(bytes), "{value:?}");
        }
        for value in [
            "a",
            "Y",
            "YW",
            "YW==",
            "YWJjZB",
            "YWJjZGV",
            "Y=WJj",
            "YW Jj",
            "-_-_",
            "YWJj\t",
            "!!!*",
            "@@@@",
            "not base64!",
        ] {
            assert_eq!(decoded(value), None, "{value:?}");
        }
    }

    #[test]
    fn a_binary_value_that_grpc_s_core_does_not_decode_fails_the_call_as_it_fails_it() {
        // Of grpcio 1.85's text, `INTERNAL`; of the details of the status of an error, which the
        // call still has, empty, as gRPC's core gives them.
        let bad = |name: &str| {
            checked(
                None,
                &headers(&[("grpc-status", "8".to_string()), (name, "YW==".to_string())]),
                false,
                || 0.9,
            )
            .unwrap()
        };
        for name in ["x-d-bin", STATUS_DETAILS] {
            let status = bad(name);
            assert_eq!(status.code(), tonic::Code::Internal);
            assert!(is_refused(&status));
            assert_eq!(
                status.message(),
                format!("Error parsing '{name}' metadata: illegal base64 encoding")
            );
            assert_eq!(has_empty_details(&status), name == STATUS_DETAILS);
        }
        assert!(checked(
            None,
            &headers(&[("x-d-bin", "YWJj=".to_string())]),
            false,
            || 0.9
        )
        .is_none());
        // Listed of the bytes that gRPC's core decodes.
        assert_eq!(encoded_length(STATUS_DETAILS, b"YWJj="), 3);
        assert_eq!(encoded_length("x-d-bin", b"YQ"), 1);
    }

    #[test]
    fn the_details_of_an_error_are_read_of_the_bytes_that_grpc_s_core_decodes() {
        // tonic, which would panic of a value that its base64 refuses, reads the bytes that gRPC's
        // core decodes, and of empty details, which the call has, an error of its code and message
        // of the details empty but present; the metadata of a success of them is emptied.
        for value in ["", "=", "=="] {
            let mut trailers = headers(&[
                ("grpc-status", "8".to_string()),
                ("grpc-message", "boom".to_string()),
                (STATUS_DETAILS, value.to_string()),
                ("x-custom", "c".to_string()),
            ]);
            let status = read_details(&mut trailers).unwrap();
            assert_eq!(
                (status.code(), status.message()),
                (tonic::Code::ResourceExhausted, "boom")
            );
            assert!(status.details().is_empty() && has_empty_details(&status));
            assert!(!is_refused(&status));
            assert_eq!(status.metadata().get("x-custom").unwrap(), "c");
            assert!(status.metadata().get(STATUS_DETAILS).is_none());
            assert_eq!(trailers.get(STATUS_DETAILS).unwrap(), "");
        }
        let mut success = headers(&[
            ("grpc-status", "0".to_string()),
            (STATUS_DETAILS, "=".to_string()),
        ]);
        assert!(read_details(&mut success).is_none());
        assert_eq!(success.get(STATUS_DETAILS).unwrap(), "");
        // Of bytes, of the value that tonic decodes to them.
        for (value, read) in [
            ("YWJj=", "YWJj"),
            ("YWJj===", "YWJj"),
            ("YQ", "YQ=="),
            ("abc", "abc="),
            ("CA0SBGJvb20", "CA0SBGJvb20="),
            ("CA0SBGJvb20=", "CA0SBGJvb20="),
        ] {
            let mut trailers = headers(&[
                ("grpc-status", "13".to_string()),
                (STATUS_DETAILS, value.to_string()),
            ]);
            assert!(read_details(&mut trailers).is_none());
            assert_eq!(trailers.get(STATUS_DETAILS).unwrap(), read, "{value:?}");
            let status = tonic::Status::from_header_map(&trailers).unwrap();
            assert_eq!(status.details(), &unbase64(value.as_bytes()).unwrap()[..]);
        }
        assert!(read_details(&mut headers(&[("grpc-status", "13".to_string())])).is_none());
    }

    #[test]
    fn metadata_past_the_soft_limit_keeps_the_details_of_the_server_s_error() {
        // As gRPC's core, which still gives the call the metadata that it refuses at the soft
        // limit: the details of the error and the other entries, but its status and message.
        let details = b"\x08\x0d\x12\x06failed";
        let trailers = headers(&[
            ("grpc-status-details-bin", "CA0SBmZhaWxlZA".to_string()),
            ("grpc-status", "13".to_string()),
            ("grpc-message", "failed".to_string()),
            ("x-custom", "c".to_string()),
            ("x-pad", "a".repeat(9000)),
        ]);
        let status = checked(None, &trailers, true, || 0.0).unwrap();
        assert_eq!(status.code(), tonic::Code::ResourceExhausted);
        assert!(is_refused(&status));
        assert_eq!(
            status.message(),
            "received metadata size exceeds soft limit (9242 vs. 8192);  grpc-status:45B \
             grpc-message:50B grpc-status-details-bin:65B x-custom:41B x-pad:9037B"
        );
        assert_eq!(status.details(), &details[..]);
        let names: Vec<&str> = status
            .metadata()
            .keys()
            .map(|key| match key {
                tonic::metadata::KeyRef::Ascii(key) => key.as_str(),
                tonic::metadata::KeyRef::Binary(key) => key.as_str(),
            })
            .collect();
        assert_eq!(names, ["x-custom", "x-pad"]);
        // At the hard limit, nothing of the metadata is kept.
        let past = headers(&[
            ("grpc-status-details-bin", "CA0SBmZhaWxlZA".to_string()),
            ("x-pad", "a".repeat(17000)),
        ]);
        let status = checked(None, &past, true, || 0.0).unwrap();
        assert!(status.details().is_empty() && status.metadata().is_empty());
    }

    #[test]
    fn the_metadata_of_a_source_is_checked_as_grpc_s_core_checks_it() {
        let pair = |key: &str, value: &[u8]| (key.as_bytes().to_vec(), value.to_vec());
        let metadata = call_metadata(&[
            pair("x-a", b"one"),
            pair("x-a", b"two"),
            pair("x-bin", &[0, 1, 255]),
            pair("user-agent", b"mine"),
        ])
        .unwrap();
        let values: Vec<&str> = metadata
            .get_all("x-a")
            .iter()
            .map(|value| value.to_str().unwrap())
            .collect();
        assert_eq!(values, ["one", "two"]);
        assert_eq!(
            metadata
                .get_bin("x-bin")
                .unwrap()
                .to_bytes()
                .unwrap()
                .as_ref(),
            &[0, 1, 255]
        );
        assert!(metadata.get("user-agent").is_none());
        for pairs in [
            vec![pair("Bad:Key", b"v")],
            vec![pair("", b"v")],
            vec![pair("x-a", b"v\n")],
            vec![pair("x-a", &[0xc3, 0xa9])],
        ] {
            let status = call_metadata(&pairs).unwrap_err();
            assert_eq!(status.code(), tonic::Code::Internal);
            assert_eq!(status.message(), "Invalid metadata");
        }
    }

    #[test]
    fn the_metadata_of_a_call_is_sent_of_its_order_and_the_token_last() {
        // Of the review of SPARK-62220: grpcio sends every value of a key, in its order, and the
        // `authorization` of its call credentials after the metadata of the call.
        let pair = |key: &str, value: &str| (key.as_bytes().to_vec(), value.as_bytes().to_vec());
        let metadata = call_metadata(&[
            pair("x-seq", "2"),
            pair("authorization", "Bearer FROM-METADATA"),
            pair("x-dup", "1"),
            pair("x-dup", "2"),
        ])
        .unwrap();
        let token = (
            tonic::metadata::AsciiMetadataKey::from_static("authorization"),
            tonic::metadata::AsciiMetadataValue::from_static("Bearer TOKEN"),
        );
        let mut headers = http::HeaderMap::new();
        headers.insert("te", http::HeaderValue::from_static("trailers"));
        append_metadata(&mut headers, metadata, &[token]);
        let values = |key: &str| -> Vec<&str> {
            headers
                .get_all(key)
                .iter()
                .map(|value| value.to_str().unwrap())
                .collect()
        };
        assert_eq!(values("x-seq"), ["2"]);
        assert_eq!(values("x-dup"), ["1", "2"]);
        assert_eq!(
            values("authorization"),
            ["Bearer FROM-METADATA", "Bearer TOKEN"]
        );
        assert_eq!(values("te"), ["trailers"]);
    }

    #[test]
    fn the_keys_that_grpc_sets_itself_are_not_sent_of_a_source() {
        // Of the probes of grpcio 1.85.1: a call of `te`, `content-type`, `content-length` or
        // `grpc-timeout` of the metadata of a builder succeeds, of gRPC's own.
        let pair = |key: &str, value: &str| (key.as_bytes().to_vec(), value.as_bytes().to_vec());
        let metadata = call_metadata(&[
            pair("te", "foo"),
            pair("content-type", "text/plain"),
            pair("content-length", "5"),
            pair("grpc-timeout", "1n"),
            pair("grpc-status", "5"),
            pair("grpc-message", "injected"),
            pair("x-kept", "v"),
        ])
        .unwrap();
        let keys: Vec<&str> = metadata
            .keys()
            .map(|key| match key {
                tonic::metadata::KeyRef::Ascii(key) => key.as_str(),
                tonic::metadata::KeyRef::Binary(key) => key.as_str(),
            })
            .collect();
        assert_eq!(keys, ["x-kept"]);
        // A value that gRPC does not send fails the call all the same.
        let status = call_metadata(&[pair("te", "a\nb")]).unwrap_err();
        assert_eq!(status.message(), "Invalid metadata");
        assert!(sent_of_an_application("x-kept") && !sent_of_an_application("grpc-timeout"));
    }

    #[test]
    fn a_field_of_a_connection_fails_its_call_as_grpcs_server_resets_it() {
        // Of the probes: grpcio sends `connection`, `keep-alive`, `transfer-encoding` and
        // `upgrade`, of which the server resets the stream (RFC 9113, 8.2.2).
        let mut headers = http::HeaderMap::new();
        headers.insert("connection", http::HeaderValue::from_static("close"));
        assert!(of_connection_fields(&headers, false).is_none());
        let status = of_connection_fields(&headers, true).unwrap();
        assert_eq!(status.code(), tonic::Code::Internal);
        assert_eq!(
            status.message(),
            "Stream removed (RST_STREAM (Received RST_STREAM with error code 1))"
        );
        assert!(of_connection_fields(&http::HeaderMap::new(), true).is_none());
    }

    #[test]
    fn the_error_of_a_source_is_the_error_of_its_call() {
        let error = crate::error::SparkError::builtin_value_msg("of the source");
        let mut status = tonic::Status::unknown(error.to_string());
        status.set_source(std::sync::Arc::new(MetadataFailed(error)));
        let failure = metadata_failure(&status).unwrap();
        assert!(failure.to_string().contains("of the source"));
        assert!(metadata_failure(&tonic::Status::unknown("other")).is_none());
    }
}
