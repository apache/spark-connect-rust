//! The codec of the generated gRPC client, which tags the status of a response that the client
//! fails to decode as the client's own, and decodes a response within a limit of the nesting of
//! its messages.

use std::marker::PhantomData;
use std::sync::Arc;

use tonic::codec::{BufferSettings, Codec, DecodeBuf, Decoder};
use tonic::Status;
use tonic_prost::{ProstDecoder, ProstEncoder};

use crate::depth::{self, on_stack_of, TooDeep};

/// The codec of [`crate::spark_connect_service_client::SparkConnectServiceClient`]: prost's, as
/// `tonic_prost::ProstCodec` encodes and decodes, but that the status of a response that does not
/// decode, tonic-prost's `INTERNAL` of prost's error, has the source [`ClientDecodeError`]. A
/// server that fails to decode a request sends a status of the same code and text, of no source,
/// so [`is_client_decode_error`] tells the two apart.
///
/// A response whose messages nest more levels below it than the limit of the client (100, the
/// limit of prost's decoder, unless the Python binding follows upb's limit) fails so too, of
/// prost's error of its limit, whatever the features of prost in the build: of the levels counted
/// as upb and prost count them, before prost decodes it, which it then does on a stack that holds
/// its levels. Within a higher limit, a response whose kept messages (of the fields that their
/// types have) nest more than 1024 levels fails of the client's error of that bound ("maximum
/// recursion depth exceeded while decoding a response; ..."), which is the source of the status.
///
/// The limit is of the responses of the service that the client decodes. The generated server
/// (of the `server` feature), whose bindings share this codec, decodes the requests that it
/// receives as `tonic_prost::ProstCodec` does, within prost's own limit, of the features of prost
/// in the build.
#[derive(Debug, Clone)]
pub struct ClientCodec<T, U> {
    _types: PhantomData<(T, U)>,
}

impl<T, U> Default for ClientCodec<T, U> {
    fn default() -> Self {
        Self {
            _types: PhantomData,
        }
    }
}

impl<T, U> Codec for ClientCodec<T, U>
where
    T: prost::Message + Send + 'static,
    U: prost::Message + prost::Name + Default + Send + 'static,
{
    type Encode = T;
    type Decode = U;
    type Encoder = ProstEncoder<T>;
    type Decoder = ClientDecoder<U>;

    fn encoder(&mut self) -> Self::Encoder {
        ProstEncoder::new(BufferSettings::default())
    }

    fn decoder(&mut self) -> Self::Decoder {
        ClientDecoder {
            decoder: ProstDecoder::new(BufferSettings::default()),
            message: depth::response_type_index(&U::full_name()),
        }
    }
}

/// The decoder of [`ClientCodec`]: tonic-prost's, of the status of its error tagged, within the
/// limit of the nesting of a response.
#[derive(Debug, Clone)]
pub struct ClientDecoder<U> {
    decoder: ProstDecoder<U>,
    /// The index of the type of the messages in the table of the scan, if it is a response of the
    /// service, which the client decodes within the limit; none of a request, of the server.
    message: Option<u32>,
}

impl<U: prost::Message + Default> Decoder for ClientDecoder<U> {
    type Item = U;
    type Error = Status;

    fn decode(&mut self, buf: &mut DecodeBuf<'_>) -> Result<Option<Self::Item>, Self::Error> {
        use prost::bytes::Buf;
        let Some(message) = self.message else {
            return self.decoder.decode(buf).map_err(tagged);
        };
        let len = buf.remaining();
        if buf.chunk().len() < len {
            // Never of tonic's buffers, which hold a message in one piece: of a copy of it.
            let bytes = buf.copy_to_bytes(len);
            let depth = depth::scan(message, &bytes, crate::depth_limit()).map_err(too_deep)?;
            return on_stack_of(depth, move || U::decode(bytes))
                .map(Some)
                .map_err(|error| tagged(Status::internal(error.to_string())));
        }
        let depth =
            depth::scan(message, &buf.chunk()[..len], crate::depth_limit()).map_err(too_deep)?;
        let decoder = &mut self.decoder;
        on_stack_of(depth, move || decoder.decode(buf)).map_err(tagged)
    }

    fn buffer_settings(&self) -> BufferSettings {
        self.decoder.buffer_settings()
    }
}

/// The status of a response that nests past the limit: tonic-prost's of prost's error of it, which
/// the client tags as its own; of one that nests its kept messages past `MAX_KEPT_DEPTH`, of the
/// error as its source.
fn too_deep(error: TooDeep) -> Status {
    let mut status = Status::internal(error.to_string());
    if error.is_kept() {
        status.set_source(Arc::new(error));
        return status;
    }
    tagged(status)
}

fn tagged(mut status: Status) -> Status {
    status.set_source(Arc::new(ClientDecodeError));
    status
}

/// The source of the status of a response that the client failed to decode (see
/// [`ClientCodec`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClientDecodeError;

impl std::fmt::Display for ClientDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the client failed to decode the response")
    }
}

impl std::error::Error for ClientDecodeError {}

/// Whether `status` is that of a response that the client failed to decode, of [`ClientCodec`],
/// rather than one of the server.
pub fn is_client_decode_error(status: &Status) -> bool {
    std::iter::successors(std::error::Error::source(status), |error| error.source())
        .any(|error| error.is::<ClientDecodeError>())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AnalyzePlanRequest, AnalyzePlanResponse, ExecutePlanRequest, ExecutePlanResponse};

    #[test]
    fn the_client_scans_its_responses_and_the_server_decodes_its_requests_as_prost_does() {
        // The generated client decodes the responses (of `Decode` the response) within the limit;
        // the generated server, of the same codec, its requests (of `Decode` the request) of none.
        let mut client = ClientCodec::<ExecutePlanRequest, ExecutePlanResponse>::default();
        assert!(client.decoder().message.is_some());
        let mut client = ClientCodec::<AnalyzePlanRequest, AnalyzePlanResponse>::default();
        assert!(client.decoder().message.is_some());
        let mut server = ClientCodec::<ExecutePlanResponse, ExecutePlanRequest>::default();
        assert!(server.decoder().message.is_none());
        let mut server = ClientCodec::<AnalyzePlanResponse, AnalyzePlanRequest>::default();
        assert!(server.decoder().message.is_none());
    }
}
