//! The codec of the generated gRPC client, which tags the status of a response that the client
//! fails to decode as the client's own.

use std::marker::PhantomData;
use std::sync::Arc;

use tonic::codec::{BufferSettings, Codec, DecodeBuf, Decoder};
use tonic::Status;
use tonic_prost::{ProstDecoder, ProstEncoder};

/// The codec of [`crate::spark_connect_service_client::SparkConnectServiceClient`]: prost's, as
/// `tonic_prost::ProstCodec` encodes and decodes, but that the status of a response that does not
/// decode, tonic-prost's `INTERNAL` of prost's error, has the source [`ClientDecodeError`]. A
/// server that fails to decode a request sends a status of the same code and text, of no source,
/// so [`is_client_decode_error`] tells the two apart.
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
    U: prost::Message + Default + Send + 'static,
{
    type Encode = T;
    type Decode = U;
    type Encoder = ProstEncoder<T>;
    type Decoder = ClientDecoder<U>;

    fn encoder(&mut self) -> Self::Encoder {
        ProstEncoder::new(BufferSettings::default())
    }

    fn decoder(&mut self) -> Self::Decoder {
        ClientDecoder(ProstDecoder::new(BufferSettings::default()))
    }
}

/// The decoder of [`ClientCodec`]: tonic-prost's, of the status of its error tagged.
#[derive(Debug, Clone)]
pub struct ClientDecoder<U>(ProstDecoder<U>);

impl<U: prost::Message + Default> Decoder for ClientDecoder<U> {
    type Item = U;
    type Error = Status;

    fn decode(&mut self, buf: &mut DecodeBuf<'_>) -> Result<Option<Self::Item>, Self::Error> {
        self.0.decode(buf).map_err(|mut status| {
            status.set_source(Arc::new(ClientDecodeError));
            status
        })
    }

    fn buffer_settings(&self) -> BufferSettings {
        self.0.buffer_settings()
    }
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
