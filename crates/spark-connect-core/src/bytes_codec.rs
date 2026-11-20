//! A passthrough gRPC codec that treats message bodies as raw bytes.
//!
//! The Spark Connect requests our transport carries are already-serialized protobuf
//! (built by the reference client). Re-decoding them with prost imposes a recursion
//! limit that deeply nested plans exceed. This codec sends/receives the raw bytes
//! unchanged, so the server (which has no such client-side limit) handles them.

use std::marker::PhantomData;
use std::sync::Arc;

use tonic::codec::{Codec, DecodeBuf, Decoder, EncodeBuf, Encoder};
use tonic::Status;

#[derive(Default, Clone)]
pub struct BytesCodec;

pub struct BytesEncoder;
pub struct BytesDecoder;

impl Codec for BytesCodec {
    type Encode = Vec<u8>;
    type Decode = Vec<u8>;
    type Encoder = BytesEncoder;
    type Decoder = BytesDecoder;

    fn encoder(&mut self) -> Self::Encoder {
        BytesEncoder
    }

    fn decoder(&mut self) -> Self::Decoder {
        BytesDecoder
    }
}

impl Encoder for BytesEncoder {
    type Item = Vec<u8>;
    type Error = Status;

    fn encode(&mut self, item: Self::Item, dst: &mut EncodeBuf<'_>) -> Result<(), Self::Error> {
        use bytes::BufMut;
        dst.put_slice(&item);
        Ok(())
    }
}

impl Decoder for BytesDecoder {
    type Item = Vec<u8>;
    type Error = Status;

    fn decode(&mut self, src: &mut DecodeBuf<'_>) -> Result<Option<Self::Item>, Self::Error> {
        use bytes::Buf;
        let remaining = src.remaining();
        let mut out = vec![0u8; remaining];
        src.copy_to_slice(&mut out);
        Ok(Some(out))
    }
}

/// [`BytesCodec`] of a client whose responses, of the type `U`, nest within the limit of the
/// responses of the client, as those of the generated client do
/// (`spark_connect_proto::ClientCodec`): a deeper response fails with the status of prost's error
/// of the limit, which the client tags as its own (`spark_connect_proto::ClientDecodeError`), as
/// gRPC's deserializer of PySpark's client fails it where the bytes are parsed.
pub struct ClientBytesCodec<U>(PhantomData<fn() -> U>);

impl<U> Default for ClientBytesCodec<U> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// The decoder of [`ClientBytesCodec`].
pub struct ClientBytesDecoder<U>(PhantomData<fn() -> U>);

impl<U: prost::Name + Send + 'static> Codec for ClientBytesCodec<U> {
    type Encode = Vec<u8>;
    type Decode = Vec<u8>;
    type Encoder = BytesEncoder;
    type Decoder = ClientBytesDecoder<U>;

    fn encoder(&mut self) -> Self::Encoder {
        BytesEncoder
    }

    fn decoder(&mut self) -> Self::Decoder {
        ClientBytesDecoder(PhantomData)
    }
}

impl<U: prost::Name> Decoder for ClientBytesDecoder<U> {
    type Item = Vec<u8>;
    type Error = Status;

    fn decode(&mut self, src: &mut DecodeBuf<'_>) -> Result<Option<Self::Item>, Self::Error> {
        let out = BytesDecoder.decode(src)?.unwrap_or_default();
        if let Err(too_deep) = spark_connect_proto::response_depth::<U>(&out) {
            let mut status = Status::internal(too_deep.to_string());
            if too_deep.is_kept() {
                status.set_source(Arc::new(too_deep));
            } else {
                status.set_source(Arc::new(spark_connect_proto::ClientDecodeError));
            }
            return Err(status);
        }
        Ok(Some(out))
    }
}
