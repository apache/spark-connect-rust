//! A passthrough gRPC codec that treats message bodies as raw bytes.
//!
//! The Spark Connect requests our transport carries are already-serialized protobuf
//! (built by the reference client). Re-decoding them with prost imposes a recursion
//! limit that deeply nested plans exceed. This codec sends/receives the raw bytes
//! unchanged, so the server (which has no such client-side limit) handles them.

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
