//! Generated Spark Connect protobuf and gRPC types.
//!
//! All messages live in the `spark.connect` protobuf package and are re-exported
//! here at the crate root. The generated gRPC client is
//! [`spark_connect_service_client::SparkConnectServiceClient`], of the codec [`ClientCodec`],
//! which tags the status of a response that the client fails to decode, and decodes a response
//! only within the limit of 100 levels of nested messages of prost's decoder. Each message
//! implements `prost::Name`, of its full name in the `spark.connect` package.
//! Enable the optional `server` feature for `spark_connect_service_server`
//! bindings, which decode the requests as prost does, of no limit of the client's.
//! Embeddings configure tonic's transport server features separately.
//!
//! Baseline protocol version: see `proto/PROTO_VERSION.txt` / `proto/SPARK_SHA.txt`.

#![allow(clippy::large_enum_variant)] // Generated protobuf code has large enum variants

// tonic-build emits `google.protobuf` well-known types referenced by the schema.
pub mod spark {
    pub mod connect {
        tonic::include_proto!("spark.connect");
    }
}

pub use spark::connect::*;

/// The encoded `google.protobuf.FileDescriptorSet` of the Spark Connect protos and of the protos
/// they import, such as `google/protobuf/any.proto`, without their source info. The Python
/// drop-in builds the messages of `pyspark.sql.connect.proto` from it when the `protobuf`
/// package is not installed.
pub const FILE_DESCRIPTOR_SET: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/spark_connect_descriptor_set.bin"
));

mod codec;
mod depth;

pub use codec::{is_client_decode_error, ClientCodec, ClientDecodeError, ClientDecoder};
#[doc(hidden)]
pub use depth::{
    decode_status_detail, depth_limit, kept_too_deep, probed_depth_limit, response_depth,
    response_depth_of, set_depth_limit, set_depth_probe, Depth, TooDeep, DEFAULT_DEPTH_LIMIT,
    MAX_KEPT_DEPTH,
};
