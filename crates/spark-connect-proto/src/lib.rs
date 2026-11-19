//! Generated Spark Connect protobuf and gRPC types.
//!
//! All messages live in the `spark.connect` protobuf package and are re-exported
//! here at the crate root. The generated gRPC client is
//! [`spark_connect_service_client::SparkConnectServiceClient`].
//! Enable the optional `server` feature for `spark_connect_service_server`
//! bindings. Embeddings configure tonic's transport server features separately.
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
