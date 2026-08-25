//! Generated Spark Connect protobuf and gRPC types.
//!
//! All messages live in the `spark.connect` protobuf package and are re-exported
//! here at the crate root. The generated gRPC client is
//! [`spark_connect_service_client::SparkConnectServiceClient`].
//!
//! Baseline protocol version: see `proto/PROTO_VERSION.txt` / `proto/SPARK_SHA.txt`.

// tonic-build emits `google.protobuf` well-known types referenced by the schema.
pub mod spark {
    pub mod connect {
        tonic::include_proto!("spark.connect");
    }
}

pub use spark::connect::*;
