//! Spark Connect client transport layer.
//!
//! Mirrors `pyspark.sql.connect.client.*`: connection-string parsing and channel
//! configuration ([`channel`]), the gRPC client, retries, reattach, artifacts,
//! and structured errors ([`error`]).

pub mod artifact;
pub mod bytes_codec;
pub mod channel;
pub mod client;
pub mod client_log;
mod connector;
pub mod error;
#[doc(hidden)]
pub mod fork;
pub mod grpc_log;
mod metadata;
mod proxy;
pub mod reattach;
pub mod retries;
mod rpc_status;
pub mod runtime;
// Shared with `apache-spark-connect`, whose session keeps per-thread tags the same way.
#[doc(hidden)]
pub mod thread_entries;

pub use artifact::{
    build_artifact_request_stream, compute_crc32, ArtifactData, FileArtifact, InMemoryArtifact,
    NamedArtifact, CHUNK_SIZE,
};
pub use channel::ChannelBuilder;
pub use client::{ExecutePlanHook, ResultChunking, RpcDeadlines, SparkConnectClient};
#[doc(hidden)]
pub use connector::set_grpcio_dialect;
pub use error::{QueryContext, QueryContextType, Result, SparkError, SparkErrorKind};
pub use reattach::ExecutePlanResponseReattachableIterator;
pub use retries::{RetryPolicy, RetryPolicyState, DEFAULT_MAX_RETRY_EXCEPTION_ELAPSED_TIME};
pub use runtime::{block_on, get_runtime};
