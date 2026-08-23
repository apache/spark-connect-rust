//! Spark Connect client transport layer.
//!
//! Mirrors `pyspark.sql.connect.client.*`: connection-string parsing and channel
//! configuration ([`channel`]), the gRPC client, retries, reattach, artifacts,
//! and structured errors ([`error`]).

pub mod artifact;
pub mod bytes_codec;
pub mod channel;
pub mod client;
pub mod error;
pub mod reattach;
pub mod retries;
pub mod runtime;

pub use artifact::{
    build_artifact_request_stream, compute_crc32, ArtifactData, FileArtifact, InMemoryArtifact,
    NamedArtifact, CHUNK_SIZE,
};
pub use channel::ChannelBuilder;
pub use client::SparkConnectClient;
pub use error::{QueryContext, QueryContextType, Result, SparkError, SparkErrorKind};
pub use reattach::ExecutePlanResponseReattachableIterator;
pub use retries::{RetryPolicy, RetryPolicyState, DEFAULT_MAX_RETRY_EXCEPTION_ELAPSED_TIME};
pub use runtime::{block_on, get_runtime};
