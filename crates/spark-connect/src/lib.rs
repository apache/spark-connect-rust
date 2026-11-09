//! Pure-Rust Spark Connect DataFrame client mirroring the PySpark API surface.
//!
//! This crate provides the core data types and operations for interacting with
//! Apache Spark via the Spark Connect protocol.
//!
//! # Synchronous API
//!
//! Every method is synchronous: it returns once the server has answered. The API may be called
//! from any thread, including inside a Tokio runtime of either flavor. The call blocks the
//! calling thread until it completes, as any blocking call does, so avoid it in a task that
//! must keep other tasks of the same runtime running. In a multi-thread runtime the thread is
//! blocked in place. In a `current_thread` runtime (the default of `#[tokio::test]` and of
//! `#[tokio::main(flavor = "current_thread")]`) the work runs on a thread of its own while the
//! calling thread waits, and it uses the tags and user context extensions of the calling thread.

#![allow(clippy::result_large_err)] // SparkError is large; changing the type would break the API

mod arrow_input;
pub mod catalog;
pub mod column;
pub mod conf;
pub mod dataframe;
pub mod datasource;
pub mod expression;
pub mod functions;
pub mod group;
pub mod merge;
pub mod ml;
pub mod observation;
pub mod origin;
pub mod pipelines;
pub mod plan;
pub mod profiler;
pub mod progress;
pub mod readwriter;
pub mod resource;
pub mod row;
pub mod session;
pub mod storage;
pub mod streaming;
pub mod table_arg;
pub mod tvf;
pub mod types;
pub mod udf;
/// Run Rust UDFs on Spark via WebAssembly. Requires the `wasm-udf` feature.
#[cfg(feature = "wasm-udf")]
pub mod wasm_udf;
pub mod window;

// Re-export commonly used types
#[doc(hidden)]
pub use arrow_input::ArrowInputOptions;
pub use column::{col, lit, lit_boolean, lit_double, lit_string, Column};
pub use dataframe::{DataFrame, LocalRowIterator};
pub use datasource::{CommonInlineUserDefinedDataSourceExpression, PythonDataSourcePayload};
pub use expression::{
    Alias, CaseWhen, Cast, ColumnReference, Expression, ExpressionNode, LiteralExpression,
    SortOrder, UnresolvedFunction,
};
pub use group::{CoGroupedData, GroupedData};
pub use merge::MergeIntoWriter;
pub use observation::Observation;
pub use progress::{ProgressUpdate, StageInfo};
pub use readwriter::{
    DataFrameReader, DataFrameWriter, DataFrameWriterV2, ReadType, SaveMode, TableSaveMethod,
};
pub use resource::{
    ExecutorResourceRequests, ResourceProfile, ResourceProfileBuilder, TaskResourceRequests,
};
pub use session::{SparkSession, SparkSessionBuilder};
pub use spark_connect_core::client::ExecutePlanHook;
/// Re-exported so fallible APIs (e.g. the `#[spark_wasm_udf]`-generated UDF
/// constructors) can name the client's `Result`/error types.
pub use spark_connect_core::error::{
    BreakingChangeInfo, QueryContext, QueryContextType, Result, ServerErrorInfo, SparkError,
    SparkErrorKind,
};
/// Re-exported so `persist(...)` / `storage_level()` have a public storage-level type.
pub use spark_connect_proto::StorageLevel;
/// Named `StorageLevel` presets (`MEMORY_AND_DISK`, `DISK_ONLY`, ...) mirroring `pyspark.StorageLevel`.
pub use storage::StorageLevelExt;
pub use streaming::{
    DataStreamReader, DataStreamWriter, StreamingQuery, StreamingQueryManager, Trigger,
};
pub use types::{
    DataType, DateType, DayTimeIntervalType, GeographyType, GeometryType, SpatialType, StructField,
    YearMonthIntervalType,
};
pub use window::{FrameBound, FrameType, Window, WindowSpec};
