//! Pure-Rust Spark Connect DataFrame client mirroring the PySpark API surface.
//!
//! This crate provides the core data types and operations for interacting with
//! Apache Spark via the Spark Connect protocol.

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
pub mod plan;
pub mod profiler;
pub mod readwriter;
pub mod resource;
pub mod row;
pub mod session;
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
pub use column::{col, lit, lit_boolean, lit_double, lit_string, Column};
pub use dataframe::{DataFrame, LocalRowIterator};
pub use datasource::{CommonInlineUserDefinedDataSourceExpression, PythonDataSourcePayload};
pub use expression::{
    Alias, CaseWhen, Cast, ColumnReference, Expression, LiteralExpression, SortOrder,
    UnresolvedFunction,
};
pub use group::{CoGroupedData, GroupedData};
pub use merge::MergeIntoWriter;
pub use observation::Observation;
pub use profiler::ProfilerCollector;
pub use readwriter::{
    DataFrameReader, DataFrameWriter, DataFrameWriterV2, ReadType, SaveMode, TableSaveMethod,
};
pub use resource::{
    ExecutorResourceRequests, ResourceProfile, ResourceProfileBuilder, TaskResourceRequests,
};
pub use session::{SparkSession, SparkSessionBuilder};
/// Re-exported so fallible APIs (e.g. the `#[spark_wasm_udf]`-generated UDF
/// constructors) can name the client's `Result`/error types.
pub use spark_connect_core::error::{Result, SparkError};
/// Re-exported so `persist(...)` / `storage_level()` have a public storage-level type.
pub use spark_connect_proto::StorageLevel;
pub use streaming::{
    DataStreamReader, DataStreamWriter, StreamingQuery, StreamingQueryManager, Trigger,
};
pub use types::{DataType, StructField};
pub use window::{FrameBound, FrameType, Window, WindowSpec};
