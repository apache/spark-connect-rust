//! Pure-Rust Spark Connect DataFrame client mirroring the PySpark API surface.
//!
//! This crate provides the core data types and operations for interacting with
//! Apache Spark via the Spark Connect protocol.

pub mod catalog;
pub mod column;
pub mod conf;
pub mod dataframe;
pub mod expression;
pub mod functions;
pub mod group;
pub mod ml;
pub mod observation;
pub mod plan;
pub mod readwriter;
pub mod row;
pub mod session;
pub mod streaming;
pub mod table_arg;
pub mod tvf;
pub mod types;
pub mod udf;
pub mod window;

// Re-export commonly used types
pub use column::{col, lit, lit_boolean, lit_double, lit_string, Column};
pub use dataframe::DataFrame;
pub use expression::{
    Alias, CaseWhen, Cast, ColumnReference, Expression, LiteralExpression, SortOrder,
    UnresolvedFunction,
};
pub use observation::Observation;
pub use readwriter::{DataFrameReader, DataFrameWriter, ReadType, SaveMode, TableSaveMethod};
pub use session::{SparkSession, SparkSessionBuilder};
pub use streaming::{
    DataStreamReader, DataStreamWriter, StreamingQuery, StreamingQueryManager, Trigger,
};
pub use types::{DataType, StructField};
pub use window::{FrameBound, FrameType, Window, WindowSpec};
