//! DataFrameReader and DataFrameWriter implementation mirroring `pyspark.sql.connect.readwriter`.
//!
//! Provides the API for reading data from various sources and writing DataFrames to files/tables.

use std::collections::HashMap;

use crate::dataframe::DataFrame;
use crate::plan::LogicalPlan;
use crate::session::SparkSession;

/// DataFrameReader for reading data from various sources.
///
/// Mirrors `pyspark.sql.connect.readwriter.DataFrameReader`.
pub struct DataFrameReader {
    session: SparkSession,
    format: Option<String>,
    schema: String,
    options: HashMap<String, String>,
}

impl DataFrameReader {
    /// Create a new DataFrameReader.
    pub(crate) fn new(session: SparkSession) -> Self {
        DataFrameReader {
            session,
            format: None,
            schema: String::new(),
            options: HashMap::new(),
        }
    }

    /// Set the format/source type (e.g., "json", "parquet", "csv").
    pub fn format(mut self, source: &str) -> Self {
        self.format = Some(source.to_string());
        self
    }

    /// Set the schema from a DDL string or JSON string.
    pub fn schema(mut self, schema: String) -> Self {
        self.schema = schema;
        self
    }

    /// Set a single option key-value pair.
    pub fn option(mut self, key: &str, value: &str) -> Self {
        self.options.insert(key.to_string(), value.to_string());
        self
    }

    /// Set multiple options.
    pub fn options(mut self, options: HashMap<String, String>) -> Self {
        self.options.extend(options);
        self
    }

    /// Load data from the specified path(s) with the configured format, schema, and options.
    pub fn load(self, path: Option<&str>) -> DataFrame {
        let paths = path.map(|p| vec![p.to_string()]);
        let plan = LogicalPlan::Read {
            read_type: ReadType::DataSource {
                format: self.format.clone(),
                schema: if self.schema.is_empty() {
                    None
                } else {
                    Some(self.schema.clone())
                },
                options: self.options.clone(),
                paths: paths.unwrap_or_default(),
                predicates: vec![],
                source_name: None,
            },
            is_streaming: false,
        };
        DataFrame::new(self.session, plan)
    }

    /// Read from a named table.
    pub fn table(mut self, table_name: &str) -> DataFrame {
        let plan = LogicalPlan::Read {
            read_type: ReadType::NamedTable {
                table_name: table_name.to_string(),
                options: self.options.clone(),
            },
            is_streaming: false,
        };
        DataFrame::new(self.session, plan)
    }

    /// Read JSON data.
    pub fn json(mut self, path: &str) -> DataFrame {
        self.format = Some("json".to_string());
        let paths = vec![path.to_string()];
        let plan = LogicalPlan::Read {
            read_type: ReadType::DataSource {
                format: self.format.clone(),
                schema: if self.schema.is_empty() {
                    None
                } else {
                    Some(self.schema.clone())
                },
                options: self.options.clone(),
                paths,
                predicates: vec![],
                source_name: None,
            },
            is_streaming: false,
        };
        DataFrame::new(self.session, plan)
    }

    /// Read Parquet data.
    pub fn parquet(mut self, path: &str) -> DataFrame {
        self.format = Some("parquet".to_string());
        let paths = vec![path.to_string()];
        let plan = LogicalPlan::Read {
            read_type: ReadType::DataSource {
                format: self.format.clone(),
                schema: if self.schema.is_empty() {
                    None
                } else {
                    Some(self.schema.clone())
                },
                options: self.options.clone(),
                paths,
                predicates: vec![],
                source_name: None,
            },
            is_streaming: false,
        };
        DataFrame::new(self.session, plan)
    }

    /// Read CSV data.
    pub fn csv(mut self, path: &str) -> DataFrame {
        self.format = Some("csv".to_string());
        let paths = vec![path.to_string()];
        let plan = LogicalPlan::Read {
            read_type: ReadType::DataSource {
                format: self.format.clone(),
                schema: if self.schema.is_empty() {
                    None
                } else {
                    Some(self.schema.clone())
                },
                options: self.options.clone(),
                paths,
                predicates: vec![],
                source_name: None,
            },
            is_streaming: false,
        };
        DataFrame::new(self.session, plan)
    }

    /// Read ORC data.
    pub fn orc(mut self, path: &str) -> DataFrame {
        self.format = Some("orc".to_string());
        let paths = vec![path.to_string()];
        let plan = LogicalPlan::Read {
            read_type: ReadType::DataSource {
                format: self.format.clone(),
                schema: if self.schema.is_empty() {
                    None
                } else {
                    Some(self.schema.clone())
                },
                options: self.options.clone(),
                paths,
                predicates: vec![],
                source_name: None,
            },
            is_streaming: false,
        };
        DataFrame::new(self.session, plan)
    }

    /// Read text data.
    pub fn text(mut self, path: &str) -> DataFrame {
        self.format = Some("text".to_string());
        let paths = vec![path.to_string()];
        let plan = LogicalPlan::Read {
            read_type: ReadType::DataSource {
                format: self.format.clone(),
                schema: if self.schema.is_empty() {
                    None
                } else {
                    Some(self.schema.clone())
                },
                options: self.options.clone(),
                paths,
                predicates: vec![],
                source_name: None,
            },
            is_streaming: false,
        };
        DataFrame::new(self.session, plan)
    }

    /// Read from JDBC data source.
    pub fn jdbc(mut self, url: &str, table: &str, predicates: Option<Vec<String>>) -> DataFrame {
        self.format = Some("jdbc".to_string());
        self.options.insert("url".to_string(), url.to_string());
        self.options
            .insert("dbtable".to_string(), table.to_string());

        let plan = LogicalPlan::Read {
            read_type: ReadType::DataSource {
                format: self.format.clone(),
                schema: if self.schema.is_empty() {
                    None
                } else {
                    Some(self.schema.clone())
                },
                options: self.options.clone(),
                paths: vec![],
                predicates: predicates.unwrap_or_default(),
                source_name: None,
            },
            is_streaming: false,
        };
        DataFrame::new(self.session, plan)
    }
}

/// ReadType variant for Read relation.
#[derive(Debug, Clone)]
pub enum ReadType {
    /// DataSource read (file-based or other formats).
    DataSource {
        format: Option<String>,
        schema: Option<String>,
        options: HashMap<String, String>,
        paths: Vec<String>,
        predicates: Vec<String>,
        source_name: Option<String>,
    },
    /// Named table read.
    NamedTable {
        table_name: String,
        options: HashMap<String, String>,
    },
}

/// SaveMode for write operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaveMode {
    Append,
    Overwrite,
    ErrorIfExists,
    Ignore,
}

impl SaveMode {
    /// Convert SaveMode to proto i32 value.
    pub fn to_proto(&self) -> i32 {
        match self {
            SaveMode::Append => 1i32,
            SaveMode::Overwrite => 2i32,
            SaveMode::ErrorIfExists => 3i32,
            SaveMode::Ignore => 4i32,
        }
    }

    /// Parse SaveMode from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "append" => Some(SaveMode::Append),
            "overwrite" => Some(SaveMode::Overwrite),
            "error" | "errorifexists" => Some(SaveMode::ErrorIfExists),
            "ignore" => Some(SaveMode::Ignore),
            _ => None,
        }
    }
}

/// TableSaveMethod for WriteOperation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableSaveMethod {
    SaveAsTable,
    InsertInto,
}

impl TableSaveMethod {
    /// Convert to proto i32 value.
    pub fn to_proto(&self) -> i32 {
        match self {
            TableSaveMethod::SaveAsTable => 1i32,
            TableSaveMethod::InsertInto => 2i32,
        }
    }
}

/// DataFrameWriter for writing DataFrames to various destinations.
///
/// Mirrors `pyspark.sql.connect.readwriter.DataFrameWriter`.
pub struct DataFrameWriter {
    session: SparkSession,
    input_plan: LogicalPlan,
    format: Option<String>,
    mode: SaveMode,
    options: HashMap<String, String>,
    partition_cols: Vec<String>,
    bucket_cols: Vec<String>,
    sort_cols: Vec<String>,
    num_buckets: Option<i32>,
}

impl DataFrameWriter {
    /// Create a new DataFrameWriter.
    pub(crate) fn new(session: SparkSession, input_plan: LogicalPlan) -> Self {
        DataFrameWriter {
            session,
            input_plan,
            format: None,
            mode: SaveMode::ErrorIfExists,
            options: HashMap::new(),
            partition_cols: vec![],
            bucket_cols: vec![],
            sort_cols: vec![],
            num_buckets: None,
        }
    }

    /// Set the save mode.
    pub fn mode(mut self, mode: &str) -> Self {
        if let Some(m) = SaveMode::from_str(mode) {
            self.mode = m;
        }
        self
    }

    /// Set the format/source type.
    pub fn format(mut self, source: &str) -> Self {
        self.format = Some(source.to_string());
        self
    }

    /// Set a single option.
    pub fn option(mut self, key: &str, value: &str) -> Self {
        self.options.insert(key.to_string(), value.to_string());
        self
    }

    /// Set multiple options.
    pub fn options(mut self, options: HashMap<String, String>) -> Self {
        self.options.extend(options);
        self
    }

    /// Set the partition columns.
    pub fn partition_by(mut self, cols: Vec<String>) -> Self {
        self.partition_cols = cols;
        self
    }

    /// Set the bucketing specification.
    pub fn bucket_by(mut self, num_buckets: i32, cols: Vec<String>) -> Self {
        self.num_buckets = Some(num_buckets);
        self.bucket_cols = cols;
        self
    }

    /// Set the sort columns.
    pub fn sort_by(mut self, cols: Vec<String>) -> Self {
        self.sort_cols = cols;
        self
    }

    /// Save the DataFrame to a file path.
    ///
    /// This method builds a WriteOperation command and would submit it to the server.
    /// Currently implemented as a placeholder that does not execute remotely.
    pub fn save(self, _path: Option<&str>) {
        // In a full implementation, this would construct a proto::WriteOperation
        // command and submit it to the server via the gRPC client.
        // For now, we keep the signature for API parity.
    }

    /// Save the DataFrame as a table.
    ///
    /// This method builds a WriteOperation command and would submit it to the server.
    /// Currently implemented as a placeholder that does not execute remotely.
    pub fn save_as_table(self, _table_name: &str) {
        // In a full implementation, this would construct a proto::WriteOperation
        // command with SaveAsTable mode and submit it to the server.
    }

    /// Insert into a table.
    ///
    /// This method builds a WriteOperation command and would submit it to the server.
    /// Currently implemented as a placeholder that does not execute remotely.
    pub fn insert_into(self, _table_name: &str) {
        // In a full implementation, this would construct a proto::WriteOperation
        // command with InsertInto mode and submit it to the server.
    }

    /// Write as JSON.
    pub fn json(mut self, path: &str) {
        self.format = Some("json".to_string());
        self.save(Some(path));
    }

    /// Write as Parquet.
    pub fn parquet(mut self, path: &str) {
        self.format = Some("parquet".to_string());
        self.save(Some(path));
    }

    /// Write as CSV.
    pub fn csv(mut self, path: &str) {
        self.format = Some("csv".to_string());
        self.save(Some(path));
    }

    /// Write as ORC.
    pub fn orc(mut self, path: &str) {
        self.format = Some("orc".to_string());
        self.save(Some(path));
    }

    /// Write as text.
    pub fn text(mut self, path: &str) {
        self.format = Some("text".to_string());
        self.save(Some(path));
    }
}
