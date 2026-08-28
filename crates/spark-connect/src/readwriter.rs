//! DataFrameReader and DataFrameWriter implementation mirroring `pyspark.sql.connect.readwriter`.
//!
//! Provides the API for reading data from various sources and writing DataFrames to files/tables.

use std::collections::HashMap;

use spark_connect_core::error::Result;
use spark_connect_proto as proto;

use crate::column::Column;
use crate::dataframe::{build_input_relation, execute_command, DataFrame};
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

    /// Read the CDC changes of a named table. Mirrors `DataFrameReader.changes`.
    pub fn changes(self, table_name: &str) -> DataFrame {
        let plan = LogicalPlan::RelationChanges {
            table_name: table_name.to_string(),
            options: self.options.clone(),
            is_streaming: None,
        };
        DataFrame::new(self.session, plan)
    }

    /// Read from a named table.
    pub fn table(self, table_name: &str) -> DataFrame {
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

    /// Read XML file(s). Mirrors `DataFrameReader.xml`.
    pub fn xml(mut self, path: &str) -> DataFrame {
        self.format = Some("xml".to_string());
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
    cluster_cols: Vec<String>,
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
            cluster_cols: vec![],
            bucket_cols: vec![],
            sort_cols: vec![],
            num_buckets: None,
        }
    }

    /// Cluster the output by the given columns (liquid clustering).
    pub fn cluster_by<S: Into<String>>(mut self, cols: impl IntoIterator<Item = S>) -> Self {
        self.cluster_cols = cols.into_iter().map(Into::into).collect();
        self
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
    pub fn partition_by<S: Into<String>>(mut self, cols: impl IntoIterator<Item = S>) -> Self {
        self.partition_cols = cols.into_iter().map(Into::into).collect();
        self
    }

    /// Set the bucketing specification.
    pub fn bucket_by<S: Into<String>>(
        mut self,
        num_buckets: i32,
        cols: impl IntoIterator<Item = S>,
    ) -> Self {
        self.num_buckets = Some(num_buckets);
        self.bucket_cols = cols.into_iter().map(Into::into).collect();
        self
    }

    /// Set the sort columns.
    pub fn sort_by<S: Into<String>>(mut self, cols: impl IntoIterator<Item = S>) -> Self {
        self.sort_cols = cols.into_iter().map(Into::into).collect();
        self
    }

    /// Build the `WriteOperation` command proto for this writer.
    pub(crate) fn build_write_operation(
        &self,
        save_type: Option<proto::write_operation::SaveType>,
    ) -> Result<proto::WriteOperation> {
        let mut op = proto::WriteOperation::default();
        op.input = Some(build_input_relation(&self.input_plan, &self.session)?);
        op.source = self.format.clone();
        op.mode = self.mode.to_proto();
        op.sort_column_names = self.sort_cols.clone();
        op.partitioning_columns = self.partition_cols.clone();
        op.clustering_columns = self.cluster_cols.clone();
        op.options = self.options.clone();
        op.save_type = save_type;
        if let Some(num_buckets) = self.num_buckets {
            let mut bucket_by = proto::write_operation::BucketBy::default();
            bucket_by.num_buckets = num_buckets;
            bucket_by.bucket_column_names = self.bucket_cols.clone();
            op.bucket_by = Some(bucket_by);
        }
        Ok(op)
    }

    fn save_table(self, table_name: &str, method: TableSaveMethod) -> Result<()> {
        let mut table = proto::write_operation::SaveTable::default();
        table.table_name = table_name.to_string();
        table.save_method = method.to_proto();
        let op =
            self.build_write_operation(Some(proto::write_operation::SaveType::Table(table)))?;
        execute_command(
            &self.session,
            proto::command::CommandType::WriteOperation(op),
        )
    }

    /// Save the DataFrame to a file path (or to a path-less sink such as `noop`).
    pub fn save(self, path: Option<&str>) -> Result<()> {
        let save_type = path.map(|p| proto::write_operation::SaveType::Path(p.to_string()));
        let op = self.build_write_operation(save_type)?;
        execute_command(
            &self.session,
            proto::command::CommandType::WriteOperation(op),
        )
    }

    /// Save the DataFrame as a managed table.
    pub fn save_as_table(self, table_name: &str) -> Result<()> {
        self.save_table(table_name, TableSaveMethod::SaveAsTable)
    }

    /// Insert the DataFrame into an existing table.
    pub fn insert_into(self, table_name: &str) -> Result<()> {
        self.save_table(table_name, TableSaveMethod::InsertInto)
    }

    /// Write as JSON.
    pub fn json(mut self, path: &str) -> Result<()> {
        self.format = Some("json".to_string());
        self.save(Some(path))
    }

    /// Write as Parquet.
    pub fn parquet(mut self, path: &str) -> Result<()> {
        self.format = Some("parquet".to_string());
        self.save(Some(path))
    }

    /// Write as CSV.
    pub fn csv(mut self, path: &str) -> Result<()> {
        self.format = Some("csv".to_string());
        self.save(Some(path))
    }

    /// Write as ORC.
    pub fn orc(mut self, path: &str) -> Result<()> {
        self.format = Some("orc".to_string());
        self.save(Some(path))
    }

    /// Write as text.
    pub fn text(mut self, path: &str) -> Result<()> {
        self.format = Some("text".to_string());
        self.save(Some(path))
    }

    /// Write as XML. Mirrors `DataFrameWriter.xml`.
    pub fn xml(mut self, path: &str) -> Result<()> {
        self.format = Some("xml".to_string());
        self.save(Some(path))
    }
}

/// DataFrameWriterV2 for the v2 write API (`DataFrame.writeTo`).
///
/// Mirrors `pyspark.sql.connect.readwriter.DataFrameWriterV2`.
pub struct DataFrameWriterV2 {
    session: SparkSession,
    input_plan: LogicalPlan,
    table_name: String,
    provider: Option<String>,
    options: HashMap<String, String>,
    table_properties: HashMap<String, String>,
    partition_cols: Vec<Column>,
    cluster_cols: Vec<String>,
}

impl DataFrameWriterV2 {
    /// Create a new DataFrameWriterV2 targeting `table_name`.
    pub(crate) fn new(session: SparkSession, input_plan: LogicalPlan, table_name: &str) -> Self {
        DataFrameWriterV2 {
            session,
            input_plan,
            table_name: table_name.to_string(),
            provider: None,
            options: HashMap::new(),
            table_properties: HashMap::new(),
            partition_cols: vec![],
            cluster_cols: vec![],
        }
    }

    /// Cluster the output table by the given columns (liquid clustering).
    pub fn cluster_by<S: Into<String>>(mut self, cols: impl IntoIterator<Item = S>) -> Self {
        self.cluster_cols = cols.into_iter().map(Into::into).collect();
        self
    }

    /// Specify the underlying output data source provider (e.g. "parquet").
    pub fn using(mut self, provider: &str) -> Self {
        self.provider = Some(provider.to_string());
        self
    }

    /// Add a write option.
    pub fn option(mut self, key: &str, value: &str) -> Self {
        self.options.insert(key.to_string(), value.to_string());
        self
    }

    /// Add multiple write options.
    pub fn options(mut self, options: HashMap<String, String>) -> Self {
        self.options.extend(options);
        self
    }

    /// Add a table property.
    pub fn table_property(mut self, property: &str, value: &str) -> Self {
        self.table_properties
            .insert(property.to_string(), value.to_string());
        self
    }

    /// Partition the output table by the given columns.
    pub fn partition_by<C: Into<Column>>(mut self, columns: impl IntoIterator<Item = C>) -> Self {
        self.partition_cols = columns.into_iter().map(Into::into).collect();
        self
    }

    /// Build the `WriteOperationV2` command proto for the given mode.
    pub(crate) fn build_operation(
        &self,
        mode: proto::write_operation_v2::Mode,
        overwrite_condition: Option<proto::Expression>,
    ) -> Result<proto::WriteOperationV2> {
        let mut op = proto::WriteOperationV2::default();
        op.input = Some(build_input_relation(&self.input_plan, &self.session)?);
        op.table_name = self.table_name.clone();
        op.provider = self.provider.clone();
        op.partitioning_columns = self.partition_cols.iter().map(|c| c.to_proto()).collect();
        op.clustering_columns = self.cluster_cols.clone();
        op.options = self.options.clone();
        op.table_properties = self.table_properties.clone();
        op.mode = mode as i32;
        op.overwrite_condition = overwrite_condition;
        Ok(op)
    }

    fn execute(self, mode: proto::write_operation_v2::Mode) -> Result<()> {
        let op = self.build_operation(mode, None)?;
        execute_command(
            &self.session,
            proto::command::CommandType::WriteOperationV2(op),
        )
    }

    /// Create a new table from the DataFrame.
    pub fn create(self) -> Result<()> {
        self.execute(proto::write_operation_v2::Mode::Create)
    }

    /// Replace an existing table with the DataFrame.
    pub fn replace(self) -> Result<()> {
        self.execute(proto::write_operation_v2::Mode::Replace)
    }

    /// Create the table, or replace it if it already exists.
    pub fn create_or_replace(self) -> Result<()> {
        self.execute(proto::write_operation_v2::Mode::CreateOrReplace)
    }

    /// Append the DataFrame's rows to the table.
    pub fn append(self) -> Result<()> {
        self.execute(proto::write_operation_v2::Mode::Append)
    }

    /// Overwrite rows matching `condition` with the DataFrame's rows.
    pub fn overwrite(self, condition: Column) -> Result<()> {
        let op = self.build_operation(
            proto::write_operation_v2::Mode::Overwrite,
            Some(condition.to_proto()),
        )?;
        execute_command(
            &self.session,
            proto::command::CommandType::WriteOperationV2(op),
        )
    }

    /// Overwrite all partitions touched by the DataFrame (dynamic overwrite).
    pub fn overwrite_partitions(self) -> Result<()> {
        self.execute(proto::write_operation_v2::Mode::OverwritePartitions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::SparkSession;

    // The gRPC channel connects lazily, so a session can be built offline for
    // tests that only construct request protos (no server round-trip).
    fn session() -> SparkSession {
        SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .expect("failed to build session")
    }

    #[test]
    fn v1_write_operation_to_path() {
        let spark = session();
        let df = spark.range(3).unwrap();
        let op = df
            .write()
            .format("parquet")
            .mode("overwrite")
            .option("compression", "snappy")
            .partition_by(vec!["a".to_string()])
            .build_write_operation(Some(proto::write_operation::SaveType::Path(
                "/tmp/out".to_string(),
            )))
            .unwrap();

        assert!(op.input.is_some());
        assert_eq!(op.source.as_deref(), Some("parquet"));
        assert_eq!(op.mode, SaveMode::Overwrite.to_proto());
        assert_eq!(
            op.options.get("compression").map(String::as_str),
            Some("snappy")
        );
        assert_eq!(op.partitioning_columns, vec!["a".to_string()]);
        match op.save_type {
            Some(proto::write_operation::SaveType::Path(p)) => assert_eq!(p, "/tmp/out"),
            other => panic!("expected Path save_type, got {other:?}"),
        }
    }

    #[test]
    fn v1_write_operation_save_as_table() {
        let spark = session();
        let df = spark.range(3).unwrap();
        let mut table = proto::write_operation::SaveTable::default();
        table.table_name = "db.people".to_string();
        table.save_method = TableSaveMethod::SaveAsTable.to_proto();
        let op = df
            .write()
            .build_write_operation(Some(proto::write_operation::SaveType::Table(table)))
            .unwrap();

        match op.save_type {
            Some(proto::write_operation::SaveType::Table(t)) => {
                assert_eq!(t.table_name, "db.people");
                assert_eq!(t.save_method, TableSaveMethod::SaveAsTable.to_proto());
            }
            other => panic!("expected Table save_type, got {other:?}"),
        }
    }

    #[test]
    fn v2_write_operation_fields_and_modes() {
        let spark = session();
        let df = spark.range(3).unwrap();
        let op = df
            .write_to("db.tbl")
            .using("delta")
            .option("mergeSchema", "true")
            .table_property("owner", "eng")
            .partition_by(vec![crate::column::col("a")])
            .build_operation(proto::write_operation_v2::Mode::Append, None)
            .unwrap();

        assert!(op.input.is_some());
        assert_eq!(op.table_name, "db.tbl");
        assert_eq!(op.provider.as_deref(), Some("delta"));
        assert_eq!(op.mode, proto::write_operation_v2::Mode::Append as i32);
        assert_eq!(op.partitioning_columns.len(), 1);
        assert_eq!(
            op.options.get("mergeSchema").map(String::as_str),
            Some("true")
        );
        assert_eq!(
            op.table_properties.get("owner").map(String::as_str),
            Some("eng")
        );

        // Terminal modes map to the correct proto enum values.
        let create = df
            .write_to("t")
            .build_operation(proto::write_operation_v2::Mode::Create, None)
            .unwrap();
        assert_eq!(create.mode, 1);
        let cor = df
            .write_to("t")
            .build_operation(proto::write_operation_v2::Mode::CreateOrReplace, None)
            .unwrap();
        assert_eq!(cor.mode, 6);
    }

    #[test]
    fn v2_overwrite_sets_condition() {
        let spark = session();
        let df = spark.range(3).unwrap();
        let op = df
            .write_to("t")
            .build_operation(
                proto::write_operation_v2::Mode::Overwrite,
                Some(crate::column::col("id").to_proto()),
            )
            .unwrap();
        assert_eq!(op.mode, proto::write_operation_v2::Mode::Overwrite as i32);
        assert!(op.overwrite_condition.is_some());
    }

    #[test]
    fn v1_write_operation_cluster_by() {
        let spark = session();
        let df = spark.range(3).unwrap();
        let op = df
            .write()
            .format("parquet")
            .cluster_by(vec!["col1".to_string(), "col2".to_string()])
            .build_write_operation(Some(proto::write_operation::SaveType::Path(
                "/tmp/out".to_string(),
            )))
            .unwrap();

        assert_eq!(op.clustering_columns, vec!["col1", "col2"]);
    }

    #[test]
    fn v2_write_operation_cluster_by() {
        let spark = session();
        let df = spark.range(3).unwrap();
        let op = df
            .write_to("t")
            .cluster_by(vec!["col1".to_string(), "col2".to_string()])
            .build_operation(proto::write_operation_v2::Mode::Create, None)
            .unwrap();

        assert_eq!(op.clustering_columns, vec!["col1", "col2"]);
    }

    #[test]
    fn reader_jdbc_with_options() {
        let spark = session();
        let reader = spark.read();
        let df = reader
            .option("url", "jdbc:mysql://localhost:3306/db")
            .option("user", "root")
            .option("password", "secret")
            .jdbc("jdbc:mysql://localhost:3306/db", "table_name", None);

        match &df.plan {
            LogicalPlan::Read {
                read_type:
                    ReadType::DataSource {
                        options, format, ..
                    },
                ..
            } => {
                assert_eq!(format.as_deref(), Some("jdbc"));
                assert_eq!(
                    options.get("url").map(String::as_str),
                    Some("jdbc:mysql://localhost:3306/db")
                );
                assert_eq!(options.get("user").map(String::as_str), Some("root"));
                assert_eq!(options.get("password").map(String::as_str), Some("secret"));
            }
            _ => panic!("expected Read plan"),
        }
    }

    #[test]
    fn reader_jdbc_with_predicates() {
        let spark = session();
        let reader = spark.read();
        let predicates = vec!["col1 > 10".to_string(), "col2 = 'value'".to_string()];
        let df = reader.jdbc(
            "jdbc:mysql://localhost/db",
            "table",
            Some(predicates.clone()),
        );

        match &df.plan {
            LogicalPlan::Read {
                read_type:
                    ReadType::DataSource {
                        predicates: preds, ..
                    },
                ..
            } => {
                assert_eq!(preds.len(), 2);
            }
            _ => panic!("expected Read plan with predicates"),
        }
    }

    #[test]
    fn v1_write_partition_and_cluster() {
        let spark = session();
        let df = spark.range(3).unwrap();
        let op = df
            .write()
            .format("delta")
            .partition_by(vec!["date".to_string()])
            .cluster_by(vec!["user_id".to_string()])
            .build_write_operation(Some(proto::write_operation::SaveType::Path(
                "/tmp/data".to_string(),
            )))
            .unwrap();

        assert_eq!(op.partitioning_columns, vec!["date"]);
        assert_eq!(op.clustering_columns, vec!["user_id"]);
    }

    #[test]
    fn v1_write_bucket_by() {
        let spark = session();
        let df = spark.range(3).unwrap();
        let op = df
            .write()
            .format("parquet")
            .bucket_by(10, vec!["col1".to_string()])
            .build_write_operation(Some(proto::write_operation::SaveType::Path(
                "/tmp/out".to_string(),
            )))
            .unwrap();

        assert!(op.bucket_by.is_some());
        let bucket_by = op.bucket_by.unwrap();
        assert_eq!(bucket_by.num_buckets, 10);
        assert_eq!(bucket_by.bucket_column_names, vec!["col1"]);
    }

    #[test]
    fn v1_write_sort_by() {
        let spark = session();
        let df = spark.range(3).unwrap();
        let op = df
            .write()
            .format("parquet")
            .sort_by(vec!["col1".to_string()])
            .build_write_operation(Some(proto::write_operation::SaveType::Path(
                "/tmp/out".to_string(),
            )))
            .unwrap();

        assert_eq!(op.sort_column_names, vec!["col1"]);
    }

    #[test]
    fn reader_format_options() {
        let spark = session();
        let mut opts = std::collections::HashMap::new();
        opts.insert("delimiter".to_string(), ";".to_string());
        opts.insert("header".to_string(), "true".to_string());

        let df = spark
            .read()
            .format("csv")
            .options(opts)
            .load(Some("/data.csv"));

        match &df.plan {
            LogicalPlan::Read {
                read_type:
                    ReadType::DataSource {
                        options, format, ..
                    },
                ..
            } => {
                assert_eq!(format.as_deref(), Some("csv"));
                assert_eq!(options.get("delimiter").map(String::as_str), Some(";"));
                assert_eq!(options.get("header").map(String::as_str), Some("true"));
            }
            _ => panic!("expected Read plan"),
        }
    }
}
