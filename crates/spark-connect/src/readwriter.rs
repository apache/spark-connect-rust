//! DataFrameReader and DataFrameWriter implementation mirroring `pyspark.sql.connect.readwriter`.
//!
//! Provides the API for reading data from various sources and writing DataFrames to files/tables.

use std::collections::HashMap;
use std::sync::Arc;

use spark_connect_core::error::Result;
use spark_connect_proto as proto;

use crate::column::Column;
use crate::dataframe::{build_input_relation, DataFrame};
use crate::plan::LogicalPlan;
use crate::session::SparkSession;

/// DataFrameReader for reading data from various sources.
///
/// Mirrors `pyspark.sql.connect.readwriter.DataFrameReader`.
#[derive(Clone)]
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

    /// Unset the format, so that the server reads its default source
    /// (`spark.sql.sources.default`), as PySpark's `format(None)`.
    pub fn clear_format(mut self) -> Self {
        self.format = None;
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

    /// Unset a read option, as PySpark's `DataFrameReader.option(key, None)` does.
    pub fn unset_option(mut self, key: &str) -> Self {
        self.options.remove(key);
        self
    }

    /// Set multiple options.
    pub fn options(mut self, options: HashMap<String, String>) -> Self {
        self.options.extend(options);
        self
    }

    /// The configured source format, if any.
    pub fn get_format(&self) -> Option<&str> {
        self.format.as_deref()
    }

    /// The configured schema (a DDL/JSON string; empty when unset).
    pub fn get_schema(&self) -> &str {
        &self.schema
    }

    /// The configured read options.
    pub fn get_options(&self) -> &HashMap<String, String> {
        &self.options
    }

    /// Load data from the specified path(s) with the configured format, schema, and options.
    pub fn load(self, path: Option<&str>) -> DataFrame {
        self.load_paths(path.map(|p| vec![p.to_string()]).unwrap_or_default())
    }

    /// Load data from `paths`, mirroring `DataFrameReader.load` of a list of paths, with the
    /// configured format, schema, and options.
    pub fn load_paths(self, paths: Vec<String>) -> DataFrame {
        let paths = Some(paths);
        let plan = LogicalPlan::Read {
            read_type: ReadType::DataSource {
                format: self.format,
                schema: if self.schema.is_empty() {
                    None
                } else {
                    Some(self.schema)
                },
                options: self.options,
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
            options: self.options,
            is_streaming: None,
        };
        DataFrame::new(self.session, plan)
    }

    /// Parse the strings of `input`, a DataFrame of one string column, as `format`, with the
    /// schema and the options of this reader, mirroring `DataFrameReader.json`, `csv` and `xml`
    /// of a DataFrame: a schema of JSON, or else of DDL, which the server parses, as PySpark's
    /// `StructType.fromDDL`.
    pub fn parse(self, format: crate::plan::ParseFormat, input: &DataFrame) -> Result<DataFrame> {
        let schema = if self.schema.is_empty() {
            None
        } else if self.schema.starts_with('{') {
            Some(crate::types::DataType::from_json_str(&self.schema)?)
        } else {
            Some(self.session.parse_ddl(&self.schema)?)
        };
        let plan = LogicalPlan::Parse {
            input: Arc::clone(&input.plan),
            format,
            schema,
            options: self.options,
        };
        Ok(DataFrame::new(self.session, plan))
    }

    /// Read from a named table.
    pub fn table(self, table_name: &str) -> DataFrame {
        let plan = LogicalPlan::Read {
            read_type: ReadType::NamedTable {
                table_name: table_name.to_string(),
                options: self.options,
            },
            is_streaming: false,
        };
        DataFrame::new(self.session, plan)
    }

    /// Read JSON data.
    pub fn json(self, path: &str) -> DataFrame {
        self.format("json").load(Some(path))
    }

    /// Read Parquet data.
    pub fn parquet(self, path: &str) -> DataFrame {
        self.format("parquet").load(Some(path))
    }

    /// Read CSV data.
    pub fn csv(self, path: &str) -> DataFrame {
        self.format("csv").load(Some(path))
    }

    /// Read ORC data.
    pub fn orc(self, path: &str) -> DataFrame {
        self.format("orc").load(Some(path))
    }

    /// Read text data.
    pub fn text(self, path: &str) -> DataFrame {
        self.format("text").load(Some(path))
    }

    /// Read XML file(s). Mirrors `DataFrameReader.xml`.
    pub fn xml(self, path: &str) -> DataFrame {
        self.format("xml").load(Some(path))
    }

    /// Read from JDBC data source.
    pub fn jdbc(mut self, url: &str, table: &str, predicates: Option<Vec<String>>) -> DataFrame {
        self.format = Some("jdbc".to_string());
        self.options.insert("url".to_string(), url.to_string());
        self.options
            .insert("dbtable".to_string(), table.to_string());

        let plan = LogicalPlan::Read {
            read_type: ReadType::DataSource {
                format: self.format,
                schema: if self.schema.is_empty() {
                    None
                } else {
                    Some(self.schema)
                },
                options: self.options,
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
#[derive(Clone)]
pub struct DataFrameWriter {
    session: SparkSession,
    input_plan: Arc<LogicalPlan>,
    format: Option<String>,
    mode: SaveMode,
    /// The last mode set when it is not a save mode, which fails the write.
    unsupported_mode: Option<String>,
    options: HashMap<String, String>,
    partition_cols: Vec<String>,
    cluster_cols: Vec<String>,
    bucket_cols: Vec<String>,
    sort_cols: Vec<String>,
    num_buckets: Option<i32>,
    execution: Option<crate::dataframe::ExecutionSlot>,
}

impl DataFrameWriter {
    #[cfg(test)]
    pub(crate) fn plan(&self) -> &LogicalPlan {
        self.input_plan.as_ref()
    }

    /// Create a new DataFrameWriter.
    pub(crate) fn new(session: SparkSession, input_plan: Arc<LogicalPlan>) -> Self {
        DataFrameWriter {
            session,
            input_plan,
            format: None,
            mode: SaveMode::ErrorIfExists,
            unsupported_mode: None,
            options: HashMap::new(),
            partition_cols: vec![],
            cluster_cols: vec![],
            bucket_cols: vec![],
            sort_cols: vec![],
            num_buckets: None,
            execution: None,
        }
    }

    /// Keep the information of this write's execution as the last execution of its
    /// DataFrame, as PySpark's writers report it to theirs.
    pub(crate) fn recording_into(mut self, execution: crate::dataframe::ExecutionSlot) -> Self {
        self.execution = Some(execution);
        self
    }

    /// Run the command of the write, keeping its execution as its DataFrame's last one.
    fn run(&self, command_type: proto::command::CommandType) -> Result<()> {
        let (_, info) = crate::dataframe::execute_command_with_info(&self.session, command_type)?;
        if let Some(execution) = &self.execution {
            *execution.lock().unwrap() = Some(info);
        }
        Ok(())
    }

    /// Cluster the output by the given columns (liquid clustering).
    pub fn cluster_by<S: Into<String>>(mut self, cols: impl IntoIterator<Item = S>) -> Self {
        self.cluster_cols = cols.into_iter().map(Into::into).collect();
        self
    }

    /// Set the save mode: `append`, `overwrite`, `error` (or `errorifexists`) or `ignore`, in
    /// any case. As PySpark's, the write fails with `UNSUPPORTED_OPERATION` when the last mode
    /// set is any other string.
    pub fn mode(mut self, mode: &str) -> Self {
        match SaveMode::from_str(mode) {
            Some(m) => {
                self.mode = m;
                self.unsupported_mode = None;
            }
            None => self.unsupported_mode = Some(mode.to_string()),
        }
        self
    }

    /// Set the format/source type.
    pub fn format(mut self, source: &str) -> Self {
        self.format = Some(source.to_string());
        self
    }

    /// Unset the format, so that the server writes its default source
    /// (`spark.sql.sources.default`), as PySpark's `format(None)`.
    pub fn clear_format(mut self) -> Self {
        self.format = None;
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
        if let Some(mode) = &self.unsupported_mode {
            return Err(spark_connect_core::error::SparkError::value(
                "UNSUPPORTED_OPERATION",
                &[("operation", mode)],
            ));
        }
        let mut op = proto::WriteOperation::default();
        op.input = Some(build_input_relation(&self.input_plan, &self.session));
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
        self.run(proto::command::CommandType::WriteOperation(op))
    }

    /// Save the DataFrame to a file path (or to a path-less sink such as `noop`).
    pub fn save(self, path: Option<&str>) -> Result<()> {
        let save_type = path.map(|p| proto::write_operation::SaveType::Path(p.to_string()));
        let op = self.build_write_operation(save_type)?;
        self.run(proto::command::CommandType::WriteOperation(op))
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
#[derive(Clone)]
pub struct DataFrameWriterV2 {
    session: SparkSession,
    input_plan: Arc<LogicalPlan>,
    table_name: String,
    provider: Option<String>,
    options: HashMap<String, String>,
    table_properties: HashMap<String, String>,
    partition_cols: Vec<Column>,
    cluster_cols: Vec<String>,
    execution: Option<crate::dataframe::ExecutionSlot>,
}

impl DataFrameWriterV2 {
    #[cfg(test)]
    pub(crate) fn plan(&self) -> &LogicalPlan {
        self.input_plan.as_ref()
    }

    /// Create a new DataFrameWriterV2 targeting `table_name`.
    pub(crate) fn new(
        session: SparkSession,
        input_plan: Arc<LogicalPlan>,
        table_name: &str,
    ) -> Self {
        DataFrameWriterV2 {
            session,
            input_plan,
            table_name: table_name.to_string(),
            provider: None,
            options: HashMap::new(),
            table_properties: HashMap::new(),
            partition_cols: vec![],
            cluster_cols: vec![],
            execution: None,
        }
    }

    /// Keep the information of this write's execution as the last execution of its
    /// DataFrame, as PySpark's writers report it to theirs.
    pub(crate) fn recording_into(mut self, execution: crate::dataframe::ExecutionSlot) -> Self {
        self.execution = Some(execution);
        self
    }

    /// Run the command of the write, keeping its execution as its DataFrame's last one.
    fn run(&self, command_type: proto::command::CommandType) -> Result<()> {
        let (_, info) = crate::dataframe::execute_command_with_info(&self.session, command_type)?;
        if let Some(execution) = &self.execution {
            *execution.lock().unwrap() = Some(info);
        }
        Ok(())
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

    /// Leave the table property `property` unset, as PySpark's `tableProperty(property, None)`,
    /// which the command does not send.
    pub fn unset_table_property(mut self, property: &str) -> Self {
        self.table_properties.remove(property);
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
        op.input = Some(build_input_relation(&self.input_plan, &self.session));
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
        self.run(proto::command::CommandType::WriteOperationV2(op))
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
        self.run(proto::command::CommandType::WriteOperationV2(op))
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
    fn v1_write_operation_of_an_unsupported_mode_fails() {
        // As PySpark's: the last mode set decides, in any case, and any other string fails the
        // write with UNSUPPORTED_OPERATION.
        let spark = session();
        let df = spark.range(3).unwrap();
        let path = || {
            Some(proto::write_operation::SaveType::Path(
                "/tmp/out".to_string(),
            ))
        };
        for mode in ["bad", "default", "", " append"] {
            let err = df
                .write()
                .mode("append")
                .mode(mode)
                .build_write_operation(path())
                .unwrap_err();
            assert_eq!(
                err.get_condition().as_deref(),
                Some("UNSUPPORTED_OPERATION")
            );
            assert_eq!(err.params.get("operation").map(String::as_str), Some(mode));
        }
        for (mode, expected) in [
            ("APPEND", SaveMode::Append),
            ("Overwrite", SaveMode::Overwrite),
            ("error", SaveMode::ErrorIfExists),
            ("errorifexists", SaveMode::ErrorIfExists),
            ("ignore", SaveMode::Ignore),
        ] {
            let op = df
                .write()
                .mode("bad")
                .mode(mode)
                .build_write_operation(path())
                .unwrap();
            assert_eq!(op.mode, expected.to_proto());
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

        match df.plan.as_ref() {
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

        match df.plan.as_ref() {
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

        match df.plan.as_ref() {
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

    #[test]
    fn parse_sends_the_strings_with_the_schema_and_options_of_the_reader() {
        use crate::plan::ParseFormat;
        use crate::types::DataType;
        use proto::relation::RelType;

        let spark = session();
        // A lazy plan, as `spark.sql` runs on the call.
        let strings = spark.range(1).unwrap();
        let parse_of = |reader: DataFrameReader, format| match reader
            .parse(format, &strings)
            .unwrap()
            .relation()
            .rel_type
        {
            Some(RelType::Parse(parse)) => parse,
            other => panic!("expected Parse, got {other:?}"),
        };
        // A schema of JSON, as PySpark's `StructType.fromJson`; one of DDL the server parses
        // (the e2e tests check it).
        let json = DataType::from_ddl("a INT, b STRING").unwrap().json();
        let parse = parse_of(
            spark.read().schema(json).option("sep", ";"),
            ParseFormat::Csv,
        );
        assert_eq!(parse.format, proto::parse::ParseFormat::Csv as i32);
        assert_eq!(
            parse.schema,
            Some(DataType::from_ddl("a INT, b STRING").unwrap().to_proto())
        );
        assert_eq!(parse.options.get("sep").map(String::as_str), Some(";"));
        assert!(matches!(
            parse.input.and_then(|input| input.rel_type),
            Some(RelType::Range(_))
        ));
        let json = DataType::from_ddl("c BIGINT").unwrap().json();
        let parse = parse_of(spark.read().schema(json), ParseFormat::Json);
        assert_eq!(parse.format, proto::parse::ParseFormat::Json as i32);
        assert_eq!(
            parse.schema,
            Some(DataType::from_ddl("c BIGINT").unwrap().to_proto())
        );
        // Without one, the server infers it.
        let parse = parse_of(spark.read(), ParseFormat::Xml);
        assert_eq!(parse.format, proto::parse::ParseFormat::Xml as i32);
        assert_eq!(parse.schema, None);
        assert!(spark
            .read()
            .schema("{not json".to_string())
            .parse(ParseFormat::Json, &strings)
            .is_err());
    }
}
