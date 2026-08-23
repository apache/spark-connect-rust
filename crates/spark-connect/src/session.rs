//! SparkSession implementation mirroring `pyspark.sql.SparkSession`.
//!
//! Provides the entry point for DataFrame operations and SQL queries.

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use spark_connect_core::channel::ChannelBuilder;
use spark_connect_core::client::SparkConnectClient;
use spark_connect_core::error::{Result, SparkError};
use spark_connect_core::runtime::block_on;

use crate::catalog::Catalog;
use crate::dataframe::DataFrame;
use crate::plan::LogicalPlan;
use crate::row::{Row, Value};
use crate::types::DataType;

/// A Spark Connect session for interacting with a remote Spark cluster.
///
/// Mirrors `pyspark.sql.SparkSession`.
pub struct SparkSession {
    /// Shared gRPC client
    client: Arc<SparkConnectClient>,
    /// Plan ID counter for unique IDs in nested plans
    plan_id_counter: Arc<AtomicI64>,
}

impl SparkSession {
    /// Create a new SparkSession with a client.
    fn new(client: SparkConnectClient) -> Self {
        SparkSession {
            client: Arc::new(client),
            plan_id_counter: Arc::new(AtomicI64::new(1)),
        }
    }

    /// Get the next plan ID (atomic post-increment).
    pub(crate) fn next_plan_id(&self) -> i64 {
        self.plan_id_counter.fetch_add(1, Ordering::SeqCst)
    }

    /// Get the underlying client.
    pub(crate) fn client(&self) -> &Arc<SparkConnectClient> {
        &self.client
    }

    /// Create a builder for a new SparkSession.
    pub fn builder() -> SparkSessionBuilder {
        SparkSessionBuilder::new()
    }

    /// Create a DataFrame representing a range of integers.
    ///
    /// Mirrors `pyspark.sql.SparkSession.range(start, end=None, step=1, numPartitions=None)`.
    pub fn range(&self, end: i64) -> Result<DataFrame> {
        self.range_full(0, end, 1, None)
    }

    /// Create a DataFrame representing a range with full parameters.
    pub fn range_full(
        &self,
        start: i64,
        end: i64,
        step: i64,
        num_partitions: Option<i32>,
    ) -> Result<DataFrame> {
        let plan = LogicalPlan::Range {
            start,
            end,
            step,
            num_partitions,
        };
        Ok(DataFrame::new(self.clone(), plan))
    }

    /// Execute a SQL query and return a DataFrame.
    ///
    /// Mirrors `pyspark.sql.SparkSession.sql(sqlQuery)`.
    pub fn sql(&self, query: &str) -> Result<DataFrame> {
        let plan = LogicalPlan::Sql {
            query: query.to_string(),
        };
        Ok(DataFrame::new(self.clone(), plan))
    }

    /// Create a DataFrame from a collection of Rows and a schema.
    ///
    /// Mirrors `pyspark.sql.SparkSession.createDataFrame(rows, schema)`.
    pub fn create_dataframe(&self, rows: Vec<Row>, schema: DataType) -> Result<DataFrame> {
        // Build Arrow RecordBatch from rows and schema
        let arrow_data = rows_to_arrow_ipc(&rows, &schema)?;

        let plan = LogicalPlan::LocalRelation {
            schema,
            data: Some(arrow_data),
        };
        Ok(DataFrame::new(self.clone(), plan))
    }

    /// Create a DataFrameReader for reading data from various sources.
    ///
    /// Mirrors `pyspark.sql.SparkSession.read`.
    pub fn read(&self) -> crate::readwriter::DataFrameReader {
        crate::readwriter::DataFrameReader::new(self.clone())
    }

    /// Create a DataStreamReader for reading streaming data from various sources.
    ///
    /// Mirrors `pyspark.sql.SparkSession.readStream`.
    pub fn read_stream(&self) -> crate::streaming::DataStreamReader {
        crate::streaming::DataStreamReader::new(self.clone())
    }

    /// Get the streaming query manager.
    ///
    /// Mirrors `pyspark.sql.SparkSession.streams`.
    pub fn streams(&self) -> crate::streaming::StreamingQueryManager {
        crate::streaming::StreamingQueryManager::new(self.clone())
    }

    /// Get the catalog for this session.
    ///
    /// Mirrors `pyspark.sql.SparkSession.catalog`.
    pub fn catalog(&self) -> crate::catalog::Catalog {
        crate::catalog::Catalog::new(self.clone())
    }

    /// Get the runtime configuration for this session.
    ///
    /// Mirrors `pyspark.sql.SparkSession.conf`.
    pub fn conf(&self) -> crate::conf::RuntimeConf {
        crate::conf::RuntimeConf::new(Arc::clone(&self.client))
    }

    /// Get table-valued functions for this session.
    ///
    /// Mirrors `pyspark.sql.SparkSession.tvf`.
    pub fn tvf(&self) -> crate::tvf::TableValuedFunction {
        crate::tvf::TableValuedFunction::new(self.clone())
    }

    /// Stop this Spark session.
    pub fn stop(&self) -> Result<()> {
        block_on(self.client.release_session())?;
        Ok(())
    }
}

impl Clone for SparkSession {
    fn clone(&self) -> Self {
        SparkSession {
            client: Arc::clone(&self.client),
            plan_id_counter: Arc::clone(&self.plan_id_counter),
        }
    }
}

/// Builder for creating a SparkSession.
pub struct SparkSessionBuilder {
    remote_url: Option<String>,
}

impl SparkSessionBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        SparkSessionBuilder { remote_url: None }
    }

    /// Set the remote Spark Connect server URL.
    pub fn remote(mut self, url: &str) -> Self {
        self.remote_url = Some(url.to_string());
        self
    }

    /// Build and return a SparkSession.
    pub fn get_or_create(self) -> Result<SparkSession> {
        let url = self.remote_url.ok_or_else(|| {
            SparkError::value(
                "NO_REMOTE_URL",
                &[("detail", "Must call .remote(url) before .get_or_create()")],
            )
        })?;

        let builder = ChannelBuilder::parse(&url)?;
        let client = block_on(SparkConnectClient::connect(&builder))?;
        Ok(SparkSession::new(client))
    }
}

impl Default for SparkSessionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert Arrow RecordBatch data to IPC bytes.
/// Used by createDataFrame to package local data.
fn rows_to_arrow_ipc(rows: &[Row], schema: &DataType) -> Result<Vec<u8>> {
    use arrow::array::*;
    use arrow::datatypes::Schema as ArrowSchema;
    use arrow::ipc::writer::StreamWriter;
    use arrow::record_batch::RecordBatch;
    use std::io::Cursor;

    if rows.is_empty() {
        return Ok(vec![]);
    }

    // Convert our schema to Arrow schema
    let arrow_schema_vec = schema_to_arrow_fields(schema)?;
    let arrow_schema = ArrowSchema::new(arrow_schema_vec);

    // Build column arrays from rows
    let mut columns = vec![];
    let num_fields = match schema {
        DataType::Struct { fields } => fields.len(),
        _ => return Err(SparkError::connect_msg("Schema is not a struct type")),
    };

    for field_idx in 0..num_fields {
        let array = build_arrow_array(&rows, field_idx)?;
        columns.push(array);
    }

    // Create RecordBatch
    let batch = RecordBatch::try_new(Arc::new(arrow_schema), columns)
        .map_err(|e| SparkError::connect_msg(format!("Failed to create Arrow batch: {}", e)))?;

    // Write to IPC stream
    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &batch.schema())
            .map_err(|e| SparkError::connect_msg(format!("Failed to create IPC writer: {}", e)))?;
        writer
            .write(&batch)
            .map_err(|e| SparkError::connect_msg(format!("Failed to write batch to IPC: {}", e)))?;
        writer
            .finish()
            .map_err(|e| SparkError::connect_msg(format!("Failed to finish IPC writer: {}", e)))?;
    }

    Ok(buffer)
}

/// Convert DataType fields to Arrow fields.
fn schema_to_arrow_fields(schema: &DataType) -> Result<Vec<arrow::datatypes::Field>> {
    use arrow::datatypes::{DataType as ArrowDataType, Field, TimeUnit};

    match schema {
        DataType::Struct { fields } => {
            let mut arrow_fields = vec![];
            for field in fields {
                let arrow_type = match &field.data_type {
                    DataType::Null => ArrowDataType::Null,
                    DataType::Boolean => ArrowDataType::Boolean,
                    DataType::Byte => ArrowDataType::Int8,
                    DataType::Short => ArrowDataType::Int16,
                    DataType::Integer => ArrowDataType::Int32,
                    DataType::Long => ArrowDataType::Int64,
                    DataType::Float => ArrowDataType::Float32,
                    DataType::Double => ArrowDataType::Float64,
                    DataType::String { .. } => ArrowDataType::Utf8,
                    DataType::Binary => ArrowDataType::Binary,
                    DataType::Date => ArrowDataType::Date32,
                    DataType::Timestamp => ArrowDataType::Timestamp(TimeUnit::Microsecond, None),
                    _ => {
                        return Err(SparkError::connect_msg(
                            "Unsupported type for Arrow conversion",
                        ))
                    }
                };
                arrow_fields.push(Field::new(&field.name, arrow_type, field.nullable));
            }
            Ok(arrow_fields)
        }
        _ => Err(SparkError::connect_msg("Schema is not a struct")),
    }
}

/// Build an Arrow array for a specific field across all rows.
fn build_arrow_array(rows: &[Row], field_idx: usize) -> Result<Arc<dyn arrow::array::Array>> {
    use arrow::array::*;

    if rows.is_empty() {
        return Ok(Arc::new(NullArray::new(0)));
    }

    // Peek at the first row to determine the type
    let first_val = rows[0]
        .get(field_idx)
        .ok_or_else(|| SparkError::connect_msg("Invalid row index"))?;

    match first_val {
        Value::Null => Ok(Arc::new(NullArray::new(rows.len()))),
        Value::Bool(_) => {
            let values: Result<Vec<_>> = rows
                .iter()
                .map(|r| {
                    r.get(field_idx)
                        .and_then(|v| v.as_bool())
                        .ok_or_else(|| SparkError::connect_msg("Type mismatch in row data"))
                })
                .collect();
            Ok(Arc::new(BooleanArray::from(values?)))
        }
        Value::Byte(_) => {
            let values: Result<Vec<_>> = rows
                .iter()
                .map(|r| {
                    r.get(field_idx)
                        .map(|v| match v {
                            Value::Byte(b) => Ok(*b),
                            _ => Err(SparkError::connect_msg("Type mismatch in row data")),
                        })
                        .transpose()
                })
                .collect();
            Ok(Arc::new(Int8Array::from(values?)))
        }
        Value::Short(_) => {
            let values: Result<Vec<_>> = rows
                .iter()
                .map(|r| {
                    r.get(field_idx)
                        .map(|v| match v {
                            Value::Short(s) => Ok(*s),
                            _ => Err(SparkError::connect_msg("Type mismatch in row data")),
                        })
                        .transpose()
                })
                .collect();
            Ok(Arc::new(Int16Array::from(values?)))
        }
        Value::Integer(_) => {
            let values: Result<Vec<_>> = rows
                .iter()
                .map(|r| {
                    r.get(field_idx)
                        .map(|v| match v {
                            Value::Integer(i) => Ok(*i),
                            _ => Err(SparkError::connect_msg("Type mismatch in row data")),
                        })
                        .transpose()
                })
                .collect();
            Ok(Arc::new(Int32Array::from(values?)))
        }
        Value::Long(_) => {
            let values: Result<Vec<_>> = rows
                .iter()
                .map(|r| {
                    r.get(field_idx)
                        .map(|v| match v {
                            Value::Long(l) => Ok(*l),
                            _ => Err(SparkError::connect_msg("Type mismatch in row data")),
                        })
                        .transpose()
                })
                .collect();
            Ok(Arc::new(Int64Array::from(values?)))
        }
        Value::Float(_) => {
            let values: Result<Vec<_>> = rows
                .iter()
                .map(|r| {
                    r.get(field_idx)
                        .map(|v| match v {
                            Value::Float(f) => Ok(*f),
                            _ => Err(SparkError::connect_msg("Type mismatch in row data")),
                        })
                        .transpose()
                })
                .collect();
            Ok(Arc::new(Float32Array::from(values?)))
        }
        Value::Double(_) => {
            let values: Result<Vec<_>> = rows
                .iter()
                .map(|r| {
                    r.get(field_idx)
                        .map(|v| match v {
                            Value::Double(d) => Ok(*d),
                            _ => Err(SparkError::connect_msg("Type mismatch in row data")),
                        })
                        .transpose()
                })
                .collect();
            Ok(Arc::new(Float64Array::from(values?)))
        }
        Value::String(_) => {
            let values: Result<Vec<_>> = rows
                .iter()
                .map(|r| {
                    r.get(field_idx)
                        .map(|v| match v {
                            Value::String(s) => Ok(s.as_str()),
                            _ => Err(SparkError::connect_msg("Type mismatch in row data")),
                        })
                        .transpose()
                })
                .collect();
            Ok(Arc::new(StringArray::from(values?)))
        }
        Value::Binary(_) => {
            let values: Result<Vec<_>> = rows
                .iter()
                .map(|r| {
                    r.get(field_idx)
                        .map(|v| match v {
                            Value::Binary(b) => Ok(b.as_slice()),
                            _ => Err(SparkError::connect_msg("Type mismatch in row data")),
                        })
                        .transpose()
                })
                .collect();
            Ok(Arc::new(BinaryArray::from(values?)))
        }
        _ => Err(SparkError::connect_msg("Unsupported value type")),
    }
}
