//! Arrow local relations and session-scoped chunked uploads.

use arrow::datatypes::SchemaRef;
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use spark_connect_core::client::SparkConnectClient;
use spark_connect_core::error::{Result, SparkError, SparkErrorKind};
use spark_connect_core::runtime::block_on;
use std::collections::HashMap;

use crate::plan::LogicalPlan;
use crate::types::DataType;

/// Internal settings snapshot shared with the Python Arrow importer; not a stable API.
#[doc(hidden)]
pub struct ArrowInputOptions {
    timestamp_type: String,
    timezone: String,
    use_large_var_types: bool,
    cache_threshold: usize,
    upload: Options,
}

impl ArrowInputOptions {
    const KEYS: [&'static str; 8] = [
        "spark.sql.timestampType",
        "spark.sql.session.timeZone",
        "spark.sql.execution.arrow.useLargeVarTypes",
        "spark.sql.session.localRelationCacheThreshold",
        "spark.sql.session.localRelationSizeLimit",
        "spark.sql.session.localRelationChunkSizeRows",
        "spark.sql.session.localRelationChunkSizeBytes",
        "spark.sql.session.localRelationBatchOfChunksSizeBytes",
    ];
    pub(crate) fn from_client(client: &SparkConnectClient) -> Result<Self> {
        let values = block_on(client.get_configs(&Self::KEYS))?;
        let values = Self::KEYS
            .into_iter()
            .zip(values)
            .filter_map(|(key, value)| value.map(|value| (key, value)))
            .collect();
        Self::from_values(values)
    }

    fn from_values(values: HashMap<&str, String>) -> Result<Self> {
        let required = |key: &str| {
            values.get(key).map(String::as_str).ok_or_else(|| {
                SparkError::classed(
                    SparkErrorKind::SparkNoSuchElement,
                    "SQL_CONF_NOT_FOUND",
                    &[("sqlConf", key)],
                )
            })
        };
        let invalid = |key: &str| SparkError::connect_msg(format!("Invalid configuration: {key}"));
        let integer = |key: &str, allow_zero: bool| {
            required(key)?
                .parse::<usize>()
                .ok()
                .filter(|value| allow_zero || *value > 0)
                .ok_or_else(|| invalid(key))
        };
        let chunk_bytes = integer("spark.sql.session.localRelationChunkSizeBytes", false)?;
        let upload_batch_bytes = integer(
            "spark.sql.session.localRelationBatchOfChunksSizeBytes",
            false,
        )?;
        Ok(Self {
            timestamp_type: required("spark.sql.timestampType")?.to_owned(),
            timezone: required("spark.sql.session.timeZone")?.to_owned(),
            use_large_var_types: required("spark.sql.execution.arrow.useLargeVarTypes")? == "true",
            cache_threshold: integer("spark.sql.session.localRelationCacheThreshold", true)?,
            upload: Options {
                size_limit: integer("spark.sql.session.localRelationSizeLimit", false)?,
                chunk_rows: integer("spark.sql.session.localRelationChunkSizeRows", false)?,
                // Match upstream's min(chunk size, upload batch size).
                chunk_bytes: chunk_bytes.min(upload_batch_bytes),
                upload_batch_bytes,
            },
        })
    }

    /// Server timestamp type, including its registered default.
    pub fn timestamp_type(&self) -> Option<&str> {
        Some(&self.timestamp_type)
    }

    /// Server session timezone; do not substitute a client-local timezone.
    pub fn timezone(&self) -> Option<&str> {
        Some(&self.timezone)
    }

    /// Whether the server prefers large Arrow string and binary offsets.
    pub fn use_large_var_types(&self) -> Option<&str> {
        Some(if self.use_large_var_types {
            "true"
        } else {
            "false"
        })
    }
}

struct Options {
    size_limit: usize,
    chunk_rows: usize,
    chunk_bytes: usize,
    upload_batch_bytes: usize,
}

fn arrow_error(error: arrow::error::ArrowError) -> SparkError {
    SparkError::connect_msg(error.to_string())
}

fn validate(batches: &[RecordBatch]) -> Result<()> {
    if let Some(first) = batches.first() {
        if batches.iter().any(|batch| batch.schema() != first.schema()) {
            return Err(SparkError::connect_msg("Arrow batch schemas must match"));
        }
    }
    Ok(())
}

pub(crate) fn batch_data_size(batch: &RecordBatch) -> usize {
    batch
        .columns()
        .iter()
        .map(|array| {
            array
                .to_data()
                .get_slice_memory_size()
                .unwrap_or_else(|_| array.get_buffer_memory_size())
        })
        .sum()
}

pub(crate) fn inline_relation(
    batches: &[RecordBatch],
    schema: Option<DataType>,
) -> Result<LogicalPlan> {
    validate(batches)?;
    let data = if let Some(first) = batches.first() {
        let mut writer = StreamWriter::try_new(Vec::new(), &first.schema()).map_err(arrow_error)?;
        for batch in batches {
            writer.write(batch).map_err(arrow_error)?;
        }
        Some(writer.into_inner().map_err(arrow_error)?)
    } else {
        None
    };
    Ok(LogicalPlan::LocalRelation { schema, data })
}

pub(crate) fn create_relation(
    client: &SparkConnectClient,
    batches: Vec<RecordBatch>,
    schema: Option<DataType>,
    options: Option<&ArrowInputOptions>,
) -> Result<LogicalPlan> {
    // Validate before any RPC, and preserve the offline empty-input path.
    validate(&batches)?;
    if batches.iter().all(|batch| batch.num_rows() == 0) {
        return inline_relation(&batches, schema);
    }
    let fetched;
    let options = match options {
        Some(options) => options,
        None => {
            fetched = ArrowInputOptions::from_client(client)?;
            &fetched
        }
    };
    // Count the logical slices rather than their potentially much larger backing buffers.
    let bytes: usize = batches.iter().map(batch_data_size).sum();
    if bytes < options.cache_threshold {
        return inline_relation(&batches, schema);
    }
    let options = &options.upload;

    let mut pending = Vec::new();
    let mut pending_size = 0;
    let mut total_size = 0;
    let mut hashes = Vec::new();
    let mut enqueue = |chunk: Vec<u8>, is_data: bool| -> Result<()> {
        total_size += chunk.len();
        // As PySpark, the schema chunk counts toward the size, and the limit applies to the data.
        if is_data && total_size > options.size_limit {
            return Err(SparkError::classed(
                SparkErrorKind::Analysis,
                "LOCAL_RELATION_SIZE_LIMIT_EXCEEDED",
                &[
                    ("actualSize", &total_size.to_string()),
                    ("sizeLimit", &options.size_limit.to_string()),
                ],
            ));
        }
        if !pending.is_empty() && pending_size + chunk.len() > options.upload_batch_bytes {
            hashes.extend(block_on(
                client.cache_artifacts(std::mem::take(&mut pending)),
            )?);
            pending_size = 0;
        }
        pending_size += chunk.len();
        pending.push(chunk);
        Ok(())
    };
    if let Some(schema) = &schema {
        enqueue(schema.json().into_bytes(), false)?;
    }
    serialize_chunks(&batches, options, &mut |chunk| enqueue(chunk, true))?;
    hashes.extend(block_on(client.cache_artifacts(pending))?);
    let schema_hash = schema.map(|_| hashes.remove(0));
    Ok(LogicalPlan::ChunkedCachedLocalRelation {
        data_hashes: hashes,
        schema_hash,
    })
}

fn new_writer(schema: &SchemaRef) -> Result<StreamWriter<Vec<u8>>> {
    StreamWriter::try_new(Vec::new(), schema).map_err(arrow_error)
}

/// Match PySpark's pre-write estimate: stream schema plus each batch's data size.
/// The row setting caps individual record batches, not the complete IPC stream.
fn serialize_chunks(
    batches: &[RecordBatch],
    options: &Options,
    emit: &mut impl FnMut(Vec<u8>) -> Result<()>,
) -> Result<()> {
    let schema = batches[0].schema();
    let schema_size = new_writer(&schema)?
        .into_inner()
        .map_err(arrow_error)?
        .len();
    let mut writer = new_writer(&schema)?;
    let mut current_size = schema_size;
    let mut has_data = false;
    for batch in batches {
        let mut offset = 0;
        while offset < batch.num_rows() {
            let length = (batch.num_rows() - offset)
                .min(1024)
                .min(options.chunk_rows);
            let slice = batch.slice(offset, length);
            let batch_size = batch_data_size(&slice);
            if has_data && current_size + batch_size > options.chunk_bytes {
                emit(writer.into_inner().map_err(arrow_error)?)?;
                writer = new_writer(&schema)?;
                current_size = schema_size;
            }
            writer.write(&slice).map_err(arrow_error)?;
            current_size += batch_size;
            has_data = true;
            offset += length;
        }
    }
    if has_data {
        emit(writer.into_inner().map_err(arrow_error)?)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Array;
    use arrow::datatypes::{DataType as ArrowType, Field, Schema};
    use arrow::ipc::reader::StreamReader;
    use std::io::Cursor;
    use std::ops::Range;
    use std::sync::Arc;

    fn batch(values: Range<i64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("id", ArrowType::Int64, true)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from_iter_values(values))]).unwrap()
    }

    fn chunks(batch: &RecordBatch, chunk_rows: usize, chunk_bytes: usize) -> Vec<Vec<RecordBatch>> {
        let options = Options {
            size_limit: usize::MAX,
            chunk_rows,
            chunk_bytes,
            upload_batch_bytes: usize::MAX,
        };
        let mut chunks = Vec::new();
        serialize_chunks(std::slice::from_ref(batch), &options, &mut |chunk| {
            chunks.push(chunk);
            Ok(())
        })
        .unwrap();
        chunks
            .into_iter()
            .map(|chunk| {
                let reader = StreamReader::try_new(Cursor::new(chunk), None).unwrap();
                assert_eq!(reader.schema(), batch.schema());
                reader.collect::<std::result::Result<Vec<_>, _>>().unwrap()
            })
            .collect()
    }

    fn chunk_row_counts(chunks: &[Vec<RecordBatch>]) -> Vec<usize> {
        chunks
            .iter()
            .map(|chunk| chunk.iter().map(RecordBatch::num_rows).sum())
            .collect()
    }

    #[test]
    fn row_setting_limits_batches_without_splitting_the_stream() {
        let chunks = chunks(&batch(0..2500), 700, usize::MAX);
        assert_eq!(chunks.len(), 1);
        assert_eq!(
            chunks[0]
                .iter()
                .map(RecordBatch::num_rows)
                .collect::<Vec<_>>(),
            vec![700, 700, 700, 400]
        );
    }

    #[test]
    fn chunks_flush_before_a_batch_exceeds_the_data_size_estimate() {
        let batch = batch(0..25);
        let schema_size = new_writer(&batch.schema())
            .unwrap()
            .into_inner()
            .unwrap()
            .len();
        // Two four-row Int64 batches fit exactly, despite IPC message overhead.
        let chunks = chunks(&batch, 4, schema_size + 8 * 8);
        assert_eq!(chunk_row_counts(&chunks), vec![8, 8, 8, 1]);
        let actual: Vec<_> = chunks
            .iter()
            .flatten()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied()
            })
            .collect();
        assert_eq!(actual, (0..25).collect::<Vec<_>>());
    }

    #[test]
    fn an_oversized_batch_does_not_emit_an_empty_stream() {
        let batch = batch(0..10);
        let schema_size = new_writer(&batch.schema())
            .unwrap()
            .into_inner()
            .unwrap()
            .len();
        let chunks = chunks(&batch, 4, schema_size + 4 * 8 - 1);
        assert_eq!(chunk_row_counts(&chunks), vec![4, 4, 2]);
    }

    #[test]
    fn inline_schema_is_sent_only_when_explicit() {
        let batch = batch(0..2);
        let explicit = DataType::Struct {
            fields: vec![crate::types::StructField {
                name: "id".to_owned(),
                data_type: DataType::Long,
                nullable: true,
                metadata: Default::default(),
            }],
        };
        for schema in [None, Some(explicit)] {
            let relation = inline_relation(std::slice::from_ref(&batch), schema.clone())
                .unwrap()
                .to_proto();
            let Some(spark_connect_proto::relation::RelType::LocalRelation(local)) =
                relation.rel_type
            else {
                panic!("expected inline relation");
            };
            assert_eq!(local.schema, schema.as_ref().map(DataType::json));
            let data = local.data.unwrap();
            let decoded = StreamReader::try_new(Cursor::new(data), None)
                .unwrap()
                .collect::<std::result::Result<Vec<_>, _>>()
                .unwrap();
            assert_eq!(decoded, vec![batch.clone()]);
        }
    }
}
