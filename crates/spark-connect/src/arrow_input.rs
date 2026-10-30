//! Arrow local relations and session-scoped chunked uploads.

use arrow::datatypes::SchemaRef;
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use spark_connect_core::client::SparkConnectClient;
use spark_connect_core::error::{Result, SparkError, SparkErrorKind};
use spark_connect_core::runtime::block_on;

use crate::plan::LogicalPlan;
use crate::types::DataType;

struct Options {
    cache_threshold: usize,
    size_limit: usize,
    chunk_rows: usize,
    chunk_bytes: usize,
    upload_batch_bytes: usize,
}

impl Options {
    fn from_client(client: &SparkConnectClient) -> Result<Self> {
        let keys = [
            "spark.sql.session.localRelationCacheThreshold",
            "spark.sql.session.localRelationSizeLimit",
            "spark.sql.session.localRelationChunkSizeRows",
            "spark.sql.session.localRelationChunkSizeBytes",
            "spark.sql.session.localRelationBatchOfChunksSizeBytes",
        ];
        let values = block_on(client.get_configs(&keys))?;
        let mut parsed = Vec::new();
        for (i, key) in keys.iter().enumerate() {
            let value = values
                .get(i)
                .and_then(Option::as_deref)
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|v| i == 0 || *v > 0)
                .ok_or_else(|| {
                    SparkError::connect_msg(format!("Invalid or missing configuration: {key}"))
                })?;
            parsed.push(value);
        }
        Ok(Self {
            cache_threshold: parsed[0],
            size_limit: parsed[1],
            chunk_rows: parsed[2],
            chunk_bytes: parsed[3].min(parsed[4]),
            upload_batch_bytes: parsed[4],
        })
    }
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

pub(crate) fn inline_relation(batches: &[RecordBatch], schema: DataType) -> Result<LogicalPlan> {
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
    schema: DataType,
) -> Result<LogicalPlan> {
    // Validate before any RPC, and preserve the offline empty-input path.
    validate(&batches)?;
    if batches.iter().all(|batch| batch.num_rows() == 0) {
        return inline_relation(&batches, schema);
    }
    let options = Options::from_client(client)?;
    // Count the logical slices rather than their potentially much larger backing buffers.
    let bytes: usize = batches
        .iter()
        .flat_map(|b| b.columns())
        .map(|a| {
            a.to_data()
                .get_slice_memory_size()
                .unwrap_or_else(|_| a.get_buffer_memory_size())
        })
        .sum();
    if bytes < options.cache_threshold {
        return inline_relation(&batches, schema);
    }

    let mut pending = Vec::new();
    let mut pending_size = 0;
    let mut total_size = 0;
    let mut hashes = Vec::new();
    let mut enqueue = |chunk: Vec<u8>| -> Result<()> {
        total_size += chunk.len();
        if total_size > options.size_limit {
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
    enqueue(schema.json().into_bytes())?;
    serialize_chunks(&batches, &options, &mut enqueue)?;
    hashes.extend(block_on(client.cache_artifacts(pending))?);
    let schema_hash = hashes.remove(0);
    Ok(LogicalPlan::ChunkedCachedLocalRelation {
        data_hashes: hashes,
        schema_hash: Some(schema_hash),
    })
}

fn new_writer(schema: &SchemaRef) -> Result<StreamWriter<Vec<u8>>> {
    StreamWriter::try_new(Vec::new(), schema).map_err(arrow_error)
}

/// Each chunk is an independently readable IPC stream. Check limits at most every
/// 1024 rows, as reference PySpark does; a single oversized row remains indivisible.
fn serialize_chunks(
    batches: &[RecordBatch],
    options: &Options,
    emit: &mut impl FnMut(Vec<u8>) -> Result<()>,
) -> Result<()> {
    let schema = batches[0].schema();
    let mut writer = new_writer(&schema)?;
    let mut rows = 0;
    for batch in batches {
        let mut offset = 0;
        while offset < batch.num_rows() {
            let length = (batch.num_rows() - offset)
                .min(1024)
                .min(options.chunk_rows - rows);
            writer
                .write(&batch.slice(offset, length))
                .map_err(arrow_error)?;
            offset += length;
            rows += length;
            if rows >= options.chunk_rows || writer.get_ref().len() >= options.chunk_bytes {
                emit(writer.into_inner().map_err(arrow_error)?)?;
                writer = new_writer(&schema)?;
                rows = 0;
            }
        }
    }
    if rows > 0 {
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
    use std::sync::Arc;

    #[test]
    fn chunks_preserve_rows_schema_and_order() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", ArrowType::Int64, true)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from_iter_values(0..2500))],
        )
        .unwrap();
        for (chunk_rows, chunk_bytes) in [(700, usize::MAX), (10000, 1024)] {
            let options = Options {
                cache_threshold: 0,
                size_limit: usize::MAX,
                chunk_rows,
                chunk_bytes,
                upload_batch_bytes: usize::MAX,
            };
            let mut chunks = Vec::new();
            serialize_chunks(std::slice::from_ref(&batch), &options, &mut |c| {
                chunks.push(c);
                Ok(())
            })
            .unwrap();
            assert!(chunks.len() > 1);
            let mut actual = Vec::new();
            for chunk in chunks {
                let reader = StreamReader::try_new(Cursor::new(chunk), None).unwrap();
                assert_eq!(reader.schema(), schema);
                let mut rows = 0;
                for batch in reader {
                    let batch = batch.unwrap();
                    rows += batch.num_rows();
                    actual.extend(
                        batch
                            .column(0)
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .unwrap()
                            .values()
                            .iter()
                            .copied(),
                    );
                }
                assert!(rows <= chunk_rows);
            }
            assert_eq!(actual, (0..2500).collect::<Vec<_>>());
        }
    }
}
