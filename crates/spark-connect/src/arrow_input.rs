//! Arrow local relations and session-scoped chunked uploads.

use arrow::array::{make_array, ArrayData};
use arrow::datatypes::{DataType as ArrowType, FieldRef, Schema, SchemaRef};
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::{RecordBatch, RecordBatchOptions};
use spark_connect_core::client::SparkConnectClient;
use spark_connect_core::error::{Result, SparkError, SparkErrorKind};
use spark_connect_core::runtime::block_on;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use crate::plan::{LogicalPlan, LogicalPlanKind};
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

fn normalize_map_field(field: &FieldRef) -> Result<FieldRef> {
    let data_type = normalize_map_type(field.data_type())?;
    if &data_type == field.data_type() {
        return Ok(Arc::clone(field));
    }
    Ok(Arc::new(field.as_ref().clone().with_data_type(data_type)))
}

fn normalize_map_type(data_type: &ArrowType) -> Result<ArrowType> {
    use ArrowType as A;
    Ok(match data_type {
        A::Map(entries, sorted) => {
            let entries = normalize_map_field(entries)?;
            let A::Struct(fields) = entries.data_type() else {
                return Err(SparkError::connect_msg(
                    "Arrow map entries must be a struct",
                ));
            };
            if fields.len() != 2 {
                return Err(SparkError::connect_msg(
                    "Arrow maps must have key and value children",
                ));
            }
            let fields = fields
                .iter()
                .zip(["key", "value"])
                .map(|(field, name)| Arc::new(field.as_ref().clone().with_name(name)))
                .collect();
            A::Map(
                Arc::new(
                    entries
                        .as_ref()
                        .clone()
                        .with_name("entries")
                        .with_data_type(A::Struct(fields)),
                ),
                *sorted,
            )
        }
        A::Struct(fields) => A::Struct(
            fields
                .iter()
                .map(normalize_map_field)
                .collect::<Result<Vec<_>>>()?
                .into(),
        ),
        A::List(field) => A::List(normalize_map_field(field)?),
        A::LargeList(field) => A::LargeList(normalize_map_field(field)?),
        A::ListView(field) => A::ListView(normalize_map_field(field)?),
        A::LargeListView(field) => A::LargeListView(normalize_map_field(field)?),
        A::FixedSizeList(field, size) => A::FixedSizeList(normalize_map_field(field)?, *size),
        A::Union(fields, mode) => {
            let fields = fields
                .iter()
                .map(|(id, field)| Ok((id, normalize_map_field(field)?)))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .collect();
            A::Union(fields, *mode)
        }
        A::Dictionary(key, value) => A::Dictionary(
            Box::new(normalize_map_type(key)?),
            Box::new(normalize_map_type(value)?),
        ),
        A::RunEndEncoded(ends, values) => {
            A::RunEndEncoded(normalize_map_field(ends)?, normalize_map_field(values)?)
        }
        other => other.clone(),
    })
}

fn child_types(data_type: &ArrowType) -> Vec<&ArrowType> {
    use ArrowType as A;
    match data_type {
        A::Map(field, _)
        | A::List(field)
        | A::LargeList(field)
        | A::ListView(field)
        | A::LargeListView(field)
        | A::FixedSizeList(field, _) => vec![field.data_type()],
        A::Struct(fields) => fields.iter().map(|field| field.data_type()).collect(),
        A::Union(fields, _) => fields.iter().map(|(_, field)| field.data_type()).collect(),
        A::Dictionary(_, value) => vec![value.as_ref()],
        A::RunEndEncoded(ends, values) => vec![ends.data_type(), values.data_type()],
        _ => vec![],
    }
}

/// Reconcile schema field names and metadata only through checked construction.
/// Arrow's name-insensitive equality also accepts reordered union fields, whose
/// children cannot be relabeled by position, so reject those before recursion.
fn with_schema_type(data: ArrayData, data_type: &ArrowType) -> Result<ArrayData> {
    if data.data_type() == data_type {
        return Ok(data);
    }
    let mismatch = || SparkError::connect_msg("Arrow column type does not match its schema field");
    if !data.data_type().equals_datatype(data_type) {
        return Err(mismatch());
    }
    if let (ArrowType::Union(before, _), ArrowType::Union(after, _)) = (data.data_type(), data_type)
    {
        if !before
            .iter()
            .map(|(id, _)| id)
            .eq(after.iter().map(|(id, _)| id))
        {
            return Err(mismatch());
        }
    }
    let child_types = child_types(data_type);
    if child_types.len() != data.child_data().len() {
        return Err(mismatch());
    }
    let children = data
        .child_data()
        .iter()
        .zip(child_types)
        .map(|(child, data_type)| with_schema_type(child.clone(), data_type))
        .collect::<Result<Vec<_>>>()?;
    // Safe Arrow union constructors can retain inconsistent declared child
    // types. Keep validation even when only changing map field names.
    data.into_builder()
        .data_type(data_type.clone())
        .child_data(children)
        .build()
        .map_err(arrow_error)
}

/// Arrow permits arbitrary map child names, but Spark's ArrowColumnVector looks
/// up map keys and values by name. Rewrite only those fields, including nested
/// maps, while sharing the original value, offset, and validity buffers.
/// When a map schema changes, checked reconstruction also reconciles column
/// names and metadata with that schema, rejecting incompatible union order.
fn normalize_map_names(batches: &[RecordBatch]) -> Result<Cow<'_, [RecordBatch]>> {
    // Do not let canonicalization conceal a mismatch in the supplied schemas.
    validate(batches)?;
    let Some(first) = batches.first() else {
        return Ok(Cow::Borrowed(batches));
    };
    let original = first.schema();
    let fields = original
        .fields()
        .iter()
        .map(normalize_map_field)
        .collect::<Result<Vec<_>>>()?;
    if fields.as_slice() == original.fields().as_ref() {
        return Ok(Cow::Borrowed(batches));
    }
    let schema = Arc::new(Schema::new_with_metadata(
        fields,
        original.metadata().clone(),
    ));
    batches
        .iter()
        .map(|batch| {
            let columns = batch
                .columns()
                .iter()
                .zip(schema.fields())
                .map(|(array, field)| {
                    if array.data_type() == field.data_type() {
                        Ok(Arc::clone(array))
                    } else {
                        with_schema_type(array.to_data(), field.data_type()).map(make_array)
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            RecordBatch::try_new_with_options(
                Arc::clone(&schema),
                columns,
                &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
            )
            .map_err(arrow_error)
        })
        .collect::<Result<Vec<_>>>()
        .map(Cow::Owned)
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
    let batches = normalize_map_names(batches)?;
    write_inline_relation(&batches, schema)
}

fn write_inline_relation(batches: &[RecordBatch], schema: Option<DataType>) -> Result<LogicalPlan> {
    let data = if let Some(first) = batches.first() {
        let mut writer = StreamWriter::try_new(Vec::new(), &first.schema()).map_err(arrow_error)?;
        for batch in batches {
            writer.write(batch).map_err(arrow_error)?;
        }
        Some(writer.into_inner().map_err(arrow_error)?)
    } else {
        None
    };
    Ok(LogicalPlan::from_builder(
        LogicalPlanKind::LocalRelation { schema, data },
        [],
    ))
}

pub(crate) fn create_relation(
    client: &SparkConnectClient,
    batches: Vec<RecordBatch>,
    schema: Option<DataType>,
    options: Option<&ArrowInputOptions>,
) -> Result<LogicalPlan> {
    // Preserve the offline empty-input path. Both paths validate the original
    // schemas and normalize map names exactly once, before any RPC.
    if batches.iter().all(|batch| batch.num_rows() == 0) {
        return inline_relation(&batches, schema);
    }
    let batches = normalize_map_names(&batches)?;
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
        return write_inline_relation(&batches, schema);
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
    Ok(LogicalPlan::from_builder(
        LogicalPlanKind::ChunkedCachedLocalRelation {
            data_hashes: hashes,
            schema_hash,
        },
        [],
    ))
}

fn new_writer(schema: &SchemaRef) -> Result<StreamWriter<Vec<u8>>> {
    StreamWriter::try_new(Vec::new(), schema).map_err(arrow_error)
}

/// Match PySpark's pre-write estimate: stream schema plus each batch's data size.
/// The row setting caps individual record batches, not the complete IPC stream.
/// Callers supply batches whose schemas and map field names are already prepared.
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
    use arrow::array::builder::Int64Builder;
    use arrow::array::{
        Array, ArrayRef, DictionaryArray, FixedSizeListArray, Int32Array, Int64Array, Int8Array,
        ListArray, MapArray, MapBuilder, RunArray, StructArray, UnionArray,
    };
    use arrow::buffer::{NullBuffer, OffsetBuffer};
    use arrow::datatypes::{DataType as ArrowType, Field, Int32Type, Int8Type, Schema};
    use arrow::ipc::reader::StreamReader;
    use std::io::Cursor;
    use std::ops::Range;
    use std::sync::Arc;

    fn batch(values: Range<i64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("id", ArrowType::Int64, true)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from_iter_values(values))]).unwrap()
    }

    fn map(rows: usize) -> MapArray {
        let mut builder = MapBuilder::new(None, Int64Builder::new(), Int64Builder::new());
        for row in 0..rows {
            if row % 3 == 1 {
                builder.append(false).unwrap();
            } else {
                if row % 3 == 0 {
                    builder.keys().append_value(row as i64);
                    builder.values().append_value(row as i64 + 10);
                    builder.keys().append_value(row as i64 + 100);
                    builder.values().append_null();
                }
                builder.append(true).unwrap();
            }
        }
        builder.finish()
    }

    fn map_batch(rows: usize) -> RecordBatch {
        let map: ArrayRef = Arc::new(map(rows));
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "mapping",
                map.data_type().clone(),
                true,
            )])),
            vec![map],
        )
        .unwrap()
    }

    fn map_fields(data_type: &ArrowType) -> (&FieldRef, &arrow::datatypes::Fields, bool) {
        let ArrowType::Map(entries, sorted) = data_type else {
            panic!("expected map");
        };
        let ArrowType::Struct(fields) = entries.data_type() else {
            panic!("expected entries struct");
        };
        (entries, fields, *sorted)
    }

    fn assert_canonical_map(data_type: &ArrowType) {
        let (entries, fields, _) = map_fields(data_type);
        assert_eq!(entries.name(), "entries");
        assert_eq!(fields[0].name(), "key");
        assert_eq!(fields[1].name(), "value");
    }

    fn assert_shared_buffers(before: &ArrayData, after: &ArrayData) {
        assert_eq!(
            (before.len(), before.offset()),
            (after.len(), after.offset())
        );
        assert_eq!(before.buffers().len(), after.buffers().len());
        for (before, after) in before.buffers().iter().zip(after.buffers()) {
            assert!(before.ptr_eq(after), "normalization copied an Arrow buffer");
        }
        match (before.nulls(), after.nulls()) {
            (Some(before), Some(after)) => assert!(before.inner().ptr_eq(after.inner())),
            (None, None) => {}
            _ => panic!("normalization changed the validity bitmap"),
        }
        assert_eq!(before.child_data().len(), after.child_data().len());
        for (before, after) in before.child_data().iter().zip(after.child_data()) {
            assert_shared_buffers(before, after);
        }
    }

    fn normalize_array(array: ArrayRef) -> ArrayRef {
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "user_column",
                array.data_type().clone(),
                true,
            )])),
            vec![Arc::clone(&array)],
        )
        .unwrap();
        let normalized = normalize_map_names(std::slice::from_ref(&batch)).unwrap();
        let result = Arc::clone(normalized[0].column(0));
        assert_shared_buffers(&array.to_data(), &result.to_data());
        result.to_data().validate_full().unwrap();
        result
    }

    fn assert_map_values_unchanged(before: &ArrayRef, after: &ArrayRef) {
        let before = before.as_any().downcast_ref::<MapArray>().unwrap();
        let after = after.as_any().downcast_ref::<MapArray>().unwrap();
        assert_canonical_map(after.data_type());
        assert_eq!(before.len(), after.len());
        assert_eq!(before.offsets(), after.offsets());
        assert_eq!(before.nulls(), after.nulls());
        assert_eq!(before.keys().to_data(), after.keys().to_data());
        assert_eq!(before.values().to_data(), after.values().to_data());
    }

    #[test]
    fn map_names_are_canonical_in_inline_and_chunked_ipc() {
        let batch = map_batch(5).slice(1, 4);
        let expected = normalize_map_names(std::slice::from_ref(&batch))
            .unwrap()
            .into_owned()
            .remove(0);
        let before = batch.column(0).as_any().downcast_ref::<MapArray>().unwrap();
        let after = expected
            .column(0)
            .as_any()
            .downcast_ref::<MapArray>()
            .unwrap();
        assert_canonical_map(after.data_type());
        assert_eq!(before.keys().to_data(), after.keys().to_data());
        assert_eq!(before.values().to_data(), after.values().to_data());
        assert_eq!(before.offsets(), after.offsets());

        let relation = inline_relation(std::slice::from_ref(&batch), None).unwrap();
        let LogicalPlanKind::LocalRelation {
            data: Some(data), ..
        } = relation.kind()
        else {
            panic!("expected inline data");
        };
        let mut reader = StreamReader::try_new(Cursor::new(data.clone()), None).unwrap();
        assert_canonical_map(reader.schema().field(0).data_type());
        assert_eq!(reader.next().unwrap().unwrap(), expected);
        assert!(reader.next().is_none());

        let options = Options {
            size_limit: usize::MAX,
            chunk_rows: 2,
            chunk_bytes: 1,
            upload_batch_bytes: usize::MAX,
        };
        let mut streams = Vec::new();
        serialize_chunks(std::slice::from_ref(&expected), &options, &mut |data| {
            streams.push(data);
            Ok(())
        })
        .unwrap();
        assert_eq!(streams.len(), 2);
        let mut offset = 0;
        for data in streams {
            let reader = StreamReader::try_new(Cursor::new(data), None).unwrap();
            assert_canonical_map(reader.schema().field(0).data_type());
            for batch in reader {
                let batch = batch.unwrap();
                assert_eq!(batch, expected.slice(offset, batch.num_rows()));
                offset += batch.num_rows();
            }
        }
        assert_eq!(offset, expected.num_rows());
    }

    #[test]
    fn nested_sliced_maps_share_buffers_and_preserve_field_attributes() {
        let tagged = |name: &str, data_type, nullable| {
            Field::new(name, data_type, nullable)
                .with_metadata(HashMap::from([("tag".into(), name.into())]))
        };
        let nested: ArrayRef = Arc::new(map(4));
        let entries = StructArray::new(
            vec![
                tagged("keys", ArrowType::Int64, false),
                tagged("values", nested.data_type().clone(), true),
            ]
            .into(),
            vec![Arc::new(Int64Array::from(vec![1, 2, 3, 4])), nested],
            None,
        );
        let outer: ArrayRef = Arc::new(MapArray::new(
            Arc::new(tagged("pairs", entries.data_type().clone(), false)),
            OffsetBuffer::new(vec![0, 1, 2, 3, 4].into()),
            entries,
            Some(NullBuffer::from(vec![true, false, true, true])),
            true,
        ));
        let list: ArrayRef = Arc::new(ListArray::new(
            Arc::new(tagged("user_items", outer.data_type().clone(), true)),
            OffsetBuffer::new(vec![0, 1, 3, 4].into()),
            outer,
            Some(NullBuffer::from(vec![true, false, true])),
        ));
        let record: ArrayRef = Arc::new(StructArray::new(
            vec![tagged("user_lists", list.data_type().clone(), true)].into(),
            vec![list],
            Some(NullBuffer::from(vec![true, false, true])),
        ));
        let schema = Schema::new_with_metadata(
            vec![
                tagged("user_record", record.data_type().clone(), true),
                tagged("untouched", ArrowType::Int64, false),
            ],
            HashMap::from([("schema_tag".into(), "preserved".into())]),
        );
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![record, Arc::new(Int64Array::from(vec![11, 22, 33]))],
        )
        .unwrap()
        .slice(1, 2);
        let normalized = normalize_map_names(std::slice::from_ref(&batch))
            .unwrap()
            .into_owned()
            .remove(0);
        assert_eq!(normalized.num_rows(), 2);
        assert_eq!(batch.schema().metadata(), normalized.schema().metadata());
        assert!(Arc::ptr_eq(batch.column(1), normalized.column(1)));
        for (before, after) in batch.columns().iter().zip(normalized.columns()) {
            assert_shared_buffers(&before.to_data(), &after.to_data());
            after.to_data().validate_full().unwrap();
        }
        let original = batch.schema();
        let result = normalized.schema();
        let mut before = original.field(0);
        let mut after = result.field(0);
        for expected_name in ["user_record", "user_lists", "user_items"] {
            assert_eq!(after.name(), expected_name);
            assert_eq!(before.metadata(), after.metadata());
            assert_eq!(before.is_nullable(), after.is_nullable());
            if expected_name != "user_items" {
                before = match before.data_type() {
                    ArrowType::Struct(fields) => fields[0].as_ref(),
                    ArrowType::List(field) => field.as_ref(),
                    _ => panic!("expected user container"),
                };
                after = match after.data_type() {
                    ArrowType::Struct(fields) => fields[0].as_ref(),
                    ArrowType::List(field) => field.as_ref(),
                    _ => panic!("expected user container"),
                };
            }
        }
        let (old_entries, old_fields, old_sorted) = map_fields(before.data_type());
        let (entries, fields, sorted) = map_fields(after.data_type());
        assert_canonical_map(after.data_type());
        assert!(old_sorted && sorted);
        assert_eq!(old_entries.metadata(), entries.metadata());
        assert_eq!(old_entries.is_nullable(), entries.is_nullable());
        for (before, after) in old_fields.iter().zip(fields) {
            assert_eq!(before.metadata(), after.metadata());
            assert_eq!(before.is_nullable(), after.is_nullable());
        }
        assert_canonical_map(fields[1].data_type());
        assert!(!map_fields(fields[1].data_type()).2);
    }

    #[test]
    fn original_map_schema_mismatches_are_rejected_before_normalization() {
        fn assert_schema_error<T>(result: Result<T>) {
            let Err(error) = result else {
                panic!("expected the original batch-schema mismatch");
            };
            assert_eq!(error.message(), "Arrow batch schemas must match");
        }
        let session = crate::SparkSession::builder()
            .remote("sc://127.0.0.1:1")
            .get_or_create()
            .unwrap();
        for rows in [0, 1] {
            let raw = map_batch(rows);
            let canonical = normalize_map_names(std::slice::from_ref(&raw))
                .unwrap()
                .into_owned()
                .remove(0);
            let batches = vec![raw, canonical];
            assert_schema_error(normalize_map_names(&batches));
            assert_schema_error(inline_relation(&batches, None));
            assert_schema_error(session.create_dataframe_from_arrow(batches, None));
        }
    }

    #[test]
    fn reordered_union_schema_fields_are_rejected_before_relabeling_children() {
        let maps: ArrayRef = Arc::new(map(1));
        let fields = [
            (0, Arc::new(Field::new("small", ArrowType::Int32, false))),
            (1, Arc::new(Field::new("large", ArrowType::Int64, false))),
            (
                2,
                Arc::new(Field::new("maps", maps.data_type().clone(), true)),
            ),
        ];
        let array: ArrayRef = Arc::new(
            UnionArray::try_new(
                fields.clone().into_iter().collect(),
                vec![0, 1, 2].into(),
                Some(vec![0, 0, 0].into()),
                vec![
                    Arc::new(Int32Array::from(vec![42])),
                    Arc::new(Int64Array::from(vec![84])),
                    maps,
                ],
            )
            .unwrap(),
        );
        array.to_data().validate_full().unwrap();
        let schema_type = ArrowType::Union(
            [fields[1].clone(), fields[0].clone(), fields[2].clone()]
                .into_iter()
                .collect(),
            arrow::datatypes::UnionMode::Dense,
        );
        assert_ne!(array.data_type(), &schema_type);
        assert!(array.data_type().equals_datatype(&schema_type));
        let batch = RecordBatch::try_new_with_options(
            Arc::new(Schema::new(vec![Field::new(
                "user_union",
                schema_type,
                true,
            )])),
            vec![array],
            &RecordBatchOptions::new().with_match_field_names(false),
        )
        .unwrap();
        let session = crate::SparkSession::builder()
            .remote("sc://127.0.0.1:1")
            .get_or_create()
            .unwrap();
        for result in [
            normalize_map_names(std::slice::from_ref(&batch)).map(|_| ()),
            inline_relation(std::slice::from_ref(&batch), None).map(|_| ()),
            session
                .create_dataframe_from_arrow(vec![batch], None)
                .map(|_| ()),
        ] {
            let Err(error) = result else {
                panic!("reordered union fields must not relabel physical children");
            };
            assert_eq!(
                error.message(),
                "Arrow column type does not match its schema field"
            );
        }
    }

    #[test]
    fn inconsistent_union_child_types_are_rejected_without_reinterpretation() {
        let session = crate::SparkSession::builder()
            .remote("sc://127.0.0.1:1")
            .get_or_create()
            .unwrap();
        // The second fixture's Int32 buffer has enough bytes for one Int64;
        // checking buffer sizes alone would not prove retyping was rejected.
        for values in [vec![42], vec![42, 43]] {
            let maps: ArrayRef = Arc::new(map(1));
            let array: ArrayRef = Arc::new(
                UnionArray::try_new(
                    [
                        (0, Arc::new(Field::new("wide", ArrowType::Int64, false))),
                        (
                            1,
                            Arc::new(Field::new("maps", maps.data_type().clone(), true)),
                        ),
                    ]
                    .into_iter()
                    .collect(),
                    vec![0, 1].into(),
                    Some(vec![0, 0].into()),
                    vec![Arc::new(Int32Array::from(values)), maps],
                )
                .unwrap(),
            );
            // Arrow 59.3 accepts this through its safe constructor despite the
            // declared Int64 field containing an Int32 child.
            assert!(array.to_data().validate_full().is_err());
            let batch = RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "user_union",
                    array.data_type().clone(),
                    true,
                )])),
                vec![array],
            )
            .unwrap();
            for result in [
                normalize_map_names(std::slice::from_ref(&batch)).map(|_| ()),
                inline_relation(std::slice::from_ref(&batch), None).map(|_| ()),
                session
                    .create_dataframe_from_arrow(vec![batch], None)
                    .map(|_| ()),
            ] {
                let Err(error) = result else {
                    panic!("declared union types must not reinterpret physical children");
                };
                assert_eq!(
                    error.message(),
                    "Arrow column type does not match its schema field"
                );
            }
        }
    }

    #[test]
    fn schema_field_renames_around_maps_use_checked_reconstruction() {
        let maps: ArrayRef = Arc::new(map(3));
        let array: ArrayRef = Arc::new(StructArray::new(
            vec![Field::new("array_maps", maps.data_type().clone(), true)].into(),
            vec![Arc::clone(&maps)],
            None,
        ));
        let field = Field::new("schema_maps", maps.data_type().clone(), true)
            .with_metadata(HashMap::from([("schema_tag".into(), "preserved".into())]));
        let batch = RecordBatch::try_new_with_options(
            Arc::new(Schema::new(vec![Field::new(
                "user_record",
                ArrowType::Struct(vec![field].into()),
                true,
            )])),
            vec![Arc::clone(&array)],
            &RecordBatchOptions::new().with_match_field_names(false),
        )
        .unwrap();
        let normalized = normalize_map_names(std::slice::from_ref(&batch)).unwrap();
        let after = normalized[0].column(0);
        assert_shared_buffers(&array.to_data(), &after.to_data());
        after.to_data().validate_full().unwrap();
        let ArrowType::Struct(fields) = after.data_type() else {
            panic!("expected struct around map");
        };
        assert_eq!(fields[0].name(), "schema_maps");
        assert_eq!(fields[0].metadata().get("schema_tag").unwrap(), "preserved");
        let after = after.as_any().downcast_ref::<StructArray>().unwrap();
        assert_map_values_unchanged(&maps, after.column(0));
        let relation = inline_relation(std::slice::from_ref(&batch), None).unwrap();
        let LogicalPlanKind::LocalRelation {
            data: Some(data), ..
        } = relation.kind()
        else {
            panic!("expected inline relation");
        };
        let mut reader = StreamReader::try_new(Cursor::new(data.clone()), None).unwrap();
        let decoded = reader.next().unwrap().unwrap();
        decoded.column(0).to_data().validate_full().unwrap();
        assert_eq!(decoded.schema(), normalized[0].schema());
    }

    #[test]
    fn public_zero_row_arrow_input_canonicalizes_maps_without_a_config_rpc() {
        let session = crate::SparkSession::builder()
            .remote("sc://127.0.0.1:1")
            .get_or_create()
            .unwrap();
        let dataframe = session
            .create_dataframe_from_arrow(vec![map_batch(0)], None)
            .unwrap();
        let Some(spark_connect_proto::relation::RelType::LocalRelation(local)) =
            dataframe.to_relation().unwrap().rel_type
        else {
            panic!("expected zero-row inline relation");
        };
        assert!(local.schema.is_none());
        let reader = StreamReader::try_new(Cursor::new(local.data.unwrap()), None).unwrap();
        assert_canonical_map(reader.schema().field(0).data_type());
        let batches = reader.collect::<std::result::Result<Vec<_>, _>>().unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 0);
    }

    #[test]
    fn batches_without_maps_preserve_schema_columns_and_zero_column_row_counts() {
        let batch = batch(0..2);
        let batches = std::slice::from_ref(&batch);
        let normalized = normalize_map_names(batches).unwrap();
        assert!(matches!(normalized, Cow::Borrowed(_)));
        assert!(std::ptr::eq(normalized.as_ptr(), batches.as_ptr()));
        let normalized = &normalized[0];
        assert!(Arc::ptr_eq(&batch.schema(), &normalized.schema()));
        assert!(Arc::ptr_eq(batch.column(0), normalized.column(0)));
        let no_columns = RecordBatch::try_new_with_options(
            Arc::new(Schema::empty()),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(7)),
        )
        .unwrap();
        assert_eq!(normalize_map_names(&[no_columns]).unwrap()[0].num_rows(), 7);
    }

    #[test]
    fn canonical_schema_with_default_map_column_writes_a_canonical_ipc_header() {
        let original = map_batch(3);
        let field = normalize_map_field(&original.schema().fields()[0]).unwrap();
        let batch = RecordBatch::try_new_with_options(
            Arc::new(Schema::new(vec![field])),
            original.columns().to_vec(),
            &RecordBatchOptions::new().with_match_field_names(false),
        )
        .unwrap();
        let prepared = normalize_map_names(std::slice::from_ref(&batch)).unwrap();
        assert!(matches!(prepared, Cow::Borrowed(_)));
        let relation = inline_relation(std::slice::from_ref(&batch), None).unwrap();
        let LogicalPlanKind::LocalRelation {
            data: Some(data), ..
        } = relation.kind()
        else {
            panic!("expected inline relation");
        };
        let mut reader = StreamReader::try_new(Cursor::new(data.clone()), None).unwrap();
        assert_canonical_map(reader.schema().field(0).data_type());
        let decoded = reader.next().unwrap().unwrap();
        assert_canonical_map(decoded.schema().field(0).data_type());
        decoded.column(0).to_data().validate_full().unwrap();
        assert_map_values_unchanged(original.column(0), decoded.column(0));
        assert!(reader.next().is_none());
    }

    #[test]
    fn maps_are_normalized_inside_every_arrow_container() {
        fn containers(map: ArrowType) -> Vec<ArrowType> {
            let field = Arc::new(Field::new("user_item", map.clone(), true));
            vec![
                ArrowType::LargeList(Arc::clone(&field)),
                ArrowType::FixedSizeList(Arc::clone(&field), 2),
                ArrowType::ListView(Arc::clone(&field)),
                ArrowType::LargeListView(Arc::clone(&field)),
                ArrowType::Dictionary(Box::new(ArrowType::Int8), Box::new(map)),
                ArrowType::Union(
                    [(5, Arc::clone(&field))].into_iter().collect(),
                    arrow::datatypes::UnionMode::Dense,
                ),
                ArrowType::RunEndEncoded(
                    Arc::new(Field::new("user_ends", ArrowType::Int32, false)),
                    field,
                ),
            ]
        }
        let raw = map(0).data_type().clone();
        let canonical = ArrowType::Map(
            Arc::new(Field::new(
                "entries",
                ArrowType::Struct(
                    vec![
                        Field::new("key", ArrowType::Int64, false),
                        Field::new("value", ArrowType::Int64, true),
                    ]
                    .into(),
                ),
                false,
            )),
            false,
        );
        for (before, expected) in containers(raw).into_iter().zip(containers(canonical)) {
            let batch = RecordBatch::new_empty(Arc::new(Schema::new(vec![Field::new(
                "user_column",
                before,
                true,
            )])));
            let normalized = normalize_map_names(&[batch])
                .unwrap()
                .into_owned()
                .remove(0);
            assert_eq!(normalized.schema().field(0).data_type(), &expected);
            normalized.column(0).to_data().validate_full().unwrap();
        }
    }

    #[test]
    fn sliced_fixed_size_lists_keep_map_values_and_parent_validity() {
        let values: ArrayRef = Arc::new(map(8));
        let before = FixedSizeListArray::new(
            Arc::new(Field::new("user_item", values.data_type().clone(), true)),
            2,
            values,
            Some(NullBuffer::from(vec![true, false, true, true])),
        )
        .slice(1, 2);
        let normalized = normalize_array(Arc::new(before.clone()));
        let after = normalized
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        assert_eq!(after.len(), 2);
        assert_eq!(after.value_length(), 2);
        assert_eq!(before.nulls(), after.nulls());
        assert_eq!(before.value_offset(1), after.value_offset(1));
        assert_map_values_unchanged(before.values(), after.values());
    }

    #[test]
    fn sliced_dictionaries_keep_keys_and_unsliced_map_values() {
        let before = DictionaryArray::<Int8Type>::try_new(
            Int8Array::from(vec![Some(3), Some(0), None, Some(2), Some(3)]),
            Arc::new(map(4)),
        )
        .unwrap()
        .slice(1, 3);
        let normalized = normalize_array(Arc::new(before.clone()));
        let after = normalized
            .as_any()
            .downcast_ref::<DictionaryArray<Int8Type>>()
            .unwrap();
        assert_eq!(before.keys().to_data(), after.keys().to_data());
        assert_eq!(
            after.keys_iter().collect::<Vec<_>>(),
            vec![Some(0), None, Some(2)]
        );
        assert_eq!(after.values().len(), 4);
        assert_map_values_unchanged(before.values(), after.values());
    }

    #[test]
    fn sliced_dense_unions_keep_nonsequential_map_offsets_and_type_ids() {
        let values: ArrayRef = Arc::new(map(4));
        let before = UnionArray::try_new(
            [(
                5,
                Arc::new(Field::new("user_maps", values.data_type().clone(), true)),
            )]
            .into_iter()
            .collect(),
            vec![5, 5, 5, 5, 5].into(),
            Some(vec![3, 0, 2, 1, 3].into()),
            vec![values],
        )
        .unwrap()
        .slice(1, 3);
        let normalized = normalize_array(Arc::new(before.clone()));
        let after = normalized.as_any().downcast_ref::<UnionArray>().unwrap();
        assert_eq!(before.type_ids(), after.type_ids());
        assert_eq!(after.type_ids().as_ref(), &[5, 5, 5]);
        assert_eq!(before.offsets(), after.offsets());
        assert_eq!(
            (0..after.len())
                .map(|i| after.value_offset(i))
                .collect::<Vec<_>>(),
            vec![0, 2, 1]
        );
        assert_map_values_unchanged(before.child(5), after.child(5));
    }

    #[test]
    fn sliced_run_arrays_keep_physical_map_runs_and_logical_offsets() {
        let before = RunArray::<Int32Type>::try_new(&Int32Array::from(vec![2, 5, 6, 9]), &map(4))
            .unwrap()
            .slice(1, 6);
        let normalized = normalize_array(Arc::new(before.clone()));
        let after = normalized
            .as_any()
            .downcast_ref::<RunArray<Int32Type>>()
            .unwrap();
        assert_eq!(before.offset(), after.offset());
        assert_eq!(after.offset(), 1);
        assert_eq!(before.run_ends().values(), after.run_ends().values());
        assert_eq!(after.run_ends().values(), &[2, 5, 6, 9]);
        assert_eq!(
            (0..after.len())
                .map(|i| after.get_physical_index(i))
                .collect::<Vec<_>>(),
            vec![0, 1, 1, 1, 2, 3]
        );
        assert_map_values_unchanged(before.values(), after.values());
    }

    fn chunks(batch: &RecordBatch, chunk_rows: usize, chunk_bytes: usize) -> Vec<Vec<RecordBatch>> {
        let options = Options {
            size_limit: usize::MAX,
            chunk_rows,
            chunk_bytes,
            upload_batch_bytes: usize::MAX,
        };
        let mut chunks = Vec::new();
        let normalized = normalize_map_names(std::slice::from_ref(batch)).unwrap();
        serialize_chunks(&normalized, &options, &mut |chunk| {
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
