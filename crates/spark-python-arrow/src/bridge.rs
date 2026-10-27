// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Export Arrow record batches to PyArrow through the Arrow C Stream Interface.
//!
//! The caller supplies valid Rust record batches and owns the Python interpreter.
//! PyArrow's importers consume the exported descriptors and retain their buffers.

use std::ffi::{c_char, c_int, CString};
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use arrow::error::ArrowError;
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use arrow::ffi_stream::FFI_ArrowArrayStream;
use arrow::ipc::writer::FileWriter;
use arrow::record_batch::{RecordBatch, RecordBatchIterator};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

// Called after table export has routed NUL names and timezones to the IPC fallback.
fn export_c_schema(schema: &Schema) -> Result<FFI_ArrowSchema, ArrowError> {
    let data_type = DataType::Struct(schema.fields().clone());
    export_c_data_type(&data_type)?.with_metadata(schema.metadata())
}

fn validate_pyarrow_schema(py: Python<'_>, schema: &Schema) -> PyResult<()> {
    let mut schema = export_c_schema(schema).map_err(arrow_error)?;
    let address = &mut schema as *mut FFI_ArrowSchema as usize;
    py.import("pyarrow")?
        .getattr("Schema")?
        .call_method1("_import_from_c", (address,))?;
    Ok(())
}

/// Build a PyArrow table, sharing data buffers for C-compatible schemas.
///
/// Batch boundaries, schema metadata, and zero-column row counts are preserved.
/// An empty batch vector produces an empty table with an empty schema.
/// Value and offset buffers are shared; sliced validity bitmaps may be realigned.
/// Inconsistent batch schemas, including metadata and ordering flags, raise `ValueError`.
pub fn record_batches_to_pyarrow(py: Python<'_>, batches: Vec<RecordBatch>) -> PyResult<Py<PyAny>> {
    let schema = batches
        .first()
        .map(|batch| batch.schema())
        .unwrap_or_else(|| Arc::new(Schema::empty()));
    if batches.iter().any(|batch| {
        !Arc::ptr_eq(batch.schema_ref(), &schema) && !schemas_equal(batch.schema_ref(), &schema)
    }) {
        return Err(PyValueError::new_err(
            "Arrow record batches have inconsistent schemas",
        ));
    }
    let pa = py.import("pyarrow")?;
    if schema.fields().iter().any(|field| field_needs_ipc(field)) {
        // C schema strings cannot contain NUL. Use the already collected
        // batches for this rare compatibility path without repeating any RPC.
        let bytes = py
            .detach(|| -> arrow::error::Result<Vec<u8>> {
                let mut bytes = Vec::new();
                let mut writer = FileWriter::try_new(&mut bytes, &schema)?;
                for batch in &batches {
                    writer.write(batch)?;
                }
                writer.finish()?;
                drop(writer);
                Ok(bytes)
            })
            .map_err(arrow_error)?;
        return Ok(pa
            .getattr("ipc")?
            .call_method1("open_file", (PyBytes::new(py, &bytes),))?
            .call_method0("read_all")?
            .unbind());
    }

    // Older PyArrow versions can abort on unsupported stream schemas. Validate
    // the same schema through the fallible standalone importer before streaming.
    validate_pyarrow_schema(py, &schema)?;
    let batches = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    let mut stream = with_exported_schema(FFI_ArrowArrayStream::new(Box::new(batches)), schema);
    let address = &mut stream as *mut FFI_ArrowArrayStream as usize;
    // The importer moves the stream and clears its release callback. Rust drops
    // any unconsumed stream on failure; imported arrays retain their own buffers.
    // The pointer importer is available at our minimum supported PyArrow (10).
    let reader = pa
        .getattr("RecordBatchReader")?
        .call_method1("_import_from_c", (address,))?;
    Ok(reader.call_method0("read_all")?.unbind())
}

fn arrow_error(error: ArrowError) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

// Arrow 58's stream exporter needs the same sorted-map schema repair as the
// standalone exporter below. Own the native stream and override only its schema
// conversion; array export, iteration, and remaining-batch cleanup stay in Arrow.
struct ExportStream {
    inner: FFI_ArrowArrayStream,
    schema: Arc<Schema>,
    schema_error: Option<CString>,
}

fn with_exported_schema(inner: FFI_ArrowArrayStream, schema: Arc<Schema>) -> FFI_ArrowArrayStream {
    let state = Box::new(ExportStream {
        inner,
        schema,
        schema_error: None,
    });
    FFI_ArrowArrayStream {
        get_schema: Some(stream_schema),
        get_next: Some(stream_next),
        get_last_error: Some(stream_error),
        release: Some(stream_release),
        private_data: Box::into_raw(state).cast(),
    }
}

// These callbacks run without the GIL and must not call Python. The C Stream
// contract gives each callback exclusive access to a live stream and, where
// applicable, writable output storage. private_data is our Box until release.
unsafe extern "C" fn stream_schema(
    stream: *mut FFI_ArrowArrayStream,
    out: *mut FFI_ArrowSchema,
) -> c_int {
    // SAFETY: the consumer supplies the live stream constructed above.
    let state = unsafe { &mut *(*stream).private_data.cast::<ExportStream>() };
    state.schema_error = None;
    match export_c_schema(&state.schema) {
        Ok(schema) => {
            // SAFETY: out is writable. Each call transfers a fresh descriptor,
            // which the consumer releases independently of the stream.
            unsafe { std::ptr::write(out, schema) };
            0
        }
        Err(error) => {
            // Error strings may contain NUL; never panic across the C boundary.
            state.schema_error =
                Some(CString::new(error.to_string().replace('\0', "\\0")).unwrap_or_default());
            22 // EINVAL, specified by the C Stream interface.
        }
    }
}

unsafe extern "C" fn stream_next(
    stream: *mut FFI_ArrowArrayStream,
    out: *mut FFI_ArrowArray,
) -> c_int {
    // SAFETY: the consumer supplies the live stream constructed above.
    let state = unsafe { &mut *(*stream).private_data.cast::<ExportStream>() };
    state.schema_error = None;
    match state.inner.get_next {
        // SAFETY: delegate with the inner stream's own private_data, not ours.
        Some(next) => unsafe { next(&mut state.inner, out) },
        None => {
            state.schema_error = Some(c"Arrow stream has no get_next callback".into());
            22
        }
    }
}

unsafe extern "C" fn stream_error(stream: *mut FFI_ArrowArrayStream) -> *const c_char {
    // SAFETY: the consumer supplies the live stream constructed above.
    let state = unsafe { &mut *(*stream).private_data.cast::<ExportStream>() };
    if let Some(error) = &state.schema_error {
        return error.as_ptr();
    }
    match state.inner.get_last_error {
        // SAFETY: the inner error string stays owned by the inner stream.
        Some(error) => unsafe { error(&mut state.inner) },
        None => std::ptr::null(),
    }
}

unsafe extern "C" fn stream_release(stream: *mut FFI_ArrowArrayStream) {
    if stream.is_null() {
        return;
    }
    // SAFETY: the consumer supplies writable storage for this stream.
    let stream = unsafe { &mut *stream };
    if stream.release.is_none() {
        return;
    }
    let state = stream.private_data.cast::<ExportStream>();
    stream.release = None;
    stream.private_data = std::ptr::null_mut();
    stream.get_schema = None;
    stream.get_next = None;
    stream.get_last_error = None;
    // SAFETY: this is the Box installed above, consumed exactly once. Dropping
    // the inner stream releases unread batches, not independently exported arrays.
    drop(unsafe { Box::from_raw(state) });
}

// Arrow 58's Field conversion overwrites a map's type flags with its field flags.
// Rebuild the descriptor tree using public constructors, retaining both. Only
// descriptors are rebuilt: array buffers are not involved in schema conversion.
fn export_c_data_type(data_type: &DataType) -> Result<FFI_ArrowSchema, ArrowError> {
    fn has_sorted_map(data_type: &DataType) -> bool {
        matches!(data_type, DataType::Map(_, true))
            || child_fields(data_type)
                .iter()
                .any(|field| has_sorted_map(field.data_type()))
            || matches!(data_type, DataType::Dictionary(_, values) if has_sorted_map(values))
    }

    fn rebuild(
        source: &FFI_ArrowSchema,
        data_type: &DataType,
    ) -> Result<FFI_ArrowSchema, ArrowError> {
        let children = child_fields(data_type)
            .into_iter()
            .enumerate()
            .map(|(index, field)| rebuild(source.child(index), field.data_type()))
            .collect::<Result<Vec<_>, _>>()?;
        let dictionary = match (source.dictionary(), data_type) {
            (Some(schema), DataType::Dictionary(_, values)) => Some(rebuild(schema, values)?),
            _ => None,
        };
        let mut schema = FFI_ArrowSchema::try_new(source.format(), children, dictionary)?
            .with_metadata(source.metadata()?)?;
        if let Some(name) = source.name() {
            schema = schema.with_name(name)?;
        }
        schema.flags = source.flags;
        if matches!(data_type, DataType::Map(_, true)) {
            // ARROW_FLAG_MAP_KEYS_SORTED in the public Arrow C Data Interface.
            schema.flags |= 0b100;
        }
        Ok(schema)
    }

    let schema = FFI_ArrowSchema::try_from(data_type)?;
    if has_sorted_map(data_type) {
        rebuild(&schema, data_type)
    } else {
        Ok(schema)
    }
}

fn child_fields(data_type: &DataType) -> Vec<&Field> {
    match data_type {
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => vec![field],
        DataType::Struct(fields) => fields.iter().map(AsRef::as_ref).collect(),
        DataType::Union(fields, _) => fields.iter().map(|(_, field)| field.as_ref()).collect(),
        DataType::RunEndEncoded(run_ends, values) => vec![run_ends, values],
        _ => vec![],
    }
}

fn schemas_equal(left: &Schema, right: &Schema) -> bool {
    fn same_order(left: &DataType, right: &DataType) -> bool {
        let fields_match =
            child_fields(left)
                .into_iter()
                .zip(child_fields(right))
                .all(|(left, right)| {
                    left.dict_is_ordered() == right.dict_is_ordered()
                        && same_order(left.data_type(), right.data_type())
                });
        fields_match
            && match (left, right) {
                (DataType::Dictionary(_, left), DataType::Dictionary(_, right)) => {
                    same_order(left, right)
                }
                _ => true,
            }
    }

    // Arrow's structural equality intentionally excludes dictionary-order flags.
    // PyArrow includes them, so a table must not relabel later batches' ordering.
    left == right
        && same_order(
            &DataType::Struct(left.fields().clone()),
            &DataType::Struct(right.fields().clone()),
        )
}

fn field_needs_ipc(field: &Field) -> bool {
    field.name().contains('\0') || data_type_needs_ipc(field.data_type())
}

fn data_type_needs_ipc(data_type: &DataType) -> bool {
    match data_type {
        DataType::Timestamp(_, Some(timezone)) => timezone.contains('\0'),
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => field_needs_ipc(field),
        DataType::Struct(fields) => fields.iter().any(|field| field_needs_ipc(field)),
        DataType::Union(fields, _) => fields.iter().any(|(_, field)| field_needs_ipc(field)),
        DataType::RunEndEncoded(run_ends, values) => {
            field_needs_ipc(run_ends) || field_needs_ipc(values)
        }
        DataType::Dictionary(keys, values) => {
            data_type_needs_ipc(keys) || data_type_needs_ipc(values)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::ffi::CStr;

    use arrow::datatypes::TimeUnit;

    use super::*;

    #[test]
    fn invalid_time_type_is_rejected_without_panicking() {
        assert!(export_c_data_type(&DataType::Time32(TimeUnit::Nanosecond)).is_err());
    }

    #[test]
    fn stream_schema_can_be_requested_and_released_repeatedly() {
        let entries = Field::new(
            "entries",
            DataType::Struct(
                vec![
                    Field::new("key", DataType::Utf8, false),
                    Field::new("value", DataType::Int64, true),
                ]
                .into(),
            ),
            false,
        );
        let schema = Arc::new(Schema::new_with_metadata(
            vec![Field::new(
                "maps",
                DataType::Map(Arc::new(entries), true),
                true,
            )],
            HashMap::from([("key".into(), "value\0metadata".into())]),
        ));
        let batches = RecordBatchIterator::new(std::iter::empty(), schema.clone());
        let mut stream = with_exported_schema(FFI_ArrowArrayStream::new(Box::new(batches)), schema);
        let mut exported = Vec::new();
        for _ in 0..3 {
            let mut out = FFI_ArrowSchema::empty();
            // SAFETY: both descriptors are live and out is writable.
            assert_eq!(unsafe { stream_schema(&mut stream, &mut out) }, 0);
            assert_ne!(out.child(0).flags & 0b100, 0);
            assert_eq!(out.metadata().unwrap()["key"], "value\0metadata");
            exported.push(out);
        }
        drop(stream);
        // Schema descriptors have their own owners after the stream is gone.
        for schema in exported {
            assert_eq!(schema.child(0).name(), Some("maps"));
        }
    }

    #[test]
    fn stream_schema_errors_have_stable_storage_until_next_operation() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "invalid",
            DataType::Time32(TimeUnit::Nanosecond),
            true,
        )]));
        let batches = RecordBatchIterator::new(std::iter::empty(), schema.clone());
        let mut stream = with_exported_schema(FFI_ArrowArrayStream::new(Box::new(batches)), schema);
        let mut out = FFI_ArrowSchema::empty();
        // SAFETY: the stream and output are live, writable descriptors.
        assert_eq!(unsafe { stream_schema(&mut stream, &mut out) }, 22);
        // SAFETY: the error pointer is owned by the stream until the next operation.
        let error = unsafe { stream_error(&mut stream) };
        assert!(!error.is_null());
        assert_eq!(error, unsafe { stream_error(&mut stream) });
        assert!(!unsafe { CStr::from_ptr(error) }.to_bytes().is_empty());
        let mut array = FFI_ArrowArray::empty();
        assert_eq!(unsafe { stream_next(&mut stream, &mut array) }, 0);
        assert!(unsafe { stream_error(&mut stream) }.is_null());
    }

    #[test]
    fn stream_forwards_inner_iteration_errors() {
        let schema = Arc::new(Schema::empty());
        let batches = RecordBatchIterator::new(
            std::iter::once(Err(ArrowError::InvalidArgumentError(
                "test read failure".into(),
            ))),
            schema.clone(),
        );
        let mut stream = with_exported_schema(FFI_ArrowArrayStream::new(Box::new(batches)), schema);
        let mut array = FFI_ArrowArray::empty();
        // SAFETY: the live stream owns the error string, and array is writable.
        assert_eq!(unsafe { stream_next(&mut stream, &mut array) }, 22);
        let message = unsafe { CStr::from_ptr(stream_error(&mut stream)) };
        assert!(message.to_str().unwrap().contains("test read failure"));
    }
}
