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

//! Export Spark result batches through arrow-rs's PyArrow table conversion.
//!
//! The caller supplies valid Rust record batches and owns the Python interpreter.
//! PyArrow's importers consume the exported descriptors and retain their buffers.

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use arrow::error::ArrowError;
use arrow::ffi::FFI_ArrowSchema;
use arrow::ipc::writer::FileWriter;
use arrow::pyarrow::{IntoPyArrow, Table};
use arrow::record_batch::RecordBatch;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

// Called after table export has routed NUL names and timezones to the IPC fallback.
fn validate_pyarrow_schema(py: Python<'_>, schema: &Schema) -> PyResult<()> {
    let mut schema = FFI_ArrowSchema::try_from(schema).map_err(arrow_error)?;
    let address = &mut schema as *mut FFI_ArrowSchema as usize;
    py.import("pyarrow")?
        .getattr("Schema")?
        .call_method1("_import_from_c", (address,))?;
    Ok(())
}

/// Build a PyArrow table from Spark result batches, sharing compatible buffers.
///
/// Batch boundaries, schema metadata, and zero-column row counts are preserved.
/// An empty batch vector produces an empty table with an empty schema.
/// Value and offset buffers are shared; sliced validity bitmaps may be realigned.
/// Inconsistent batch schemas, including metadata, raise `ValueError`.
/// Sorted-map flags and dictionary encoding are outside the Spark result contract.
pub fn record_batches_to_pyarrow(py: Python<'_>, batches: Vec<RecordBatch>) -> PyResult<Py<PyAny>> {
    let schema = batches
        .first()
        .map(|batch| batch.schema())
        .unwrap_or_else(|| Arc::new(Schema::empty()));
    // Validate before either export path, including the NUL compatibility path.
    let table = Table::try_new(batches, schema.clone())
        .map_err(|_| PyValueError::new_err("Arrow record batches have inconsistent schemas"))?;
    let pa = py.import("pyarrow")?;
    if schema.fields().iter().any(|field| field_needs_ipc(field)) {
        // C schema strings cannot contain NUL. Use the already collected
        // batches for this rare compatibility path without repeating any RPC.
        let bytes = py
            .detach(|| -> arrow::error::Result<Vec<u8>> {
                let mut bytes = Vec::new();
                let mut writer = FileWriter::try_new(&mut bytes, &schema)?;
                for batch in table.record_batches() {
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

    // Arrow's table helper imports batches before its final schema. Preflight
    // first so older PyArrow versions can reject unsupported stream schemas.
    validate_pyarrow_schema(py, &schema)?;
    Ok(table.into_pyarrow(py)?.unbind())
}

fn arrow_error(error: ArrowError) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
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
    use arrow::datatypes::TimeUnit;

    use super::*;

    #[test]
    fn invalid_time_type_is_rejected_without_panicking() {
        Python::initialize();
        Python::attach(|py| {
            let schema = Schema::new(vec![Field::new(
                "invalid",
                DataType::Time32(TimeUnit::Nanosecond),
                true,
            )]);
            let error = validate_pyarrow_schema(py, &schema).unwrap_err();
            assert!(error.is_instance_of::<PyRuntimeError>(py));
        });
    }
}
