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

//! Export Arrow record batches to PyArrow through the Arrow C Data Interface.
//!
//! The caller supplies valid Rust record batches and owns the Python interpreter.
//! PyArrow's importers consume the exported descriptors and retain their buffers.

use std::sync::Arc;

use arrow::array::{Array, StructArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::error::ArrowError;
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use arrow::ipc::writer::FileWriter;
use arrow::record_batch::RecordBatch;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};

// Called after table export has routed NUL names and timezones to the IPC fallback.
fn export_schema(py: Python<'_>, schema: &Schema) -> PyResult<Py<PyAny>> {
    let data_type = DataType::Struct(schema.fields().clone());
    let mut schema = export_c_data_type(&data_type)
        .and_then(|exported| exported.with_metadata(schema.metadata()))
        .map_err(arrow_error)?;
    let address = &mut schema as *mut FFI_ArrowSchema as usize;
    Ok(py
        .import("pyarrow")?
        .getattr("Schema")?
        .call_method1("_import_from_c", (address,))?
        .unbind())
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

    // Import once before any arrays. This also lets older PyArrow versions
    // reject an unsupported schema without consuming any data descriptors.
    let py_schema = export_schema(py, &schema)?;
    let import_batch = pa.getattr("RecordBatch")?.getattr("_import_from_c")?;
    let py_batches = PyList::empty(py);
    for batch in batches {
        let data = StructArray::from(batch).to_data();
        let mut array = FFI_ArrowArray::new(&data);
        let address = &mut array as *mut FFI_ArrowArray as usize;
        // Import consumes the array and clears its release callback. If import
        // fails before consumption, Rust releases the remaining ownership.
        py_batches.append(import_batch.call1((address, py_schema.bind(py)))?)?;
    }
    Ok(pa
        .getattr("Table")?
        .call_method1("from_batches", (py_batches, py_schema.bind(py)))?
        .unbind())
}

fn arrow_error(error: ArrowError) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
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
    use arrow::datatypes::TimeUnit;

    use super::*;

    #[test]
    fn invalid_time_type_is_rejected_without_panicking() {
        assert!(export_c_data_type(&DataType::Time32(TimeUnit::Nanosecond)).is_err());
    }
}
