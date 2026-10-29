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

//! Export decoded Spark results using arrow-rs's PyArrow reader conversion.

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::FileWriter;
use arrow::pyarrow::IntoPyArrow;
use arrow::record_batch::{RecordBatch, RecordBatchIterator, RecordBatchReader};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Share compatible value and offset buffers with PyArrow in one reader import.
///
/// Batches have already been collected; this does not stream the network response.
/// Spark result maps are unsorted and result fields are not dictionary-encoded.
/// An empty vector retains the existing empty-schema behavior.
pub(crate) fn record_batches_to_pyarrow(
    py: Python<'_>,
    batches: Vec<RecordBatch>,
) -> PyResult<Py<PyAny>> {
    let schema = batches
        .first()
        .map(|batch| batch.schema())
        .unwrap_or_else(|| Arc::new(Schema::empty()));
    // RecordBatchIterator does not validate schemas. Check before either path,
    // including metadata, so later batches are not relabeled with the first schema.
    if batches.iter().any(|batch| batch.schema_ref() != &schema) {
        return Err(PyValueError::new_err(
            "Arrow record batches have inconsistent schemas",
        ));
    }
    if schema.fields().iter().any(|field| field_needs_ipc(field)) {
        // Arrow 58.3 panics when exporting NUL-containing C strings. Reuse
        // the collected batches for IPC without executing the request again.
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
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        return Ok(py
            .import("pyarrow")?
            .getattr("ipc")?
            .call_method1("open_file", (PyBytes::new(py, &bytes),))?
            .call_method0("read_all")?
            .unbind());
    }

    let reader: Box<dyn RecordBatchReader + Send> = Box::new(RecordBatchIterator::new(
        batches.into_iter().map(Ok),
        schema,
    ));
    Ok(reader.into_pyarrow(py)?.call_method0("read_all")?.unbind())
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
