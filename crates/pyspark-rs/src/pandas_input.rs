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

//! Fused validation and inline encoding of guarded pandas string columns.

use std::sync::Arc;

use arrow::array::{ArrayRef, StringBuilder};
use arrow::datatypes::{DataType as ArrowType, Field, Schema};
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};

use crate::types::PyStructType;

/// The Python eligibility helper authenticates pandas storage and native schema.
/// Validate each object before accepting it: extraction must never call a custom
/// numeric/string conversion hook or turn a pandas missing sentinel into null.
#[pyfunction]
pub(crate) fn _pandas_string_ipc<'py>(
    py: Python<'py>,
    columns: Vec<Bound<'py, PyAny>>,
    schema: PyRef<'py, PyStructType>,
) -> PyResult<Option<Bound<'py, PyBytes>>> {
    if columns.len() != schema.fields.len() {
        return Ok(None);
    }
    // Reject unsupported objects before copying any string data. Otherwise a
    // late subclass would pay for most of an Arrow build before the row path.
    // Keep this pass to exact type/null checks; UTF-8 extraction stays below.
    for (column, field) in columns.iter().zip(&schema.fields) {
        if !matches!(&field.data_type, spark_connect::types::DataType::String { collation }
            if collation == "UTF8_BINARY")
        {
            return Ok(None);
        }
        for value in column.try_iter()? {
            let value = value?;
            if !(value.is_none() && field.nullable || value.is_exact_instance_of::<PyString>()) {
                return Ok(None);
            }
        }
    }
    // A lone surrogate must fall back before allocating partial Arrow buffers.
    // Drop each temporary UTF-8 bytes object immediately rather than retaining
    // an additional full copy of every column. Keep the cheap type pass above
    // so a late subclass does not pay for this UTF-8 traversal.
    for (column, field) in columns.iter().zip(&schema.fields) {
        for value in column.try_iter()? {
            let value = value?;
            if value.is_none() && field.nullable {
                continue;
            }
            let Ok(value) = value.cast_exact::<PyString>() else {
                return Ok(None);
            };
            if value.encode_utf8().is_err() {
                return Ok(None);
            }
        }
    }
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(columns.len());
    let mut fields = Vec::with_capacity(columns.len());
    for (column, field) in columns.iter().zip(&schema.fields) {
        let mut builder = StringBuilder::with_capacity(column.len()?, column.len()? * 8);
        for value in column.try_iter()? {
            let value = value?;
            if value.is_none() && field.nullable {
                builder.append_null();
            } else if let Ok(value) = value.cast_exact::<PyString>() {
                let Ok(bytes) = value.encode_utf8() else {
                    return Ok(None);
                };
                let Ok(text) = std::str::from_utf8(bytes.as_bytes()) else {
                    return Ok(None);
                };
                // append_value copies while the bytes object and GIL are live.
                // This avoids to_cow's owned String on the abi3-py39 build.
                builder.append_value(text);
            } else {
                return Ok(None);
            }
        }
        arrays.push(Arc::new(builder.finish()));
        // Like the row encoder, Spark metadata lives in the logical schema;
        // Arrow fields carry the name, standard string type and nullability.
        fields.push(Field::new(&field.name, ArrowType::Utf8, field.nullable));
    }
    let bytes = py
        .detach(move || -> Result<Vec<u8>, arrow::error::ArrowError> {
            let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?;
            let mut writer = StreamWriter::try_new(Vec::new(), &batch.schema())?;
            writer.write(&batch)?;
            writer.into_inner()
        })
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    Ok(Some(PyBytes::new(py, &bytes)))
}
